# -*- coding: utf-8 -*-
"""在线双态蠕变观测器 K9 —— Python 实现（按《抗时漂 v3.4 算法代码说明与移植指南》§6 逐帧伪码）。

口径声明
--------
* 本文件是 **指南 §6 伪码的忠实实现**，逐帧对齐 C++ `src/domain/drift_v6/creep_observer.cpp`
  （K9 版，param_set = plan-v4 observer-k9）。状态量、参数名、执行顺序、限幅位置一一对应。
* **本文不含**抗下漂变体（D1-A/B/C，见《抗时漂 v3.4 抗下漂变体设计（草案D1）》）：
  先把基线跑出来，再决定是否叠加变体。
* 唯一外部依赖：numpy。数值用 float64（与 C++ 的 double 同口径）。

逐帧执行顺序（不可调换，对应 creep_observer.cpp 行号）
-------------------------------------------------------
  首帧初始化            :50-67
  dt 截断 + K7 全局旁路  :69-94
  ① 去趋势空载门/包络/零点 :102-115
  ② 低通导数 slope       :118-121
  ③ 沿检测 t_edge        :126-130
  ④ 缓坡计时 ramp_dwell  :131-135
  ⑤ 快变清 dwell         :136-138
  ⑥ tc1（沿后前馈+缓坡混合）:139-146
  ⑦ 快态 x1 / v_lp 更新   :147-152
  ⑧ 慢态 x2（软冻结/dwell/快泄放）:153-181
  ⑨ K6 预留池 applied     :182-199
  输出（旁路帧/正常帧）    :202-211
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass
class Params:
    """27 个参数，默认值 = 现役值（与 creep_observer.h::Params 逐项一致）。"""

    # ---- v3.4 基线：快态 x1 ----
    r_fast: float = 0.10            # 幅度比：终值 = r_fast*e（闭环吃掉载荷 9.09%）
    tau_c_fast_s: float = 12.0      # 受载收敛 τ；e>0 时 x1 朝 e 收敛，故它也是变载后残余释放的 τ
    tau_r_fast_s: float = 2.0       # 空载恢复 τ，仅 e_now<=0 生效
    r_slow_max: float = 0.35        # 慢态上限（相对 e）
    tau_r_slow_s: float = 150.0     # 非 idle 恢复 τ，仅 e<=0 且未进近零带
    y_max_tau_s: float = 600.0      # 量程包络衰减 τ

    # ---- v3.4 基线：慢态 x2 ----
    tau_slope_s: float = 1.0        # 低通导数 τ
    slope_gate_frac: float = 0.05   # 沿门：|slope| < frac*max(e,1) 才积分
    slope_cap_frac: float = 0.01    # 积分速率上限（1%*e/s）
    tau_zero_s: float = 8.0         # 零点跟踪 τ（仅近零带内）
    idle_frac: float = 0.05         # 近零带宽（去趋势口径）

    # ---- H3：沿检测 / 前馈 / 软冻结 ----
    edge_slope_thres: float = 60.0  # 沿阈（|slope|，双向使用）
    edge_refract_s: float = 2.0     # 沿不应期
    edge_boost_s: float = 2.0       # 沿后前馈窗
    tau_c_fast_boost_s: float = 2.0  # 前馈窗内快态 τ
    soft_unfreeze_s: float = 4.0    # x2 软冻结窗（1-exp(-t/w)>0.5 ⟺ t > w*ln2 = 2.773s）

    # ---- K6：补偿预留池 ----
    hold_eps: float = 2.0           # 限额余量（ADC/s），≤0 即关闭预留池
    hold_tau_s: float = 0.5         # 限额用的快速低通 τ

    # ---- K7：慢态门控 + 全局旁路 ----
    slow_confirm_s: float = 10.0    # x2 需连续受载确认时长
    y_floor_tau_s: float = 300.0    # 去趋势基线【上行】跟踪 τ（下行瞬时跟随）
    tau_r_slow_idle_s: float = 8.0  # idle 期 x2 快泄放 τ（≤0 回落 tau_r_slow_s）
    bypass_release_frac: float = 1.15   # 直通判据（只由它决定）
    bypass_engage_frac: float = 1.25    # 只门控"是否继续跟踪基线/噪声"
    bypass_noise_sigma: float = 3.0
    bypass_base_tau_s: float = 10.0

    # ---- K9：缓坡前馈（本轮增量） ----
    ramp_slope_min: float = 0.5     # 缓坡带下限（ADC/s）
    ramp_full_s: float = 4.0        # 缓坡满该时长后 tc1 全额过渡到 boost

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


LN2 = 0.69314718056
T_EDGE_INITIAL = 1.0e6


class CreepObserverK9:
    """K9 在线双态蠕变观测器（单目标；多目标由外层按 key/签名隔离）。"""

    def __init__(self, params: Params | None = None) -> None:
        self.p = params or Params()
        self._n = 0
        self._first = True
        self._last_ts = 0.0
        # 诊断记录（不影响算法）
        self.trace: dict[str, list] = {}

    # ---------------- 生命周期 ----------------
    def _reset_for(self, n: int) -> None:
        self._n = n
        self._first = True
        self._last_ts = 0.0
        z = np.zeros(n, dtype=np.float64)
        self.x_fast = z.copy()
        self.x_slow = z.copy()
        self.applied = z.copy()
        self.load_dwell = z.copy()
        self.ramp_dwell = z.copy()
        self.t_edge = np.full(n, T_EDGE_INITIAL, dtype=np.float64)
        # 以下 5 个由首帧分支赋值（与 C++ 一致：ResetFor 不构造它们）
        self.v_lp = z.copy()
        self.zero = z.copy()
        self.y_max = z.copy()
        self.v_fast_lp = z.copy()
        self.y_floor = z.copy()
        self.total_baseline = 0.0
        self.total_noise = 0.0
        self.trace = {}

    def _trace(self, key: str, arr) -> None:
        self.trace.setdefault(key, []).append(np.array(arr, dtype=np.float64, copy=True))

    # ---------------- 主流程 ----------------
    def process(self, timestamp_s: float, values: np.ndarray) -> np.ndarray:
        """就地风格：返回补偿后的显示值（不修改入参）。"""
        v = np.array(values, dtype=np.float64, copy=True)
        n = v.size
        if n <= 0:
            return v

        if self._n != n:
            self._reset_for(n)

        p = self.p

        # ── 首帧 :50-67 ──
        if self._first:
            self._first = False
            self._last_ts = float(timestamp_s)
            self.zero = v.copy()
            self.y_max = np.zeros(n)
            self.v_lp = v.copy()
            self.v_fast_lp = v.copy()
            self.y_floor = v.copy()
            self.load_dwell = np.zeros(n)
            self.ramp_dwell = np.zeros(n)
            self.t_edge = np.full(n, T_EDGE_INITIAL)
            self.applied = np.zeros(n)
            self.total_baseline = float(np.sum(v)) if np.isfinite(np.sum(v)) else 0.0
            self.total_noise = 0.0
            out = v - self.applied
            self._trace_frame(v, out, False)
            return out

        # ── 时间步 :69-71 ──
        dt = float(timestamp_s) - self._last_ts
        self._last_ts = float(timestamp_s)
        dt = min(dt, 0.1) if dt > 0.0 else 0.0

        # ── K7 全局总值旁路 :78-94（dt>0 才更新统计；直通判据只用 release）──
        bypass = False
        if dt > 0.0:
            total = float(np.sum(v))
            if np.isfinite(total):
                alpha = min(dt / p.bypass_base_tau_s, 1.0)
                release = max(p.bypass_release_frac * self.total_baseline,
                              self.total_baseline + p.bypass_noise_sigma * self.total_noise)
                engage = max(p.bypass_engage_frac * self.total_baseline,
                             self.total_baseline + (p.bypass_noise_sigma + 1.0) * self.total_noise)
                if total < engage:
                    self.total_baseline += alpha * (total - self.total_baseline)
                    self.total_noise += alpha * (abs(total - self.total_baseline) - self.total_noise)
                    if self.total_noise < 0.0:
                        self.total_noise = 0.0
                bypass = bool(total < release)

        if dt > 0.0:
            y_decay = float(np.exp(-dt / p.y_max_tau_s))
            y = v - self.zero
            slope = (v - self.v_lp) / p.tau_slope_s
            e_now = np.maximum(y - self.x_fast - self.x_slow, 0.0)
            self._slope = slope
            self._e_now = e_now

            for k in range(n):
                # ① 去趋势空载门 + 包络 + 零点跟踪
                y0 = v[k] - self.zero[k]
                if y0 < self.y_floor[k]:
                    self.y_floor[k] = y0
                else:
                    self.y_floor[k] += (dt / p.y_floor_tau_s) * (y0 - self.y_floor[k])
                self.y_max[k] = max(self.y_max[k] * y_decay, max(y0, 0.0))
                span = max(self.y_max[k] - self.y_floor[k], 1.0)
                idle_k = (y[k] - self.y_floor[k]) < p.idle_frac * span
                self.load_dwell[k] = 0.0 if idle_k else self.load_dwell[k] + dt
                if idle_k:
                    self.zero[k] += (dt / p.tau_zero_s) * (v[k] - self.zero[k])

                # ③ 沿检测（先 t_edge += dt，再判触发）
                self.t_edge[k] += dt
                if slope[k] > p.edge_slope_thres and \
                        self.t_edge[k] > max(p.edge_refract_s, p.edge_boost_s):
                    self.t_edge[k] = 0.0

                # ④ 缓坡计时（正斜率带内累加，出带清零）
                ramp_on = (slope[k] > p.ramp_slope_min) and (slope[k] <= p.edge_slope_thres)
                self.ramp_dwell[k] = self.ramp_dwell[k] + dt if ramp_on else 0.0

                # ⑤ 任一快变（升或降）重计受载确认窗
                if abs(slope[k]) > p.edge_slope_thres:
                    self.load_dwell[k] = 0.0

                # ⑥ 快态收敛 τ：沿后前馈 → 缓坡线性混合
                tc1 = p.tau_c_fast_boost_s \
                    if (p.edge_boost_s > 0.0 and self.t_edge[k] < p.edge_boost_s) \
                    else p.tau_c_fast_s
                if p.ramp_full_s > 0.0:
                    w = min(self.ramp_dwell[k] / p.ramp_full_s, 1.0)
                    tc1 = tc1 * (1.0 - w) + p.tau_c_fast_boost_s * w

                # ⑦ 快态 x1（斜率口径：用更新前的 x1 与 e_now）
                dx1_rate = (p.r_fast * e_now[k] - self.x_fast[k]) / tc1 \
                    if e_now[k] > 0.0 else -self.x_fast[k] / p.tau_r_fast_s
                self.x_fast[k] = max(self.x_fast[k] + dx1_rate * dt, 0.0)
                self.v_lp[k] += (dt / p.tau_slope_s) * (v[k] - self.v_lp[k])

                # ⑧ 慢态 x2（e 用本帧更新后的 x1 重算）
                e = max(y[k] - self.x_fast[k] - self.x_slow[k], 0.0)
                if e > 0.0:
                    base = max(e, 1.0)
                    cap = p.slope_cap_frac * base
                    soft_ok = (1.0 - float(np.exp(-self.t_edge[k] / p.soft_unfreeze_s))) > 0.5
                    dwell_ok = self.load_dwell[k] >= p.slow_confirm_s
                    if dwell_ok and soft_ok and abs(slope[k]) < p.slope_gate_frac * base:
                        dx2 = min(max(slope[k] - dx1_rate, -cap), cap)
                        self.x_slow[k] += dt * dx2
                    hi = p.r_slow_max * base
                    if self.x_slow[k] > hi:
                        self.x_slow[k] = hi
                    if self.x_slow[k] < 0.0:
                        self.x_slow[k] = 0.0
                if e <= 0.0 or idle_k:
                    tau_r2 = p.tau_r_slow_idle_s \
                        if (idle_k and p.tau_r_slow_idle_s > 0.0) else p.tau_r_slow_s
                    self.x_slow[k] -= (dt / tau_r2) * self.x_slow[k]
                    if self.x_slow[k] < 0.0:
                        self.x_slow[k] = 0.0

                # ⑨ K6 预留池
                if p.hold_eps > 0.0:
                    slope_a = (v[k] - self.v_fast_lp[k]) / p.hold_tau_s
                    self.v_fast_lp[k] += (dt / p.hold_tau_s) * (v[k] - self.v_fast_lp[k])
                    d_des = self.x_fast[k] + self.x_slow[k]
                    a_up = self.applied[k] + dt * max(slope_a + p.hold_eps, 0.0)
                    a_dn = self.applied[k] + dt * (slope_a - p.hold_eps)
                    a = min(d_des, a_up) if d_des > self.applied[k] else max(d_des, a_dn)
                    if e > 0.0:
                        a = min(max(a, 0.0), max(d_des, 0.0))
                    else:
                        a = d_des
                    self.applied[k] = a

        # ── 输出 :202-211 ──
        if bypass:
            self.applied = self.x_fast + self.x_slow   # 状态已更新，只是不施加
            out = v.copy()
        elif p.hold_eps > 0.0:
            out = v - self.applied
        else:
            out = v - self.x_fast - self.x_slow

        self._trace_frame(v, out, bypass)
        return out

    def _trace_frame(self, v: np.ndarray, out: np.ndarray, bypass: bool) -> None:
        """每帧固定记录同样的键（保证 trace 行数与帧数严格对齐，不受 dt<=0 分支影响）。"""
        self._trace("total_in", v)
        self._trace("total_out", out)
        self._trace("bypass", np.array([bypass], dtype=bool))
        self._trace("x_fast", self.x_fast)
        self._trace("x_slow", self.x_slow)
        self._trace("applied", self.applied)
        self._trace("zero", self.zero)
        self._trace("y_floor", self.y_floor)
        self._trace("y_max", self.y_max)
        self._trace("dwell", self.load_dwell)
        self._trace("t_edge", self.t_edge)
        self._trace("ramp_dwell", self.ramp_dwell)
        self._trace("slope", getattr(self, "_slope", np.zeros(self._n)))
        self._trace("e_now", getattr(self, "_e_now", np.zeros(self._n)))

    # ---------------- 诊断：把 trace 变成 ndarray ----------------
    def traces(self) -> dict[str, np.ndarray]:
        out = {}
        for key, seq in self.trace.items():
            if not seq:
                continue
            if key == "bypass":
                out[key] = np.array([bool(x[0]) for x in seq], dtype=bool)
            else:
                out[key] = np.vstack(seq)
        return out
