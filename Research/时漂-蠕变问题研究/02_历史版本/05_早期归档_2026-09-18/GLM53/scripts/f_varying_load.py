# -*- coding: utf-8 -*-
"""GLM53 分析第六步：变化负载下的混合算法验证与 v2 阶跃感知方案

问题：v1（当前 C++ 产品版）的蠕变场假设幅度 A 恒定——负载中途变化时
     A 失效，补偿把真实负载阶跃当作蠕变扣除（受 clip 限制显示被"钉"在旧幅度），
     部分卸载则被当作完全卸载导致补偿直接丢失。

v2 方案：阶跃感知混合算法
  - 新增双 EMA 阶跃检测器: fast(τ=0.7s) vs slow(τ=6s) 的相对发散 >15%
    （+绝对下限 1% 历史最大电平）判定负载阶跃——蠕变的发散理论上界 ≈6-11%，
    与阶跃可分;
  - 阶跃统一驱动三种迁移: 空载→负载(onset)、负载→负载(重捕获 A)、
    负载→空载(unload, 回到最小电平附近);
  - onset 后 10s 抑制窗避开蠕变快相; u>3s 起允许"回到空载"快速判定;
  - 迁移时快慢 EMA 与电平参考对齐当前值, 发散归零。

本脚本:
  1) 在 变化负载/ 两组数据上运行 raw / v1 / v2, 输出总量+主通道对比图
  2) 在 9 组恒载数据(右/左拇指、四指)上回归: v2 相对 v1 的漂移指标与误触发
输出: figures/e1_varying_*.png, e2_regression.png; results/e_varying_metrics.csv
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # temp
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                    # GLM53
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")

# ── 与 C++ 产品版一致的核心参数 ──────────────────────────────
P = dict(tau_total=0.3, tau_level=10.0, tau_base=2.0, tau_creep=3.0,
         onset_ratio=1.8, unload_ratio=0.5, loaded_frac=0.10,
         gamma_lo=0.3, gamma_hi=2.0, creep_lo=-0.5, creep_hi=1.5,
         g_enable=0.02, a_w0=1.0, a_w1=3.0,
         unlatch_ratio=3.0)  # v1 修正: 锁存释放比率(见 CompV1 注释)
# v2 新增
P2 = dict(P, tau_fast=0.7, tau_slow=6.0,
          onset_rel=0.5,          # 空载→负载: 总量阶跃 ≥50%(高阈免疫近零空载域噪声)
          step_rel=0.18,          # 负载内阶跃: 相对发散阈(蠕变上界 ~11%)
          step_abs_frac=0.01,     # 负载内阶跃: 绝对下限(1% 历史最大电平)
          step_suppress=6.0,      # onset 后抑制窗(避开蠕变快相)
          unload_fast=3.0,        # 快速卸载判定最小 u
          step_persist=2.5,       # 阶跃确认持续时长(瞬态拒绝)
          idle_frac=0.10,         # 卸载判据: 电平参考 10%
          base_gate_frac=0.20)    # 基线跟踪门控: 历史最大电平 20%


class CompV1:
    """当前 C++ 产品版的忠实复刻"""

    def __init__(self, p=P):
        self.p = p
        self.reset()

    def reset(self):
        self.n = 0
        self.init = False
        self.last_ts = 0.0
        self.ts = 0.0
        self.min_ts = 0.0
        self.max_ts = 0.0
        self.level = 0.0
        self.jump_latched = False
        self.drop_latched = False
        self.armed = False
        self.in_load = False
        self.onset = 0.0
        self.b = None
        self.a_acc = None
        self.a_frames = 0
        self.a_captured = False
        self.A = None
        self.loaded = None
        self.g = 0.0
        self.g2 = 0.0
        self.g_rel = None
        self.gamma = None
        self.events = []

    def _reset_for(self, n):
        self.reset()
        self.n = n
        self.b = np.zeros(n)
        self.a_acc = np.zeros(n)
        self.A = np.zeros(n)
        self.g_rel = np.zeros(n)
        self.gamma = np.ones(n)
        self.loaded = np.zeros(n, dtype=bool)

    def _onset(self, ts, tag):
        self.in_load = True
        self.onset = ts
        self.a_captured = False
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
        self.g = 0.0
        self.g2 = 0.0
        self.g_rel = np.zeros(self.n)
        self.gamma = np.ones(self.n)
        self.events.append((ts, tag))

    def process(self, ts, v):
        n = len(v)
        if n <= 0:
            return
        if n != self.n:
            self._reset_for(n)
        if not self.init:
            self.init = True
            self.last_ts = ts
        dt = min(max(ts - self.last_ts, 0.0), 0.1)
        self.last_ts = ts

        p = self.p
        total = float(np.sum(v))
        if dt > 0:
            self.ts += (dt / p["tau_total"]) * (total - self.ts)
        else:
            self.ts = total
        self.min_ts = min(self.min_ts, self.ts)
        self.max_ts = max(self.max_ts, self.ts)
        self.level += (dt / p["tau_level"]) * (self.ts - self.level)
        eps = 1e-6 * (1.0 + abs(self.max_ts))

        jump = self.ts > p["onset_ratio"] * self.level + eps
        if jump and not self.jump_latched:
            self.jump_latched = True
            self._onset(ts, "onset")
        elif not jump:
            self.jump_latched = False

        # 已知缺陷(忠实保留, 作为"当前产品版"行为): 空载总量极小(如 N 域 ~0.15)
        # 时 ts(τ=0.3) 相对 level(τ=10) 持续偏大, jump 锁存可能无法释放,
        # 真实加载 onset 不触发 → 补偿失效; v2 解决。

        u = ts - self.onset if self.in_load else 0.0
        if self.in_load and u > 1.0:
            drop = self.ts < p["unload_ratio"] * self.level - eps
            if drop and not self.drop_latched:
                self.drop_latched = True
                self.in_load = False
                self.armed = True
                self.level = self.ts
                self.events.append((ts, "unload"))
            elif not drop:
                self.drop_latched = False

        if not self.in_load and self.armed and self.ts < 1.5 * self.min_ts + eps:
            self.b += (dt / p["tau_base"]) * (v - self.b)

        Z = v - self.b
        if not self.in_load:
            return Z
        if not self.a_captured:
            if p["a_w0"] <= u <= p["a_w1"]:
                self.a_acc += Z
                self.a_frames += 1
            if u > p["a_w1"]:
                self.A = self.a_acc / max(self.a_frames, 1)
                amax = float(np.max(self.A))
                self.loaded = self.A > p["loaded_frac"] * amax if amax > 1e-9 else np.zeros(n, bool)
                self.a_captured = True
            else:
                return Z

        ld = self.loaded & (self.A > 1e-9)
        rel = (Z[ld] - self.A[ld]) / self.A[ld]
        if rel.size:
            self.g += (dt / p["tau_creep"]) * (float(np.median(rel)) - self.g)
        if self.g > p["g_enable"]:
            self.g2 += dt * self.g * self.g
            full = np.zeros(n)
            full[ld] = rel
            self.g_rel += dt * self.g * full
            if self.g2 > 1e-8:
                g_new = np.ones(n)
                g_new[ld] = np.clip(self.g_rel[ld] / self.g2, p["gamma_lo"], p["gamma_hi"])
                self.gamma = g_new
        out = Z.copy()
        creep = np.zeros(n)
        creep[ld] = np.clip(self.gamma[ld] * self.A[ld] * self.g,
                            p["creep_lo"] * self.A[ld], p["creep_hi"] * self.A[ld])
        out[ld] -= creep[ld]
        return out


class CompV2(CompV1):
    """v2: 阶跃感知（双 EMA 发散检测 + 统一 onset/step/unload 迁移）"""

    def __init__(self):
        super().__init__(p=P2)
        self.fast = 0.0
        self.slow = 0.0

    def reset(self):
        super().reset()
        self.fast = 0.0
        self.slow = 0.0
        self.t0 = None
        self.pend_t = None

    def _reset_for(self, n):
        super()._reset_for(n)

    def _snap(self, val):
        self.level = val
        self.fast = val
        self.slow = val
        self.pend_t = None

    def process(self, ts, v):
        n = len(v)
        if n <= 0:
            return
        if n != self.n:
            self._reset_for(n)
            self.fast = self.slow = 0.0

        p = self.p
        total = float(np.sum(v))
        # 修正: EMA 初始化只在首帧; 后续 dt==0(时间戳重复帧, 本传感器约 3/4)
        # 只跳过状态更新——绝不能快照, 否则 fast/slow 在阶跃处瞬间同时收敛,
        # 发散检测失效 (v1/C++ 产品版存在同源问题, 忠实保留作对照)
        if not self.init:
            self.init = True
            self.last_ts = ts
            self.t0 = ts
            self.ts = self.fast = self.slow = total
            self.min_ts = self.max_ts = total
            dt = 0.0
        else:
            dt = min(max(ts - self.last_ts, 0.0), 0.1)
            self.last_ts = ts
            if dt > 0:
                self.ts += (dt / p["tau_total"]) * (total - self.ts)
                self.fast += (dt / p["tau_fast"]) * (total - self.fast)
                self.slow += (dt / p["tau_slow"]) * (total - self.slow)
        self.min_ts = min(self.min_ts, self.ts)
        self.max_ts = max(self.max_ts, self.ts)
        if dt > 0:
            self.level += (dt / p["tau_level"]) * (self.ts - self.level)
        eps = 1e-6 * (1.0 + abs(self.max_ts))

        # 阶跃检测(双阈值):
        #   空载→负载: onset_rel(0.5×) 高阈——负载总量通常远大于空载偏置,
        #              高阈可免疫近零空载域(N 域空载总量 ~0.15±40%)的噪声;
        #   负载内阶跃: step_rel(0.18×) + 绝对下限(1% 历史最大电平, 此时
        #              max_ts 已是负载量级)——蠕变发散理论上界 ~11%, 与
        #              真实阶跃可分。
        div_onset_thr = p["onset_rel"] * max(self.slow, eps)
        div_step_thr = max(p["step_rel"] * max(self.slow, eps),
                           p["step_abs_frac"] * self.max_ts)
        div = abs(self.fast - self.slow)
        u = (ts - self.onset) if self.in_load else 0.0

        thr_for_state = div_onset_thr if not self.in_load else div_step_thr
        step_now = div > thr_for_state
        # 阶跃确认状态机: 发散须持续 step_persist 秒, 短暂回落视为瞬态(手指调整)
        if step_now:
            if self.pend_t is None:
                self.pend_t = ts
        else:
            if self.pend_t is not None and div < 0.5 * thr_for_state:
                self.pend_t = None
        step_confirmed = (self.pend_t is not None
                          and ts - self.pend_t > p["step_persist"])

        # 卸载判据(电平参考 10% 或 历史最小 1.5×, 任一成立):
        # 电平参考判据解决近零空载域 min_ts 噪声下 1.5×min 失效的问题
        def is_idle():
            return (self.ts < p["idle_frac"] * max(self.level, eps)
                    or self.ts < 1.5 * self.min_ts + eps)

        if not self.in_load:
            if self.t0 is None or ts <= self.t0 + 1e-9:
                # 首帧: 带载使能/连接即受载 → 直接 onset(空载时无害, A=空载
                # 偏置且 g≈0); 除此之外不再使用比率路径(微小空载域会抖动)
                if total > eps:
                    self._onset(ts, "onset")
                    self._snap(self.ts)
            elif step_confirmed:
                # 空载→负载(阶跃路径)
                self._onset(ts, "onset")
                self._snap(self.fast)
        else:
            if u > p["unload_fast"] and is_idle():
                # 快速卸载判定(不依赖发散, 避开抑制窗)
                self.in_load = False
                self.armed = True
                self._snap(self.ts)
                self.events.append((ts, "unload"))
            elif u > p["step_suppress"] and step_confirmed:
                if is_idle():
                    self.in_load = False
                    self.armed = True
                    self._snap(self.ts)
                    self.events.append((ts, "unload"))
                else:
                    # 负载阶跃(加重/减轻): 重捕获幅度
                    self._onset(ts, "restep")
                    self._snap(self.fast)

        # 基线跟踪门控: 空载态 + 已确认循环 + 电平远低于历史最大(≤20%)
        if not self.in_load and self.armed and self.ts < p["base_gate_frac"] * self.max_ts:
            self.b += (dt / p["tau_base"]) * (v - self.b)

        Z = v - self.b
        if not self.in_load:
            return Z
        if not self.a_captured:
            if p["a_w0"] <= u <= p["a_w1"]:
                self.a_acc += Z
                self.a_frames += 1
            if u > p["a_w1"]:
                self.A = self.a_acc / max(self.a_frames, 1)
                amax = float(np.max(self.A))
                self.loaded = self.A > p["loaded_frac"] * amax if amax > 1e-9 else np.zeros(n, bool)
                self.a_captured = True
            else:
                return Z

        ld = self.loaded & (self.A > 1e-9)
        rel = (Z[ld] - self.A[ld]) / self.A[ld]
        if rel.size:
            self.g += (dt / p["tau_creep"]) * (float(np.median(rel)) - self.g)
        if self.g > p["g_enable"]:
            self.g2 += dt * self.g * self.g
            full = np.zeros(n)
            full[ld] = rel
            self.g_rel += dt * self.g * full
            if self.g2 > 1e-8:
                g_new = np.ones(n)
                g_new[ld] = np.clip(self.g_rel[ld] / self.g2, p["gamma_lo"], p["gamma_hi"])
                self.gamma = g_new
        out = Z.copy()
        creep = np.zeros(n)
        creep[ld] = np.clip(self.gamma[ld] * self.A[ld] * self.g,
                            p["creep_lo"] * self.A[ld], p["creep_hi"] * self.A[ld])
        out[ld] -= creep[ld]
        return out


class CompV3(CompV2):
    """v3: v2 + 切换连续性修复（用户实测反馈: 切换负载处显示有大跳变）

    缺陷机制（v2）: 阶跃待确认(persist 2.5s)期间仍用旧段 A/g 补偿新信号,
    rel=(Z-A)/A 爆炸 → g 暴涨 → 补偿把显示反向拖低; restep 确认瞬间
    g/γ/A 清零 → 显示单帧跳回 raw, 跳变量 = pending 期累积的虚假蠕变量。

    修复:
    1) pending 期冻结蠕变补偿: 不积分 g/γ, 补偿保持 pend 起始值
       → 显示立即跟随真实阶跃, 不再被反向拖低;
    2) restep 原子迁移: A 取 pending 期 Z(=v-b) 的逐通道均值并当帧生效
       (无 1~3s 直通黑障); g 按补偿连续性锚定 median(冻结补偿/(γ·A_new)),
       γ 逐通道增益保留 → restep 前后显示连续, 无单帧跳变;
    3) 空载→负载 onset 加绝对下限(1% 历史最大电平) → 免疫近零空载域
       噪声假 onset(数据A 的 35.0/54.3s、数据B 的 34.4s 均为此类);
    4) 抑制窗(6s)保留 v2 值; 因 1)+2), 误触发 restep 变为无害
       (仅重锚 A 且补偿连续, 无显示跳变)。
    """

    def __init__(self):
        super().__init__()
        self.hold = False
        self.hold_comp = None
        self.a_new_acc = None
        self.a_new_frames = 0

    def reset(self):
        super().reset()
        self.hold = False
        self.hold_comp = None
        self.a_new_acc = None
        self.a_new_frames = 0

    def _reset_for(self, n):
        super()._reset_for(n)
        self.a_new_acc = np.zeros(n)

    def process(self, ts, v):
        n = len(v)
        if n <= 0:
            return
        if n != self.n:
            self._reset_for(n)
            self.fast = self.slow = 0.0

        p = self.p
        total = float(np.sum(v))
        if not self.init:
            self.init = True
            self.last_ts = ts
            self.t0 = ts
            self.ts = self.fast = self.slow = total
            self.min_ts = self.max_ts = total
            dt = 0.0
        else:
            dt = min(max(ts - self.last_ts, 0.0), 0.1)
            self.last_ts = ts
            if dt > 0:
                self.ts += (dt / p["tau_total"]) * (total - self.ts)
                self.fast += (dt / p["tau_fast"]) * (total - self.fast)
                self.slow += (dt / p["tau_slow"]) * (total - self.slow)
        self.min_ts = min(self.min_ts, self.ts)
        self.max_ts = max(self.max_ts, self.ts)
        if dt > 0:
            self.level += (dt / p["tau_level"]) * (self.ts - self.level)
        eps = 1e-6 * (1.0 + abs(self.max_ts))

        div = abs(self.fast - self.slow)
        div_onset_thr = p["onset_rel"] * max(self.slow, eps)
        div_step_thr = max(p["step_rel"] * max(self.slow, eps),
                           p["step_abs_frac"] * self.max_ts)
        u = (ts - self.onset) if self.in_load else 0.0

        if self.in_load:
            thr_for_state = div_step_thr
            step_now = div > thr_for_state
        else:
            # 空载→负载: 0.5×slow 高阈 + 绝对下限(1% 历史最大电平) + 方向门
            # (fast>slow, 仅上升沿): 卸载后 fast 快于 slow 回落的 settling
            # 发散(slow>fast)不满足方向门 → 免疫卸载后的假 onset
            thr_for_state = max(div_onset_thr, p["step_abs_frac"] * self.max_ts)
            step_now = div > thr_for_state and self.fast > self.slow
        if step_now:
            if self.pend_t is None:
                self.pend_t = ts
        else:
            if self.pend_t is not None and div < 0.5 * thr_for_state:
                self.pend_t = None
        step_confirmed = (self.pend_t is not None
                          and ts - self.pend_t > p["step_persist"])

        def is_idle():
            return (self.ts < p["idle_frac"] * max(self.level, eps)
                    or self.ts < 1.5 * self.min_ts + eps)

        if not self.in_load:
            if self.t0 is None or ts <= self.t0 + 1e-9:
                if total > eps:
                    self._onset(ts, "onset")
                    self._snap(self.ts)
            elif step_confirmed:
                self._onset(ts, "onset")
                self._snap(self.fast)
            self.hold = False
        else:
            if u > p["unload_fast"] and is_idle():
                self.in_load = False
                self.armed = True
                self._snap(self.ts)
                self.hold = False
                self.hold_comp = None
                self.pend_t = None
                self.events.append((ts, "unload"))
            elif u > p["step_suppress"] and step_confirmed:
                if is_idle():
                    self.in_load = False
                    self.armed = True
                    self._snap(self.ts)
                    self.hold = False
                    self.hold_comp = None
                    self.pend_t = None
                    self.events.append((ts, "unload"))
                else:
                    # restep: 原子迁移, A 取 pending 期 Z 均值, 补偿连续(修复#2)
                    Z = v - (self.b if self.b is not None else 0.0)
                    self.in_load = True
                    self.onset = ts
                    self.a_acc = np.zeros(n)
                    self.a_frames = 0
                    self.A = (self.a_new_acc / max(self.a_new_frames, 1)
                              if self.a_new_frames else Z.copy())
                    amax = float(np.max(self.A))
                    self.loaded = (self.A > p["loaded_frac"] * amax
                                   if amax > 1e-9 else np.zeros(n, bool))
                    self.a_captured = True
                    ld = self.loaded & (self.A > 1e-9)
                    if np.any(ld) and self.hold_comp is not None:
                        ratio = self.hold_comp[ld] / np.maximum(
                            self.gamma[ld] * self.A[ld], 1e-9)
                        self.g = float(np.clip(np.median(ratio), 0.0, 1.0))
                    else:
                        self.g = 0.0
                    # g2/g_rel 累加器保留(不重置): 新段 rel 从 0 起步而 g 为
                    # 携带值, 若重置则首帧 rel/g≈0.3 会把 γ 一帧砸到下限 0.3,
                    # 补偿坍缩(实测 2365→638); 保留旧质量使 γ 平滑延续。
                    self.hold = False
                    self.hold_comp = None
                    self.a_new_acc = np.zeros(n)
                    self.a_new_frames = 0
                    self.pend_t = None
                    self._snap(self.fast)
                    self.events.append((ts, "restep"))
            elif self.pend_t is not None:
                # pending 期: 冻结补偿并积累新电平样本(修复#1)
                self.hold = True
                if self.hold_comp is None and self.a_captured and self.b is not None:
                    ld0 = self.loaded & (self.A > 1e-9)
                    comp0 = np.zeros(n)
                    comp0[ld0] = np.clip(self.gamma[ld0] * self.A[ld0] * self.g,
                                         p["creep_lo"] * self.A[ld0],
                                         p["creep_hi"] * self.A[ld0])
                    self.hold_comp = comp0
                if self.b is not None:
                    self.a_new_acc += v - self.b
                    self.a_new_frames += 1
            else:
                self.hold = False
                self.hold_comp = None
                self.a_new_acc = np.zeros(n)
                self.a_new_frames = 0

        # 基线跟踪门控: 空载态 + 已确认循环 + 电平远低于历史最大(≤20%)
        if not self.in_load and self.armed and self.ts < p["base_gate_frac"] * self.max_ts:
            self.b += (dt / p["tau_base"]) * (v - self.b)

        Z = v - self.b
        if not self.in_load:
            return Z
        if not self.a_captured:
            if p["a_w0"] <= u <= p["a_w1"]:
                self.a_acc += Z
                self.a_frames += 1
            if u > p["a_w1"]:
                self.A = self.a_acc / max(self.a_frames, 1)
                amax = float(np.max(self.A))
                self.loaded = self.A > p["loaded_frac"] * amax if amax > 1e-9 else np.zeros(n, bool)
                self.a_captured = True
            else:
                return Z

        ld = self.loaded & (self.A > 1e-9)
        if self.hold:
            # 惰性初始化: pend 可能在 A 捕获窗内开始, 当帧捕获完成后补算冻结值
            if self.hold_comp is None:
                comp0 = np.zeros(n)
                comp0[ld] = np.clip(self.gamma[ld] * self.A[ld] * self.g,
                                    p["creep_lo"] * self.A[ld], p["creep_hi"] * self.A[ld])
                self.hold_comp = comp0
            out = Z.copy()
            out[ld] -= self.hold_comp[ld]
            return out

        rel = (Z[ld] - self.A[ld]) / self.A[ld]
        if rel.size:
            self.g += (dt / p["tau_creep"]) * (float(np.median(rel)) - self.g)
        if self.g > p["g_enable"]:
            self.g2 += dt * self.g * self.g
            full = np.zeros(n)
            full[ld] = rel
            self.g_rel += dt * self.g * full
            if self.g2 > 1e-8:
                g_new = np.ones(n)
                g_new[ld] = np.clip(self.g_rel[ld] / self.g2, p["gamma_lo"], p["gamma_hi"])
                self.gamma = g_new
        out = Z.copy()
        creep = np.zeros(n)
        creep[ld] = np.clip(self.gamma[ld] * self.A[ld] * self.g,
                            p["creep_lo"] * self.A[ld], p["creep_hi"] * self.A[ld])
        out[ld] -= creep[ld]
        return out


def load_csv(path):
    df = pd.read_csv(path, skiprows=24)
    tc = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy()
    t = t - t[0]
    return t, df[tc].to_numpy(), tc


def find_segment(total):
    thr = 0.15 * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if loaded[0]:
        s = np.r_[0, s]
    if loaded[-1]:
        e = np.r_[e, len(loaded)]
    return list(zip(s, e))


def run_case(comp_cls, X, t):
    comp = comp_cls()
    Y = np.empty_like(X, dtype=float)
    for i in range(len(t)):
        Y[i] = comp.process(t[i], X[i].astype(float))
    return Y, comp


# ══════════ 1. 变化负载数据 ══════════
VARY = ["零负载-切换负载-零负载-再切换负载", "零负载-中途切换负载-零负载-切换负载"]
vary_rows = []
vary_store = {}
for loc in VARY:
    t, X, tc = load_csv(os.path.join(BASE, "变化负载", loc, "device_001_seg000.csv"))
    total = X.sum(axis=1)
    main = int(np.argmax(X.max(axis=0)))
    Y1, c1 = run_case(CompV1, X, t)
    Y2, c2 = run_case(CompV2, X, t)
    Y3, c3 = run_case(CompV3, X, t)
    vary_store[loc] = dict(t=t, X=X, Y1=Y1, Y2=Y2, Y3=Y3, main=main, c1=c1, c2=c2, c3=c3,
                           total=total, tot1=Y1.sum(1), tot2=Y2.sum(1), tot3=Y3.sum(1))
    print(f"[{loc}] dur={t[-1]:.0f}s main=ch{main}")
    print(f"  v1 events: {[(round(ts,1), tag) for ts, tag in c1.events]}")
    print(f"  v2 events: {[(round(ts,1), tag) for ts, tag in c2.events]}")
    print(f"  v3 events: {[(round(ts,1), tag) for ts, tag in c3.events]}")
    # 恒定电平段平坦度(以离线检测的电平段为单元)
    segs = find_segment(total)
    for si, (s0, s1) in enumerate(segs):
        s1e = min(s1, len(t) - 1)
        if t[s1e] - t[s0] < 5:
            continue
        seg = slice(s0, s1e)
        lvl = total[seg].mean()
        for name, tot in [("raw", total), ("v1", vary_store[loc]["tot1"]),
                          ("v2", vary_store[loc]["tot2"]), ("v3", vary_store[loc]["tot3"])]:
            y = tot[seg]
            tt = t[seg] - t[s0]
            k = np.polyfit(tt, y, 1)
            flat = 100 * (y - np.polyval(k, tt)).std() / max(lvl, 1e-9)
            vary_rows.append(dict(dataset=loc, seg=si, t0=round(t[s0], 1), dur=round(t[s1e]-t[s0], 1),
                                  level=round(lvl), algo=name,
                                  drift_pct=100 * (y[-len(y)//10:].mean() - y[:len(y)//10].mean()) / max(lvl, 1e-9),
                                  flatness_pct=flat))
pd.DataFrame(vary_rows).to_csv(os.path.join(RES, "e_varying_metrics.csv"),
                               index=False, encoding="utf-8-sig")

# 图 e1: 变化负载对比
for loc in VARY:
    d = vary_store[loc]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    ax = axes[0]
    ax.plot(d["t"], d["total"], lw=0.5, color="k", label="原始总量")
    for ts_, tag in d["c2"].events:
        ax.axvline(ts_, color="gray", ls=":", lw=0.8)
        ax.text(ts_, d["total"].max() * 0.95, tag, fontsize=7, rotation=90, va="top")
    ax.set_title(f"{loc} · 阵列总量（灰虚线=v2 事件）", fontsize=11)
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("ADC 总和")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(d["t"], d["total"], lw=0.5, color="k", alpha=0.4, label="原始")
    ax.plot(d["t"], d["tot1"], lw=0.7, color="tab:red", label="v1 当前产品版")
    ax.plot(d["t"], d["tot2"], lw=0.7, color="tab:blue", label="v2 阶跃感知")
    ax.plot(d["t"], d["tot3"], lw=0.7, color="tab:green", label="v3 切换连续性修复")
    ax.set_title("补偿后总量对比", fontsize=11)
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("ADC 总和")
    ax.legend(fontsize=8)
    ax = axes[2]
    m = d["main"]
    ax.plot(d["t"], d["X"][:, m], lw=0.5, color="k", alpha=0.4, label="原始")
    ax.plot(d["t"], d["Y1"][:, m], lw=0.7, color="tab:red", label="v1")
    ax.plot(d["t"], d["Y2"][:, m], lw=0.7, color="tab:blue", label="v2")
    ax.plot(d["t"], d["Y3"][:, m], lw=0.7, color="tab:green", label="v3")
    ax.set_title(f"主通道 ch{m} 对比", fontsize=11)
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("ADC")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, f"e1_varying_{'a' if VARY[0]==loc else 'b'}.png"), dpi=140)
    plt.close(fig)

# ══════════ 2. 恒载数据回归（v2 不得劣化、不得误触发 restep） ══════════
CONST = [("右拇指指尖", "数据1"), ("右拇指指尖", "数据2"), ("右拇指指尖", "数据3"),
         ("左拇指指尖", "数据1"), ("左拇指指尖", "数据2"), ("左拇指指尖", "数据3"),
         ("四指指尖", "数据1"), ("四指指尖", "数据2"), ("四指指尖", "数据3")]
reg_rows = []
for loc, name in CONST:
    t, X, tc = load_csv(os.path.join(BASE, loc, name, "device_001_seg000.csv"))
    total = X.sum(axis=1)
    segs = find_segment(total)
    if not segs:
        continue
    s0, s1 = max(segs, key=lambda z: z[1] - z[0])
    main = int(np.argmax(X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)))
    amp = X[s0:s1, main].mean() - X[:s0, main].mean()
    row = dict(dataset=f"{loc}/{name}")
    for tag, cls in [("v1", CompV1), ("v2", CompV2), ("v3", CompV3)]:
        Y, comp = run_case(cls, X, t)
        L = Y[s0:s1, main]
        nL = len(L)
        drift = 100 * (L[-nL // 10:].mean() - L[: nL // 10].mean()) / amp
        resteps = sum(1 for _, tg in comp.events if tg == "restep")
        row[f"{tag}_drift_main"] = round(drift, 1)
        row[f"{tag}_events"] = len(comp.events)
        row[f"{tag}_resteps"] = resteps
    reg_rows.append(row)
rdf = pd.DataFrame(reg_rows)
rdf.to_csv(os.path.join(RES, "e_regression_metrics.csv"), index=False, encoding="utf-8-sig")
print("\n===== 恒载回归（v1 / v2 / v3 主通道漂移%, 误触发 restep 次数） =====")
print(rdf.to_string(index=False))

# 图 e2: 回归柱状图
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
xs = np.arange(len(rdf))
w = 0.26
for ax, col, title in [
        (axes[0], "drift_main", "恒载回归：负载段漂移残余（主通道，%）"),
        (axes[1], "resteps", "恒载回归：误触发重捕获次数（应=0）")]:
    ax.bar(xs - w, rdf[f"v1_{col}"], width=w, color="tab:red", label="v1")
    ax.bar(xs, rdf[f"v2_{col}"], width=w, color="tab:blue", label="v2")
    ax.bar(xs + w, rdf[f"v3_{col}"], width=w, color="tab:green", label="v3")
    ax.set_xticks(xs)
    ax.set_xticklabels([d.replace("指尖/", "\n") for d in rdf["dataset"]], fontsize=6.5)
    ax.set_title(title, fontsize=10)
    if col == "drift_main":
        ax.axhline(0, color="k", lw=0.8)
    ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "e2_regression.png"), dpi=140)
plt.close(fig)

print("\nsaved: e1_varying_a/b.png, e2_regression.png")
print("results: e_varying_metrics.csv, e_regression_metrics.csv")
