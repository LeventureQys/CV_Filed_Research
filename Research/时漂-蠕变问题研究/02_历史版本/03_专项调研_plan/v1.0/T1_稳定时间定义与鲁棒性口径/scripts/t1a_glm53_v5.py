# -*- coding: utf-8 -*-
"""GLM53 v5：针对实测暴露的两个确定性缺陷做的修复版（v3 + 四项改造）。

修复的缺陷（均由 2026-09-17 实测复核定位）：
  P1 负载内变载识别门限被 EMA 分离增益(0.665)悄悄放大：
     门限写作 `div=|fast-slow| > 0.18·slow`，而双 EMA(τf=0.7s/τs=6.0s) 对阶跃的
     分离峰值只有台阶量的 0.665 倍 ⇒ 名义 18% 实际等效 ≈27~29% 电平。
     低于该比例的加载被判为蠕变积分掉，显示被"拉回"原电平（用户实测：10N 上加 2.5N
     时显示停在 1.00×10N；13ffca@27.66s 的 +4672 台阶只透传 41%）。
  P2 restep 重捕获幅度时把"旧载已累积的蠕变"锁进新基线 A：
     `A_new = mean(pending 期 Z)`，于是旧蠕变被当成新基线的一部分，
     此后不再被扣除 ⇒ 每次变载都留下一个 ≈已累积蠕变的永久正偏（实测 +5~8%）。
  P3 快相免责期（v4）只挂在"空载→负载"上（`_restep` 不经 `_begin`），
     与文档 §3.3 描述不符；且免责期内仍按旧路径输出，导致"旧扣除"丢失。
  P4 原型 v4 在免责期内提前 return，整段跳过 v3 状态机 ⇒ 免责期内无法识别卸载。

v5 的四项改造：
  ① 变载检测增加"直接电平差"判据：近期窗(0.4s)均值 vs 参考窗(0.4~3.4s)均值，
     差超过 max(6%×参考电平, 1%×历史最大) 即判为阶跃（与原有 div 判据取"或"，
     只增敏不降敏）。有效最小可识别台阶由 ≈27% 降到 ≈6%（且不被 0.665 放大）。
  ② restep 时把已累积的蠕变从新基线上扣掉：`A_new = mean(pending Z) − hold_comp`，
     g 仍按"补偿连续性"锚定。此后 g_raw 会自然收敛到 旧蠕变/A_new，
     旧蠕变继续被扣除 —— 显示既连续、又不会留下永久正偏。
  ③ 免责期同时挂在 onset 与 restep 上（`_begin` 与 `_restep` 都复位），
     与文档语义一致；且免责期内**保留 epoch 起点已生效的扣除(carry)**，
     输出 = Z − carry，保证变载瞬间显示不跳变。
  ④ 免责期分支插在状态机**之后**（与 `04-C++实现骨架.md` §3.4 的插入点一致），
     状态机（阶跃/卸载/基线跟踪）在免责期内照常运行。

其余部分与 v3 逐字相同。
"""
import numpy as np

from t1a_glm53_v3 import GLM53v3


class GLM53v5(GLM53v3):
    # ── ① 直接电平差变载检测 ──
    # 门限含义明确：|Δ电平| > max(LEV_REL×参考电平, LEV_ABS_FRAC×历史最大)
    # 与 div 判据的差别：div 的分离峰值只有台阶量的 0.665 倍，等效门限被放大到 ≈27%；
    # 这里直接测「近期 0.3s 窗」与「0.3~0.8s 前窗」的电平差，门限就是 5%，
    # 而且**天然区分台阶与蠕变斜坡**：2%/s 的蠕变在 0.5s 内只变 1%（远低于 5%），
    # 砝码台阶是一次性跳变、全额呈现。命中后锁存"变化前参考电平"，只要电平没回到
    # latch±0.5×门限就算仍在阶跃中，避免参考窗自己吃进台阶导致 pending 提前复位
    # （v5 初版用 3s 参考窗时正是这样漏检 20N+5N，并被快相误触发）。
    LEV_FAST_S = 0.3          # 近期窗长度
    LEV_LAG_S = 0.5           # 滞后期（参考窗紧邻近期窗之前）
    LEV_REL = 0.05            # 相对门限：|Δ电平| > 5% × 参考电平
    LEV_ABS_FRAC = 0.01       # 绝对下限：|Δ电平| > 1% × 历史最大电平
    LEV_ARM_S = 3.0           # epoch 起点后多久才武装（=免责期长度，避开快相）

    # ── ③④ 快相免责期 ──
    FAST_S = 3.0              # 免责期长度（2026-09-17 用户指定 5s → 3s）
    EXEMPT_AWIN = 1.0         # 幅度采集窗长度（紧贴免责期末端 → [2.0,3.0]s）

    CAP = 1024                # 电平环形缓冲（≥10s@100Hz）

    def __init__(self, n):
        super().__init__(n)
        self._buf = np.zeros((2, self.CAP))
        self._bn = 0
        self.fast_done = True
        self.carry = np.zeros(n)          # epoch 起点已生效的逐通道扣除
        self.ex_a = np.zeros(n)
        self.ex_n = 0
        self.lev_latch = None             # 锁存的变化前参考电平
        self.lev_thr = 0.0

    # ---------------- 电平环形缓冲与窗口均值 ----------------
    def _push(self, ts, total):
        i = self._bn % self.CAP
        self._buf[0, i] = ts
        self._buf[1, i] = total
        self._bn += 1

    def _win_mean(self, t0, t1):
        n = min(self._bn, self.CAP)
        ts, vs = self._buf[0, :n], self._buf[1, :n]
        m = (ts > t0) & (ts <= t1)
        return float(vs[m].mean()) if m.any() else None

    def _deduction(self):
        """当前逐通道蠕变扣除（与输出口径一致，含限幅）"""
        d = np.zeros(self.n)
        m = self.loaded & (self.A > 1e-9)
        if m.any():
            d[m] = np.clip(self.gamma[m] * self.A[m] * self.g,
                           self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
        return d

    # ---------------- epoch 起点 ----------------
    def _begin(self, ts):
        super()._begin(ts)                # in_load=True, onset_ts, a_captured=False, g/γ 清零
        self.fast_done = False            # ③ 空载→负载也起免责期
        self.carry = np.zeros(self.n)     # 空载态无扣除
        self.ex_a = np.zeros(self.n)
        self.ex_n = 0
        self.lev_latch = None             # 新 epoch：参考窗重新对齐

    def _restep(self, ts, z_now):
        hc = None if self.hold_comp is None else np.array(self.hold_comp, dtype=float)
        super()._restep(ts, z_now)        # v3 原子迁移（A=mean(pending Z)，g 锚定连续）
        if hc is not None and hc.size == self.n:
            # ② 基线扣除旧载已累积的蠕变：A_new = mean(pending Z) − 旧扣除
            self.A = np.maximum(self.A - hc, 0.0)
            amax = self.A.max()
            self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                else np.zeros(self.n, bool)
            m = self.loaded & (self.A > 1e-9)
            if m.any():
                # 分母含 γ：扣除量 = γ·A·g，锚定要让 γ·A·g_new = 旧扣除 hc
                # （v3 的 restep 锚定同样含 γ；漏掉 γ 会在 γ≠1 时引入 (γ−1)·hc 的偏差）
                self.g = float(np.clip(
                    np.median(hc[m] / (self.gamma[m] * self.A[m])), 0.0, 1.0))
            else:
                self.g = 0.0
        self.fast_done = False            # ③ 负载内变载同样起免责期
        self.carry = self._deduction()    # 供免责期保持连续
        self.ex_a = np.zeros(self.n)
        self.ex_n = 0
        self.lev_latch = None

    # ---------------- 主流程（v3 + 四处改动） ----------------
    def process(self, ts, v):
        v = v.copy()
        total = v.sum()
        self._push(ts, total)
        dt = 0.0
        if self.first:
            self.first = False
            self.last_ts = self.t0 = ts
            self.ts_smooth = self.fast = self.slow = total
            self.min_ts = self.max_ts = self.level_ref = total
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            dt = 0.0 if not (dt > 0.0) else min(dt, 0.1)
            if dt > 0.0:
                self.ts_smooth += (dt / self.TAU_TOTAL) * (total - self.ts_smooth)
                self.fast += (dt / self.TAU_FAST) * (total - self.fast)
                self.slow += (dt / self.TAU_SLOW) * (total - self.slow)
                self.level_ref += (dt / self.TAU_LEVEL) * (self.ts_smooth - self.level_ref)
        self.min_ts = min(self.min_ts, self.ts_smooth)
        self.max_ts = max(self.max_ts, self.ts_smooth)
        eps = self._eps()
        div = abs(self.fast - self.slow)
        thr = (max(self.STEP_REL * max(self.slow, eps), self.STEP_ABS * self.max_ts)
               if self.in_load else
               max(self.ONSET_REL * max(self.slow, eps), self.STEP_ABS * self.max_ts))

        # ── ① 直接电平差判据（仅负载内启用；与 div 判据取或）──
        #    命中即锁存"变化前参考电平"，只要电平没回到 latch±0.5×门限就算仍在阶跃中，
        #    从而不会被"参考窗自己吃进台阶"提前复位（这是 v5 初版漏检 20N+5N 的原因）。
        lev_ok = False
        if self.in_load and self._bn > 3:
            lv_now = self._win_mean(ts - self.LEV_FAST_S, ts)
            if lv_now is not None:
                if self.lev_latch is not None:
                    lev_ok = abs(lv_now - self.lev_latch) > self.PENDING_RESET * self.lev_thr
                    if not lev_ok:
                        self.lev_latch = None          # 电平回来了 → 瞬态，取消
                elif (ts - self.onset_ts) > self.LEV_ARM_S:
                    lv_ref = self._win_mean(ts - self.LEV_FAST_S - self.LEV_LAG_S,
                                            ts - self.LEV_FAST_S)
                    if lv_ref is not None:
                        self.lev_thr = max(self.LEV_REL * max(lv_ref, eps),
                                           self.LEV_ABS_FRAC * self.max_ts)
                        if abs(lv_now - lv_ref) > self.lev_thr:
                            self.lev_latch = lv_ref
                            lev_ok = True

        step_now = (div > thr) if self.in_load else (div > thr and self.fast > self.slow)
        if self.in_load:
            step_now = step_now or lev_ok
        if step_now:
            if not self.pending:
                self.pending, self.pending_ts = True, ts
        elif self.pending and (not lev_ok) and div < self.PENDING_RESET * thr:
            self.pending = False
        step_conf = self.pending and (ts - self.pending_ts > self.STEP_PERSIST)
        u = ts - self.onset_ts if self.in_load else 0.0
        idle = (self.ts_smooth < self.IDLE_FRAC * max(self.level_ref, eps) or
                self.ts_smooth < self.UNLOAD_MIN_RATIO * self.min_ts + eps)
        if not self.in_load:
            if abs(ts - self.t0) <= 1e-9:
                if total > eps:
                    self._begin(ts)
                    self._align(self.ts_smooth)
            elif step_conf:
                self._begin(ts)
                self._align(self.fast)
            self.hold = False
        elif u > self.UNLOAD_FAST and idle:
            self.in_load, self.armed = False, True
            self._align(self.ts_smooth)
            self.hold, self.hold_comp = False, None
            self.pending, self.pending_ts = False, 0.0
            self.fast_done = True
            self.lev_latch = None
        elif u > self.STEP_SUPPRESS and step_conf:
            if idle:
                self.in_load, self.armed = False, True
                self._align(self.ts_smooth)
                self.hold, self.hold_comp = False, None
                self.pending, self.pending_ts = False, 0.0
                self.fast_done = True
                self.lev_latch = None
            else:
                self._restep(ts, v - self.b)
                self._align(self.fast)
        elif self.pending:
            self.hold = True
            self.a_new_acc = self.a_new_acc + (v - self.b)
            self.a_new_frames += 1
        else:
            self.hold, self.hold_comp = False, None
            if self.a_new_frames > 0:
                self.a_new_acc = np.zeros(self.n)
                self.a_new_frames = 0
        if (not self.in_load) and self.armed and self.ts_smooth < self.BASE_GATE_FRAC * self.max_ts:
            self.b = self.b + (dt / self.TAU_BASE) * (v - self.b)

        # ═══════════ ③④ 快相免责期（插在状态机之后、扣除计算之前）═══════════
        if self.in_load and not self.fast_done:
            ux = ts - self.onset_ts
            if ux < self.FAST_S:
                # 只冻结"蠕变估计"：不用 v3 的 A 窗、不积分 g；状态机已照常运行。
                # 注意：**不动 g2_acc_/g_rel_acc_/gamma_** —— v3 的 restep 同样刻意保留它们
                # （γ 是器件属性、不是 epoch 属性；重置后任何一次台阶都会把 γ 一帧拉飞）。
                self.a_captured = False
                self.a_acc = np.zeros(self.n)
                self.a_frames = 0
                self.g = 0.0
                if self.hold and self.hold_comp is None:
                    self.hold_comp = self.carry.copy()
                Zx = v - self.b
                if ux >= self.FAST_S - self.EXEMPT_AWIN:
                    self.ex_a = self.ex_a + (Zx - self.carry)
                    self.ex_n += 1
                return Zx - self.carry          # 直通 + 保留 epoch 起点扣除（连续）
            # ── 免责期结束当帧：捕获幅度并锚定 g（保持扣除连续）──
            self.fast_done = True
            Ax = (self.ex_a / self.ex_n) if self.ex_n > 0 else (v - self.b - self.carry)
            self.A = np.maximum(Ax, 0.0)
            amax = self.A.max()
            self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                else np.zeros(self.n, bool)
            self.a_captured = True
            self.g = 0.0
            m = self.loaded & (self.A > 1e-9)
            if m.any():
                # 同样含 γ：让 γ·A·g = carry（免责期内保持的扣除），display 不跳变
                self.g = float(np.clip(
                    np.median(self.carry[m] / (self.gamma[m] * self.A[m])), 0.0, 1.0))
            self.carry = np.zeros(self.n)
            self.ex_a = np.zeros(self.n)
            self.ex_n = 0
            # 不 return：当帧继续走下面的 v3 蠕变补偿逻辑
        # ═══════════════════════════════════════════════════════════════════

        Z = v - self.b
        if not self.in_load:
            return Z
        if not self.a_captured:
            if self.A_W0 <= u <= self.A_W1:
                self.a_acc = self.a_acc + Z
                self.a_frames += 1
            if u > self.A_W1:
                self.A = self.a_acc / self.a_frames if self.a_frames > 0 else Z.copy()
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                    else np.zeros(self.n, bool)
                self.a_captured = True
            else:
                return Z
        if self.hold:
            if self.hold_comp is None:
                self.hold_comp = self._deduction()
            out = Z.copy()
            m = self.loaded & (self.A > 1e-9)
            out[m] = Z[m] - self.hold_comp[m]
            return out
        m = self.loaded & (self.A > 1e-9)
        if m.any():
            g_raw = float(np.median((Z[m] - self.A[m]) / self.A[m]))
            self.g += (dt / self.TAU_G) * (g_raw - self.g)
        if self.g > self.G_ENABLE:
            self.g2 += dt * self.g * self.g
            if m.any():
                self.g_rel[m] += dt * self.g * ((Z[m] - self.A[m]) / self.A[m])
            if self.g2 > 1e-8:
                self.gamma = np.where(self.loaded,
                                      np.clip(self.g_rel / self.g2, self.GAMMA_MIN, self.GAMMA_MAX),
                                      1.0)
        out = Z.copy()
        m = self.loaded & (self.A > 1e-9)
        if m.any():
            out[m] = Z[m] - np.clip(self.gamma[m] * self.A[m] * self.g,
                                    self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
        return out
