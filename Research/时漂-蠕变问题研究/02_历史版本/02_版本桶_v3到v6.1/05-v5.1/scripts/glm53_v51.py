# -*- coding: utf-8 -*-
"""GLM53 v5.1 = v5 + 「取消空载强制归零」（用户要求：算法一直行为，但零点不要被强制拉回 0）。

与 v5（`glm53_v5.py`，含 v3 的"卸载门控自动归零"）的差异只有两处：

  ① **取消空载基线自动归零**：v3/v5 会在空载期用 τ=2s 跟踪基线 b_ 并把它从显示里
     减掉（零点被强制拉回 0）；v5.1 完全移除该链路 —— `b` 不再更新，输出不再减基线，
     空载显示即传感器当前读数。
  ② **扣除量逐通道封顶 扣除 ≤ max(当前读数, 0)**：只封顶输出值，不改任何内部状态
     （g/A/γ/carry 照常演进），避免负值被显示层 `ProcessingPipeline::ApplyThreshold`
     钳成硬 0（那就是"零点快速塌陷进 0"的直接成因）。

**算法行为保持连续**：状态机、pending 冻结、免责期 carry、蠕变积分与扣除照常运行，
不因空载而停扣、不停跑、不重置（这是与"空载带内一律不扣"写法的关键区别）。
"""
import numpy as np

from glm53_v5 import GLM53v5


def _cap(z, ded):
    """扣除量封顶：不得超过当前读数（只封顶输出值，不改内部状态）。"""
    return max(z, 0.0) if ded > z else ded


class GLM53v51(GLM53v5):
    NO_ZERO_BASELINE = True      # 供离线脚本识别

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

        # ── ① 直接电平差判据（近期 0.3s 窗均值）──
        lv_now = self._win_mean(ts - self.LEV_FAST_S, ts) if self._bn > 3 else None
        lev_ok = False
        if self.in_load and lv_now is not None:
            if self.lev_latch is not None:
                lev_ok = abs(lv_now - self.lev_latch) > self.PENDING_RESET * self.lev_thr
                if not lev_ok:
                    self.lev_latch = None
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
            self.in_load = False
            self._align(self.ts_smooth)
            self.hold, self.hold_comp = False, None
            self.pending, self.pending_ts = False, 0.0
            self.fast_done = True
            self.lev_latch = None
        elif u > self.STEP_SUPPRESS and step_conf:
            if idle:
                self.in_load = False
                self._align(self.ts_smooth)
                self.hold, self.hold_comp = False, None
                self.pending, self.pending_ts = False, 0.0
                self.fast_done = True
                self.lev_latch = None
            else:
                self._restep(ts, v)                 # v5.1: 不再减基线
                self._align(self.fast)
        elif self.pending:
            self.hold = True
            self.a_new_acc = self.a_new_acc + v      # v5.1: 不再减基线
            self.a_new_frames += 1
        else:
            self.hold, self.hold_comp = False, None
            if self.a_new_frames > 0:
                self.a_new_acc = np.zeros(self.n)
                self.a_new_frames = 0
        # ① 基线跟踪(b_)整条已移除 —— 空载不再归零

        # ── 免责期（与 v5 相同；输出只做"不超过当前读数"封顶）──
        if self.in_load and not self.fast_done:
            ux = ts - self.onset_ts
            if ux < self.FAST_S:
                self.a_captured = False
                self.a_acc = np.zeros(self.n)
                self.a_frames = 0
                self.g = 0.0
                if self.hold and self.hold_comp is None:
                    self.hold_comp = self.carry.copy()
                Zx = v.copy()
                if ux >= self.FAST_S - self.EXEMPT_AWIN:
                    self.ex_a = self.ex_a + (Zx - self.carry)
                    self.ex_n += 1
                return np.array([Zx[i] - _cap(Zx[i], self.carry[i])
                                 for i in range(self.n)])
            self.fast_done = True
            Ax = (self.ex_a / self.ex_n) if self.ex_n > 0 else (v - self.carry)
            self.A = np.maximum(Ax, 0.0)
            amax = self.A.max()
            self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                else np.zeros(self.n, bool)
            self.a_captured = True
            self.g = 0.0
            m = self.loaded & (self.A > 1e-9)
            if m.any():
                self.g = float(np.clip(
                    np.median(self.carry[m] / (self.gamma[m] * self.A[m])), 0.0, 1.0))
            self.carry = np.zeros(self.n)
            self.ex_a = np.zeros(self.n)
            self.ex_n = 0

        Z = v.copy()
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
            m = self.loaded & (self.A > 1e-9)
            out = Z.copy()
            if m.any():
                out[m] = [Z[i] - _cap(Z[i], self.hold_comp[i])
                          for i in range(self.n) if m[i]]
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
        if m.any():
            out[m] = [Z[i] - _cap(
                Z[i],
                float(np.clip(self.gamma[i] * self.A[i] * self.g,
                              self.CREEP_LO * self.A[i], self.CREEP_HI * self.A[i])))
                for i in range(self.n) if m[i]]
        return out
