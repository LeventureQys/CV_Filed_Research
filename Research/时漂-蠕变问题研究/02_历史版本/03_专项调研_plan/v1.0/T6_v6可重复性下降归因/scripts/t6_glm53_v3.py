# -*- coding: utf-8 -*-
# [T6] 由 temp\v4.1flash\progress\04-v5\scripts\glm53_v3.py 复制并改写 import（T6-00 设置脚本生成）。
# 改写: 0 处 import
# -*- coding: utf-8 -*-
"""GLM53 v3 的 Python 忠实移植（对应 src/domain/drift/drift_compensator.{h,cpp}）。

参数与状态机逐条对应 C++ 原文件，不做任何调整；用于离线对拍与算法对比。
时间轴必须用 timestamp（秒，单调），dt==0 的重复帧按 C++ 语义「只跳过状态更新」。
"""
import numpy as np


class GLM53v3:
    TAU_TOTAL, TAU_FAST, TAU_SLOW, TAU_LEVEL = 0.3, 0.7, 6.0, 10.0
    TAU_BASE, TAU_G = 2.0, 3.0
    ONSET_REL, STEP_REL, STEP_ABS, STEP_PERSIST = 0.5, 0.18, 0.01, 2.5
    STEP_SUPPRESS, UNLOAD_FAST = 6.0, 3.0
    IDLE_FRAC, UNLOAD_MIN_RATIO, BASE_GATE_FRAC, PENDING_RESET = 0.10, 1.5, 0.20, 0.5
    LOADED_FRAC, GAMMA_MIN, GAMMA_MAX = 0.10, 0.3, 2.0
    CREEP_LO, CREEP_HI, G_ENABLE = -0.5, 1.5, 0.02
    A_W0, A_W1 = 1.0, 3.0

    def __init__(self, n):
        self.n = n
        self.b = np.zeros(n)
        self.first = True
        self.last_ts = self.t0 = 0.0
        self.ts_smooth = self.fast = self.slow = self.level_ref = 0.0
        self.min_ts = self.max_ts = 0.0
        self.pending = False
        self.pending_ts = 0.0
        self.armed = self.in_load = False
        self.onset_ts = 0.0
        self.hold = False
        self.hold_comp = None
        self.a_new_acc = np.zeros(n)
        self.a_new_frames = 0
        self.a_captured = False
        self.a_acc = np.zeros(n)
        self.a_frames = 0
        self.A = np.zeros(n)
        self.loaded = np.zeros(n, bool)
        self.g = self.g2 = 0.0
        self.g_rel = np.zeros(n)
        self.gamma = np.ones(n)

    def _eps(self):
        return 1e-6 * (1.0 + abs(self.max_ts))

    def _begin(self, ts):
        self.in_load, self.onset_ts, self.a_captured = True, ts, False
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
        self.g = self.g2 = 0.0
        self.g_rel = np.zeros(self.n)
        self.gamma = np.ones(self.n)

    def _align(self, lv):
        self.level_ref = self.fast = self.slow = lv
        self.pending = False
        self.pending_ts = 0.0

    def _restep(self, ts, z_now):
        self.in_load, self.onset_ts = True, ts
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
        self.A = self.a_new_acc / self.a_new_frames if self.a_new_frames > 0 else z_now.copy()
        amax = self.A.max()
        self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 else np.zeros(self.n, bool)
        self.a_captured = True
        self.g = 0.0
        if self.hold_comp is not None:
            den = self.gamma * self.A
            m = self.loaded & (self.A > 1e-9) & (np.abs(den) > 1e-9)
            if m.any():
                self.g = float(np.clip(np.median(self.hold_comp[m] / den[m]), 0.0, 1.0))
        self.hold, self.hold_comp = False, None
        self.a_new_acc = np.zeros(self.n)
        self.a_new_frames = 0
        self.pending, self.pending_ts = False, 0.0

    def process(self, ts, v):
        v = v.copy()
        total = v.sum()
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
        step_now = (div > thr) if self.in_load else (div > thr and self.fast > self.slow)
        if step_now:
            if not self.pending:
                self.pending, self.pending_ts = True, ts
        elif self.pending and div < self.PENDING_RESET * thr:
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
        elif u > self.STEP_SUPPRESS and step_conf:
            if idle:
                self.in_load, self.armed = False, True
                self._align(self.ts_smooth)
                self.hold, self.hold_comp = False, None
                self.pending, self.pending_ts = False, 0.0
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
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 else np.zeros(self.n, bool)
                self.a_captured = True
            else:
                return Z
        if self.hold:
            if self.hold_comp is None:
                self.hold_comp = np.zeros(self.n)
                m = self.loaded & (self.A > 1e-9)
                self.hold_comp[m] = np.clip(self.gamma[m] * self.A[m] * self.g,
                                            self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
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
                                      np.clip(self.g_rel / self.g2, self.GAMMA_MIN, self.GAMMA_MAX), 1.0)
        out = Z.copy()
        if m.any():
            out[m] = Z[m] - np.clip(self.gamma[m] * self.A[m] * self.g,
                                    self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
        return out


def run_glm53_v3(t, X):
    c = GLM53v3(X.shape[1])
    Y = np.empty_like(X)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y
