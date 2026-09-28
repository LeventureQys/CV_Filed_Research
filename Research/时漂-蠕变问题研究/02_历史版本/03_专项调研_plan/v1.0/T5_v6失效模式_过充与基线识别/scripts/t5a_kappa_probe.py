# -*- coding: utf-8 -*-
"""**κ 上限探针补丁**（`t5a_kappa_probe.py`）—— 为「κ 是单侧 cap，不是增益」提供可测的触发信息。

κ 在原型里只在**一处**生效（`t5a_glm53_v6.py` / `t5a_glm53_v61.py` 的 `_inv_est` 结尾）：
    `return float(min(A, kappa * inc)), True`
即反演目标 `Â` 被封顶在 `κ·inc`（`inc` = 当前实测增量），
**只有未封顶反演值 `A_raw > κ·inc` 时 κ 才起作用**。

本模块提供 `KV6P` / `KV61P`：在 `t5a_common.KV6 / KV61` 之上覆写**各自那一份** `_inv_est`
（逐行复制原型、只在返回前多记三个数），**不改变任何返回值与数值路径**：
    `A_raw`       —— 电平域最小二乘结果（含原型自带的 `max(A, inc)` 下限，封顶前）
    `inc_probe`   —— 该次调用里的 `inc`
    `kappa_probe` —— 本次调用收到的 κ
逐帧落盘只在 `enable_probe=True` 时开。
**补丁零差证明**：`t5a_patch_ab_zero.csv` 的 V6/V7 行（探针开 vs 关，逐帧输出 0 差）。

用法：
    from t5a_kappa_probe import KV6P
    d = KV6P(n).run(tu, X)      # d 里多出 A_raw / inc_probe / kappa_probe / kind_probe
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                       # noqa: E402
from t5a_glm53_v6 import g_shape                             # noqa: E402


class _ProbeBase:
    """公共：逐帧落盘 + `run()` 包装。数值路径一行不改。"""

    enable_probe = True

    def __init__(self, n):
        super().__init__(n)
        self.probe_A_raw = np.nan
        self.probe_inc = np.nan
        self.probe_kappa = np.nan
        self._p_A_raw = None
        self._p_inc = None
        self._p_kappa = None
        self._p_kind = None
        self._p_i = 0

    def _mark(self, A_raw, inc, kappa):
        self.probe_A_raw = float(A_raw)
        self.probe_inc = float(inc)
        self.probe_kappa = float(kappa)

    def process(self, ts, v):
        self.KAPPA_ONSET = self.kappa_onset
        self.KAPPA_RESTEP = self.kappa_restep
        out = super().process(ts, v)
        if self.enable_probe and self._p_A_raw is not None and self._p_i < len(self._p_A_raw):
            i = self._p_i
            self._p_A_raw[i] = self.probe_A_raw
            self._p_inc[i] = self.probe_inc
            self._p_kappa[i] = self.probe_kappa
            self._p_kind[i] = (self.ev["kind"] if self.ev is not None else "")
            self._p_i += 1
        return out

    def run(self, tu, X):
        n = len(tu)
        if self.enable_probe:
            self._p_A_raw = np.full(n, np.nan)
            self._p_inc = np.full(n, np.nan)
            self._p_kappa = np.full(n, np.nan)
            self._p_kind = np.array([""] * n, dtype=object)
            self._p_i = 0
        d = super().run(tu, X)
        d["A_raw"] = self._p_A_raw
        d["inc_probe"] = self._p_inc
        d["kappa_probe"] = self._p_kappa
        d["kind_probe"] = self._p_kind
        return d


class KV6P(_ProbeBase, C.KV6):
    """v6 + 探针：`_inv_est` 逐行复制 `GLM53v6._inv_est`，只在返回前 `_mark`。"""

    def _inv_est(self, hist, tau, kappa):
        if len(hist) < 4 or tau < self.TAU_REF:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        tt = np.array([h[0] for h in hist])
        yy = np.array([h[1] for h in hist])
        m = (tt >= self.TAU_REF) & (tt <= min(tau, self.TAU_REF + self.AWIN))
        if m.sum() < 5:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        g = g_shape(tt[m])
        den = float((g * g).sum())
        if den < 1e-12:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        inc = float(yy[-1])
        if inc <= 0.0:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        A = float((yy[m] * g).sum()) / den
        A = max(A, inc)
        self._mark(A, inc, kappa)
        return float(min(A, kappa * inc)), True


class KV61P(_ProbeBase, C.KV61):
    """v6.1 + 探针：`_inv_est` 逐行复制 `GLM53v61._inv_est`（用 g61 形状），只多 `_mark`。"""

    def _inv_est(self, hist, tau, kappa):
        from t5a_glm53_v61 import g61_shape
        if len(hist) < 4 or tau < self.TAU_REF:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        tt = np.array([h[0] for h in hist])
        yy = np.array([h[1] for h in hist])
        m = (tt >= self.TAU_REF) & (tt <= min(tau, self.TAU_REF + self.AWIN))
        if m.sum() < 5:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        scale = self._rom_scale(self.ev["kind"] if self.ev is not None else "onset")
        g = g61_shape(tt[m], scale)
        den = float((g * g).sum())
        if den < 1e-12:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        inc = float(yy[-1])
        if inc <= 0.0:
            self._mark(np.nan, np.nan, kappa)
            return 0.0, False
        A = float((yy[m] * g).sum()) / den
        A = max(A, inc)
        self._mark(A, inc, kappa)
        return float(min(A, kappa * inc)), True


def run_probe(Cls, tu, X, kappa_onset=1.30, kappa_restep=1.12, enable_probe=True, **kw):
    c = Cls(X.shape[1])
    c.kappa_onset, c.kappa_restep = float(kappa_onset), float(kappa_restep)
    c.enable_probe = bool(enable_probe)
    for k, v in kw.items():
        setattr(c, k, v)
    return c.run(tu, X)
