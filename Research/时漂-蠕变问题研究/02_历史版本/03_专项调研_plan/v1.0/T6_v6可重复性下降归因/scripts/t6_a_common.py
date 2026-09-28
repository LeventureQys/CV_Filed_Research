# -*- coding: utf-8 -*-
# [T6] 由 temp\v4.1flash\progress\13-v6-assessment\scripts\a_common.py 复制并改写 import（T6-00 设置脚本生成）。
# 改写: 4 处 import
# -*- coding: utf-8 -*-
"""a_* 公共工具：数据载入 / v6 与 v5.1 的仪表化运行 / 扰动注入。

命名前缀 a_ = 13-v6-assessment 的 SubAgent「快速扰动下基线识别」分支，不与其它并行工作冲突。
只读取原型类并子类化，不修改 temp/ 下任何既有文件。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))   # 仓库根
if HERE not in sys.path:
    sys.path.insert(0, HERE)          # 只从本任务 scripts/ 取模块，不依赖既有桶

from t6_ad_lib import load_rec, med_smooth          # noqa: E402
from t6_glm53_v6 import GLM53v6                     # noqa: E402
from t6_glm53_v51 import GLM53v51                   # noqa: E402

REC = {
    "切换负载-快相无责": os.path.join(
        ROOT, "temp", "变化负载", "切换负载-快相无责的测试",
        "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"),
    "零负载-切换负载-零负载-再切换负载": os.path.join(
        ROOT, "temp", "变化负载", "零负载-切换负载-零负载-再切换负载",
        "device_001_seg000.csv"),
    "零负载-中途切换负载-零负载-切换负载": os.path.join(
        ROOT, "temp", "变化负载", "零负载-中途切换负载-零负载-切换负载",
        "device_001_seg000.csv"),
    "中途切换-最终测试目标": os.path.join(
        ROOT, "temp", "变化负载", "零负载-中途切换负载-零负载-切换负载",
        "最终测试目标", "device_001_seg000.csv"),
}


def load_uniform(path, fs=100.0, tmax=None):
    """读到统一 100 Hz 网格（原型与 C++ 的实际链路口径 ~100 Hz）。"""
    t, X = load_rec(path)
    if tmax is not None:
        m = t <= tmax
        t, X = t[m], X[m]
    span = t[-1] - t[0]
    dtm = 1.0 / fs
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return dict(t=t, X=X, tu=tu, Xu=Xu, dtm=dtm, span=span)


# ───────────────────────── 扰动注入 ─────────────────────────

def _bandlimited(n, fs, f_lo, f_hi, rng):
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    W[(f < f_lo) | (f > f_hi)] = 0.0
    x = np.fft.irfft(W, n)
    s = x.std()
    return x / s if s > 1e-12 else x


def make_perturb(X, A, kind, rng, f_lo=0.3, f_hi=40.0, corr=0.0):
    """加性扰动矩阵，**总量扰动的 RMS = A (ADC)**。

    kind: 'white' 每通道白噪声 | 'band' 带限噪声 [f_lo,f_hi] |
          'common' 同相（按通道平均电平分摊，总量 RMS = A）
    corr: 白噪声的通道间相关比例（1 = 完全同相）
    """
    n, n_ch = X.shape
    if kind == "common":
        s = rng.standard_normal(n)
        s = s / s.std()
        base = np.abs(X).mean(axis=0)
        w = base / base.sum() if base.sum() > 1e-9 else np.ones(n_ch) / n_ch
        E = np.outer(s, w)
    else:
        E = np.zeros((n, n_ch))
        for c in range(n_ch):
            E[:, c] = (rng.standard_normal(n) if kind == "white"
                       else _bandlimited(n, 100.0, f_lo, f_hi, rng))
        if corr > 0:
            sc = rng.standard_normal(n)
            sc = sc / sc.std()
            E = (1.0 - corr) * E + corr * np.outer(sc, np.ones(n_ch))
    E = E - E.mean(axis=0, keepdims=True)
    tot = E.sum(axis=1)
    rms = tot.std()
    return np.zeros_like(E) if rms < 1e-12 else E * (A / rms)


def _profile(X, i0, ch_mask):
    prof = np.where(ch_mask > 0, np.maximum(np.abs(X[i0]), 1.0), 0.0)
    return prof / prof.sum() if prof.sum() > 1e-9 else np.ones(X.shape[1]) / X.shape[1]


def add_tap(X, i0, amp, fs=100.0, rise_ms=50.0, hold_ms=100.0, fall_ms=None,
            ch_mask=None, shape="tap"):
    """拍击形态：上升沿 rise + 保持 hold + 回落 fall；总量扰动峰值 = amp。"""
    Y = X.copy()
    if fall_ms is None:
        fall_ms = rise_ms
    nr = max(1, int(round(rise_ms / 1000.0 * fs)))
    nh = max(1, int(round(hold_ms / 1000.0 * fs)))
    nf = max(1, int(round(fall_ms / 1000.0 * fs)))
    up = amp * (1.0 - np.cos(np.linspace(0, np.pi, nr))) / 2.0
    hd = np.full(nh, amp)
    dn = amp * (1.0 + np.cos(np.linspace(0, np.pi, nf))) / 2.0
    w = np.concatenate([up, hd, dn])
    i1 = min(len(X), i0 + len(w))
    w = w[: i1 - i0]
    if ch_mask is None:
        ch_mask = np.ones(X.shape[1])
    Y[i0:i1] += np.outer(w, _profile(X, i0, ch_mask))
    return Y, nr


def add_step(X, i0, amp, fs=100.0, rise_ms=50.0, ch_mask=None):
    """一次真实 restep：同样的上升沿但**不回落**。"""
    Y = X.copy()
    nr = max(1, int(round(rise_ms / 1000.0 * fs)))
    w = amp * (1.0 - np.cos(np.linspace(0, np.pi, nr))) / 2.0
    n = len(X)
    ramp = np.concatenate([w, np.full(max(0, n - i0 - nr), amp)])[: n - i0]
    if ch_mask is None:
        ch_mask = np.ones(X.shape[1])
    Y[i0:] += np.outer(ramp, _profile(X, i0, ch_mask))
    return Y


# ───────────────────────── 仪表化运行 ─────────────────────────

def make_traced(cls):
    """子类化原型，只记录状态变化，不改变行为。"""
    class T(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.tr_epoch = []     # (t0, kind, t_det, base, v0sum, y0sum)
            self.tr_revoke = []    # (ts, kind, tau)
            self.tr_handoff = []   # (ts, kind, A_hat, c_applied, A_sum, g)
            self.tr_unload = []    # (ts,)
            self.tr_gevent = []    # (ts, kind, d_like) 反向事件
            self.tr_deny = []      # (ts,) 检测命中但未被接受（冷却/静默/backdate 失败）

        def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
            super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
            self.tr_epoch.append((float(t0), str(kind), float(self.last_ts),
                                  float(base), float(np.sum(v0)), float(np.sum(y0))))

        def _handoff(self, ts, v, ev=None):
            ev = ev if ev is not None else self.ev
            rec = (float(ts), str(ev["kind"]), float(ev["A_hat"]), float(ev["c_applied"]))
            super()._handoff(ts, v, ev)
            self.tr_handoff.append(rec + (float(np.sum(self.A)), float(self.g)))

        def _to_idle(self):
            super()._to_idle()
            self.tr_unload.append(float(self.last_ts))

        def _run_event(self, ts, v, total, dt, eps, idle_now):
            ev = self.ev
            kind = str(ev["kind"]) if ev is not None else None
            n_rev = self.n_revoke
            e0 = None
            if ev is not None and kind == "decrease":
                e0 = float(ev["t0"])
            out = super()._run_event(ts, v, total, dt, eps, idle_now)
            if kind == "decrease" and self.n_revoke > n_rev:
                self.tr_gevent.append((float(ts), kind, e0))
            return out

        def process(self, ts, v):
            epoch_n = len(self.tr_epoch)
            rev_n = self.n_revoke
            prev_ts = self.last_ts
            out = super().process(ts, v)
            if self.n_revoke > rev_n and len(self.tr_epoch) == epoch_n:
                # 正事件撤销发生在 _run_event 内（无新增 epoch）
                self.tr_revoke.append((float(ts), "revoke"))
            return out
    return T


TRACED_V6 = make_traced(GLM53v6)


def run_traced(Cls, tu, X, **kw):
    c = Cls(X.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    n = len(tu)
    Y = np.empty_like(X)
    tot = np.empty(n)
    ev_flag = np.zeros(n, bool)
    st_code = np.zeros(n, np.int8)
    for i in range(n):
        Y[i] = c.process(tu[i], X[i])
        tot[i] = X[i].sum()
        ev_flag[i] = c.ev is not None
        st_code[i] = {"idle": 0, "event": 1, "slow": 2}[c.state]
    return dict(Y=Y, tot=tot, ev=ev_flag, st=st_code, comp=c,
                epoch=c.tr_epoch, revoke=c.tr_revoke, handoff=c.tr_handoff,
                unload=c.tr_unload, gevent=c.tr_gevent)


def run_plain(Cls, tu, X, **kw):
    c = Cls(X.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    Y = np.empty_like(X)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], X[i])
    return Y, c
