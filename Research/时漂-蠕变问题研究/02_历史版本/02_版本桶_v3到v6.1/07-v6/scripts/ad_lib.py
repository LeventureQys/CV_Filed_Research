# -*- coding: utf-8 -*-
"""v4.1flash 离线复算的公共工具（数据载入 / 事件检测 / 算法运行）。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3          # noqa: E402
from ad_v4 import GLM53v4, GLM53v4r   # noqa: E402

ALGOS = [("raw", None, {}),
         ("v3", GLM53v3, {}),
         ("v4_fast5", GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
         ("v4r_fast5", GLM53v4r, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0))]
LBL = {"raw": "原始（无补偿）", "v3": "GLM53 v3（现役）",
       "v4_fast5": "v4 免责5s（最新算法）", "v4r_fast5": "v4r（免责5s + 变载也免责）"}
ORDER = ["raw", "v3", "v4_fast5", "v4r_fast5"]


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def med_smooth(x, k):
    return pd.Series(x).rolling(max(1, int(k)), center=True, min_periods=1).median().to_numpy()


def make_traced(cls):
    class Traced(cls):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    return Traced


def run_algo(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def prep(path):
    t, X = load_rec(path)
    span = t[-1] - t[0]
    dtm = span / (len(t) - 1)
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return dict(t=t, X=X, tu=tu, Xu=Xu, span=span, dtm=dtm, tot=Xu.sum(axis=1),
                dup=int((np.diff(t) <= 0).sum()))


def find_periods(tot, dt, frac=0.05, min_s=3.0):
    thr = frac * tot.max()
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    return sorted([(a, b) for a, b in zip(s, e) if (b - a) * dt >= min_s])


def detect_events(tot, dt, rel=0.15, absfrac=0.08):
    n = len(tot)
    pn = int(2.0 / dt)
    cand = []
    for i in range(pn, n - pn, max(1, int(0.1 / dt))):
        pre = np.median(tot[i - pn:i])
        post = np.median(tot[i:i + pn])
        if abs(post - pre) > max(rel * abs(pre), absfrac * tot.max()):
            cand.append((i, post - pre))
    ev = []
    for i, dl in cand:
        if ev and i - ev[-1][0] <= int(1.5 / dt):
            if abs(dl) > abs(ev[-1][1]):
                ev[-1] = (i, dl)
        else:
            ev.append((i, dl))
    ref, seen = [], set()
    for i, dl in ev:
        a, b = max(0, i - int(2 / dt)), min(n - 1, i + int(2 / dt))
        sm = med_smooth(tot[a:b], max(3, int(0.15 / dt)))
        k = int(np.argmax(np.abs(np.diff(sm))))
        if a + k in seen:
            continue
        seen.add(a + k)
        ref.append((a + k, dl))
    return ref


def event_table(d, events, Ys, epochs_v3, gain_s=20.0, algos=None):
    """每个事件的：变载前/后读数、台阶、比值、是否被 v3 重捕获、后续最大偏差、显示增益比"""
    ks = list(algos) if algos else [k for k in ORDER if k != "raw" and k in Ys]
    tu, tot, dtm = d["tu"], d["tot"], d["dtm"]
    tot_s = med_smooth(tot, 0.5 / dtm)
    rows = []
    for e in events:
        pre = float(np.median(tot_s[max(0, e - int(2 / dtm)):e]))
        post = float(np.median(tot_s[min(len(tu) - 1, e + int(4 / dtm)):min(len(tu), e + int(6 / dtm))]))
        dl = post - pre
        if abs(dl) < 1e-6:
            continue
        hit = [x for x in epochs_v3 if 0 < x - float(tu[e]) <= 6.0]
        a, b = max(0, int(e - 1 / dtm)), min(len(tu), int(e + 12 / dtm))
        row = dict(t=float(tu[e]), pre=pre, post=post, jump=dl, ratio=abs(dl) / max(pre, 1e-9),
                   restep=(round(hit[0] - float(tu[e]), 2) if hit else None))
        # 显示增益比：事件后 gain_s 秒内「显示的增量」/「原始的增量」（1.0=台阶被完整透传）
        i0 = max(0, int(e - 1 / dtm))
        i1 = min(len(tu) - 1, int(e + gain_s / dtm))
        if i1 > i0:
            row["raw_gain"] = float(tot_s[i1] - tot_s[i0])
            for k in ks:
                y = med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)
                row[f"gain_{k}"] = float(y[i1] - y[i0])
            if "v3" in ks:
                row["gain_ratio_v3"] = (row["gain_v3"] / row["raw_gain"]) if abs(row["raw_gain"]) > 1e-9 else np.nan
        for k in ks:
            y = med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)
            row[f"gap_{k}"] = float(np.abs(y[a:b] - tot_s[a:b]).max())
        rows.append(row)
    return pd.DataFrame(rows)


def mid_load_events(ev):
    """只保留「负载内变载」：变载前后都在受载电平上（排除空载→负载与卸载）"""
    if not len(ev):
        return ev
    thr = 0.30 * ev.pre.max()
    return ev[(ev.pre > thr) & (ev.post > thr)].copy()


def slow_windows(d, guard_s=8.0, min_win_s=6.0, algos=None):
    """慢相稳定窗：只在负载段内部切（事件后 guard_s 起，到下一个事件/负载段末止）"""
    ks = list(algos) if algos else [k for k in ORDER if k in d["Ys"]]
    tu, tot = d["tu"], d["tot"]
    dtm = d["dtm"]
    tot_s = med_smooth(tot, 0.5 / dtm)
    out = []
    for pa, pb in d["periods"]:
        pb = min(int(pb), len(tu) - 1)
        cuts = [float(tu[pa])] + [float(tu[e]) for e in d["events"] if pa <= e <= pb] + [float(tu[pb])]
        cuts = sorted(set(cuts))
        for i in range(len(cuts) - 1):
            w0 = int(np.searchsorted(tu, cuts[i] + guard_s))
            w1 = int(np.searchsorted(tu, cuts[i + 1]))
            if (w1 - w0) * dtm >= min_win_s:
                out.append((w0, w1))
    rows = []
    for w0, w1 in out:
        lvl = float(np.median(tot_s[w0:w1]))
        nL = w1 - w0
        for k in ks:
            L = d["Ys"][k][w0:w1].sum(axis=1)
            sd = L[-max(1, nL // 5):].mean() - L[:max(1, nL // 5)].mean()
            rows.append(dict(window_s=float(tu[w0]), end_s=float(tu[w1]), dur_s=(w1 - w0) * dtm,
                             algo=k, level=lvl, drift_pct=100 * sd / max(lvl, 1e-9)))
    return pd.DataFrame(rows)
