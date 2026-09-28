# -*- coding: utf-8 -*-
"""v5 回归：恒载 9 组基准（口径与 Document/03 完全一致，可与 1.58/1.49、2.60/0.95 对比）。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]
ALGOS = [("raw", None, {}),
         ("v3", GLM53v3, {}),
         ("v4_fast5", L.GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
         ("v5", GLM53v5, {})]


def find_segment(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


def metrics(Y, X, tt, s0, s1, amp, loaded, m, dtm, n_on):
    nL = s1 - s0
    L = Y[s0:s1]
    by, bx = Y[:s0, m].mean(), X[:s0, m].mean()
    dr = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
    a5 = max(s0, n_on + int(5.0 / dtm))
    L5 = Y[a5:s1]
    n5 = max(1, len(L5))
    dr5 = L5[-n5 // 10:].mean(axis=0) - L5[: n5 // 10].mean(axis=0)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    sx = X[i1:i2, m].mean() - bx
    step = ((Y[i1:i2, m].mean() - by) / sx) if abs(sx) > 1e-9 else np.nan
    tts = tt[s0:s1] - tt[s0]

    def dstd(sig, tq):
        k, b0 = np.polyfit(tq, sig, 1)
        return (sig - (k * tq + b0)).std()

    ny = dstd(L[10:, m], tts[10:])
    nx = dstd(X[s0 + 10:s1, m], tts[10:])
    j1, j2 = s1 + int(5.0 / dtm), min(len(tt), s1 + int(30.0 / dtm))
    dY, dX = np.diff(Y[:, m]), np.diff(X[:, m])
    jump = float(np.abs(dY - dX)[s0:s1].max()) if s1 > s0 else np.nan
    return dict(drift_main=100 * dr[m] / amp, drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=(ny / nx) if nx > 1e-12 else np.nan,
                flat_main=100 * ny / amp,
                zero_resid=(100 * (Y[j1:j2, m].mean() - by) / amp) if j2 > j1 else np.nan,
                step_ratio=step, jump_excess=jump)


rows = []
for loc, name in DATASETS:
    p = os.path.join(TEMP, loc, name, "device_001_seg000.csv")
    if not os.path.exists(p):
        print(f"[skip] {p}")
        continue
    d = L.prep(p)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    segs = find_segment(d["tot"])
    s0r, s1r = segs[0]
    s0 = int(np.searchsorted(tu, tu[min(s0r, len(tu) - 1)]))
    s1 = int(np.searchsorted(tu, min(t0 := d["t"][s1r], d["span"])))
    amp_v = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp_v))
    amp = amp_v[m]
    loaded = amp_v > 0.10 * amp_v.max()
    tot = d["tot"]
    b0 = tot[:max(1, s0)].mean()
    n_on = next((i for i in range(s0, min(s0 + 400, len(tu) - 1))
                 if tot[i] > b0 + 0.05 * (tot[max(0, s0):s1].max() - b0)), s0)
    line = f"{loc}/{name}  ch{m}  amp={amp:.0f}"
    for k, cls, kw in ALGOS:
        if cls is None:
            Y = Xu.copy()
            det = None
        else:
            Y, c = L.run_algo(tu, Xu, cls, **kw)
            det = f"A={c.A.max():.0f} g={c.g:+.3f}"
        mm = metrics(Y, Xu, tu, s0, s1, amp, loaded, m, dtm, n_on)
        mm.update(location=loc, dataset=name, algo=k)
        rows.append(mm)
        line += f" | {k}:{mm['drift_main']:+.2f}/{mm['drift_slow']:+.2f}%"
        if det:
            line += f"({det})"
    print(line)

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "v5_static9_metrics.csv"), index=False, encoding="utf-8-sig")
agg = dfm.groupby("algo").agg(
    时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
    时漂残余_慢相段=("drift_slow", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    噪声比=("noise_ratio", "mean"), 平坦度=("flat_main", "mean"),
    零漂残余=("zero_resid", lambda s: s.abs().mean()),
    阶跃保真=("step_ratio", "mean"), 事件跳变超额=("jump_excess", "max"),
).reindex([k for k, _, _ in ALGOS])
print("\n===== 恒载 9 组汇总（口径同 Document/03；参考：v3 1.58/1.49、v4 2.60/0.95）=====")
print(agg.round(2).to_string())
agg.to_csv(os.path.join(RES, "v5_static9_summary.csv"), encoding="utf-8-sig")
