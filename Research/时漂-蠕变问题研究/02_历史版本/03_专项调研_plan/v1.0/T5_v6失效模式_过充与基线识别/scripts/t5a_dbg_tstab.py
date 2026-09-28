# -*- coding: utf-8 -*-
"""调试：`metrics_at` 的 T_stable 与逐点扫描在个别事件上不一致的原因。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t5a_common as C

recs = C.recordings()
for key, t0 in [("SW2", 35.96), ("LT3", 10.87), ("F43", 11.00), ("SW4", 8.41)]:
    d = recs[key]
    out = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12)
    Ys = out["Y"].sum(axis=1)
    Zs = d["Z"]
    tu = d["tu"]
    m = C.metrics_at(Zs, Ys, tu, t0, d["span"])
    Zf = m["Z_final"]
    J = m["J"]
    thr = 0.05 * abs(J)
    seg = np.abs(Ys - Zf)
    dt = tu[1] - tu[0]
    nwin = int(round(30.0 / dt))
    i0 = int(np.searchsorted(tu, t0))
    n = len(seg)
    print(f"\n{key} @{t0}: n={n} i0={i0} nwin={nwin} J={J:.3f} Zf={Zf:.3f} "
          f"thr={thr:.4f} | impl T30={m['T_stable30']} T10={m['T_stable10']}")
    if i0 + nwin <= n:
        sl = seg[i0:i0 + nwin]
        print(f"  完整窗 max={sl.max():.5f} idx_of_max={i0+int(np.argmax(sl))} "
              f"| thr={thr:.5f} | 窗内<=thr 的点数={int((sl<=thr).sum())}")
        # 非零起始的第一次满足
        idx = np.where(sl <= thr)[0]
        if len(idx):
            i = i0 + int(idx[0])
            print(f"  逐点扫描首个满足点 i={i} -> tau={tu[i]-t0:.3f} ；"
                  f"该点窗口 max={seg[i:i+nwin].max():.5f}")
    else:
        print(f"  完整窗放不下（i0+nwin={i0+nwin} > n={n}）")
    # 用 impl 的滚动最大值复算
    from scipy.ndimage import maximum_filter1d
    sub = seg[i0:min(i0 + nwin + 1, n)]
    mf = C._rollmax_fwd(sub, nwin)
    print(f"  impl 滚动最大前 5 个 = {np.round(mf[:5], 5)}  thr={thr:.5f}  "
          f"首个<=thr k={int(np.argmax(mf<=thr)) if (mf<=thr).any() else -1}")
