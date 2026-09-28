# -*- coding: utf-8 -*-
"""单份逐帧核对：v6 / v6.1 在指定录制上的最大偏差位置与各事件的超调/回落。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402
from glm53_v61 import GLM53v61                              # noqa: E402

idx = [int(x) for x in (sys.argv[1:] or [10])]
for i in idx:
    tag, path = C.ALL[i]
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    raw = Xu.sum(axis=1)
    rs = L.med_smooth(raw, 0.5 / dtm)
    print("=" * 112)
    print(f"{tag}")
    for cls in (GLM53v6, GLM53v61):
        c = cls(Xu.shape[1])
        Y = np.empty_like(Xu)
        for j in range(len(tu)):
            Y[j] = c.process(tu[j], Xu[j])
        ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        err = ys - rs
        k = int(np.argmax(np.abs(err)))
        print(f"\n  {cls.__name__}  epoch={len(c.epoch_t)}  max|y-raw|={np.abs(err).max():,.0f} "
              f"@t={tu[k]:.2f}s (y-raw={err[k]:+,.0f})   max(+{err.max():,.0f})  min({err.min():,.0f})")
        for t0, kd in c.kind_log:
            i0 = int(np.searchsorted(tu, t0))
            a = min(len(tu) - 1, i0 + int(4.6 / dtm))
            b = min(len(tu), i0 + int(5.4 / dtm))
            if b - a < 3:
                continue
            pre = float(np.median(rs[max(0, i0 - int(0.3 / dtm)):max(1, i0)]))
            P = float(np.median(rs[a:b]))
            step = P - pre
            if step < 3000:
                continue
            j5 = min(len(tu) - 1, i0 + int(5.0 / dtm))
            seg = ys[i0:j5 + 1]
            kk = int(np.argmax(seg))
            e = min(len(tu), i0 + kk + int(6.0 / dtm))
            tr = float(seg[kk] - np.min(ys[i0 + kk:e]))
            print(f"    t0={t0:7.2f} {kd:<14s} step={step:9,.0f}  "
                  f"超调={100*(seg[kk]-P)/step:+6.1f}%  5s残留={100*(ys[j5]-P)/step:+6.1f}%  "
                  f"瞬态回落={100*tr/step:+6.1f}%  显示峰={seg[kk]:9,.0f} 真值={P:9,.0f}")
