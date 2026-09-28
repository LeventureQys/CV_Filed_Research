# -*- coding: utf-8 -*-
"""v6 / v6.1 单份冒烟：确认 v6.1 能跑通、耗时与 epoch 数可比。"""
import os
import sys
import time

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
    for cls, kw in ((GLM53v6, {}), (GLM53v61, {})):
        t0 = time.time()
        c = cls(Xu.shape[1])
        for k, v in kw.items():
            setattr(c, k, v)
        Y = np.empty_like(Xu)
        for j in range(len(tu)):
            Y[j] = c.process(tu[j], Xu[j])
        ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        rs = L.med_smooth(raw, 0.5 / dtm)
        print(f"{tag:>18} {cls.__name__:<9} {time.time()-t0:5.1f}s epoch={len(c.epoch_t):3d} "
              f"downclip={getattr(c, 'n_down_clip', 0):6d} "
              f"out_max={Y.sum(axis=1).max():9,.0f} raw_max={raw.max():9,.0f} "
              f"max|y-raw|={np.abs(ys-rs).max():7,.0f}")
