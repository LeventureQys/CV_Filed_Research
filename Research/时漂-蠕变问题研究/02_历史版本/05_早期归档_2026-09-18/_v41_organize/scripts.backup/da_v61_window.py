# -*- coding: utf-8 -*-
"""指定时间窗的逐 0.1 s 转储：raw / v6 / v6.1 / 事件 / A / g。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402
from glm53_v61 import GLM53v61                              # noqa: E402

tag_i = int(sys.argv[1]) if len(sys.argv) > 1 else 10
t_a = float(sys.argv[2]) if len(sys.argv) > 2 else 127.0
t_b = float(sys.argv[3]) if len(sys.argv) > 3 else 133.0
step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2

tag, path = C.ALL[tag_i]
d = L.prep(path)
tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
raw = Xu.sum(axis=1)
rs = L.med_smooth(raw, 0.5 / dtm)
out = {"t": tu, "raw": rs, "raw_inst": raw}
for cls in (GLM53v6, GLM53v61):
    c = cls(Xu.shape[1])
    Y = np.empty_like(Xu)
    for j in range(len(tu)):
        Y[j] = c.process(tu[j], Xu[j])
    ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    out[cls.__name__] = ys
    ev = np.full(len(tu), np.nan)
    for t0, kd in c.kind_log:
        i0 = int(np.searchsorted(tu, t0))
        j0 = int(np.searchsorted(tu, t0 + 3.0))
        ev[i0:min(j0, len(tu))] = {"onset": 1, "restep": 2, "restep_reload": 3,
                                   "decrease": 4}.get(kd, 5)
    if cls is GLM53v61:
        out["ev61"] = ev
df = pd.DataFrame(out)
df["d61"] = df["GLM53v61"] - df["raw"]
df["d6"] = df["GLM53v6"] - df["raw"]
sel = df[(df.t >= t_a) & (df.t <= t_b)]
k = max(1, int(round(step / dtm)))
print(f"{tag}  t={t_a}~{t_b}s  步长 {step}s")
print(sel.iloc[::k][["t", "raw_inst", "raw", "GLM53v6", "GLM53v61", "d6", "d61", "ev61"]]
      .to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
