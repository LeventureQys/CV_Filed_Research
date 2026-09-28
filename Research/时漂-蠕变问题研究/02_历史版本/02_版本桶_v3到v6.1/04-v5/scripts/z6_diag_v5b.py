# -*- coding: utf-8 -*-
"""诊断 v5：豁免期结束后父类状态为何不进入蠕变补偿。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
sys.path.insert(0, HERE)
import importlib
M = importlib.import_module("z2_v5")
from glm53_v3 import GLM53v3  # noqa: E402

p = os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv")
df = pd.read_csv(p, skiprows=24)
ch = [c for c in df.columns if c.startswith("ch")]
tr = df["timestamp"].to_numpy(float)
t = tr - tr[0]
X = df[ch].to_numpy(float)
span = t[-1] - t[0]
dt = span / (len(t) - 1)
tu = np.arange(0.0, span, dt)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
amp = Xu.sum(axis=1)
# 用与 r_fastphase 相同的分段
ld = amp > 0.15 * amp.max()
d = np.diff(ld.astype(int))
s = np.where(d == 1)[0] + 1
e = np.where(d == -1)[0] + 1
if ld[0]:
    s = np.r_[0, s]
if ld[-1]:
    e = np.r_[e, len(ld)]
s0, s1 = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]
main = int(np.argmax(Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)))

for lab, cls in [("v3", GLM53v3), ("v5", M.GLM53v5)]:
    c = cls(Xu.shape[1])
    print(f"\n===== {lab} =====")
    print(f"{'t(s)':>7}{'in_load':>9}{'u':>7}{'a_cap':>7}{'g':>8}{'g2':>9}"
          f"{'Amax':>8}{'n_ld':>6}{'gamma_main':>11}{'creep_main':>11}")
    for i in range(len(tu)):
        Y = c.process(tu[i], Xu[i])
        if i % 40 == 0 and 10.5 < tu[i] < 30.0:
            A = getattr(c, "A", None)
            amax = float(A.max()) if A is not None and len(A) else 0.0
            gm = float(c.gamma[main]) if len(c.gamma) > main else np.nan
            A_m = float(A[main]) if A is not None and len(A) > main else np.nan
            creep = gm * A_m * c.g if (getattr(c, "loaded", None) is not None and c.loaded[main]) else 0.0
            print(f"{tu[i]:7.2f}{str(c.in_load):>9}{tu[i]-c.onset_ts:7.2f}"
                  f"{str(c.a_captured):>7}{c.g:8.4f}{c.g2:9.4f}{amax:8.3f}"
                  f"{int(c.loaded.sum()):6d}{gm:11.3f}{creep:11.4f}")
