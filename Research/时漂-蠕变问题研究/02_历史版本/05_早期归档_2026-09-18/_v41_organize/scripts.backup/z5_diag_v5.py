# -*- coding: utf-8 -*-
"""诊断 v5 在恒载数据上为何不补偿（豁免期没有结束？）。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
import importlib
M = importlib.import_module("z2_v5")
GLM53v5 = M.GLM53v5

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

c = GLM53v5(Xu.shape[1])
rows = []
for i in range(len(tu)):
    Y = c.process(tu[i], Xu[i])
    if i % 25 == 0:
        rows.append(dict(t=tu[i], in_load=c.in_load, onset=c.onset_ts,
                         u=(tu[i] - c.onset_ts) if c.in_load else 0.0,
                         fast_active=getattr(c, "fast_active_", None),
                         fast_end=getattr(c, "fast_end_u_", None),
                         quiet=getattr(c, "quiet_acc_", None),
                         lvl_prev=getattr(c, "lvl_prev_", None),
                         a_captured=c.a_captured, g=c.g,
                         comp=float((Xu[i] - Y).sum())))
d = pd.DataFrame(rows)
sel = d[(d.t > 7.0) & (d.t < 25.0)]
pd.set_option("display.width", 200)
print(sel.to_string(index=False))
print("\nfull-sim 补偿总量最大 =", d.comp.abs().max())
print("fast_end 出现时刻 =", d.loc[d.fast_end.notna(), "t"].head(3).tolist() if d.fast_end.notna().any() else "从未出现")
