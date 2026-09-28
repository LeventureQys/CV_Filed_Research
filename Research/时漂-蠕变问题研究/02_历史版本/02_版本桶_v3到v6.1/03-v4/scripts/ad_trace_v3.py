# -*- coding: utf-8 -*-
"""逐帧打印 v3 在首个 onset 后 0~8s 的内部量，确认补偿为何恰为 0。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


p = os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")
t, X = load_rec(p)
span = t[-1] - t[0]
dt = span / (len(t) - 1)
tu = np.arange(0.0, span, dt)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T

c = GLM53v3(Xu.shape[1])
print("  t     in_load  u     a_cap  hold  pending  A_max    g_      gamma_med  out_max  Z_max")
for i in range(len(tu)):
    Y = c.process(tu[i], Xu[i])
    if 12.0 <= tu[i] <= 22.0 and i % 25 == 0:
        u = tu[i] - c.onset_ts if c.in_load else -1
        print(f"{tu[i]:6.2f}   {int(c.in_load)}   {u:5.2f}  {int(c.a_captured)}    {int(c.hold)}    "
              f"{int(c.pending)}     {c.A.max():8.1f} {c.g:+.5f}  {np.median(c.gamma):6.3f}  "
              f"{Y.max():7.1f} {(Xu[i]-c.b).max():7.1f}")
