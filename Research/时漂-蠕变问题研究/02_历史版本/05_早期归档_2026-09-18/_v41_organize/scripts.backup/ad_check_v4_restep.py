# -*- coding: utf-8 -*-
"""核对 v4 与 v3 在「负载内变载」上到底是否逐帧相同。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3
from glm53_v7 import GLM53v7
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    tr = []
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
        tr.append((c.in_load, getattr(c, "fast_done_", None), c.g, c.A.max()))
    return Y, c, tr


for name in ["零负载-中途切换负载-零负载-切换负载", "零负载-切换负载-零负载-再切换负载"]:
    p = os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv")
    t, X = load_rec(p)
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    Y3, c3, _ = run(tu, Xu, GLM53v3)
    Y4, c4, tr4 = run(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)
    Y7, c7, _ = run(tu, Xu, GLM53v7)
    d34 = np.abs(Y3 - Y4).max(axis=1)
    print(f"\n=== {name} (span={span:.1f}s) ===")
    print(f"  max|v3-v4| 全程 = {d34.max():.4f}   帧数(差>1e-9) = {int((d34>1e-9).sum())}/{len(tu)}")
    print(f"  max|v3-v7| 全程 = {np.abs(Y3-Y7).max():.4f}")
    idx = np.where(d34 > 1e-6)[0]
    if len(idx):
        segs = []
        a = idx[0]
        for k in range(1, len(idx)):
            if idx[k] != idx[k - 1] + 1:
                segs.append((a, idx[k - 1]))
                a = idx[k]
        segs.append((a, idx[-1]))
        print("  v3-v4 有差异的区间(t s):", [(round(float(tu[a]), 2), round(float(tu[b]), 2)) for a, b in segs][:20])
    on = [i for i in range(1, len(tr4)) if tr4[i][0] and not tr4[i - 1][0]]
    print("  v4 进入负载态的时刻:", [round(float(tu[i]), 2) for i in on])
    fd = [i for i in range(1, len(tr4)) if tr4[i][1] and not tr4[i - 1][1]]
    print("  v4 fast_done_ 置真时刻:", [round(float(tu[i]), 2) for i in fd])
