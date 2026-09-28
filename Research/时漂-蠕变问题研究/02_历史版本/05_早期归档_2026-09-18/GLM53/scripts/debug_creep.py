# -*- coding: utf-8 -*-
"""诊断 log_creep 过补偿原因：单通道实际蠕变曲线 vs 固定tau拟合"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "右拇指指尖")  # temp/右拇指指尖


def seg(total):
    thr = 0.15 * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if loaded[0]:
        s = np.r_[0, s]
    if loaded[-1]:
        e = np.r_[e, len(loaded)]
    segs = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)
    return segs[0]


for name in ["数据1", "数据2", "数据3"]:
    df = pd.read_csv(os.path.join(BASE, name, "device_001_seg000.csv"), skiprows=24)
    tc = [c for c in df.columns if c.startswith("ch")]
    X = df[tc].to_numpy()
    t = df["elapsed"].to_numpy()
    s0, s1 = seg(X.sum(1))
    tt = t[s0:s1] - t[s0]
    # A 用 1~3s 窗口
    dtm = np.median(np.diff(t))
    A = X[s0 + int(1 / dtm): s0 + int(3 / dtm), 17].mean() - X[:s0, 17].mean()
    d_act = X[s0:s1, 17] - X[:s0, 17].mean() - A
    print(f"\n=== {name} ch17  A={A:.3f}  max(X)={X[:,17].max():.3f} (是否饱和?)")
    print("u(s)   实际d(u)   log拟合τ=5   log拟合τ=20   实际/A")
    # 拟合（全段）
    for tau in [5, 20]:
        phi = np.log1p(tt / tau)
        c = np.dot(phi, d_act) / np.dot(phi, phi)
        globals()[f"c{tau}"] = c
    for u in [5, 10, 20, 40, 60, 80, 100, 120]:
        i = np.argmin(np.abs(tt - u))
        if abs(tt[i] - u) > 3:
            continue
        print(f"{u:4d}  {d_act[i]:+8.4f}  {c5*np.log1p(u/5):+9.4f}  {c20*np.log1p(u/20):+9.4f}  {d_act[i]/A*100:+6.1f}%")
    # 实际蠕变的形状: 每20s一段的增量
    inc = []
    prev = 0.0
    for u in range(0, int(tt[-1]), 20):
        i = np.argmin(np.abs(tt - u))
        inc.append(d_act[i] - prev)
        prev = d_act[i]
    print("每20s实际增量:", " ".join(f"{v:+.3f}" for v in inc))
