# -*- coding: utf-8 -*-
"""直接看主通道在加载前后的原始曲线：加载瞬间的读数是多少？之后是升还是降？"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)


def load(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def seg(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]


for loc, name in [("右拇指指尖", "数据1"), ("四指指尖", "数据1"), ("左拇指指尖", "数据1")]:
    t, X = load(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    s0, s1 = seg(X.sum(axis=1))
    amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    base = X[:s0, m].mean()
    print(f"\n=== {loc}/{name}  主通道 ch{m}  空载均值={base:.4f}  负载段 {t[s0]:.1f}~{t[s1]:.1f}s"
          f"  帧数={len(t)}  跨度={t[-1]:.1f}s")
    t0 = t[s0]
    print("   [A] 加载前后逐帧（用时间查询，不假设等间隔）")
    print("       t(s,相对onset)   读数     相对空载")
    for rel in [-0.6, -0.4, -0.2, -0.1, -0.05, 0.0, 0.016, 0.032, 0.05, 0.08, 0.1,
                0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
        i = int(np.searchsorted(t, t0 + rel))
        if 0 <= i < len(t):
            print(f"       {t[i]-t0:+8.3f}   {X[i, m]:9.4f}  {X[i, m]-base:+9.4f}")
    nL = s1 - s0
    print("   [B] 负载段均分 12 点的平均读数")
    for k in range(12):
        a, b = s0 + nL * k // 12, s0 + nL * (k + 1) // 12
        print(f"       {t[a]-t0:+8.1f}~{t[b]-t0:+8.1f}s  均值 {X[a:b, m].mean():9.4f}  "
              f"相对空载 {X[a:b, m].mean()-base:+9.4f}")
    print(f"   [C] 负载段最早 5 帧读数: {np.array2string(X[s0:s0+5, m], precision=4)}")
