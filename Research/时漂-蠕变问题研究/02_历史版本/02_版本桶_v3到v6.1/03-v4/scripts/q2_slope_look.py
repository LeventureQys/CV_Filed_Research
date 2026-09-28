# -*- coding: utf-8 -*-
"""先看事实：加载后主通道（及受载通道中位）的平滑斜率随时间怎么变。"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))


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


for loc, name in [("右拇指指尖", "数据1"), ("右拇指指尖", "数据3"),
                  ("左拇指指尖", "数据1"), ("四指指尖", "数据1")]:
    t, X = load(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    s0, s1 = seg(X.sum(axis=1))
    tot = X.sum(axis=1)
    base = tot[:s0].mean()
    thr = base + 0.05 * (tot[s0:s1].max() - base)
    n_on = next(i for i in range(s0, s0 + 300) if tot[i] > thr)
    amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    b = X[:s0, m].mean()
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    y = np.interp(tu, t, X[:, m]) - b
    i0 = int(np.searchsorted(tu, t[n_on]))
    print(f"\n=== {loc}/{name} ch{m}  dt={1000*dt:.2f}ms  onset t={t[n_on]:.2f}s  "
          f"负载段长 {t[s1]-t[s0]:.1f}s ===")
    # 分段平均斜率（每段 1s），看快慢相
    print("   加载后区间(s)   区间增量    平均斜率(/s)")
    edges = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240]
    for k in range(len(edges) - 1):
        a = i0 + int(round(edges[k] / dt))
        c = i0 + int(round(edges[k + 1] / dt))
        if c >= len(y) or c <= a:
            break
        d = y[c] - y[a]
        print(f"   {edges[k]:5.1f} ~ {edges[k+1]:5.1f}   {d:+8.4f}   {d/(edges[k+1]-edges[k]):+8.5f}")
    # 累计占比
    print(f"   加载瞬间 y={y[i0]:+.4f}")
    for tt in [1, 2, 3, 5, 10, 20, 40, 80]:
        k = i0 + int(round(tt / dt))
        if k < min(s1, len(y)):
            print(f"     +{tt:3.0f}s: y={y[k]:+.4f}  (占末端 {100*y[k]/y[s1-1]:.1f}%)", end="")
            if tt in (1, 2, 3, 5):
                print()
            else:
                print()
