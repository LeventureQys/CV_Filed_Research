# -*- coding: utf-8 -*-
"""诊断：v1 产品版在 右拇指/数据2（恒载）上为何只补偿到 27%"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f_varying_load import CompV1, load_csv, find_segment  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
t, X, tc = load_csv(os.path.join(BASE, "右拇指指尖", "数据2", "device_001_seg000.csv"))
total = X.sum(axis=1)
s0, s1 = max(find_segment(total), key=lambda z: z[1] - z[0])
main = int(np.argmax(X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)))
print(f"load seg {t[s0]:.1f}-{t[s1]:.1f}s main=ch{main} amp={X[s0:s1,main].mean()-X[:s0,main].mean():.3f}")

comp = CompV1()
Y = np.empty_like(X, dtype=float)
trace = []
for i in range(len(t)):
    y = comp.process(t[i], X[i].astype(float))
    Y[i] = y
    u = t[i] - comp.onset if comp.in_load else -1
    if comp.in_load and comp.a_captured and (abs((u) % 20) < 0.02 or u < 3.1):
        ld = comp.loaded & (comp.A > 1e-9)
        rel = (Y[i][ld] - 0 - comp.A[ld]) / comp.A[ld]  # 近似: 用当帧Z
        trace.append((t[i], u, comp.A[main], comp.g, comp.gamma[main] if comp.gamma is not None else 1,
                      comp.ts, comp.level, float(np.median(rel)) if rel.size else np.nan))

print("\n  t      u     A[main]   g      γ[main]  ts     level   med(rel)")
for r in trace[::3]:
    print("  ".join(f"{v:7.2f}" if isinstance(v, float) else str(v) for v in r))

L = Y[s0:s1, main]
nL = len(L)
amp = X[s0:s1, main].mean() - X[:s0, main].mean()
print(f"\nv1 drift_main = {100*(L[-nL//10:].mean()-L[:nL//10].mean())/amp:+.1f}%")
print("Y[main] @ 负载段 每10%位置:", " ".join(
    f"{L[int(k*nL/10)]:.3f}" for k in range(10)))
print("X[main] 同位置:", " ".join(
    f"{X[s0+int(k*nL/10), main]:.3f}" for k in range(10)))
