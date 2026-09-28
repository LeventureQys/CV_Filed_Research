# -*- coding: utf-8 -*-
"""插桩 log_creep：打印 ĉ(u) 与补偿后 Y 的轨迹"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "右拇指指尖")  # temp/右拇指指尖
TAU_CREEP = 5.0
MAIN = 17


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


name = "数据1"
df = pd.read_csv(os.path.join(BASE, name, "device_001_seg000.csv"), skiprows=24)
tc = [c for c in df.columns if c.startswith("ch")]
X = df[tc].to_numpy()
t = df["elapsed"].to_numpy()
n, m = X.shape
s0, s1 = seg(X.sum(axis=1))
dtm = np.median(np.diff(t))

# baseline（卸载门控 τ=2s）
dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.1)
total = X.sum(1)
amp_total_max = total[s0:s1].max()
unloaded = total < 0.05 * amp_total_max
b0 = np.zeros(m)
a_b = np.clip(dt / 2.0, 0, 1)
baseline = np.empty((n, m))
for i in range(n):
    if unloaded[i]:
        b0 = b0 + a_b[i] * (X[i] - b0)
    baseline[i] = b0

i_on1 = s0 + int(1.0 / dtm)
i_on2 = s0 + int(3.0 / dtm)
A = X[i_on1:i_on2].mean(axis=0) - baseline[s0]

# 在线对数蠕变（与 b_compare.py 完全一致）
Y = X.copy()
phi_acc = np.zeros(m)
phid_acc = np.zeros(m)
u_prev = 0.0
trace = []
for i in range(s0, s1):
    u = t[i] - t[s0]
    du = u - u_prev
    phi = np.log1p(u / TAU_CREEP)
    d = X[i] - baseline[s0] - A
    phi_acc += du * phi
    phid_acc += du * phi * d
    c_hat_ch = np.where(phi_acc > 1e-6, phid_acc / np.maximum(phi_acc, 1e-6), 0.0)
    if u > 8.0:
        creep = c_hat_ch * np.log1p(u / TAU_CREEP)
        creep = np.minimum(creep, 0.9 * A)
        Y[i] = X[i] - creep
    if abs(u % 10) < dtm / 2 or u < 3:
        trace.append((u, X[i, MAIN], d[MAIN], c_hat_ch[MAIN],
                      (c_hat_ch[MAIN] * np.log1p(u / TAU_CREEP)) if u > 8 else 0.0,
                      Y[i, MAIN]))
    u_prev = u

print(f"A[main]={A[MAIN]:.4f}  baseline[s0,main]={baseline[s0, MAIN]:.4f}")
print("  u     X      d      c_hat   creep   Y")
for u, x, dd, cc, cr, yy in trace:
    print(f"{u:5.1f} {x:+7.3f} {dd:+7.3f} {cc:+7.3f} {cr:+7.3f} {yy:+7.3f}")
