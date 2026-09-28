# -*- coding: utf-8 -*-
"""检查内置 Kalman 补偿输出 + 卸载后恢复动态"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "右拇指指尖")  # temp/右拇指指尖
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

for name in ["数据1", "数据2", "数据3"]:
    raw = pd.read_csv(os.path.join(BASE, name, "device_001_seg000.csv"), skiprows=24)
    kal = pd.read_csv(os.path.join(BASE, name, "device_001_seg000_kalman_compensated.csv"), skiprows=24)
    tc = [c for c in raw.columns if c.startswith("ch")]
    Xr, Xk = raw[tc].to_numpy(), kal[tc].to_numpy()
    tr, tk = raw["elapsed"].to_numpy(), kal["elapsed"].to_numpy()
    n = min(len(tr), len(tk))
    print(f"{name}: raw={Xr.shape} kal={Xk.shape} timestamps_align={np.allclose(tr[:n], tk[:n], atol=1e-3)}")
    for lbl, X, t in [("raw", Xr, tr), ("kal", Xk, tk)]:
        s0, s1 = seg(X.sum(1))
        L = X[s0:s1, MAIN]
        nL = len(L)
        drift = L[-nL // 10:].mean() - L[: nL // 10:].mean()
        amp = L.mean() - X[:s0, MAIN].mean()
        post = X[s1:, MAIN]
        z_pre = X[:s0, MAIN].mean()
        z_after = post[: len(post) // 10].mean()
        z_end = post[-len(post) // 10:].mean()
        # 恢复时间常数：卸载后基线衰减
        noise = L[nL // 5:].std()
        print(f"  {lbl}: seg {t[s0]:.1f}-{t[s1]:.1f}s  drift={drift:+.4f} ({100*drift/amp:+.1f}%)  "
              f"amp={amp:.3f}  noise_std={noise:.4f}  zero_pre={z_pre:.4f} "
              f"zero_just_after={z_after:.4f}  zero_end={z_end:.4f}")
        # 卸载后基线恢复轨迹（每10%处的均值）
        dec = [post[int(i*len(post)/10):int((i+1)*len(post)/10)].mean() for i in range(10)]
        print("    卸载后基线轨迹(10档):", " ".join(f"{v:+.4f}" for v in dec))
