# -*- coding: utf-8 -*-
"""步骤6b：单通道探针——检查各算法在真实数据上的输出形状与"参考锚点"行为。"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, DATASETS
import tac_algorithms as AL

np.set_printoptions(precision=4, suppress=True)

D = load_dataset("数据1")
fi = D["frame_index"]
k, b = np.polyfit(fi, D["t"], 1)
t = fi * k; fs = 1 / k
X = D["X"]
seg = detect_segments(X.sum(axis=1), D["t"], fs=fs)
a, bb = seg["pre"]; c0, d0 = seg["load"]
base = X[a:bb].mean(axis=0)
Xn = X - base
resp = Xn[c0:d0].max(axis=0)
main = int(np.argmax(resp))
y = Xn[:, main]
w = lambda s: int(round(s * fs))
tt = t - t[c0]
print(f"主通道={D['ch_cols'][main]}  c0={c0} d0={d0}")

cases = [
    ("Raw", AL.Raw()),
    ("Model-exp1", AL.ModelFitCompensator("exp1", 0.2)),
    ("Model-power", AL.ModelFitCompensator("power", 0.2)),
    ("ArrayShape-exp2", AL.ModelFitShapeShared(0.2, "exp2")),
    ("CMRefFit-mean-o2", AL.CommonModeRefFit("mean", 2, 0.2)),
    ("PCA-k3", AL.PCASubspace(3, False, 0.2)),
    ("ANC-0.2-32", AL.AdaptiveNoiseCanceller(0.2, 32, "mean", 0.2)),
    ("CMR-mean-5s", AL.CommonModeRemoval("mean", 5.0, 0.2, True)),
]

# 参考窗口
probes = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5),
          (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0),
          (None, None)]
hdr = "  ".join(f"{lbl:>8s}" for lbl in
                ["0-.05", ".05-.1", ".1-.2", ".2-.3", ".3-.5", ".5-1", "1-2",
                 "2-5", "5-10", "末5s"])
print(f"{'算法':<18s} {hdr}")
raw_vals = {}
for label, c in cases:
    Y = c.transform(Xn, t, fs, c0)
    yy = Y[:, main]
    vals = []
    for lo, hi in probes[:-1]:
        vals.append(yy[c0 + w(lo):c0 + w(hi)].mean() * 1000)
    vals.append(yy[d0 - w(5):d0].mean() * 1000)
    if label == "Raw":
        raw_vals = dict(zip([f"{p}" for p in probes], vals))
    print(f"{label:<18s} " + "  ".join(f"{v:8.2f}" for v in vals))

# 检查拟合参数
print("\n各算法拟合细节：")
for label, c in cases:
    if hasattr(c, "fitted") and c.fitted:
        j = main
        if j in c.fitted:
            print(f"  {label} 主通道拟合参数 = {c.fitted[j]}")
    if hasattr(c, "shape") and c.shape is not None:
        print(f"  {label} 共享形态参数 = {c.shape}")
    if hasattr(c, "amps") and c.amps is not None:
        print(f"  {label} 通道幅度 amps[受载] = "
              f"{np.round(c.amps[np.abs(c.amps)>1e-6][:8], 3)}")

# 拟合质量：exp1 / power 对负载段的 R²
from scipy.optimize import curve_fit
tl = t[c0:d0] - t[c0]
yy = y[c0:d0]
def f1(t_, a, tau, c): return a * (1 - np.exp(-t_ / tau)) + c
def fp(t_, a, p, c): return a * np.power(np.maximum(t_, 1e-9), p) + c
for nm, f, p0 in (("exp1", f1, [0.6, 63.0, 1.18]), ("power", fp, [0.1, 0.4, 1.0])):
    try:
        popt, _ = curve_fit(f, tl, yy, p0=p0, maxfev=200000)
        r2 = 1 - np.var(yy - f(tl, *popt)) / np.var(yy)
        print(f"  主通道整段 {nm}: params={np.round(popt,4)} R²={r2:.5f} "
              f"f(0)={f(np.array([0.0]),*popt)[0]:.4f} f(0.2)={f(np.array([0.2]),*popt)[0]:.4f} "
              f"f(损末)={f(np.array([tl[-1]]),*popt)[0]:.4f}")
    except Exception as e:
        print(f"  主通道整段 {nm}: 失败 {e}")

# 只拟合早期 10s
m = tl < 10.0
for nm, f, p0 in (("exp1-10s", f1, [0.6, 5.0, 0.85]), ("power-10s", fp, [0.1, 0.4, 0.85])):
    try:
        popt, _ = curve_fit(f, tl[m], yy[m], p0=p0, maxfev=200000)
        r2 = 1 - np.var(yy[m] - f(tl[m], *popt)) / np.var(yy[m])
        print(f"  主通道前10s {nm}: params={np.round(popt,4)} R²={r2:.5f}")
    except Exception as e:
        print(f"  主通道前10s {nm}: 失败 {e}")
