# -*- coding: utf-8 -*-
"""§6.1 的 IIR vs 并行实现：用真实拟合参数逐点比冲激响应，判定是否等价。"""
import numpy as np
from scipy import signal

FS = 100.5
DT = 1.0 / FS


def iir(a1, t1, a2, t2, fs):
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0]


def parallel_imp(a1, t1, a2, t2, n, dt):
    """1/H_A = N0·[1 + Σ Kᵢ HPᵢ],  HPᵢ = (1-z⁻¹)/(1-e^{-dt/τᵢ}z⁻¹),  Kᵢ=(aᵢ/N0)/(1+aᵢ)"""
    N0 = 1 + a1 + a2
    imp = np.zeros(n)
    imp[0] = 1.0
    y = N0 * imp
    for ai, ti in ((a1, t1), (a2, t2)):
        if ai <= 0:
            continue
        Ki = (ai / N0) / (1 + ai)
        y = y + N0 * Ki * signal.lfilter([1.0, -1.0], [1.0, -np.exp(-dt / ti)], imp)
    return y


CASES = [
    ("dsp.md 标称", 0.15, 1.2, 0.20, 45.0),
    ("左拇指拟合", 0.099, 3.58, 0.155, 1660.0),
    ("右拇指拟合", 0.192, 3.46, 6.000, 1504.5),
    ("四指拟合", 0.197, 4.28, 0.245, 37.8),
]
N = 60000
print(f"{'参数组':<14}|{'κ=‖b‖1/|b.sum()|':>18}|{'Σ冲激响应 IIR':>15}|{'Σ冲激响应 并行':>16}|"
      f"{'max|h_IIR−h_par|':>18}|{'相对误差':>10}")
print("-" * 110)
for label, a1, t1, a2, t2 in CASES:
    b, a = iir(a1, t1, a2, t2, FS)
    imp = np.zeros(N)
    imp[0] = 1.0
    hi = signal.lfilter(b, a, imp)
    hp = parallel_imp(a1, t1, a2, t2, N, DT)
    kappa = float(np.sum(np.abs(b)) / abs(b.sum()))
    err = np.abs(hi - hp).max()
    rel = err / max(np.abs(hp).max(), 1e-12)
    print(f"{label:<14}|{kappa:18.4g}|{hi.sum():15.6f}|{hp.sum():16.6f}|{err:18.4g}|{rel:10.3g}")

print("\n若两列 Σ冲激响应 都等于 N0 且 max 差 ≈ 0 ⇒ 两种实现等价（IIR 无实质数值缺陷）。")
