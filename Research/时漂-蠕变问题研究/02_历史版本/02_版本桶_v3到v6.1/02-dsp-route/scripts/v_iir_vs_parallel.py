# -*- coding: utf-8 -*-
"""诊断：§6.1 的 IIR 与并行实现到底哪个对？（用频域严格比对，绕开时域数值问题）"""
import numpy as np
from scipy import signal


def N_D(a1, t1, a2, t2):
    return ([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2], [t1 * t2, t1 + t2, 1.0])


def iir_from_params(a1, t1, a2, t2, fs):
    N, D = N_D(a1, t1, a2, t2)
    num_s = [D[0] * N[2], D[1] * N[2], D[2] * N[2]]
    b, a = signal.bilinear(num_s, N, fs=fs)
    return b / a[0], a / a[0]


CASES = [
    # (a1, t1, a2, t2) 覆盖「慢相很长」与「快慢相近」两种极端
    (0.192, 3.46, 6.000, 1504.5),
    (0.196, 5.05, 4.958, 2194.3),
    (0.116, 2.34, 0.310, 68.2),
    (0.116, 3.04, 0.141, 210.3),
    (0.099, 3.58, 0.155, 1660.0),
    (0.197, 4.28, 0.245, 37.8),
    (0.211, 6.01, 0.178, 82.5),
    (0.15, 1.2, 0.20, 45.0),
]
FS = 100.5
DT = 1.0 / FS

print(f"{'a1':>6}{'T1':>8}{'a2':>7}{'T2':>9}|{'N0':>7}|{'||b||1':>11}|{'|B(1)|':>11}|{'kappa':>10}|"
      f"{'IIR 1/H_A 误差':>15}|{'并行 1/H_A 误差':>17}")
print("-" * 112)
for a1, t1, a2, t2 in CASES:
    b, a = iir_from_params(a1, t1, a2, t2, FS)
    N0 = 1 + a1 + a2
    w = np.linspace(1e-9, np.pi, 4001)
    z = np.exp(1j * w)
    # §6.1 的 IIR 在 s 域的解析值（与离散系数无关，用于交叉验证）
    H_iir = np.polyval(b, z) / np.polyval(a, z)
    # 目标：1/H_A = N0·[1 + Σ Kᵢ·HPᵢ]，HPᵢ = sτᵢ/(1+sτᵢ)
    s = (2 / DT) * (1 - z ** -1) / (1 + z ** -1)
    K1 = (a1 / N0) / (1 + a1)
    K2 = (a2 / N0) / (1 + a2)
    HP1 = s * t1 / (1 + s * t1)
    HP2 = s * t2 / (1 + s * t2)
    H_target = N0 * (1 + K1 * HP1 + K2 * HP2)
    # 并行实现（逐通道 lfilter 版本）的频域等价式
    tau1, tau2 = np.exp(-DT / t1), np.exp(-DT / t2)
    H_par = N0 * (1 + K1 * (1 - z ** -1) / (1 - tau1 * z ** -1)
                    + K2 * (1 - z ** -1) / (1 - tau2 * z ** -1))
    e_iir = np.abs(H_iir - H_target).max() / np.abs(H_target).max()
    e_par = np.abs(H_par - H_target).max() / np.abs(H_target).max()
    dc_b = float(np.polyval(b, 1.0))
    print(f"{a1:6.3f}{t1:8.2f}{a2:7.3f}{t2:9.1f}|{N0:7.3f}|{np.sum(np.abs(b)):11.4g}|{dc_b:11.4g}|"
          f"{np.sum(np.abs(b))/abs(dc_b):10.3g}|{e_iir:15.3e}|{e_par:17.3e}")

print("\n说明：H_target 是用 s 域解析式算的「理想 1/H_A」（Tustin 映射），")
print("      e_iir / e_par 分别是两种离散实现相对它的最大相对偏差。")
print("      若 e_par ≈ 0 而 e_iir 很大 ⇒ §6.1 的二阶多项式系数本身已不可用（数值条件问题）。")

# 时域验证：直接过滤一段确定性信号，看两者差多少
print("\n时域交叉验证（1 Hz 正弦 + 阶跃，各 60s）：")
n = int(60 * FS)
tt = np.arange(n) * DT
x = 1.0 + 0.3 * np.sin(2 * np.pi * 0.05 * tt)
x[: int(5 * FS)] = 0.0
for a1, t1, a2, t2 in CASES:
    b, a = iir_from_params(a1, t1, a2, t2, FS)
    N0 = 1 + a1 + a2
    yi = signal.lfilter(b, a, x)
    yp = N0 * x.copy()
    for ai, ti in ((a1, t1), (a2, t2)):
        tp = np.exp(-DT / ti)
        Ki = (ai / N0) / (1 + ai)
        yp = yp + N0 * Ki * signal.lfilter([1.0, -1.0], [1.0, -tp], x)
    print(f"  a1={a1:.3f} T1={t1:6.2f} a2={a2:.3f} T2={t2:7.1f}: "
          f"max|IIR−并行|={np.abs(yi-yp).max():10.4f}  "
          f"IIR 范围=[{yi.min():.3f},{yi.max():.3f}]  并行范围=[{yp.min():.3f},{yp.max():.3f}]")
