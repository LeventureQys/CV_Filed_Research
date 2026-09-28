# -*- coding: utf-8 -*-
"""唯一裁决：把 §6.1 的系数作用在 §2.1 时域蠕变信号上，到底收敛到哪个电平？
用卷积（解析可控）而不是 stateful lfilter，排除初始条件/索引干扰。
"""
import numpy as np
from scipy import signal

FS = 100.5
DT = 1.0 / FS


def coeffs(a1, t1, a2, t2, fs):
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0], N


A1, T1, A2, T2 = 0.15, 1.2, 0.20, 45.0
b, a, N = coeffs(A1, T1, A2, T2, FS)
N0 = N[2]
print(f"b = {b}\na = {a}\nN0 = {N0}")

# 1) 阶跃响应（离散，直接用 lfilter）
n = int(1200 * FS)
u = np.ones(n)
s = signal.lfilter(b, a, u)
print(f"\n[1] 单位阶跃响应: t=1s {s[int(FS)]:.6f}  20s {s[int(20*FS)]:.6f}  "
      f"100s {s[int(100*FS)]:.6f}  600s {s[int(600*FS)]:.6f}  1199s {s[-1]:.6f}")
print(f"    b.sum()={b.sum():.9f}  a.sum()={a.sum():.9f}  b.sum()/a.sum()={b.sum()/a.sum():.6f}")

# 2) E·h_A 的解析构造（用卷积核）
imp = np.zeros(n)
imp[0] = 1.0
hA = signal.lfilter(
    *signal.bilinear(N / N0, [T1 * T2, T1 + T2, 1.0], fs=FS)[::1], imp) if False else None
# 更直接：H_A = N(s)/(N0 D(s)) 的离散
bA, aA = signal.bilinear(N / N0, [T1 * T2, T1 + T2, 1.0], fs=FS)
bA, aA = bA / aA[0], aA / aA[0]
hA = signal.lfilter(bA, aA, imp)
print(f"\n[2] H_A 冲激响应累加（= H_A 直流增益）: {np.cumsum(hA)[-1]:.6f} 应为 1")
print(f"    H_A 阶跃响应稳态: {signal.lfilter(bA, aA, u)[-1]:.6f} 应为 1")

# 3) 时域公式 vs H_A 的阶跃响应（两者是否同一个系统）
E = 1000.0
t = np.arange(n) * DT
i_on = int(60 * FS)
uu = np.clip(t - t[i_on], 0, None)
X_formula = np.zeros(n)
X_formula[i_on:] = E * (1 + A1 * (1 - np.exp(-uu[i_on:] / T1)) + A2 * (1 - np.exp(-uu[i_on:] / T2)))
X_conv = E * np.convolve(hA, np.r_[np.zeros(i_on), np.ones(n - i_on)])[:n]
print(f"\n[3] 时域公式 X vs E·(H_A 阶跃响应)：")
for k in [i_on, i_on + int(FS), i_on + int(20 * FS), i_on + int(300 * FS), n - 1]:
    print(f"    t={t[k]:7.1f}s: 公式={X_formula[k]:10.3f}   卷积={X_conv[k]:10.3f}   差={X_formula[k]-X_conv[k]:+.4f}")

# 4) 把 doc 的逆滤波器作用在 X_formula 上
Y = signal.lfilter(b, a, X_formula)
print(f"\n[4] 逆滤波器作用在 X_formula 上（真值 1000）：")
for k in [i_on, i_on + int(FS), i_on + int(5 * FS), i_on + int(60 * FS), i_on + int(600 * FS), n - 1]:
    print(f"    t={t[k]:7.1f}s: Y={Y[k]:10.3f}  (Y/X={Y[k]/X_formula[k]:.6f})")

# 5) 逆滤波器作用在 X_conv 上（= 真正的 E·h_A）
Yc = signal.lfilter(b, a, X_conv)
print(f"\n[5] 逆滤波器作用在 E·h_A 上（真值 1000）：")
for k in [i_on + int(FS), i_on + int(5 * FS), i_on + int(60 * FS), i_on + int(600 * FS), n - 1]:
    print(f"    t={t[k]:7.1f}s: Y={Yc[k]:10.3f}")

# 6) 结论性检查：级联
bc = np.convolve(bA, b)
ac = np.convolve(aA, a)
w, h = signal.freqz(bc, ac, worN=50000)
print(f"\n[6] 级联 H_A→逆滤波器 的频响：|H(0)|={np.abs(h[0]):.6f}  最大={np.abs(h).max():.6f}  最小={np.abs(h).min():.6f}")
