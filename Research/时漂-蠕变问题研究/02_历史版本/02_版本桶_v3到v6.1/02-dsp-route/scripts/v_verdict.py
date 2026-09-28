# -*- coding: utf-8 -*-
"""最终裁决：§6.1 的 IIR 与并行实现，冲激响应/阶跃响应/频响三路比对。"""
import numpy as np
from scipy import signal

FS = 100.5
DT = 1.0 / FS
A1, T1, A2, T2 = 0.15, 1.2, 0.20, 45.0


def N_D(a1, t1, a2, t2):
    return np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2]), np.array([t1 * t2, t1 + t2, 1.0])


def iir(a1, t1, a2, t2):
    N, D = N_D(a1, t1, a2, t2)
    b, a = signal.bilinear(D * N[2], N, fs=FS)
    return b / a[0], a / a[0]


N, D = N_D(A1, T1, A2, T2)
N0 = N[2]
b, a = iir(A1, T1, A2, T2)
print(f"N0 = {N0}, N = {N}, D = {D}")
print(f"IIR  b = {b}\nIIR  a = {a}")

# --- 真值：1/H_A(s) = N0*D(s)/N(s) 的部分分式 + 常数 ---
# H_A(s) = N(s)/(N0 D(s))。1/H_A = N0 D(s)/N(s)
# 留数法：r_i = N0 D(p_i)/N'(p_i)，p_i 为 N(s) 的根
p = np.roots(N)
print(f"N(s) 的根 p = {p}  ->  时间常数 {-1/p.real}")
Np = np.polyder(N)
res = [N0 * np.polyval(D, pi) / np.polyval(Np, pi) for pi in p]
print(f"留数 r = {res}")
print(f"常数项 = N0*D2/N2 = {N0*D[0]/N[0]:.6f}")
# 系统 = 常数 + Σ r_i/(s-p_i)（p_i<0 → 指数衰减 r_i e^{p_i t}）
# 冲激响应（连续）：c0*δ(t) + Σ r_i e^{p_i t} u(t)
c0 = N0 * D[0] / N[0]
print(f"\n连续冲激响应: {c0:.6f}·δ(t) + " +
      " + ".join(f"{r.real:.4f}·e^({pi.real:.5f}t)" for r, pi in zip(res, p)))
print(f"直流增益 = c0 + Σ(-r_i/p_i) = {c0 + sum(-r.real/pi.real for r, pi in zip(res, p)):.6f}")

n = int(200 * FS)
imp = np.zeros(n)
imp[0] = 1.0
hi = signal.lfilter(b, a, imp)
hp = np.zeros(n)
hp[0] = c0 * FS                      # 离散化：δ 近似为 1/dt 脉冲 × dt
for r, pi in zip(res, p):
    hp += r.real * np.exp(pi.real * np.arange(n) * DT)
print("\n冲激响应累积和（应等于直流增益）：")
for k in [0, 1, 5, 50, 500, 5000, n - 1]:
    print(f"  k={k:6d}: IIR cumsum={np.cumsum(hi)[k]:12.6f}   解析 cumsum={np.cumsum(hp)[k]:12.6f}")

# 阶跃响应（离散 = cumsum(impulse)*dt... 直接用 lfilter 打阶跃）
u = np.ones(n)
si = signal.lfilter(b, a, u)
sp = np.cumsum(hp)
print("\n阶跃响应：")
for k in [1, 5, 50, 500, 2000, 5000, n - 1]:
    print(f"  t={k*DT:8.3f}s: IIR={si[k]:12.6f}   解析={sp[k]:12.6f}")

# 频响
w, h = signal.freqz(b, a, worN=20000)
s = (2 / DT) * (1 - np.exp(-1j * w)) / (1 + np.exp(-1j * w))
Hanalog = N0 * np.polyval(D, s) / np.polyval(N, s)
print("\n频响比对 |IIR| vs |解析|：")
for i in [0, 1, 10, 100, 1000, 5000, 19999]:
    print(f"  f={w[i]/(2*np.pi*DT):10.5f} Hz: |IIR|={abs(h[i]):10.5f}  |解析|={abs(Hanalog[i]):10.5f}")
