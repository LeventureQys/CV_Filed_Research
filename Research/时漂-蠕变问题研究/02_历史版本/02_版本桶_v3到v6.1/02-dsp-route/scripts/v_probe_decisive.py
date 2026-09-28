# -*- coding: utf-8 -*-
"""决定性实验：dsp.md §6.1 系数的直流增益、阶跃响应、以及作用在 E·h(t) 上的稳态。"""
import numpy as np
from scipy import signal


def N_D(a1, t1, a2, t2):
    return ([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2], [t1 * t2, t1 + t2, 1.0])


def iir(num_s, den_s, fs):
    b, a = signal.bilinear(num_s, den_s, fs=fs)
    return b / a[0], a / a[0]


a1, t1, a2, t2, fs = 0.15, 1.2, 0.20, 45.0, 50.0
N, D = N_D(a1, t1, a2, t2)
gc = N[2] / D[2]
bd, ad = iir([D[0] * gc, D[1] * gc, D[2] * gc], N, fs)
bf, af = iir(N, D, fs)

print("N =", N, "D =", D, "N0 =", N[2])
print("doc b =", bd, "\ndoc a =", ad)
print("fwd b =", bf, "\nfwd a =", af)
print("doc DC gain = polyval(b,1)/polyval(a,1) =", np.polyval(bd, 1.0) / np.polyval(ad, 1.0))
print("fwd DC gain = ", np.polyval(bf, 1.0) / np.polyval(af, 1.0))

n = int(fs * 6000)
u = np.zeros(n)
u[10:] = 1.0
sd = np.cumsum(signal.lfilter(bd, ad, u))   # doc 的阶跃响应
sf = np.cumsum(signal.lfilter(bf, af, u))   # 正向的阶跃响应
print("\ndoc 阶跃响应：首值 b0 =", bd[0])
for tq in [0.02, 0.1, 0.5, 1, 5, 20, 60, 200, 600, 1200, 3000, 5900]:
    k = 10 + int(tq * fs)
    print(f"  t={tq:7.1f}s  doc_step={sd[k]:.6f}   fwd_step={sf[k]:.6f}")

# 串起来：正向 → doc，看是否恒等
print("\n串联 fwd→doc（理想恒等，应恒为 1.0）：")
y = signal.lfilter(bd, ad, signal.lfilter(bf, af, u))
for tq in [0.02, 0.5, 5, 20, 60, 200, 600, 1200, 3000, 5900]:
    k = 10 + int(tq * fs)
    print(f"  t={tq:7.1f}s  串联输出={y[k]:.6f}")

# E·h(t) 送入 doc
print("\nE·h(t) 送入 doc（E=1000）：")
E = 1000.0
X = E * signal.lfilter(bf, af, u)
Y = signal.lfilter(bd, ad, X)
print("  X 首值 =", X[10], " X 600s =", X[10 + int(600 * fs)])
for tq in [0.02, 0.5, 5, 20, 60, 200, 600, 1200, 3000, 5900]:
    k = 10 + int(tq * fs)
    print(f"  t={tq:7.1f}s  X={X[k]:10.3f}  Y={Y[k]:12.3f}  Y/1000={Y[k]/1000:.4f}")
