# -*- coding: utf-8 -*-
"""诊断：为什么 /N0 与不除 N0 得到一样的结果？（怀疑 dc() 取值不是 1.35）"""
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
bn, an = iir(D, N, fs)
print("N =", N, " D =", D, " gc =", gc)
print("bd =", bd, " ad =", ad)
print("bn =", bn, " an =", an)
print("sum(bd) =", bd.sum(), " sum(ad) =", ad.sum(), " ratio =", bd.sum() / ad.sum())
print("bd/gc =", bd / gc)
print("bn 与 bd/gc 的差 =", bn - bd / gc)
d = bd / (bd.sum() / ad.sum())
print("bd/dc =", d)
print("bn - bd/dc =", bn - d)
print("an - ad    =", an - ad)
