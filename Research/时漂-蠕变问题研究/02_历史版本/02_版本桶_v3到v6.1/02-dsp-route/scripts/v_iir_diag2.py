# -*- coding: utf-8 -*-
"""§6.1 的 IIR：偏差到底来自「系数错了」还是「多项式求值相消」？
方法：不看 b/a 的频响，直接在**系数上**验证三项必要条件：
  R1 分母 a 是否等于 bilinear(N(s), D(s)) 的分母（正向系统极点的离散像）
  R2 分子零点是否等于分母零点（零极点对消的直接判据）
  R3 用「剩余传递函数」G(z)·H(z) 的直流/低频值验证对消质量
"""
import numpy as np
from scipy import signal


def N_D(a1, t1, a2, t2):
    return ([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2], [t1 * t2, t1 + t2, 1.0])


CASES = [
    (0.192, 3.46, 6.000, 1504.5),
    (0.196, 5.05, 4.958, 2194.3),
    (0.150, 1.20, 0.200, 45.0),
    (0.211, 6.01, 0.178, 82.5),
]
FS = 100.5
DT = 1.0 / FS

print(f"{'a1':>6}{'T1':>7}{'a2':>7}{'T2':>9}|{'a 与正向分母差':>15}|"
      f"{'b 零点 vs a 零点':>28}|{'零点/极点 相对距离':>20}")
print("-" * 100)
for a1, t1, a2, t2 in CASES:
    N, D = N_D(a1, t1, a2, t2)
    b, a = signal.bilinear([D[0] * N[2], D[1] * N[2], D[2] * N[2]], N, fs=FS)
    b, a = b / a[0], a / a[0]
    bf, af = signal.bilinear(N, D, fs=FS)
    af = af / af[0]
    da = np.abs(a - af).max()
    zb = np.sort(np.roots(b))
    za = np.sort(np.roots(a))
    rel = np.abs(zb - za) / np.abs(za)
    print(f"{a1:6.3f}{t1:7.2f}{a2:7.3f}{t2:9.1f}|{da:15.3e}|"
          f"{np.array2string(zb, precision=7):>28}|{np.array2string(rel, precision=3):>20}")

print("\nR3：剩余传递函数 R(z) = G(z)·H(z) 应为常数 1（=1/H_B·H_B）")
print("    这里用「正向 bf/af 与补偿 b/a 级联」在若干频率上的实测值：")
for a1, t1, a2, t2 in CASES:
    N, D = N_D(a1, t1, a2, t2)
    b, a = signal.bilinear([D[0] * N[2], D[1] * N[2], D[2] * N[2]], N, fs=FS)
    b, a = b / a[0], a / a[0]
    bf, af = signal.bilinear(N, D, fs=FS)
    bf, af = bf / af[0], af / af[0]
    # 级联：分子 bf*b，分母 af*a
    bc = np.convolve(bf, b)
    ac = np.convolve(af, a)
    w = np.array([1e-6, 1e-4, 1e-3, 1e-2, 1e-1]) * np.pi
    z = np.exp(1j * w)
    # 用 z-1 变换避免相消：逐点用 Horner 在 z 上直接算不方便；改用 freqz
    ww, hh = signal.freqz(bc, ac, worN=200000)
    sel = [0, 20, 200, 2000, 20000]
    vals = "  ".join(f"{np.abs(hh[i]):.6f}" for i in sel)
    print(f"  a1={a1:.3f} T2={t2:7.1f}: |R| @ω={['%.0e' % (np.pi*200000 and ww[i]) for i in sel]} = {vals}")

print("\nR4：把残余对消量化成「慢极点残差时间常数」")
for a1, t1, a2, t2 in CASES:
    N, D = N_D(a1, t1, a2, t2)
    b, a = signal.bilinear([D[0] * N[2], D[1] * N[2], D[2] * N[2]], N, fs=FS)
    b, a = b / a[0], a / a[0]
    za = np.sort(np.roots(a))
    zb = np.sort(np.roots(b))
    for zp, zz in zip(za, zb):
        tau_p = -DT / np.log(zp)
        tau_z = -DT / np.log(zz)
        print(f"  a1={a1:.3f} T2={t2:7.1f}: 极点 τ={tau_p:9.2f}s  最近零点 τ={tau_z:9.2f}s  "
              f"相对偏差={100*abs(tau_z-tau_p)/tau_p:6.2f}%")
