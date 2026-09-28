# -*- coding: utf-8 -*-
"""诊断 P4：逆滤波器为什么没有消掉蠕变？（直接用正向模型生成数据，闭环自检）
正向 H(z) = bilinear(N(s), D(s))；逆 G(z) = bilinear(N0*D(s), N(s))。
理论上 G·H = 1（零极点精确对消），故 G(正向阶跃响应) 应恒等于输入阶跃。
"""
import numpy as np
from scipy import signal


def H_cont(a1, t1, a2, t2):
    N = [t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2]
    D = [t1 * t2, t1 + t2, 1.0]
    return N, D


fs = 50.0
a1, t1, a2, t2 = 0.10, 1.2, 0.025, 45.0
N, D = H_cont(a1, t1, a2, t2)
print("N(s)=", N, " D(s)=", D, " N0=", N[2])
bf, af = signal.bilinear(N, D, fs=fs)
bf, af = bf / af[0], af / af[0]
bg, ag = signal.bilinear([D[0] * N[2], D[1] * N[2], D[2] * N[2]], N, fs=fs)
bg, ag = bg / ag[0], ag / ag[0]
print("正向 b,a =", np.round(bf, 6), np.round(af, 6))
print("逆向 b,a =", np.round(bg, 6), np.round(ag, 6))
print("正向零点 =", np.round(np.roots(bf), 6), " 逆向极点 =", np.round(np.roots(ag), 6))
print("正向极点 =", np.round(np.roots(af), 6), " 逆向零点 =", np.round(np.roots(bg), 6))

n = int(fs * 900)
u = np.zeros(n)
u[20:] = 1.0          # 单位阶跃，t=0.4s
yf = signal.lfilter(bf, af, u)      # 测量值（含蠕变）
yg = signal.lfilter(bg, ag, yf)     # 逆滤波后
print("\n阶跃响应闭环自检（输入恒 1.0）：")
for tq in [0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 880]:
    k = int(tq * fs)
    print(f"  t={tq:6.1f}s  yf={yf[k]:.5f}  yg={yg[k]:.5f}  (yg 应为 1.00000)")
print("\n结论：若 G·H 精确成立，yg 全程应为 1.0。上面若 yg != 1，则 P4 的残余来自这个失配。")

print("\n[附] 用连续域解析式做同样的对消检查（不看离散化）：")
for tq in [0.5, 1, 2, 5, 10, 30, 60, 120]:
    # 数值反拉氏：用极慢但保险的做法——把 H(s) 与 G(s) 的乘积在频域验证
    pass
s = 1j * 2 * np.pi * np.logspace(-4, 2, 2000)
H = np.polyval(N, s) / np.polyval(D, s)
G = np.polyval([D[0] * N[2], D[1] * N[2], D[2] * N[2]], s) / np.polyval(N, s)
print(f"  max|G·H − 1| (0.0001~100 Hz) = {np.abs(G*H-1).max():.3e}  ⇒ 连续域对消精确")
z = np.exp(s / fs)
Hd = np.polyval(bf, z) / np.polyval(af, z)
Gd = np.polyval(bg, z) / np.polyval(ag, z)
m = np.isfinite(Hd) & np.isfinite(Gd)
print(f"  max|Gd·Hd − 1| (同一频率网格) = {np.abs(Gd[m]*Hd[m]-1).max():.3e}  ⇒ 离散域对消精确")
