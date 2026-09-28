# -*- coding: utf-8 -*-
"""§6.1 的 IIR 在真实拟合参数下的精度与稳定性实测。

用「理想解析解」做基准：不做任何滤波，
直接按 §2.1 时域公式构造蠕变信号 X = E·[1 + Σaᵢ(1-e^{-u/τᵢ})]，
则 1/H_A 的理想输出恒为 E（弹性幅度）。
把 §6.1 的 IIR 与并行实现分别作用上去，看各自的输出与 E 的偏差。

同时报出 κ = ‖b‖₁/|B(1)|（二阶分子的大幅值相消程度）。
"""
import numpy as np
from scipy import signal

FS = 100.5
DT = 1.0 / FS


def N_D(a1, t1, a2, t2):
    return ([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2], [t1 * t2, t1 + t2, 1.0])


def iir(a1, t1, a2, t2):
    N, D = N_D(a1, t1, a2, t2)
    b, a = signal.bilinear([D[0] * N[2], D[1] * N[2], D[2] * N[2]], N, fs=FS)
    return b / a[0], a / a[0]


def par(X, a1, t1, a2, t2):
    N0 = 1 + a1 + a2
    out = N0 * X
    for ai, ti in ((a1, t1), (a2, t2)):
        if ai <= 0:
            continue
        tp = float(np.exp(-DT / ti))
        Ki = (ai / N0) / (1 + ai)
        out = out + N0 * Ki * signal.lfilter([1.0, -1.0], [1.0, -tp], X, axis=0)
    return out


CASES = [
    ("右拇指/数据1 拟合", 0.192, 3.46, 6.000, 1504.5),
    ("右拇指/数据2 拟合", 0.196, 5.05, 4.958, 2194.3),
    ("dsp.md §6.1 示例", 0.150, 1.20, 0.200, 45.0),
    ("四指/数据2 拟合", 0.211, 6.01, 0.178, 82.5),
]

T_END = 600.0
n = int(T_END * FS)
t = np.arange(n) * DT
i0 = int(5.0 * FS)          # 5s 处加载
E = 1.0

print(f"{'参数组':<18}|{'κ=‖b‖1/|B(1)|':>16}|"
      f"{'IIR 输出@5.2s':>14}{'@10s':>10}{'@60s':>10}{'@600s':>10}|"
      f"{'并行@600s':>10}|{'IIR 最大误差':>13}")
print("-" * 116)
for label, a1, t1, a2, t2 in CASES:
    b, a = iir(a1, t1, a2, t2)
    kappa = float(np.sum(np.abs(b)) / abs(np.polyval(b, 1.0)))
    u = np.clip(t - t[i0], 0, None)
    X = np.zeros(n)
    X[i0:] = E * (1 + a1 * (1 - np.exp(-u[i0:] / t1)) + a2 * (1 - np.exp(-u[i0:] / t2)))
    Yi = signal.lfilter(b, a, X)
    Yp = par(X, a1, t1, a2, t2)
    err = np.abs(Yi - E)          # 理想输出恒为 E
    err = err[i0:]
    idx = [i0 + int(x * FS) for x in (0.2, 5.0, 55.0, 595.0)]
    print(f"{label:<18}|{kappa:16.3g}|" +
          "".join(f"{Yi[k]:14.4f}" if j == 0 else f"{Yi[k]:10.4f}" for j, k in enumerate(idx)) +
          f"|{Yp[idx[-1]]:10.4f}|{err.max():13.4f}")

print("\n【读法】理想 1/H_A 的输出应恒等于 E=1.0000。")
print("  · κ 越大 ⇒ 二阶分子存在越严重的大幅值相消，IIR 的数值可信度越低；")
print("  · IIR 最大误差 = 加载后整段内 |Y−1| 的最大值，含慢瞬态与相消误差。")

print("\n慢瞬态时间常数检验（对 X 施加一个纯阶跃，看 IIR 多久回到稳态）：")
for label, a1, t1, a2, t2 in CASES:
    b, a = iir(a1, t1, a2, t2)
    N, D = N_D(a1, t1, a2, t2)
    za = np.sort(np.roots(a))
    taup = [-DT / np.log(z) for z in za]
    print(f"  {label:<18} 离散极点 τ = {np.array2string(np.array(taup), precision=2)} s "
          f"（设计 τ1={t1}, τ2={t2}）")
