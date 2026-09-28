# -*- coding: utf-8 -*-
"""§6.1 逆滤波器的瞬态代价：用「真实工况形状」的载荷测它在保压期偏多久。

关键事实（本脚本的基准）：
  §6.1 的 IIR 频响与解析式 1/H_A 完全一致（6 位小数），直流增益 = 1 —— 数学是对的。
  但 1/H_A 的极点是 N(s) 的根（τ ≈ 1.04s 与 38.4s），最慢那个 38.4s 决定了
  「阶跃后要多久才回到真值」。真实载荷不是理想阶跃（有机械建立时间），
  这里用 0.3s 斜坡上升的载荷，比较接近手指/气缸加载。
"""
import numpy as np
from scipy import signal

FS = 100.5
DT = 1.0 / FS


def iir(a1, t1, a2, t2, fs=FS):
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0]


CASES = [
    ("dsp.md 标称  a1=.15 τ1=1.2s a2=.20 τ2=45s", 0.15, 1.2, 0.20, 45.0),
    ("左拇指实测拟合 a1=.116 τ1=3.0s a2=.14 τ2=210s", 0.116, 3.04, 0.141, 210.3),
    ("右拇指实测拟合 a1=.19  τ1=3.5s a2=.31 τ2=68s", 0.192, 3.46, 0.310, 68.2),
]

T_END, T_ON, T_RAMP = 900.0, 60.0, 0.3
n = int(T_END * FS)
t = np.arange(n) * DT
E = 1000.0                                   # 弹性幅度（真值）

print(f"{'参数组':<44}|{'真值':>7}|" +
      "".join(f"{f'{x}s':>10}" for x in [1, 5, 15, 30, 60, 120, 300, 600]) +
      f"|{'最大过冲':>10}|{'回到±5%':>10}|{'回到±1%':>10}")
print("-" * 152)
for label, a1, t1, a2, t2 in CASES:
    b, a = iir(a1, t1, a2, t2)
    # 真实载荷：0.3s 斜坡建立，之后恒定保压
    ramp = np.clip((t - T_ON) / T_RAMP, 0.0, 1.0)
    u = np.clip(t - T_ON, 0, None)
    # 测量值 = 弹性分量 ⊗ h_A(一阶并联形式，等价且数值稳定)
    N0 = 1 + a1 + a2
    X = np.zeros(n)
    for ai, ti in ((a1, t1), (a2, t2)):
        hp = signal.lfilter([1.0, 0.0], [1.0, -np.exp(-DT / ti)], ramp * E * N0 * ai / N0, axis=0)
        # 实际用物理模型直接搭：X = E*ramp*(1 + Σ ai(1-e^-u/τi))
        pass
    X = E * ramp * (1 + a1 * (1 - np.exp(-u / t1)) + a2 * (1 - np.exp(-u / t2)))
    Y = signal.lfilter(b, a, X)
    i_on = int(T_ON * FS)
    cols = []
    for x in [1, 5, 15, 30, 60, 120, 300, 600]:
        k = i_on + int(x * FS)
        cols.append(f"{Y[k]:10.1f}" if k < n else f"{'-':>10}")
    seg = Y[i_on:]
    over = 100 * (seg.max() - E) / E
    # 回到 ±5% / ±1% 的时刻（持续满足）
    def settle(frac):
        thr = frac * E
        ok = np.abs(seg - E) <= thr
        # 找最后一个不满足点
        bad = np.where(~ok)[0]
        return (bad[-1] + 1) * DT if len(bad) else 0.0
    print(f"{label:<44}|{E:7.0f}|" + "".join(cols) +
          f"|{over:9.1f}%|{settle(0.05):9.1f}s|{settle(0.01):9.1f}s")

print("\n【说明】真值 = 弹性幅度 1000。表中数值是补偿后的显示值，理想应恒为 1000。")
print("  「回到±5%/±1%」= 加载后显示值持续落在真值 ±5%/±1% 之内所需的时长。")
print("  过冲来自逆滤波器的慢极点（= 正向系统零点，τ 介于 τ1 与 τ2 之间）。")
