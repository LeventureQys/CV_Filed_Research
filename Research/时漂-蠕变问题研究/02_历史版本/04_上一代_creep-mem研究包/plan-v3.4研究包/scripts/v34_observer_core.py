# -*- coding: utf-8 -*-
"""v3.4 在线双态观测器 · 核心原型（无事件、无记忆、参数全局固定）。

逐通道模型（Kelvin-Voigt ×2，显示 = 弹性响应）：
  e = max(v − x1 − x2, 0)          # 弹性估计（受载门自然形成）
  受载(e>0):  x_k += dt·(r_k·e − x_k)/τk_c
  空载(e=0):  x_k += dt·(−x_k)/τk_r
  display = v − x1 − x2

物理含义：x_k 是材料非弹性应变（快/慢两个时间常数），受载时朝与载荷成正比的
饱和值增长、空载时恢复；显示 = 总响应 − 非弹性 = 纯弹性响应。
跨卸载自然连续（状态不清零），不依赖任何事件分类与历史账本。
"""
import numpy as np

# 全局默认参数（由规律辨识 v34_law_identify.py 的量级定标，不做逐会话拟合）
P = dict(r1=0.12, tc1=8.0, tr1=6.0,
         r2=0.20, tc2=300.0, tr2=150.0)


def observe(ts, V, p=None):
    """ts: (n,) 秒；V: (n, ch)。返回 (display(n,ch), x_sum(n,ch))。"""
    if p is None:
        p = P
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    X = np.empty((n, ch))
    D = np.empty((n, ch))
    t_prev = ts[0]
    for i in range(n):
        dt = ts[i] - t_prev
        t_prev = ts[i]
        dt = min(max(dt, 0.0), 0.1)
        e = np.maximum(V[i] - x1 - x2, 0.0)
        if dt > 0.0:
            loaded = e > 0.0
            dx1 = np.where(loaded, (p["r1"] * e - x1) / p["tc1"], -x1 / p["tr1"])
            dx2 = np.where(loaded, (p["r2"] * e - x2) / p["tc2"], -x2 / p["tr2"])
            x1 += dt * dx1
            x2 += dt * dx2
            x1 = np.maximum(x1, 0.0)
            x2 = np.maximum(x2, 0.0)
        X[i] = x1 + x2
        D[i] = V[i] - x1 - x2
    return D, X
