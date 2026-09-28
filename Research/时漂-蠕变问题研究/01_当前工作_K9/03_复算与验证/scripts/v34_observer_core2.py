# -*- coding: utf-8 -*-
"""v3.4 观测器 v2：慢态从「固定幅度模型」改为「实测慢漂移跟踪」。

改动（相对 v34_observer_core.py）：
  x1（快态）不变：固定模型（r1/τc1/τr1），负责快相爬升；
  x2（慢态）：受载时按**输入的平滑慢漂移速率**积分（扣除快态已解释的部分），
             上限 r2max·e 钳位；空载按 τr2 恢复。
    dx2/dt = clamp(slope_v − dx1/dt, ≤ slope_cap)，其中 slope_v 是 v 的低通(τs)导数；
    沿/快速变化(|slope|>gate)时不积分（那是 x1/事件的事）。
  效果：恒载下 x2 精确吃掉实测蠕变 ⇒ 显示按构造钉在弹性电平，不随 r 假设衰减。
"""
import numpy as np

P2 = dict(r1=0.12, tc1=8.0, tr1=6.0,
          r2max=0.35, tr2=150.0,
          tau_slope=3.0,        # 慢漂移速率估计的低通时间常数
          slope_gate=0.02,      # |dv/dt| 超过 2%·e/s 视为沿/快变，不积分
          slope_cap=0.01)       # 慢态积分速率上限（1%·e/s）


def observe2(ts, V, p=None):
    if p is None:
        p = P2
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    v_lp = V[0].copy()               # 低通后的 v（估计 slope 用）
    X = np.empty((n, ch))
    D = np.empty((n, ch))
    t_prev = ts[0]
    for i in range(n):
        dt = ts[i] - t_prev
        t_prev = ts[i]
        dt = min(max(dt, 0.0), 0.1)
        v = V[i]
        if dt > 0.0:
            a1 = dt / p["tc1"]
            # 快态（固定模型）
            e_now = np.maximum(v - x1 - x2, 0.0)
            dx1 = np.where(e_now > 0.0,
                           (p["r1"] * e_now - x1) / p["tc1"],
                           -x1 / p["tr1"]) * dt
            x1 = np.maximum(x1 + dx1, 0.0)
            # 慢漂移速率估计（低通导数）
            v_lp_new = v_lp + (dt / p["tau_slope"]) * (v - v_lp)
            slope = (v_lp_new - v_lp) / dt
            v_lp = v_lp_new
            e = np.maximum(v - x1 - x2, 0.0)
            rate_cap = p["slope_cap"] * np.maximum(e, 1.0)
            gate = p["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1 / dt, -rate_cap, rate_cap)
            dx2 = np.where((e > 0.0) & (np.abs(slope) < gate), dx2, 0.0) * dt
            x2 = x2 + dx2
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, p["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / p["tr2"], 0.0))
        X[i] = x1 + x2
        D[i] = v - x1 - x2
    return D, X
