# -*- coding: utf-8 -*-
"""v3.4 观测器 v3：加零点跟踪（修复空载漂没）。

问题（录制 20260919_190545_1d4d3b，全程空载）：传感器零点偏置 ~2120 ADC，
v2 的 e=max(v−x1−x2,0) 把零点偏置当弹性载荷 ⇒ 空载下快/慢态持续生长、
显示一路下沉 −315 ADC。

修复：逐通道维护零点 zero（仅在 y=v−zero 处于近零带时以 τ0 向读数靠拢，
受载时冻结）；观测器跑在 y 上：
  y = v − zero；y_max = 慢衰减的量程包络；idle = y < idle_frac·y_max
  idle 时 zero += (dt/τ0)·(v − zero)
  e = max(y − x1 − x2, 0)；（以下与 v2 相同，作用在 y 上）
  显示 = zero + y − x1 − x2
空载 ⇒ y≈0 ⇒ 显示≈原始读数（直通）；加载 ⇒ zero 冻结，行为同 v2。
"""
import numpy as np

P3 = dict(r1=0.12, tc1=8.0, tr1=6.0,
          r2max=0.35, tr2=150.0,
          tau_slope=3.0, slope_gate=0.02, slope_cap=0.01,
          tau_zero=8.0,        # 零点跟踪时间常数（仅近零带内）
          idle_frac=0.05,      # 近零带：y < 5%·y_max 视为空载
          y_max_tau=600.0)     # 量程包络的衰减时间常数（防止历史大量程压死小量程）


def observe3(ts, V, p=None):
    if p is None:
        p = P3
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    zero = V[0].copy()
    y_max = np.maximum(V[0] - zero, 0.0)
    v_lp = V[0].copy()
    X = np.empty((n, ch))
    D = np.empty((n, ch))
    t_prev = ts[0]
    for i in range(n):
        dt = ts[i] - t_prev
        t_prev = ts[i]
        dt = min(max(dt, 0.0), 0.1)
        v = V[i]
        if dt > 0.0:
            y = v - zero
            y_max = np.maximum(y_max * np.exp(-dt / p["y_max_tau"]),
                               np.maximum(y, 0.0))
            idle = y < p["idle_frac"] * np.maximum(y_max, 1.0)
            zero = np.where(idle, zero + (dt / p["tau_zero"]) * (v - zero), zero)
            y = v - zero
            # 快态（固定模型）
            e_now = np.maximum(y - x1 - x2, 0.0)
            dx1_rate = np.where(e_now > 0.0,
                                (p["r1"] * e_now - x1) / p["tc1"],
                                -x1 / p["tr1"])
            x1 = np.maximum(x1 + dt * dx1_rate, 0.0)
            # 慢态（慢漂移跟踪）
            slope = (v - v_lp) / p["tau_slope"]
            v_lp = v_lp + (dt / p["tau_slope"]) * (v - v_lp)
            e = np.maximum(y - x1 - x2, 0.0)
            rate_cap = p["slope_cap"] * np.maximum(e, 1.0)
            gate = p["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1_rate, -rate_cap, rate_cap)
            dx2 = np.where((e > 0.0) & (np.abs(slope) < gate), dx2, 0.0) * dt
            x2 = x2 + dx2
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, p["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / p["tr2"], 0.0))
        X[i] = x1 + x2
        D[i] = v - x1 - x2
    return D, X
