# -*- coding: utf-8 -*-
"""91_observer：CreepObserverCompensator（v3.4 / K9）的 Python 忠实移植。

对照 src/domain/drift_v6/creep_observer.cpp 逐行移植（2026-09-27 版本），
仅用于离线复算；Params 默认值取自 creep_observer.h。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np


@dataclass
class Params:
    r_fast: float = 0.01
    tau_c_fast_s: float = 40.0
    tau_r_fast_s: float = 6.0
    r_slow_max: float = 0.2
    tau_r_slow_s: float = 150.0
    tau_slope_s: float = 1.0
    slope_gate_frac: float = 0.05
    slope_cap_frac: float = 0.05
    tau_zero_s: float = 8.0
    idle_frac: float = 0.05
    y_max_tau_s: float = 600.0
    edge_slope_thres: float = 60.0
    edge_refract_s: float = 2.0
    edge_boost_s: float = 2.0
    tau_c_fast_boost_s: float = 2.0
    soft_unfreeze_s: float = 8.0
    hold_eps: float = 2.0
    hold_tau_s: float = 0.5
    slow_confirm_s: float = 5.0
    y_floor_tau_s: float = 300.0
    tau_r_slow_idle_s: float = 0.5
    bypass_release_frac: float = 1.15
    bypass_engage_frac: float = 1.25
    bypass_noise_sigma: float = 3.0
    bypass_base_tau_s: float = 10.0
    ramp_slope_min: float = 0.5
    ramp_full_s: float = 4.0


def default_with(over: dict) -> Params:
    p = Params()
    names = {f.name for f in fields(Params)}
    for k, v in over.items():
        if k not in names:
            raise KeyError(k)
        setattr(p, k, float(v))
    return p


def run(t: np.ndarray, V: np.ndarray, p: Params) -> dict:
    """t: (n,) 秒；V: (n, m) 逐帧通道值。返回显示值与逐帧状态和（诊断用）。"""
    n_, m = V.shape
    out = V.astype(np.float64).copy()
    zero = V[0].astype(np.float64).copy()
    y_max = np.zeros(m)
    v_lp = V[0].astype(np.float64).copy()
    v_fast_lp = V[0].astype(np.float64).copy()
    y_floor = V[0].astype(np.float64).copy()
    load_dwell = np.zeros(m)
    ramp_dwell = np.zeros(m)
    t_edge = np.full(m, 1.0e6)
    applied = np.zeros(m)
    x_fast = np.zeros(m)
    x_slow = np.zeros(m)
    total_baseline = float(V[0].sum())
    total_noise = 0.0
    last_ts = float(t[0])
    first = True
    sum_x1 = np.zeros(n_)
    sum_x2 = np.zeros(n_)
    sum_app = np.zeros(n_)
    sum_x1[0] = sum_x2[0] = sum_app[0] = 0.0

    for i in range(1, n_):
        v = out[i]  # 注意：cpp 里 values_io 尚未扣减，与录制一致
        dt = float(t[i]) - last_ts
        last_ts = float(t[i])
        dt = min(dt, 0.1) if dt > 0.0 else 0.0

        bypass_frame = False
        if dt > 0.0:
            total = float(v.sum())
            if math.isfinite(total):
                alpha = min(dt / p.bypass_base_tau_s, 1.0)
                release = max(p.bypass_release_frac * total_baseline,
                              total_baseline + p.bypass_noise_sigma * total_noise)
                engage = max(p.bypass_engage_frac * total_baseline,
                             total_baseline + (p.bypass_noise_sigma + 1.0) * total_noise)
                if total < engage:
                    total_baseline += alpha * (total - total_baseline)
                    total_noise += alpha * (abs(total - total_baseline) - total_noise)
                    total_noise = max(total_noise, 0.0)
                if total < release:
                    bypass_frame = True

        if dt > 0.0:
            y_decay = math.exp(-dt / p.y_max_tau_s)
            y0 = v - zero
            y_floor = np.where(y0 < y_floor, y0,
                               y_floor + (dt / p.y_floor_tau_s) * (y0 - y_floor))
            y_max = np.maximum(y_max * y_decay, np.maximum(y0, 0.0))
            span = np.maximum(y_max - y_floor, 1.0)
            idle = (y0 - y_floor) < p.idle_frac * span
            load_dwell = np.where(idle, 0.0, load_dwell + dt)
            zero = np.where(idle, zero + (dt / p.tau_zero_s) * (v - zero), zero)
            y = v - zero
            slope = (v - v_lp) / p.tau_slope_s
            e_now = np.maximum(y - x_fast - x_slow, 0.0)
            refract = max(p.edge_refract_s, p.edge_boost_s)
            t_edge = t_edge + dt
            fire = (slope > p.edge_slope_thres) & (t_edge > refract)
            t_edge = np.where(fire, 0.0, t_edge)
            ramp_on = (slope > p.ramp_slope_min) & (slope <= p.edge_slope_thres)
            ramp_dwell = np.where(ramp_on, ramp_dwell + dt, 0.0)
            load_dwell = np.where(np.abs(slope) > p.edge_slope_thres, 0.0, load_dwell)
            tc1 = np.where((p.edge_boost_s > 0.0) & (t_edge < p.edge_boost_s),
                           p.tau_c_fast_boost_s, p.tau_c_fast_s)
            if p.ramp_full_s > 0.0:
                w = np.minimum(ramp_dwell / p.ramp_full_s, 1.0)
                tc1 = tc1 * (1.0 - w) + p.tau_c_fast_boost_s * w
            dx1_rate = np.where(
                e_now > 0.0,
                (p.r_fast * e_now - x_fast) / tc1,
                -x_fast / p.tau_r_fast_s)
            x_fast = np.maximum(x_fast + dx1_rate * dt, 0.0)
            v_lp = v_lp + (dt / p.tau_slope_s) * (v - v_lp)
            # 慢态
            e = np.maximum(y - x_fast - x_slow, 0.0)
            span_k = np.maximum(y_max - y_floor, 1.0)
            idle_k = (y - y_floor) < p.idle_frac * span_k
            if True:
                base = np.maximum(e, 1.0)
                cap = p.slope_cap_frac * base
                soft_ok = (1.0 - np.exp(-t_edge / p.soft_unfreeze_s)) > 0.5
                dwell_ok = load_dwell >= p.slow_confirm_s
                gate_ok = np.abs(slope) < p.slope_gate_frac * base
                ok = (e > 0.0) & dwell_ok & soft_ok & gate_ok
                dx2 = np.clip(slope - dx1_rate, -cap, cap)
                x_slow = x_slow + np.where(ok, dt * dx2, 0.0)
                hi = p.r_slow_max * base
                x_slow = np.clip(x_slow, 0.0, hi)
            rel = (e <= 0.0) | idle_k
            tau_r2 = np.where((idle_k & (p.tau_r_slow_idle_s > 0.0)),
                              p.tau_r_slow_idle_s, p.tau_r_slow_s)
            x_slow = np.where(rel, np.maximum(
                x_slow - (dt / tau_r2) * x_slow, 0.0), x_slow)
            # K6 预留池
            slope_a = (v - v_fast_lp) / p.hold_tau_s
            v_fast_lp = v_fast_lp + (dt / p.hold_tau_s) * (v - v_fast_lp)
            d_des = x_fast + x_slow
            a_up = applied + dt * np.maximum(slope_a + p.hold_eps, 0.0)
            a_dn = applied + dt * (slope_a - p.hold_eps)
            a = np.where(d_des > applied, np.minimum(d_des, a_up),
                         np.maximum(d_des, a_dn))
            a = np.where(e > 0.0, np.clip(a, 0.0, np.maximum(d_des, 0.0)), d_des)
            applied = a
        else:
            pass

        if bypass_frame:
            applied = x_fast + x_slow
        out[i] = out[i] - applied
        sum_x1[i], sum_x2[i], sum_app[i] = x_fast.sum(), x_slow.sum(), applied.sum()

    return {"out": out, "x1": sum_x1, "x2": sum_x2, "applied": sum_app,
            "in_tot": V.sum(axis=1), "out_tot": out.sum(axis=1)}
