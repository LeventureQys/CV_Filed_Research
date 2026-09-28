# -*- coding: utf-8 -*-
"""107_audit：当前参数（推荐档）在 152928 上的"补偿账本"。

逐秒拆：输入爬升(−E)、x1、x2、applied、显示−E；并记录慢态三扇门
（dwell_ok / soft_ok / 斜率门）何时打开、dx1_rate 吃掉多少 x2 预算。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
OUT = TEMP / "palm6" / "out"
E1 = 16481.0
P = {"r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0, "soft_unfreeze_s": 1.5,
     "slope_cap_frac": 0.005, "r_slow_max": 0.15, "tau_r_fast_s": 6.0,
     "tau_r_slow_idle_s": 0.5}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, pre, tin = z["t"], z["pre"], z["tot_pre"]

    # 重跑并带出逐帧门状态与积分量
    p = obs.default_with(P)
    n_, m = pre.shape
    x1s = np.zeros(n_); x2s = np.zeros(n_); aps = np.zeros(n_)
    slope_sum = np.zeros(n_); dx1r_sum = np.zeros(n_); dx2_sum = np.zeros(n_)
    open_frac = np.zeros(n_)
    # 直接复用 run() 但重算细节太重；改为在 run 结果上加门诊断
    r = obs.run(t, pre, p)

    # 门诊断（重新单趟、只算聚合门量）
    zero = pre[0].copy(); y_max = np.zeros(m); v_lp = pre[0].copy()
    y_floor = pre[0].copy(); load_dwell = np.zeros(m); t_edge = np.full(m, 1e6)
    x_fast = np.zeros(m); x_slow = np.zeros(m); last = float(t[0])
    for i in range(1, n_):
        v = pre[i]
        dt = float(t[i]) - last; last = float(t[i])
        dt = min(dt, 0.1) if dt > 0 else 0.0
        if dt <= 0:
            x1s[i], x2s[i], aps[i] = x_fast.sum(), x_slow.sum(), x_fast.sum() + x_slow.sum()
            continue
        y0 = v - zero
        y_decay = math.exp(-dt / 600.0)
        y_floor = np.where(y0 < y_floor, y0, y_floor + (dt/300.0)*(y0-y_floor))
        y_max = np.maximum(y_max*y_decay, np.maximum(y0, 0.0))
        span = np.maximum(y_max - y_floor, 1.0)
        idle = (y0 - y_floor) < 0.05*span
        load_dwell = np.where(idle, 0.0, load_dwell + dt)
        zero = np.where(idle, zero + (dt/8.0)*(v-zero), zero)
        y = v - zero
        slope = (v - v_lp)/1.0
        e_now = np.maximum(y - x_fast - x_slow, 0.0)
        t_edge = t_edge + dt
        fire = (slope > 60.0) & (t_edge > 2.0)
        t_edge = np.where(fire, 0.0, t_edge)
        load_dwell = np.where(np.abs(slope) > 60.0, 0.0, load_dwell)
        tc1 = np.where((2.0 > 0.0) & (t_edge < 2.0), 2.0, 2.0)
        dx1_rate = np.where(e_now > 0, (0.06*e_now - x_fast)/tc1, -x_fast/6.0)
        x_fast = np.maximum(x_fast + dx1_rate*dt, 0.0)
        v_lp = v_lp + dt*(v - v_lp)
        e = np.maximum(y - x_fast - x_slow, 0.0)
        base = np.maximum(e, 1.0)
        soft_ok = (1.0 - np.exp(-t_edge/1.5)) > 0.5
        dwell_ok = load_dwell >= 2.0
        gate_ok = np.abs(slope) < 0.05*base
        ok = (e > 0) & dwell_ok & soft_ok & gate_ok
        cap = 0.005*base
        dx2 = np.clip(slope - dx1_rate, -cap, cap)
        x_slow = np.clip(x_slow + np.where(ok, dt*dx2, 0.0), 0.0, 0.15*base)
        slope_sum[i] = slope.sum(); dx1r_sum[i] = dx1_rate.sum()
        dx2_sum[i] = np.where(ok, dx2, 0.0).sum(); open_frac[i] = ok.mean()
        x1s[i], x2s[i] = x_fast.sum(), x_slow.sum()
        aps[i] = (x_fast + x_slow).sum()

    print(f"{'t':>5s} {'输入−E':>8s} {'x1':>7s} {'x2':>7s} {'applied':>8s} {'显示−E':>8s} "
          f"{'slopeΣ':>7s} {'dx1rΣ':>7s} {'dx2Σ':>7s} {'门开%':>5s}")
    for tt in np.arange(1.5, 41.0, 1.5):
        i = int(np.searchsorted(t, tt))
        if i >= n_:
            break
        print(f"{t[i]:5.2f} {tin[i]-E1:8.1f} {x1s[i]:7.1f} {x2s[i]:7.1f} {aps[i]:8.1f} "
              f"{tin[i]-E1-aps[i]:8.1f} {slope_sum[i]:7.1f} {dx1r_sum[i]:7.1f} "
              f"{dx2_sum[i]:7.1f} {open_frac[i]*100:5.1f}")
    # 存量账本
    F = 1163.0  # 快相
    i34 = int(np.searchsorted(t, 3.4))
    early = (tin[i34] - E1) - F
    print(f"\n账本：快相 F≈{F:.0f}；x1 稳态份额 rf/(1+rf)·电平≈{0.06/1.06*16481:.0f}；"
          f"慢态开门前的蠕变存量（1.05~3.4s）≈{early:.0f}；"
          f"快相未覆盖 ≈{F-0.06/1.06*(E1+F):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
