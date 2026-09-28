# -*- coding: utf-8 -*-
"""153_b2d896_trace：拆解 b2d896 上「cap 小 ⇒ 尾端仍漂」的机理。

对 cap ∈ {0.0005, 0.002, 0.005, 0.012}（rsm 0.15、其余出厂默认）打印
x1/x2/applied 总值、逐帧 dx2 被 cap 钳位的比例、以及被钳位丢失的累计量。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

import math  # noqa: E402

obs = import_module("91_observer")
OUT = TEMP / "palm10" / "out"


def trace(t, V, p, marks=(3, 5, 10, 20, 60, 120, 240, 420)):
    """跑一遍并记录 x1/x2/applied 总值与被 cap 钳位的帧比例。"""
    n, m = V.shape
    out = V.astype(np.float64).copy()
    zero = V[0].astype(np.float64).copy()
    y_max = np.zeros(m); v_lp = V[0].astype(np.float64).copy()
    v_fast_lp = V[0].astype(np.float64).copy(); y_floor = V[0].astype(np.float64).copy()
    dwell = np.zeros(m); ramp = np.zeros(m); t_edge = np.full(m, 1.0e6)
    applied = np.zeros(m); x1 = np.zeros(m); x2 = np.zeros(m)
    tb = float(V[0].sum()); tn = 0.0; last = float(t[0])
    clip_frames = 0; tot_frames = 0; lost = 0.0
    rec = {k: [] for k in ("t", "x1", "x2", "ap", "out", "clip_frac")}
    idx = {mk: int(np.searchsorted(t, mk)) for mk in marks}
    for i in range(1, n):
        dt = min(max(float(t[i]) - last, 0.0), 0.1); last = float(t[i])
        v = out[i]
        cl = 0
        if dt > 0:
            total = float(v.sum())
            alpha = min(dt / p.bypass_base_tau_s, 1.0)
            rel = max(p.bypass_release_frac * tb, tb + p.bypass_noise_sigma * tn)
            eng = max(p.bypass_engage_frac * tb, tb + (p.bypass_noise_sigma + 1.0) * tn)
            if total < eng:
                tb += alpha * (total - tb)
                tn = max(tn + alpha * (abs(total - tb) - tn), 0.0)
            byp = total < rel
            y0 = v - zero
            y_floor = np.where(y0 < y_floor, y0, y_floor + (dt / p.y_floor_tau_s) * (y0 - y_floor))
            y_max = np.maximum(y_max * math.exp(-dt / p.y_max_tau_s), np.maximum(y0, 0.0))
            span = np.maximum(y_max - y_floor, 1.0)
            idle = (y0 - y_floor) < p.idle_frac * span
            dwell = np.where(idle, 0.0, dwell + dt)
            zero = np.where(idle, zero + (dt / p.tau_zero_s) * (v - zero), zero)
            y = v - zero
            slope = (v - v_lp) / p.tau_slope_s
            e_now = np.maximum(y - x1 - x2, 0.0)
            t_edge = t_edge + dt
            fire = (slope > p.edge_slope_thres) & (t_edge > max(p.edge_refract_s, p.edge_boost_s))
            t_edge = np.where(fire, 0.0, t_edge)
            ramp = np.where((slope > p.ramp_slope_min) & (slope <= p.edge_slope_thres), ramp + dt, 0.0)
            dwell = np.where(np.abs(slope) > p.edge_slope_thres, 0.0, dwell)
            tc1 = np.where((p.edge_boost_s > 0) & (t_edge < p.edge_boost_s),
                           p.tau_c_fast_boost_s, p.tau_c_fast_s)
            w = np.minimum(ramp / p.ramp_full_s, 1.0)
            tc1 = tc1 * (1 - w) + p.tau_c_fast_boost_s * w
            dx1 = np.where(e_now > 0, (p.r_fast * e_now - x1) / tc1, -x1 / p.tau_r_fast_s)
            x1 = np.maximum(x1 + dx1 * dt, 0.0)
            v_lp = v_lp + (dt / p.tau_slope_s) * (v - v_lp)
            e = np.maximum(y - x1 - x2, 0.0)
            spank = np.maximum(y_max - y_floor, 1.0)
            idlek = (y - y_floor) < p.idle_frac * spank
            base = np.maximum(e, 1.0)
            cap = p.slope_cap_frac * base
            ok = ((e > 0) & (dwell >= p.slow_confirm_s)
                  & ((1 - np.exp(-t_edge / p.soft_unfreeze_s)) > 0.5)
                  & (np.abs(slope) < p.slope_gate_frac * base))
            raw = slope - dx1
            clipped = np.clip(raw, -cap, cap)
            tot_frames += 1
            cl = int((ok & (np.abs(raw) > cap)).sum())
            clip_frames += cl
            lost += float((np.abs(raw) - cap)[ok & (np.abs(raw) > cap)].sum() * dt)
            x2 = np.clip(x2 + np.where(ok, dt * clipped, 0.0), 0.0, p.r_slow_max * base)
            rel2 = (e <= 0) | idlek
            tau2 = np.where(idlek & (p.tau_r_slow_idle_s > 0), p.tau_r_slow_idle_s, p.tau_r_slow_s)
            x2 = np.where(rel2, np.maximum(x2 - (dt / tau2) * x2, 0.0), x2)
            sa = (v - v_fast_lp) / p.hold_tau_s
            v_fast_lp = v_fast_lp + (dt / p.hold_tau_s) * (v - v_fast_lp)
            ddes = x1 + x2
            a_up = applied + dt * np.maximum(sa + p.hold_eps, 0.0)
            a_dn = applied + dt * (sa - p.hold_eps)
            a = np.where(ddes > applied, np.minimum(ddes, a_up), np.maximum(ddes, a_dn))
            applied = np.where(e > 0, np.clip(a, 0.0, np.maximum(ddes, 0.0)), ddes)
            if byp:
                applied = x1 + x2
        out[i] = out[i] - applied
        rec["t"].append(float(t[i]))
        rec["x1"].append(float(x1.sum())); rec["x2"].append(float(x2.sum()))
        rec["ap"].append(float(applied.sum())); rec["out"].append(float(out[i].sum()))
        rec["clip_frac"].append(cl / max(m, 1))
    return (rec, idx, clip_frames / max(tot_frames, 1), lost)


def clip_windows(t, V, p) -> None:
    """按时间窗统计：被 cap 钳位的通道·帧占比、钳位丢失量、平均 |slope−dx1|。"""
    n, m = V.shape
    zero = V[0].astype(np.float64).copy()
    y_max = np.zeros(m); v_lp = V[0].astype(np.float64).copy()
    y_floor = V[0].astype(np.float64).copy()
    dwell = np.zeros(m); t_edge = np.full(m, 1.0e6)
    x1 = np.zeros(m); x2 = np.zeros(m); ramp = np.zeros(m)
    last = float(t[0])
    acc: dict[float, list[float]] = {}
    for i in range(1, n):
        dt = min(max(float(t[i]) - last, 0.0), 0.1); last = float(t[i])
        if dt <= 0:
            continue
        v = V[i].astype(np.float64)
        y0 = v - zero
        y_floor = np.where(y0 < y_floor, y0, y_floor + (dt / p.y_floor_tau_s) * (y0 - y_floor))
        y_max = np.maximum(y_max * math.exp(-dt / p.y_max_tau_s), np.maximum(y0, 0.0))
        span = np.maximum(y_max - y_floor, 1.0)
        idle = (y0 - y_floor) < p.idle_frac * span
        dwell = np.where(idle, 0.0, dwell + dt)
        zero = np.where(idle, zero + (dt / p.tau_zero_s) * (v - zero), zero)
        y = v - zero
        slope = (v - v_lp) / p.tau_slope_s
        e_now = np.maximum(y - x1 - x2, 0.0)
        t_edge = t_edge + dt
        fire = (slope > p.edge_slope_thres) & (t_edge > max(p.edge_refract_s, p.edge_boost_s))
        t_edge = np.where(fire, 0.0, t_edge)
        ramp = np.where((slope > p.ramp_slope_min) & (slope <= p.edge_slope_thres), ramp + dt, 0.0)
        tc1 = np.where((p.edge_boost_s > 0) & (t_edge < p.edge_boost_s),
                       p.tau_c_fast_boost_s, p.tau_c_fast_s)
        w = np.minimum(ramp / p.ramp_full_s, 1.0)
        tc1 = tc1 * (1 - w) + p.tau_c_fast_boost_s * w
        dx1 = np.where(e_now > 0, (p.r_fast * e_now - x1) / tc1, -x1 / p.tau_r_fast_s)
        x1 = np.maximum(x1 + dx1 * dt, 0.0)
        v_lp = v_lp + (dt / p.tau_slope_s) * (v - v_lp)
        e = np.maximum(y - x1 - x2, 0.0)
        base = np.maximum(e, 1.0)
        cap = p.slope_cap_frac * base
        ok = ((e > 0) & (dwell >= p.slow_confirm_s)
              & ((1 - np.exp(-t_edge / p.soft_unfreeze_s)) > 0.5)
              & (np.abs(slope) < p.slope_gate_frac * base))
        raw = slope - dx1
        cl = ok & (np.abs(raw) > cap)
        x2 = np.clip(x2 + np.where(ok, dt * np.clip(raw, -cap, cap), 0.0), 0.0, p.r_slow_max * base)
        key = 0.0
        for edge in (3.0, 10.0, 60.0, 120.0, 240.0):
            if t[i] >= edge:
                key = edge
        a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])   # 帧数, 钳位通道帧, 丢失, |raw|和, 门开通道帧
        a[0] += 1
        a[1] += float(cl.sum())
        a[2] += float((np.abs(raw) - cap)[cl].sum() * dt)
        a[3] += float(np.abs(raw)[ok].sum())
        a[4] += float(ok.sum())
    print("    窗口        帧数   钳位通道帧占比   钳位丢失(ADC)   门开通道平均|slope−dx1|(ADC/s)")
    for k in sorted(acc):
        fr, c, lost, rabs, oks = acc[k]
        print(f"    {k:5.0f}s+ {fr:9.0f} {c/max(fr*m,1)*100:14.2f}% {lost:15.0f} "
              f"{rabs/max(oks,1):22.4f}")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "b2d896.npz")
    t, V = z["t"], z["V"]
    base = {"slope_cap_frac": 0.005, "r_slow_max": 0.15}
    for cap in (0.0005, 0.002, 0.005, 0.012):
        p = obs.default_with({**base, "slope_cap_frac": cap})
        rec, idx, cliprate, lost = trace(t, V, p)
        rt = np.array(rec["t"])
        print(f"\n===== cap={cap:g}（rsm=0.15, rf.01 τc1 40）=====")
        print(f"  被 cap 钳位的帧比例（任一通道）：{cliprate*100:.1f}%  "
              f"累计被钳掉的补偿量 ≈ {lost:.0f} ADC（= 永久欠补存量）")
        print(f"  {'t(s)':>7s} {'x1':>8s} {'x2':>8s} {'applied':>9s} {'显示':>9s} "
              f"{'显示−E':>9s}")
        for mk in (3, 5, 10, 20, 60, 120, 240, 420):
            j = int(np.searchsorted(rt, mk)) - 1
            if j < 0 or j >= len(rt):
                continue
            print(f"  {rt[j]:7.1f} {rec['x1'][j]:8.0f} {rec['x2'][j]:8.0f} "
                  f"{rec['ap'][j]:9.0f} {rec['out'][j]:9.0f} {rec['out'][j]-16502:+9.0f}")
        clip_windows(t, V, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
