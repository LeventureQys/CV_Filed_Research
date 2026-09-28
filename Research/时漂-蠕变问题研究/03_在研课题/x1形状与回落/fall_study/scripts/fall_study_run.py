# -*- coding: utf-8 -*-
"""fall_study：阶跃后 ~2 s 见峰、随后回落过大 —— 逐事件归因与修正验证。

研究对象（用户主诉）：拿到阶跃后显示在 ~2 s 到顶峰，然后回落，回落太大。
指标定义（总值口径，0.3 s 中值滤波后）：
  t_peak = 沿后 [0.3, 8] s 内显示最大值时刻
  fall   = 显示(t_peak) - median(显示[t_peak+8, t_peak+18])   # 峰后回落深度
  分解   = 同窗口内 Δ(x1总和)、Δ(x2总和)、Δ(输入)   # 谁在峰后继续涨
变体（在 observe 同一代码上改参数/开关）：
  base : v3.4 原样
  A    : tau_slope 3 -> 1        （低通导数更快跟上，减少 slope 高估）
  B    : 沿后软冻结：x2 积分权重 1-exp(-(t-t_edge)/tau_w)，tau_w=10 s
  C    : slope_gate 0.02 -> 0.05 （x2 更晚才开积分，等输入真平）
  D    : M2 常数(r1=0.08,tc1=32,沿后boost) + B 软冻结
  E    : D + A + C（组合）
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, *[".."] * 9))
sys.path.insert(0, os.path.join(REPO, "toolbox", "数据解析工具"))

from dptool.session_csv import load_table  # noqa: E402

DATA_ROOT = os.path.join(REPO, "Document", "Update", "Dev-Version",
                         "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据")
OUT_RES = os.path.join(HERE, "..", "results")
FPS = 100.5

P = dict(r1=0.12, tc1=8.0, tr1=6.0, r2max=0.35, tr2=150.0,
         tau_slope=3.0, slope_gate=0.02, slope_cap=0.01,
         tau_zero=8.0, idle_frac=0.05, y_max_tau=600.0)

VARIANTS = {
    "base": dict(P),
    "A_tsl1": dict(P, tau_slope=1.0),
    "B_soft10": dict(P, soft_w=10.0),
    "C_gate05": dict(P, slope_gate=0.05),
    "D_M2+B": dict(P, r1=0.08, tc1=32.0, boost_s=2.0, tc1_boost=2.0, soft_w=10.0),
    "E_combo": dict(P, r1=0.08, tc1=32.0, boost_s=2.0, tc1_boost=2.0,
                    soft_w=10.0, tau_slope=1.0, slope_gate=0.05),
    "F_E_tc16": dict(P, r1=0.10, tc1=16.0, boost_s=2.0, tc1_boost=2.0,
                     soft_w=10.0, tau_slope=1.0, slope_gate=0.05),
    "G_E_w6": dict(P, r1=0.08, tc1=32.0, boost_s=2.0, tc1_boost=2.0,
                   soft_w=6.0, tau_slope=1.0, slope_gate=0.05),
    "H1_F_w6": dict(P, r1=0.10, tc1=16.0, boost_s=2.0, tc1_boost=2.0,
                    soft_w=6.0, tau_slope=1.0, slope_gate=0.05),
    "H2_F_w4": dict(P, r1=0.10, tc1=16.0, boost_s=2.0, tc1_boost=2.0,
                    soft_w=4.0, tau_slope=1.0, slope_gate=0.05),
    "H3_F12_w4": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=2.0,
                      soft_w=4.0, tau_slope=1.0, slope_gate=0.05),
    "H4_F_w4_b3": dict(P, r1=0.10, tc1=16.0, boost_s=3.0, tc1_boost=2.0,
                       soft_w=4.0, tau_slope=1.0, slope_gate=0.05),
    "I1_boost03": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=0.3,
                       soft_w=4.0, tau_slope=1.0, slope_gate=0.05),
    "I2_snap06": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=0.5,
                      soft_w=4.0, tau_slope=1.0, slope_gate=0.05, snap=0.6),
    "I3_snap10": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=0.5,
                      soft_w=4.0, tau_slope=1.0, slope_gate=0.05, snap=1.0),
    "J1_cap5x": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=2.0,
                     soft_w=4.0, tau_slope=1.0, slope_gate=0.05, cap_boost=5.0,
                     cap_boost_s=10.0),
    "J2_cap3x": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=2.0,
                     soft_w=4.0, tau_slope=1.0, slope_gate=0.05, cap_boost=3.0,
                     cap_boost_s=10.0),
    "J3_cap5x_w2": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=2.0,
                        soft_w=2.0, tau_slope=1.0, slope_gate=0.05, cap_boost=5.0,
                        cap_boost_s=10.0),
    "K1_hold_e2": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=2.0,
                       soft_w=4.0, tau_slope=1.0, slope_gate=0.05, hold_eps=2.0),
    "K2_hold_e10": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=2.0,
                        soft_w=4.0, tau_slope=1.0, slope_gate=0.05, hold_eps=10.0),
    "K3_base+hold2": dict(P, hold_eps=2.0),
    "K4_base+hold10": dict(P, hold_eps=10.0),
    "K5_base+holdA2": dict(P, hold_eps=2.0, hold_tau=0.5),
    "K6_H3+holdA2": dict(P, r1=0.10, tc1=12.0, boost_s=2.0, tc1_boost=2.0,
                         soft_w=4.0, tau_slope=1.0, slope_gate=0.05,
                         hold_eps=2.0, hold_tau=0.5),
}

EDGE_THR = 60.0        # 通道级上升沿（低通导数 ADC/s）
EDGE_REFRACT = 2.0
STEP_MIN = 2000.0      # 总值台阶下限（ADC），只统计清晰加载事件


def observe(ts, V, p):
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    zero = V[0].copy()
    y_max = np.maximum(V[0] - zero, 0.0)
    v_lp = V[0].copy()
    t_edge = np.full(ch, 1e9)
    a = np.zeros(ch)   # K 系：实际施加的补偿（预留池 = x1+x2 - a）
    v_lp_a = V[0].copy()  # K 系：预留池限额用的快速低通（τ=0.5s）
    D = np.empty((n, ch))
    S1 = np.empty(n)
    S2 = np.empty(n)
    t_prev = ts[0]
    boost_s = p.get("boost_s", 0.0)
    soft_w = p.get("soft_w", 0.0)
    for i in range(n):
        dt = min(max(ts[i] - t_prev, 0.0), 0.1)
        t_prev = ts[i]
        v = V[i]
        if dt > 0.0:
            t_edge += dt
            y = v - zero
            y_max = np.maximum(y_max * np.exp(-dt / p["y_max_tau"]), np.maximum(y, 0.0))
            idle = y < p["idle_frac"] * np.maximum(y_max, 1.0)
            zero = np.where(idle, zero + (dt / p["tau_zero"]) * (v - zero), zero)
            y = v - zero
            e_now = np.maximum(y - x1 - x2, 0.0)
            slope = (v - v_lp) / p["tau_slope"]
            fire = (slope > EDGE_THR) & (t_edge > max(EDGE_REFRACT, boost_s))
            if p.get("snap", 0.0) > 0.0:
                # 沿触发瞬间把 x1 直接抬到 snap·r1·e_now（预测式置位，之后照常收敛）
                snap = np.where(fire & (e_now > 0.0),
                                p["snap"] * p["r1"] * e_now, x1)
                x1 = np.maximum(x1, snap)
            t_edge = np.where(fire, 0.0, t_edge)
            tc1_eff = np.where((boost_s > 0) & (t_edge < boost_s), p.get("tc1_boost", 2.0), p["tc1"])
            dx1_rate = np.where(e_now > 0.0, (p["r1"] * e_now - x1) / tc1_eff, -x1 / p["tr1"])
            x1 = np.maximum(x1 + dt * dx1_rate, 0.0)
            v_lp = v_lp + (dt / p["tau_slope"]) * (v - v_lp)
            e = np.maximum(y - x1 - x2, 0.0)
            cap_mult = 1.0
            if p.get("cap_boost", 0.0) > 0.0 and t_edge.min() < p.get("cap_boost_s", 0.0):
                cap_mult = np.where(t_edge < p["cap_boost_s"], p["cap_boost"], 1.0)
            rate_cap = p["slope_cap"] * np.maximum(e, 1.0) * cap_mult
            gate = p["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1_rate, -rate_cap, rate_cap)
            allow = (e > 0.0) & (np.abs(slope) < gate)
            if soft_w > 0:
                allow &= (1.0 - np.exp(-t_edge / soft_w)) > 0.5
            x2 = x2 + np.where(allow, dx2, 0.0) * dt
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, p["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / p["tr2"], 0.0))
            # K 系：显示侧"预留池"——受载期施加的补偿增速不超过 快速导数+eps，
            # 会把显示往下拉的部分先存进预留池（d-a），慢相阶段随输入爬升慢慢释放。
            # 限额用 τ=0.5s 快速导数（τ=3s 低通在减速段高估爬升率，约束会失效）
            if p.get("hold_eps", 0.0) > 0.0:
                tau_a = p.get("hold_tau", 0.5)
                slope_a = (v - v_lp_a) / tau_a
                v_lp_a = v_lp_a + (dt / tau_a) * (v - v_lp_a)
                d_des = x1 + x2
                a_up = a + dt * np.maximum(slope_a + p["hold_eps"], 0.0)
                a_dn = a + dt * (slope_a - p["hold_eps"])
                a = np.where(d_des > a, np.minimum(d_des, a_up),
                             np.maximum(d_des, a_dn))
                a = np.where(e > 0.0, np.clip(a, 0.0, np.maximum(d_des, 0.0)), d_des)
        if p.get("hold_eps", 0.0) > 0.0:
            D[i] = v - a
        else:
            D[i] = v - x1 - x2
        S1[i] = x1.sum()
        S2[i] = x2.sum()
    return D, S1, S2


def medfilt(x, k=31):
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, k), axis=1)


def find_events(ts, tot):
    totf = medfilt(tot)
    d = medfilt(np.append(np.diff(totf) * FPS, 0.0))
    events = []
    last = -1e9
    for i in np.where(d > 1500.0)[0]:
        if ts[i] - last < 6.0:
            continue
        j0 = max(0, i - int(1.5 * FPS))
        j1 = min(len(totf), i + int(4.0 * FPS))
        step = np.median(totf[i + int(2.0 * FPS):i + int(4.0 * FPS)]) - np.median(totf[j0:i])
        if step < STEP_MIN:
            continue
        if ts[-1] - ts[i] < 20.0:
            continue
        events.append(i)
        last = ts[i]
    return events, totf


def event_metrics(ts, totf, i, s1=None, s2=None, tot_in=None):
    w = (ts >= ts[i]) & (ts <= ts[i] + 8.0)
    k = int(np.argmax(totf[w]))
    ip = int(np.where(w)[0][k])
    t_peak = ts[ip] - ts[i]
    w2 = (ts >= ts[ip] + 8.0) & (ts <= ts[ip] + 18.0)
    if w2.sum() < 5:
        return None
    fall = totf[ip] - np.median(totf[w2])
    plat = np.median(totf[w2])
    band = max(0.05 * abs(plat), 50.0)
    ok = np.abs(totf - plat) < band
    t_set = float("nan")
    hold = int(5.0 * FPS)
    for j in range(ip, len(ok) - hold):
        if ok[j:j + hold].all():
            t_set = ts[j] - ts[i]
            break
    dec = {"t_set": t_set}
    for name, s in (("dx1", s1), ("dx2", s2)):
        if s is not None:
            dec[name] = s[ip] - np.median(s[w2])
    if tot_in is not None:
        dec["din"] = tot_in[ip] - np.median(tot_in[w2])
    return t_peak, fall, dec


def find_sessions():
    out = []
    for root, _dirs, files in os.walk(DATA_ROOT):
        if "device_001_pre_seg0.csv" in files:
            out.append((os.path.relpath(root, DATA_ROOT), root))
    return out


def main():
    os.makedirs(OUT_RES, exist_ok=True)
    rows = []
    for cond, root in find_sessions():
        t = load_table(os.path.join(root, "device_001_pre_seg0.csv"), use_cache=False)
        ch = np.asarray(t.channels(), dtype=np.float64)
        ch = ch[:, np.isfinite(ch).all(axis=0)]
        ts = t.time_axis("frame_index") / FPS
        tot_in = medfilt(ch.sum(axis=1))
        events, _ = find_events(ts, tot_in)
        if not events:
            continue
        for vname, vp in VARIANTS.items():
            D, S1, S2 = observe(ts, ch, vp)
            tot = medfilt(D.sum(axis=1))
            for i in events:
                m = event_metrics(ts, tot, i, S1, S2, tot_in)
                if m is None:
                    continue
                rows.append((cond, ts[i], vname, m[0], m[1], m[2].get("dx1", np.nan),
                             m[2].get("dx2", np.nan), m[2].get("din", np.nan),
                             m[2].get("t_set", np.nan)))
        for i in events:
            m = event_metrics(ts, tot_in, i)
            if m is not None:
                rows.append((cond, ts[i], "INPUT", m[0], m[1], np.nan, np.nan))
        print(f"done {cond}: {len(events)} events")

    import json
    with open(os.path.join(OUT_RES, "fall_events.json"), "w", encoding="utf-8") as fh:
        json.dump([dict(cond=r[0], t=float(r[1]), variant=r[2], t_peak=float(r[3]),
                        fall=float(r[4]), dx1=float(r[5]), dx2=float(r[6]),
                        din=float(r[7]) if len(r) > 7 else float("nan"),
                        t_set=float(r[8]) if len(r) > 8 else float("nan")) for r in rows],
                  fh, ensure_ascii=False, indent=1)

    names = [k for k in list(VARIANTS) + ["INPUT"]]
    lines = ["fall_study：阶跃后峰后回落逐事件统计（正值=显示回落；excess=fall-din 同窗口）", ""]
    for vname in names:
        sub = [r for r in rows if r[2] == vname]
        if not sub:
            continue
        falls = np.array([r[4] for r in sub])
        tp = np.array([r[3] for r in sub])
        ex = np.array([r[4] - (r[7] if len(r) > 7 and np.isfinite(r[7]) else np.nan) for r in sub])
        tst = np.array([r[8] if len(r) > 8 else np.nan for r in sub])
        lines.append(f"{vname:10s} n={len(sub):2d}  t_peak中位={np.median(tp):5.2f}s  "
                     f"落位t_set中位={np.nanmedian(tst):5.2f}s  "
                     f"fall中位={np.median(falls):7.1f}  excess中位={np.nanmedian(ex):6.1f}  "
                     f"excess_p90={np.nanpercentile(ex,90):7.1f}  "
                     f"excess_max={np.nanmax(ex):7.1f}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT_RES, "fall_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    try:
        print(txt)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((txt + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
