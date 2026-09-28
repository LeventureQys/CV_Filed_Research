# -*- coding: utf-8 -*-
"""K7 会话评估脚本（用法：python k7_eval.py <session_dir> <out_dir>）。

对单份录制做 K6（现状复放+保真校验）与 K7（修复）离线对比，输出指标、PNG 对比图与 metrics.json。
算法口径与 src/domain/drift_v6/creep_observer.cpp 逐帧对齐（转写自 K7 版本）。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

P6 = dict(
    r_fast=0.10, tau_c_fast_s=12.0, tau_r_fast_s=6.0, r_slow_max=0.35,
    tau_r_slow_s=150.0, tau_slope_s=1.0, slope_gate_frac=0.05, slope_cap_frac=0.01,
    tau_zero_s=8.0, idle_frac=0.05, y_max_tau_s=600.0, edge_slope_thres=60.0,
    edge_refract_s=2.0, edge_boost_s=2.0, tau_c_fast_boost_s=2.0,
    soft_unfreeze_s=4.0, hold_eps=2.0, hold_tau_s=0.5,
)
P7 = dict(
    P6,
    tau_r_fast_s=2.0,
    slow_confirm_s=10.0, y_floor_tau_s=300.0, tau_r_slow_idle_s=8.0,
    bypass_release_frac=1.15, bypass_engage_frac=1.25,
    bypass_noise_sigma=3.0, bypass_base_tau_s=10.0,
)
# K8：空载判定加确认窗——零点跟踪与 x2 快泄放都要求空载持续 idle_confirm_s，
# 防轻载工况 idle 闪烁把零点拖向负载、把 x2 反复放血（保压期残余上漂的根因）。
# 另：总值旁路需叠加"所有通道逐通道空载"——轻载落在总值旁路死区时不再整帧直通。
P8 = dict(P7, zero_confirm_s=2.0, discharge_confirm_s=2.0, bypass_all_idle=True)


def observer(timestamps: np.ndarray, frames: np.ndarray, p: dict, k7: bool) -> np.ndarray:
    n = frames.shape[1]
    m = len(timestamps)
    out = np.empty_like(frames)
    zero = frames[0].copy()
    y_max = np.zeros(n)
    y_floor = frames[0].copy() if k7 else None
    dwell = np.zeros(n) if k7 else None
    idle_dwell = np.zeros(n) if k7 else None
    v_lp = frames[0].copy()
    v_fast = frames[0].copy()
    x_fast = np.zeros(n)
    x_slow = np.zeros(n)
    applied = np.zeros(n)
    t_edge = np.full(n, 1.0e6)
    total_base = float(frames[0].sum())
    total_noise = 0.0
    bypass = np.zeros(n, dtype=bool)
    out[0] = frames[0]
    last_ts = timestamps[0]
    for i in range(1, m):
        v = frames[i]
        ts = timestamps[i]
        dt = min(ts - last_ts, 0.1)
        if dt < 0:
            dt = 0.0
        last_ts = ts
        if k7 and dt > 0.0:
            total = float(v.sum())
            if np.isfinite(total):
                alpha = min(dt / p["bypass_base_tau_s"], 1.0)
                release = max(p["bypass_release_frac"] * total_base,
                              total_base + p["bypass_noise_sigma"] * total_noise)
                engage = max(p["bypass_engage_frac"] * total_base,
                             total_base + (p["bypass_noise_sigma"] + 1.0) * total_noise)
                if total < engage:
                    total_base += alpha * (total - total_base)
                    total_noise += alpha * (abs(total - total_base) - total_noise)
                    if total_noise < 0.0:
                        total_noise = 0.0
                if total < release:
                    # K8：总值旁路需叠加"所有通道空载"——轻载（总值落在旁路死区）
                    # 时承载通道的逐通道 idle 判定为假，此时不得直通，否则原始蠕变穿透
                    if p.get("bypass_all_idle", False):
                        idle_pre = (v - zero - y_floor) < p["idle_frac"] * np.maximum(y_max - y_floor, 1.0)
                        bypass[:] = bool(idle_pre.all())
                    else:
                        bypass[:] = True
                else:
                    bypass[:] = False
        if dt > 0.0:
            y_decay = np.exp(-dt / p["y_max_tau_s"])
            y0 = v - zero
            if k7:
                down = y0 < y_floor
                y_floor = np.where(down, y0, y_floor + (dt / p["y_floor_tau_s"]) * (y0 - y_floor))
                y_max = np.maximum(y_max * y_decay, np.maximum(y0, 0.0))
                span = np.maximum(y_max - y_floor, 1.0)
                idle = (y0 - y_floor) < p["idle_frac"] * span
                dwell = np.where(idle, 0.0, dwell + dt)
                idle_dwell = np.where(idle, idle_dwell + dt, 0.0)
                zc = p.get("zero_confirm_s", 0.0)
                zok = idle_dwell >= zc if zc > 0.0 else idle
                zero = np.where(idle & zok, zero + (dt / p["tau_zero_s"]) * (v - zero), zero)
            else:
                y_max = np.maximum(y_max * y_decay, np.maximum(y0, 0.0))
                idle = y0 < p["idle_frac"] * np.maximum(y_max, 1.0)
                zero = np.where(idle, zero + (dt / p["tau_zero_s"]) * (v - zero), zero)
            y = v - zero
            slope = (v - v_lp) / p["tau_slope_s"]
            e_now = np.maximum(y - x_fast - x_slow, 0.0)
            refract = max(p["edge_refract_s"], p["edge_boost_s"])
            t_edge = t_edge + dt
            fire = (slope > p["edge_slope_thres"]) & (t_edge > refract)
            t_edge = np.where(fire, 0.0, t_edge)
            if k7:
                # 任一快变（升或降）都视为新加载段，确认窗重计
                dwell = np.where(np.abs(slope) > p["edge_slope_thres"], 0.0, dwell)
            tc1 = np.where(
                (p["edge_boost_s"] > 0.0) & (t_edge < p["edge_boost_s"]),
                p["tau_c_fast_boost_s"], p["tau_c_fast_s"])
            if p.get("x1_slow_release", False):
                tau_r1_eff = np.where(bypass, p["tau_r_fast_s"], 60.0)
            else:
                tau_r1_eff = p["tau_r_fast_s"]
            dx1_rate = np.where(
                e_now > 0.0,
                (p["r_fast"] * e_now - x_fast) / tc1,
                -x_fast / tau_r1_eff)
            x_fast = np.maximum(x_fast + dx1_rate * dt, 0.0)
            v_lp = v_lp + (dt / p["tau_slope_s"]) * (v - v_lp)
            e = np.maximum(y - x_fast - x_slow, 0.0)
            if k7:
                span_k = np.maximum(y_max - y_floor, 1.0)
                idle_k = (y - y_floor) < p["idle_frac"] * span_k
            else:
                idle_k = np.zeros(n, dtype=bool)
            base = np.maximum(e, 1.0)
            cap = p["slope_cap_frac"] * base
            soft_ok = 1.0 - np.exp(-t_edge / p["soft_unfreeze_s"]) > 0.5
            dwell_ok = dwell >= p["slow_confirm_s"] if k7 else np.ones(n, dtype=bool)
            # K8：慢态积分门加绝对下限——轻载时 5%·e 低于导数噪声，积分被噪声挡死
            gate_th = np.maximum(p["slope_gate_frac"] * base,
                                 p.get("slope_gate_abs", 0.0))
            gate = (np.abs(slope) < gate_th) & soft_ok & dwell_ok
            dx2 = np.clip(slope - dx1_rate, -cap, cap)
            x_slow = np.where((e > 0.0) & gate, x_slow + dt * dx2, x_slow)
            hi = p["r_slow_max"] * base
            x_slow = np.clip(x_slow, 0.0, np.where(e > 0.0, hi, np.inf))
            if k7:
                tau_r2 = np.where(idle_k, p["tau_r_slow_idle_s"], p["tau_r_slow_s"])
                dconfirm = p.get("discharge_confirm_s", 0.0)
                if p.get("latch_discharge", False):
                    discharge = (e <= 0.0) | bypass
                elif dconfirm > 0.0:
                    discharge = (e <= 0.0) | (idle_k & (idle_dwell >= dconfirm))
                else:
                    discharge = (e <= 0.0) | idle_k
            else:
                tau_r2 = np.full(n, p["tau_r_slow_s"])
                discharge = e <= 0.0
            x_slow = np.where(discharge,
                              np.maximum(x_slow - (dt / tau_r2) * x_slow, 0.0), x_slow)
            slope_a = (v - v_fast) / p["hold_tau_s"]
            v_fast = v_fast + (dt / p["hold_tau_s"]) * (v - v_fast)
            d_des = x_fast + x_slow
            a_up = applied + dt * np.maximum(slope_a + p["hold_eps"], 0.0)
            a_dn = applied + dt * (slope_a - p["hold_eps"])
            a = np.where(d_des > applied, np.minimum(d_des, a_up), np.maximum(d_des, a_dn))
            a = np.where(e > 0.0, np.clip(a, 0.0, np.maximum(d_des, 0.0)), d_des)
            applied = a
        if k7 and bypass.any():
            applied = x_fast + x_slow
            out[i] = v
        else:
            out[i] = v - applied
    return out


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=28)
    return data[:, 1], data[:, 3:]


def find_streams(session_dir):
    pre = result = None
    for name in sorted(os.listdir(session_dir)):
        if name.endswith("_pre_seg0.csv"):
            pre = os.path.join(session_dir, name)
        elif name.endswith("_seg000.csv"):
            result = os.path.join(session_dir, name)
    return pre, result


def plateau_split(t, sums):
    hi = np.percentile(sums[sums > 3], 80) if (sums > 3).any() else 1.0
    on = sums > 0.6 * hi
    idle = sums < 0.5 * hi
    segs, in_on = [], False
    for i in range(len(t)):
        if on[i] and not in_on:
            start, in_on = i, True
        elif not on[i] and in_on:
            segs.append((start, i))
            in_on = False
    if in_on:
        segs.append((start, len(t)))
    return [s for s in segs if s[1] - s[0] >= 20], idle


def main():
    session_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    sid = os.path.basename(session_dir.rstrip("\\/"))
    pre_path, result_path = find_streams(session_dir)
    if pre_path is None:
        print(f"[{sid}] no pre stream, skip")
        return
    t, pre = load_csv(pre_path)
    sum_pre = pre.sum(axis=1)

    metrics = {"session": sid, "frames": int(len(t))}
    if result_path and os.path.exists(result_path):
        _, rec = load_csv(result_path)
        k6 = observer(t, pre, P6, k7=False)
        mlen = min(len(k6), len(rec))
        err = np.abs(k6[:mlen] - rec[:mlen])
        metrics["k6_port_max_err"] = float(err.max())
        metrics["k6_port_mean_err"] = float(err.mean())
    else:
        k6 = observer(t, pre, P6, k7=False)
    k7 = observer(t, pre, P7, k7=True)
    sum_k6 = k6.sum(axis=1)
    sum_k7 = k7.sum(axis=1)

    segs, idle = plateau_split(t, sum_pre)
    k6_bias = np.array([np.mean(sum_k6[a:b] - sum_pre[a:b]) for a, b in segs])
    k7_bias = np.array([np.mean(sum_k7[a:b] - sum_pre[a:b]) for a, b in segs])
    metrics["n_cycles"] = len(segs)
    for tag, bias in (("k6", k6_bias), ("k7", k7_bias)):
        metrics[f"{tag}_cycle_bias_first"] = float(bias[0]) if len(bias) else None
        metrics[f"{tag}_cycle_bias_last"] = float(bias[-1]) if len(bias) else None
        metrics[f"{tag}_cycle_bias_min"] = float(bias.min()) if len(bias) else None
        metrics[f"{tag}_growth"] = float(abs(bias[-1]) - abs(bias[0])) if len(bias) else None
    metrics["k6_idle_bias"] = float(np.mean((sum_k6 - sum_pre)[idle])) if idle.any() else None
    metrics["k7_idle_bias"] = float(np.mean((sum_k7 - sum_pre)[idle])) if idle.any() else None
    metrics["pre_plateau_mean"] = float(np.mean(sum_pre[[i for a, b in segs for i in range(a, b)]])) if segs else None
    metrics["k7_plateau_mean"] = float(np.mean(sum_k7[[i for a, b in segs for i in range(a, b)]])) if segs else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ("Microsoft YaHei", "SimHei")
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax = axes[0]
    ax.plot(t, sum_pre, lw=0.8, color="0.4", label="算法前 pre（基准）")
    ax.plot(t, sum_k6, lw=0.9, color="tab:red", label="K6 现状输出")
    ax.plot(t, sum_k7, lw=0.9, color="tab:blue", label="K7 修复输出")
    ax.set_ylabel("总值")
    ax.set_title(f"{sid}：K6 现状 vs K7 修复（逐帧离线复放）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    span = max(t[-1] * 0.12, 35.0)
    m = t >= t[-1] - span
    ax.plot(t[m], sum_pre[m], lw=1.2, color="0.4", label="pre")
    ax.plot(t[m], sum_k6[m], lw=1.2, color="tab:red", label="K6")
    ax.plot(t[m], sum_k7[m], lw=1.2, color="tab:blue", label="K7")
    ax.set_ylabel("总值（末段放大）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, sum_k6 - sum_pre, lw=0.9, color="tab:red", label="K6 − pre")
    ax.plot(t, sum_k7 - sum_pre, lw=0.9, color="tab:blue", label="K7 − pre")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("elapsed (s)")
    ax.set_ylabel("输出 − 算法前")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sid}_k6_vs_k7.png"), dpi=110)
    plt.close(fig)

    np.savez(os.path.join(out_dir, f"{sid}_series.npz"),
             t=t, pre=sum_pre, k6=sum_k6, k7=sum_k7)
    with open(os.path.join(out_dir, f"{sid}_metrics.json"), "w", encoding="utf8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[{sid}] " + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()


