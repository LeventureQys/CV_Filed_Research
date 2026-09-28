# -*- coding: utf-8 -*-
"""K9：缓坡前馈（ramp feedforward）离线评估。

在 K7 之上新增：持续亚沿值正斜率（缓坡加载）按持续时间比例加速 x1 收敛
（等效 tc1 向沿后前馈值 tau_c_fast_boost_s 过渡），使无过充的缓坡加载获得
接近沿触发的收敛速度。沿触发路径不变。

算法口径与 C++ creep_observer.cpp K9 版逐行对齐。
用法：python k9_eval.py <session_dir> <out_dir>
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
# K9：缓坡前馈。斜率处于 (ramp_slope_min, edge_slope_thres) 且持续 ramp_full_s
# 后，x1 收敛 τ 从 tc1 线性过渡到沿后前馈值；斜率跌回 0/越沿即清零计时。
P9 = dict(P7, ramp_slope_min=0.5, ramp_full_s=4.0)
# K10：e 残差反馈（未入 C++）。e_now = y−x1−x2 的导数即显示残余漂移率，
# 低通后按 fb_gain 反馈进 x2 积分率（仅 slow_confirm 之后启用），闭环压制欠跟踪。
P10 = dict(P9, fb_gain=1.0, fb_tau=2.0)
P10_HALF = dict(P9, fb_gain=0.5, fb_tau=2.0)


def observer(timestamps: np.ndarray, frames: np.ndarray, p: dict, k9: bool) -> np.ndarray:
    n = frames.shape[1]
    m = len(timestamps)
    out = np.empty_like(frames)
    zero = frames[0].copy()
    y_max = np.zeros(n)
    y_floor = frames[0].copy() if k9 else None
    dwell = np.zeros(n) if k9 else None
    rdwell = np.zeros(n) if k9 else None
    e_prev = None
    lp_de = np.zeros(n) if k9 else None
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
        if k9 and dt > 0.0:
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
                bypass[:] = total < release
        if dt > 0.0:
            y_decay = np.exp(-dt / p["y_max_tau_s"])
            y0 = v - zero
            if k9:
                down = y0 < y_floor
                y_floor = np.where(down, y0, y_floor + (dt / p["y_floor_tau_s"]) * (y0 - y_floor))
                y_max = np.maximum(y_max * y_decay, np.maximum(y0, 0.0))
                span = np.maximum(y_max - y_floor, 1.0)
                idle = (y0 - y_floor) < p["idle_frac"] * span
                dwell = np.where(idle, 0.0, dwell + dt)
                zero = np.where(idle, zero + (dt / p["tau_zero_s"]) * (v - zero), zero)
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
            if k9:
                # K9 缓坡前馈：亚沿值正斜率持续计时，满 ramp_full_s 后 tc1 全额过渡到前馈值
                ramp_on = (slope > p["ramp_slope_min"]) & (slope <= p["edge_slope_thres"])
                rdwell = np.where(ramp_on, rdwell + dt, 0.0)
                dwell = np.where(np.abs(slope) > p["edge_slope_thres"], 0.0, dwell)
            tc1 = np.where(
                (p["edge_boost_s"] > 0.0) & (t_edge < p["edge_boost_s"]),
                p["tau_c_fast_boost_s"], p["tau_c_fast_s"])
            if k9:
                w = np.minimum(rdwell / p["ramp_full_s"], 1.0)
                tc1 = tc1 * (1.0 - w) + p["tau_c_fast_boost_s"] * w
            dx1_rate = np.where(
                e_now > 0.0,
                (p["r_fast"] * e_now - x_fast) / tc1,
                -x_fast / p["tau_r_fast_s"])
            x_fast = np.maximum(x_fast + dx1_rate * dt, 0.0)
            v_lp = v_lp + (dt / p["tau_slope_s"]) * (v - v_lp)
            e = np.maximum(y - x_fast - x_slow, 0.0)
            if k9:
                span_k = np.maximum(y_max - y_floor, 1.0)
                idle_k = (y - y_floor) < p["idle_frac"] * span_k
            else:
                idle_k = np.zeros(n, dtype=bool)
            base = np.maximum(e, 1.0)
            cap = p["slope_cap_frac"] * base
            soft_ok = 1.0 - np.exp(-t_edge / p["soft_unfreeze_s"]) > 0.5
            dwell_ok = dwell >= p["slow_confirm_s"] if k9 else np.ones(n, dtype=bool)
            gate = (np.abs(slope) < p["slope_gate_frac"] * base) & soft_ok & dwell_ok
            dx2 = np.clip(slope - dx1_rate, -cap, cap)
            x_slow = np.where((e > 0.0) & gate, x_slow + dt * dx2, x_slow)
            # K10：e 残差反馈（fb_gain>0 启用；仅 slow_confirm 后；上限随 cap 防风暴）
            if k9 and p.get("fb_gain", 0.0) > 0.0:
                if e_prev is None:
                    e_prev = e_now.copy()
                de = (e_now - e_prev) / max(dt, 1e-3)
                e_prev = e_now.copy()
                lp_de = lp_de + (dt / p["fb_tau"]) * (de - lp_de)
                fb = np.where(dwell_ok, np.clip(p["fb_gain"] * lp_de, 0.0, cap), 0.0)
                x_slow = np.where((e > 0.0) & dwell_ok, x_slow + dt * fb, x_slow)
            hi = p["r_slow_max"] * base
            x_slow = np.clip(x_slow, 0.0, np.where(e > 0.0, hi, np.inf))
            tau_r2 = np.where(idle_k, p["tau_r_slow_idle_s"], p["tau_r_slow_s"])
            discharge = (e <= 0.0) | idle_k
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
        if k9 and bypass.any():
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


def main():
    session_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    sid = os.path.basename(session_dir.rstrip("\\/"))
    pre_path, _ = find_streams(session_dir)
    if pre_path is None:
        print(f"[{sid}] no pre stream, skip")
        return
    t, pre = load_csv(pre_path)
    pre_s = pre.sum(axis=1)
    hi = np.percentile(pre_s[pre_s > 3], 90) if (pre_s > 3).any() else 1.0
    on = pre_s > 0.5 * hi

    p7 = dict(P7, ramp_slope_min=P9["ramp_slope_min"], ramp_full_s=P9["ramp_full_s"])
    k7 = observer(t, pre, p7, k9=True)  # K9 基线（fb_gain=0）：同一路径、无 e 反馈
    k10 = observer(t, pre, P10, k9=True)
    k10h = observer(t, pre, P10_HALF, k9=True)
    s9, s10, s10h = k7.sum(axis=1), k10.sum(axis=1), k10h.sum(axis=1)  # k7 即 K9 基线

    segs, in_on = [], False
    for i in range(len(t)):
        if on[i] and not in_on:
            start, in_on = i, True
        elif not on[i] and in_on:
            segs.append((start, i))
            in_on = False
    if in_on:
        segs.append((start, len(t)))
    segs = [s for s in segs if s[1] - s[0] >= 30]

    def settle(sums):
        res = []
        for a, b in segs:
            final = np.mean(sums[max(b - 5, 0):b])
            band = 0.05 * max(abs(final - sums[a]), 1.0) + 0.02 * hi
            below = np.abs(sums[a:b] - final) <= band
            idx = np.argmax(below) if below.any() else -1
            res.append(float(t[a + idx] - t[a]) if idx >= 0 else float("nan"))
        return res

    st9 = settle(s9)
    st10 = settle(s10)
    st10h = settle(s10h)
    r9 = [np.polyfit(t[(t >= t[a] + 15) & (t <= t[min(b, len(t) - 1)])], (s9 - pre_s)[(t >= t[a] + 15) & (t <= t[min(b, len(t) - 1)])], 1)[0]
          for a, b in segs if t[min(b, len(t) - 1)] - t[a] >= 35]
    r10 = [np.polyfit(t[(t >= t[a] + 15) & (t <= t[min(b, len(t) - 1)])], (s10 - pre_s)[(t >= t[a] + 15) & (t <= t[min(b, len(t) - 1)])], 1)[0]
           for a, b in segs if t[min(b, len(t) - 1)] - t[a] >= 35]
    r10h = [np.polyfit(t[(t >= t[a] + 15) & (t <= t[min(b, len(t) - 1)])], (s10h - pre_s)[(t >= t[a] + 15) & (t <= t[min(b, len(t) - 1)])], 1)[0]
            for a, b in segs if t[min(b, len(t) - 1)] - t[a] >= 35]
    bias9 = np.array([np.mean(s9[a:b] - pre_s[a:b]) for a, b in segs])
    bias10 = np.array([np.mean(s10[a:b] - pre_s[a:b]) for a, b in segs])
    idle = pre_s < 0.5 * hi
    metrics = {
        "session": sid, "n_plateaus": len(segs),
        "settle_k9_med": float(np.median(st9)) if st9 else None,
        "settle_k10_med": float(np.median(st10)) if st10 else None,
        "resid_k9": r9, "resid_k10": r10, "resid_k10_half": r10h,
        "bias_k9_last": float(bias9[-1]) if len(bias9) else None,
        "bias_k10_last": float(bias10[-1]) if len(bias10) else None,
        "idle_k9": float(np.mean((s9 - pre_s)[idle])) if idle.any() else None,
        "idle_k10": float(np.mean((s10 - pre_s)[idle])) if idle.any() else None,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ("Microsoft YaHei", "SimHei")
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax = axes[0]
    ax.plot(t, pre_s, lw=0.8, color="0.4", label="算法前 pre")
    ax.plot(t, s9, lw=0.9, color="tab:green", label="K9 现状")
    ax.plot(t, s10, lw=0.9, color="tab:purple", label="K10 e反馈 k=1.0")
    ax.plot(t, s10h, lw=0.9, color="tab:orange", label="K10 e反馈 k=0.5")
    ax.set_ylabel("总值")
    ax.set_title(f"{sid}：K9 vs K10（e 残差反馈）")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t, s9 - pre_s, lw=0.9, color="tab:green", label="K9 − pre")
    ax.plot(t, s10 - pre_s, lw=0.9, color="tab:purple", label="K10 − pre")
    ax.plot(t, s10h - pre_s, lw=0.9, color="tab:orange", label="K10半增益 − pre")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("elapsed (s)")
    ax.set_ylabel("输出 − 算法前")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sid}_k9_vs_k10.png"), dpi=110)
    plt.close(fig)
    np.savez(os.path.join(out_dir, f"{sid}_series.npz"), t=t, pre=pre_s, k9=s9,
             k10=s10, k10h=s10h)
    print(f"[{sid}] " + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()



