# -*- coding: utf-8 -*-
"""exp1：v3.4 观测器基线 vs 方案修正版，working 四工况总值对比。

算法变体（来自 plan/v4/x1_shape_study 综合结论）：
  base : v3.4 基线（r1=0.12, tc1=8, tr1=6, 其余 P3）
  M1   : 保守常数修正（方向①）：r1=0.08, tc1=32
  M2   : M1 + F-B 前馈（方向④）：上升沿后 2 s 内 tc1 临时降到 2 s
  M3   : M2 + x2 沿后冻结（方向②的快速验证）：上升沿后 3 s 内 x2 不积分
读取与绘图复用 toolbox/数据解析工具 的 dptool（load_table + export_small_multiples）。

时间轴：按 T1 教训用 frame_index × (1/100.5 Hz) 重建均匀时间轴（elapsed 有 70% 重复帧）。
指标：最大加载沿后的显示回落深度（0.3 s 中值滤波后）、保压末段漂移。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, *[".."] * 9))
DPTOOL = os.path.join(REPO, "toolbox", "数据解析工具")
sys.path.insert(0, DPTOOL)
sys.path.insert(0, os.path.join(REPO, "Document", "Update", "Dev-Version",
                                "v2.7 - 抗蠕变补偿算法", "v4.1flash", "plan", "v4",
                                "x1_shape_study", "T1_shape", "scripts"))

from dptool.session_csv import load_table  # noqa: E402
from dptool.snapshot import CurveSnapshot, PanelSnapshot, FigureSnapshot  # noqa: E402
from dptool.figure_export import export_small_multiples  # noqa: E402

DATA_ROOT = os.path.join(REPO, "Document", "Update", "Dev-Version",
                         "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据", "working")
OUT_FIG = os.path.join(HERE, "..", "figure")
OUT_RES = os.path.join(HERE, "..", "results")
FPS = 100.5

P = dict(r1=0.12, tc1=8.0, tr1=6.0, r2max=0.35, tr2=150.0,
         tau_slope=3.0, slope_gate=0.02, slope_cap=0.01,
         tau_zero=8.0, idle_frac=0.05, y_max_tau=600.0)

VARIANTS = {
    "base": dict(P, tc1=32.0 * 0 + 8.0),          # v3.4 原样
    "M1":   dict(P, r1=0.08, tc1=32.0),           # 保守常数
    "M2":   dict(P, r1=0.08, tc1=32.0, boost_s=2.0, tc1_boost=2.0),      # +前馈
    "M3":   dict(P, r1=0.08, tc1=32.0, boost_s=2.0, tc1_boost=2.0, freeze_s=3.0),  # +x2冻结
}

EDGE_THR = 60.0     # 上升沿判定：低通导数 > 60 ADC/s（T3 敏感性 5~15 为通道级，总量级放大）
EDGE_REFRACT = 2.0  # 沿触发的最小间隔 s


def observe(ts, V, p):
    """v3.4 观测器逐帧回放（与 v34_observer_core3.observe3 同构）+ M2/M3 扩展。"""
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    zero = V[0].copy()
    y_max = np.maximum(V[0] - zero, 0.0)
    v_lp = V[0].copy()
    t_boost = np.full(ch, 1e9)   # 距上次上升沿触发的时长
    t_freeze = np.full(ch, 1e9)
    D = np.empty((n, ch))
    t_prev = ts[0]
    boost_s = p.get("boost_s", 0.0)
    freeze_s = p.get("freeze_s", 0.0)
    for i in range(n):
        dt = min(max(ts[i] - t_prev, 0.0), 0.1)
        t_prev = ts[i]
        v = V[i]
        if dt > 0.0:
            t_boost += dt
            t_freeze += dt
            y = v - zero
            y_max = np.maximum(y_max * np.exp(-dt / p["y_max_tau"]), np.maximum(y, 0.0))
            idle = y < p["idle_frac"] * np.maximum(y_max, 1.0)
            zero = np.where(idle, zero + (dt / p["tau_zero"]) * (v - zero), zero)
            y = v - zero
            e_now = np.maximum(y - x1 - x2, 0.0)
            # 上升沿检测（M2/M3 用；base/M1 无扩展开关时不影响——boost 只在沿后生效）
            slope = (v - v_lp) / p["tau_slope"]
            fire = slope > EDGE_THR
            fire &= t_boost > max(EDGE_REFRACT, boost_s)
            t_boost = np.where(fire, 0.0, t_boost)
            t_freeze = np.where(fire, 0.0, t_freeze)
            tc1_eff = np.where((boost_s > 0) & (t_boost < boost_s), p.get("tc1_boost", 2.0), p["tc1"])
            dx1_rate = np.where(e_now > 0.0, (p["r1"] * e_now - x1) / tc1_eff, -x1 / p["tr1"])
            x1 = np.maximum(x1 + dt * dx1_rate, 0.0)
            v_lp = v_lp + (dt / p["tau_slope"]) * (v - v_lp)
            e = np.maximum(y - x1 - x2, 0.0)
            rate_cap = p["slope_cap"] * np.maximum(e, 1.0)
            gate = p["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1_rate, -rate_cap, rate_cap)
            allow = (e > 0.0) & (np.abs(slope) < gate)
            if freeze_s > 0:
                allow &= t_freeze >= freeze_s
            x2 = x2 + np.where(allow, dx2, 0.0) * dt
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, p["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / p["tr2"], 0.0))
        D[i] = v - x1 - x2
    return D


def medfilt(x, k):
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    s = np.lib.stride_tricks.sliding_window_view(xp, k)
    return np.median(s, axis=1)


def find_sessions():
    out = []
    for root, _dirs, files in os.walk(DATA_ROOT):
        if "device_001_pre_seg0.csv" not in files:
            continue
        cond = os.path.relpath(root, DATA_ROOT).split(os.sep)[0]
        name = os.path.basename(root)
        pre = os.path.join(root, "device_001_pre_seg0.csv")
        seg = os.path.join(root, "device_001_seg000.csv")
        out.append((cond, name, pre, seg if os.path.isfile(seg) else None))
    return out


def load_channels(path):
    t = load_table(path, use_cache=False)
    ch = t.channels()
    ch = ch[:, np.isfinite(ch).all(axis=0)] if ch.size else ch
    fi = t.time_axis("frame_index")
    ts = fi / FPS
    return ts, np.asarray(ch, dtype=np.float64)


def step_fall(ts, tot, k=31):
    """最大加载沿后的回落深度：沿后 [0.5,8]s 峰值 − [10,20]s 平台值。"""
    totf = medfilt(tot, k)
    d = np.diff(totf) / np.diff(ts)
    d = medfilt(np.append(d, d[-1]), k)
    j = int(np.argmax(d))
    w1 = (ts >= ts[j] + 0.5) & (ts <= ts[j] + 8.0)
    w2 = (ts >= ts[j] + 10.0) & (ts <= ts[j] + 20.0)
    if w1.sum() < 5 or w2.sum() < 5:
        return float("nan"), float("nan")
    peak = totf[w1].max()
    plat = np.median(totf[w2])
    return peak - plat, ts[j]


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_RES, exist_ok=True)
    sessions = find_sessions()
    lines = ["exp1：v3.4 基线 vs 修正变体（总值=全通道求和）",
             f"工况数 {len(sessions)}；时间轴 frame_index/{FPS} Hz；回落=沿后[0.5,8]s峰值-[10,20]s平台（0.3s中值滤波）", ""]
    panels = []
    for cond, name, pre, seg in sessions:
        ts, V = load_channels(pre)
        tot_in = V.sum(axis=1)
        curves = [CurveSnapshot("输入(原始)", ts, tot_in, color="0.6", width=1.0)]
        if seg:
            ts2, V2 = load_channels(seg)
            if len(ts2) == len(ts):
                curves.append(CurveSnapshot("实机算法输出", ts2, V2.sum(axis=1),
                                            color="#8c564b", style="dash", width=1.0))
        row = [cond]
        for vname, vp in VARIANTS.items():
            D = observe(ts, V, vp)
            tot = D.sum(axis=1)
            fall, tj = step_fall(ts, tot)
            row.append(fall)
            curves.append(CurveSnapshot(f"{vname}", ts, tot,
                                        color=dict(base="#1f77b4", M1="#2ca02c",
                                                   M2="#ff7f0e", M3="#d62728")[vname],
                                        width=1.2))
            print(f"{cond} {vname}: fall={fall:.1f}")
        lines.append(f"[{cond}] {name}")
        fall_in, _ = step_fall(ts, tot_in)
        lines.append(f"  输入自身回落(参考) {fall_in:8.1f} ADC   " +
                     "  ".join(f"{k}={v:8.1f}" for k, v in zip(VARIANTS, row[1:])))
        panels.append(PanelSnapshot(title=cond, x_label="t (s)", y_label="总值 (ADC)",
                                    curves=curves, show_legend=(len(panels) == 0)))
    snap = FigureSnapshot(suptitle="exp1 · v3.4 基线 vs 修正变体 · 总值对比（working 四工况）",
                          panels=panels, ncol=2, width_in=16.0, panel_height_in=4.2,
                          legend_panel=0, share_x=False, max_points_per_curve=6000)
    out = os.path.join(OUT_FIG, "exp1_总值对比.png")
    info = export_small_multiples(snap, out, check=True)
    lines += ["", f"图: {out} ({info['bytes']/1024:.0f} KB)"]
    res = os.path.join(OUT_RES, "exp1_summary.txt")
    with open(res, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
