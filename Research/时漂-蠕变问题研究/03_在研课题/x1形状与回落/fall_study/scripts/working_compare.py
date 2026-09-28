# -*- coding: utf-8 -*-
"""working 全部会话：base(v3.4) vs H3 vs H4 对比测试。
每个会话：全程总值面板 + 两个最大加载事件的沿后放大（-3~+25 s）；并输出指标摘要。"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, *[".."] * 9))
sys.path.insert(0, os.path.join(REPO, "toolbox", "数据解析工具"))

from dptool.session_csv import load_table  # noqa: E402
from dptool.snapshot import CurveSnapshot, PanelSnapshot, FigureSnapshot  # noqa: E402
from dptool.figure_export import export_small_multiples  # noqa: E402

sys.path.insert(0, HERE)
from fall_study_run import FPS, VARIANTS, observe, medfilt, find_events, event_metrics  # noqa: E402

DATA_ROOT = os.path.join(REPO, "Document", "Update", "Dev-Version",
                         "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据", "working")
OUT_FIG = os.path.join(HERE, "..", "figure")
OUT_RES = os.path.join(HERE, "..", "results")

SHOW = [("base", "#1f77b4", "base(v3.4 现行)"), ("H3_F12_w4", "#ff7f0e", "H3"),
        ("H4_F_w4_b3", "#9467bd", "H4")]


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    os.makedirs(OUT_RES, exist_ok=True)
    sessions = []
    for root, _dirs, files in os.walk(DATA_ROOT):
        if "device_001_pre_seg0.csv" in files:
            sessions.append((os.path.relpath(root, DATA_ROOT), root))
    panels = []
    lines = ["working 全会话对比：base(v3.4) vs H3 vs H4",
             "excess=同窗口显示回落-输入自身回落；t_set=进入并保持平台±5%带", ""]
    for cond, root in sorted(sessions):
        t = load_table(os.path.join(root, "device_001_pre_seg0.csv"), use_cache=False)
        ch = np.asarray(t.channels(), dtype=np.float64)
        ch = ch[:, np.isfinite(ch).all(axis=0)]
        ts = t.time_axis("frame_index") / FPS
        tot_in = medfilt(ch.sum(axis=1))
        events, _ = find_events(ts, tot_in)
        curves = [CurveSnapshot("输入", ts, tot_in, color="0.6", width=1.0)]
        line = [f"[{cond}] 事件数 {len(events)}"]
        for vname, color, lab in SHOW:
            D, S1, S2 = observe(ts, ch, VARIANTS[vname])
            tot = medfilt(D.sum(axis=1))
            curves.append(CurveSnapshot(lab, ts, tot, color=color, width=1.2))
            falls, exs, tsets = [], [], []
            for i in events:
                m = event_metrics(ts, tot, i, S1, S2, tot_in)
                if m is None:
                    continue
                falls.append(m[1])
                exs.append(m[1] - m[2]["din"])
                tsets.append(m[2]["t_set"])
            if falls:
                line.append(f"{vname:10s} fall中位={np.median(falls):7.1f} "
                            f"excess中位={np.median(exs):6.1f} excess_max={np.max(exs):7.1f} "
                            f"t_set中位={np.nanmedian(tsets):5.2f}s")
        lines.extend("  " + s for s in line)
        lines.append("")
        name = os.path.basename(root)
        panels.append(PanelSnapshot(title=f"{cond} / {name[:36]} 全程",
                                    x_label="t (s)", y_label="总值 (ADC)",
                                    curves=curves, show_legend=(len(panels) == 0)))
        # 两个最大台阶的放大
        steps = []
        for i in events:
            j0 = max(0, i - int(1.5 * FPS))
            step = np.median(tot_in[i + int(2 * FPS):i + int(4 * FPS)]) - np.median(tot_in[j0:i])
            steps.append((step, i))
        for step, i in sorted(steps, reverse=True)[:2]:
            t_ev = ts[i]
            zoom = (ts >= t_ev - 3.0) & (ts <= t_ev + 25.0)
            panels.append(PanelSnapshot(
                title=f"放大：@{t_ev:.0f}s 台阶≈{step:.0f} ADC（沿后-3~+25s）",
                x_label="t-t_edge (s)", y_label="总值 (ADC)",
                curves=[CurveSnapshot(c.name, ts[zoom] - t_ev, c.y[zoom],
                                      color=c.color, width=c.width) for c in curves]))
    snap = FigureSnapshot(suptitle="working 全会话 · base(v3.4) vs H3 vs H4 · 总值对比（灰=输入，每会话=1全程+2放大）",
                          panels=panels, ncol=3, width_in=21.0, panel_height_in=3.6,
                          legend_panel=0, max_points_per_curve=6000)
    out = os.path.join(OUT_FIG, "working对比_base_H3_H4.png")
    info = export_small_multiples(snap, out, check=True)
    lines.append(f"图: {out} ({info['bytes']/1024:.0f} KB)")
    res = os.path.join(OUT_RES, "working_compare.txt")
    with open(res, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
