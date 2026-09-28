# -*- coding: utf-8 -*-
"""出图：纯过补偿回落事件的会话，输入/base/推荐变体总值对比 + 事件窗口放大。"""
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
from fall_study_run import FPS, VARIANTS, observe, medfilt, find_events  # noqa: E402

DATA_ROOT = os.path.join(REPO, "Document", "Update", "Dev-Version",
                         "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据")
OUT_FIG = os.path.join(HERE, "..", "figure")

SHOW = [("base", "#1f77b4", "base(v3.4)"), ("E_combo", "#d62728", "E 组合"),
        ("F_E_tc16", "#2ca02c", "F"), ("H3_F12_w4", "#ff7f0e", "H3 快落位"),
        ("H4_F_w4_b3", "#9467bd", "H4 快落位+长boost")]

TARGETS = [
    (r"archived\从零基线开始 - 恒定负载 - 反复加减同一个负载\20260919_100351_single_device_f9740b", 103.82),
    (r"working\零基线-反复增减同一负载\20260919_191748_single_device_795e5e\20260919_192141_single_device_73032d", 30.1),
    (r"archived\零基线-反复增减同一负载\20260919_152749_single_device_0cb8b6", 14.29),
]


def main():
    panels = []
    for rel, t_ev in TARGETS:
        root = os.path.join(DATA_ROOT, rel)
        t = load_table(os.path.join(root, "device_001_pre_seg0.csv"), use_cache=False)
        ch = np.asarray(t.channels(), dtype=np.float64)
        ch = ch[:, np.isfinite(ch).all(axis=0)]
        ts = t.time_axis("frame_index") / FPS
        tot_in = medfilt(ch.sum(axis=1))
        curves = [CurveSnapshot("输入", ts, tot_in, color="0.6", width=1.0)]
        for vname, color, label in SHOW:
            D, _, _ = observe(ts, ch, VARIANTS[vname])
            curves.append(CurveSnapshot(label, ts, medfilt(D.sum(axis=1)),
                                        color=color, width=1.2))
        name = os.path.basename(root)
        panels.append(PanelSnapshot(title=f"{name[:44]} 全程", x_label="t (s)",
                                    y_label="总值 (ADC)", curves=curves,
                                    show_legend=(len(panels) == 0)))
        zoom = (ts >= t_ev - 5.0) & (ts <= t_ev + 35.0)
        panels.append(PanelSnapshot(title=f"事件 @{t_ev}s 放大（-5~+35s）", x_label="t (s)",
                                    y_label="总值 (ADC)",
                                    curves=[CurveSnapshot(c.name, ts[zoom], c.y[zoom],
                                                          color=c.color, width=c.width)
                                            for c in curves]))
    snap = FigureSnapshot(suptitle="fall_study · 峰后回落对比：base vs 修正变体（上=全程，下=事件放大）",
                          panels=panels, ncol=2, width_in=16.0, panel_height_in=4.0,
                          legend_panel=0, max_points_per_curve=6000)
    out = os.path.join(OUT_FIG, "fall_study_对比.png")
    info = export_small_multiples(snap, out, check=True)
    print("saved", out, info["bytes"])


if __name__ == "__main__":
    main()
