# -*- coding: utf-8 -*-
"""快落位专项图：6 个代表事件的沿后放大（-3~+25 s），base/F/H3/H4 与输入。"""
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
from fall_study_run import FPS, VARIANTS, observe, medfilt  # noqa: E402

DATA_ROOT = os.path.join(REPO, "Document", "Update", "Dev-Version",
                         "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据")
OUT_FIG = os.path.join(HERE, "..", "figure")

SHOW = [("base", "#1f77b4", "base(v3.4)"), ("F_E_tc16", "#2ca02c", "F"),
        ("H3_F12_w4", "#ff7f0e", "H3"), ("H4_F_w4_b3", "#9467bd", "H4")]

TARGETS = [
    (r"archived\从零基线开始 - 恒定负载 - 反复加减同一个负载\20260919_100351_single_device_f9740b", 103.82, "f9740b@103.8s（excess≈9.3k 最大）"),
    (r"working\零基线-反复增减同一负载\20260919_191748_single_device_795e5e\20260919_192141_single_device_73032d", 30.10, "73032d@30.1s（excess≈2.5k）"),
    (r"working\零基线-反复增减同一负载\20260919_191748_single_device_795e5e\20260919_192141_single_device_73032d", 45.55, "73032d@45.6s（excess≈2.5k）"),
    (r"archived\零基线-反复增减同一负载\20260919_152749_single_device_0cb8b6", 14.29, "0cb8b6@14.3s（excess≈2.2k）"),
    (r"archived\零基线-反复增减同一负载\20260919_152749_single_device_0cb8b6", 292.56, "0cb8b6@292.6s（excess≈2.6k）"),
    (r"working\零基线-同一荷载测试\20260919_193320_single_device_3f32c5", None, "3f32c5 恒载全程首台阶"),
]


def main():
    cache = {}
    panels = []
    for rel, t_ev, label in TARGETS:
        if rel not in cache:
            root = os.path.join(DATA_ROOT, rel)
            t = load_table(os.path.join(root, "device_001_pre_seg0.csv"), use_cache=False)
            ch = np.asarray(t.channels(), dtype=np.float64)
            ch = ch[:, np.isfinite(ch).all(axis=0)]
            ts = t.time_axis("frame_index") / FPS
            tot_in = medfilt(ch.sum(axis=1))
            if t_ev is None:
                d = medfilt(np.append(np.diff(tot_in) * FPS, 0.0))
                t_ev = ts[int(np.argmax(d))]
            cache[rel] = (ts, ch, tot_in)
        ts, ch, tot_in = cache[rel]
        curves = [CurveSnapshot("输入", ts, tot_in, color="0.6", width=1.0)]
        for vname, color, lab in SHOW:
            D, _, _ = observe(ts, ch, VARIANTS[vname])
            curves.append(CurveSnapshot(lab, ts, medfilt(D.sum(axis=1)), color=color, width=1.3))
        zoom = (ts >= t_ev - 3.0) & (ts <= t_ev + 25.0)
        panels.append(PanelSnapshot(title=label, x_label="t (s)（沿后窗口 -3~+25s）",
                                    y_label="总值 (ADC)",
                                    curves=[CurveSnapshot(c.name, ts[zoom] - t_ev, c.y[zoom],
                                                          color=c.color, width=c.width)
                                            for c in curves],
                                    show_legend=(len(panels) == 0)))
    snap = FigureSnapshot(suptitle="快相落位对比 · 沿后 -3~+25 s 放大（灰=输入，蓝=base，绿=F，橙=H3，紫=H4）",
                          panels=panels, ncol=2, width_in=16.0, panel_height_in=4.0,
                          legend_panel=0, share_x=True, max_points_per_curve=6000)
    out = os.path.join(OUT_FIG, "快落位对比.png")
    info = export_small_multiples(snap, out, check=True)
    print("saved", out, info["bytes"])


if __name__ == "__main__":
    main()
