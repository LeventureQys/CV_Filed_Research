# -*- coding: utf-8 -*-
"""95_palm4_plot：ADC 会话终版对照图（录制参数 vs 三档候选）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")))
from importlib import import_module  # noqa: E402

from dptool.figure_export import export_small_multiples  # noqa: E402
from dptool.snapshot import CurveSnapshot, FigureSnapshot, PanelSnapshot  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm4" / "out"
REC = s93.REC
CANDS = [
    ("REC0", REC, "录制参数（rf=0.3 τc1=40 cap.006 rsm.05）", "#d62728", 1.2),
    ("C6", {**REC, "r_fast": 0.10, "tau_c_fast_s": 15, "slope_cap_frac": 0.006,
            "r_slow_max": 0.10}, "稳显档（cap.006：漂得多、回拽最少）", "#2ca02c", 1.2),
    ("C8", {**REC, "r_fast": 0.10, "tau_c_fast_s": 15, "slope_cap_frac": 0.008,
            "r_slow_max": 0.10}, "推荐档（rf.10 τc1=15 cap.008 rsm.10）", "#ff7f0e", 1.8),
    ("C12", {**REC, "r_fast": 0.10, "tau_c_fast_s": 15, "slope_cap_frac": 0.012,
             "r_slow_max": 0.10}, "控漂档（cap.012：漂最少、回拽较大）", "#9467bd", 1.2),
]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, tin, seg = z["t"], z["tot_pre"], z["tot_seg"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    curves, ded = [], []
    for key, pp, label, c, w in CANDS:
        r = obs.run(t, z["pre"], obs.default_with(pp))
        y = s93.medfilt1s(r["out_tot"], fps)
        curves.append(CurveSnapshot(label, t, y, c, w))
        ded.append(CurveSnapshot(label, t, tin - r["out_tot"], c, w))
    rec_curve = curves[0]
    panels = [
        PanelSnapshot("① 算法输入（pre 流）：总值 ADC",
                      "1.45 s 加载 +12363；E≈15550，随后蠕变 +2011（+16%）",
                      "经过时间 (s)", "总值 (ADC)",
                      [CurveSnapshot("输入", t, tin, "#7f7f7f", 1.0)]),
        PanelSnapshot("② 显示总值：录制（红）vs 候选",
                      "红=录制参数，显示被拽到 E 以下 963（过减）；橙=推荐档保持在 E 上方缓漂",
                      "经过时间 (s)", "总值 (ADC)",
                      [CurveSnapshot("E≈15550", np.array([1.5, 13.7]),
                                     np.array([15550, 15550]), "#1f77b4", 1.0, "dash")]
                      + curves),
        PanelSnapshot("③ 加载后 0~5 s 放大：收敛速度",
                      "候选档 ~0.2 s 进入 ±300 ADC 带（要求 1~2 s）；红=录制参数 1.8 s",
                      "经过时间 (s)", "总值 (ADC)",
                      [CurveSnapshot("E≈15550", np.array([1.5, 5.0]),
                                     np.array([15550, 15550]), "#1f77b4", 1.0, "dash")]
                      + curves),
        PanelSnapshot("④ 扣除量（输入 − 显示）",
                      "红=录制参数 x1 终值≈0.23·E 多扣 ~1500；推荐档扣除 ≈ 蠕变量且不带过冲",
                      "经过时间 (s)", "扣除量 (ADC)",
                      [CurveSnapshot("输入蠕变 +2011", np.array([1.5, 13.7]),
                                     np.array([0, 2011]), "#1f77b4", 1.0, "dash")] + ded),
    ]
    snap = FigureSnapshot(
        suptitle="手掌 · ADC 模式 · 过减 963→~50 ADC（沿瞬态）· 收敛 1.8→0.2 s · 慢漂受控（v3.4 · K9 复算）",
        panels=panels, ncol=2, width_in=15.0, panel_height_in=3.2, dpi=110,
        legend_panel=0, max_points_per_curve=9000)
    out = TEMP / "figures" / "手掌_ADC模式_过减修复对比.png"
    r = export_small_multiples(snap, str(out), check=True)
    print("check:", r.get("check"), "bytes:", r.get("bytes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
