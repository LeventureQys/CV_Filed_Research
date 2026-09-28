# -*- coding: utf-8 -*-
"""156_b2d896_plot：b2d896 尾漂定因与最小处方对照图（6 面板）。"""
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
OUT = TEMP / "palm10" / "out"
E = 16502.0
GREY, RED, GREEN, BLUE, ORANGE, PURPLE = "#7f7f7f", "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"
CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03}
MIN = {"r_slow_max": 0.15}
REC = {"r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0, "soft_unfreeze_s": 2.0,
       "slope_cap_frac": 0.005, "r_slow_max": 0.15}
UNI = {"r_fast": 0.05, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0, "soft_unfreeze_s": 2.0,
       "slope_cap_frac": 0.009, "r_slow_max": 0.15}


def med(r, t):
    return s93.medfilt1s(r["out_tot"], (len(t) - 1) / (t[-1] - t[0]))


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "b2d896.npz")
    t, V, tot = z["t"], z["V"], z["tot"]
    y_cur = med(obs.run(t, V, obs.default_with(CUR)), t)
    y_min = med(obs.run(t, V, obs.default_with(MIN)), t)
    y_rec = med(obs.run(t, V, obs.default_with(REC)), t)
    y_uni = med(obs.run(t, V, obs.default_with(UNI)), t)
    panels = [
        PanelSnapshot("① 输入全程（ADC，429 s 单台阶，不卸载）",
                      "台阶 @1.20 s、E=16502 ADC；快漂 2~20 s 结束，20 s 后速率 9.6→3.3→1.24→"
                      "尾端 0.12 ADC/s；从 10 s 到末尾输入只再涨 +713 ADC（+4.3% E）",
                      "经过时间 (s)", "总 ADC",
                      [CurveSnapshot("输入", t, tot, GREY, 1.0),
                       CurveSnapshot("E=16502", np.array([1.2, 429.0]),
                                     np.array([E, E]), BLUE, 1.2, "dash")]),
        PanelSnapshot("② 显示全程：现参数 vs 最小改动 vs 推荐",
                      "现参数（rsm.03）在 303 s 前一直跟着输入爬（+675 ADC）；只把 rsm 改到 0.15 "
                      "就冻结在 4.6 s；推荐档同时把落点从 +1626 压到 +849，过扣仍为 0",
                      "经过时间 (s)", "总 ADC",
                      [CurveSnapshot("输入", t, tot, GREY, 0.8),
                       CurveSnapshot("现参数 rsm.03", t, y_cur, RED, 1.5),
                       CurveSnapshot("最小改动 rsm.15", t, y_min, ORANGE, 1.4),
                       CurveSnapshot("推荐 rf.06 τc1 2 cap.005 rsm.15", t, y_rec, GREEN, 1.8)]),
        PanelSnapshot("③ 尾端放大（150~429 s）：尾漂到底有没有清掉",
                      "现参数尾端仍以 +0.126 ADC/s 上漂（410 s 里 +675 ADC）；两条处方档尾端"
                      "斜率 −0.11~−0.49 ADC/s（略过头，量级可忽略）",
                      "经过时间 (s)", "总 ADC",
                      [CurveSnapshot("现参数 rsm.03", t, y_cur, RED, 1.5),
                       CurveSnapshot("最小改动 rsm.15", t, y_min, ORANGE, 1.4),
                       CurveSnapshot("推荐（含 cap.005）", t, y_rec, GREEN, 1.8)]),
        PanelSnapshot("④ r_slow_max 扫描（cap 固定 0.005）",
                      "尾漂根因是幅度上限：rsm 0.02→0.15 时「10 s→末」从 +476 降到 +5 ADC、"
                      "冻结时刻从 297 s 提前到 6 s；门槛 ≈0.10（理论 0.08），0.15 有 2 倍余量",
                      "r_slow_max", "ADC（斜率 / 10s→末变化）",
                      [CurveSnapshot("尾端斜率 ADC/s",
                                     np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]),
                                     np.array([0.045, 0.050, -0.006, -0.108, -0.279, -0.424, -0.452, -0.491, -0.511, -0.514]),
                                     RED, 1.6),
                       CurveSnapshot("10s→末 变化/100",
                                     np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]),
                                     np.array([4.76, 3.68, 2.73, 1.89, 1.22, 0.48, 0.16, 0.08, 0.05, 0.02]),
                                     BLUE, 1.6)]),
        PanelSnapshot("⑤ slope_cap_frac 扫描（rsm 固定 0.15）",
                      "cap 不是「跟尾端速率」的参数：尾端每通道只要 0.0024 ADC/s，但 cap 降到 0.0005 "
                      "「10s→末」仍 +543 ADC——早段（2~20 s）被钳掉的量不可回收；cap ≥0.005 才清零",
                      "slope_cap_frac（对数）", "ADC（10s→末）/ 钳位比例%",
                      [CurveSnapshot("10s→末 变化",
                                     np.log10([0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012]),
                                     np.array([5.43, 4.15, 2.20, 1.12, 0.05, -0.46, -0.50]),
                                     RED, 1.8),
                       CurveSnapshot("被 cap 钳位的通道·帧占比 %",
                                     np.log10([0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012]),
                                     np.array([59.6, 39.0, 27.7, 20.6, 10.1, 6.5, 4.3]),
                                     BLUE, 1.5, "dash")]),
        PanelSnapshot("⑥ r_fast 扫描（cap 0.005 / rsm 0.15 / τc1 2）：落点与过扣的分界",
                      "落点随 rf 线性下降（+1847→+214 ADC），过扣在 rf≤0.08 恒为 0；"
                      "rf=0.10 首次出现过扣 222 ADC → 零过扣的 rf 上限 ≈0.08~0.09",
                      "r_fast", "ADC",
                      [CurveSnapshot("落点（段末−E）",
                                     np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]),
                                     np.array([1847, 1697, 1550, 1406, 1264, 1125, 988, 721, 463, 214]),
                                     GREEN, 1.8),
                       CurveSnapshot("过扣",
                                     np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]),
                                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 222, 391]),
                                     RED, 1.8)]),
    ]
    snap = FigureSnapshot(
        suptitle="手掌 b2d896（ADC、429 s）· 尾端漂移定因：不是速率不够，是 rsm 幅度上限 + 早段欠账不可回收",
        panels=panels, ncol=2, width_in=15.4, panel_height_in=3.2, dpi=110,
        legend_panel=1, max_points_per_curve=9000)
    out = TEMP / "figures" / "手掌_b2d896_尾漂定因与最小处方.png"
    r = export_small_multiples(snap, str(out), check=True)
    print("check:", r.get("check"), "bytes:", r.get("bytes"))
    print("path:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
