# -*- coding: utf-8 -*-
"""102_final_plot：修正版终图——回调归因（rf.10 的 x1 稳态）+ 新旧参数在两个数据集的对比。"""
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
s100 = import_module("100_palm5_synth")
OUT4 = TEMP / "palm4" / "out"
OLD = {"r_fast": 0.10, "tau_c_fast_s": 15, "slow_confirm_s": 1, "soft_unfreeze_s": 2,
       "slope_cap_frac": 0.008, "r_slow_max": 0.10, "tau_r_fast_s": 6,
       "tau_r_slow_idle_s": 0.5}
NEW = {"r_fast": 0.01, "tau_c_fast_s": 40, "slow_confirm_s": 5, "soft_unfreeze_s": 2,
       "slope_cap_frac": 0.003, "r_slow_max": 0.05, "tau_r_fast_s": 6,
       "tau_r_slow_idle_s": 0.5}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    # ①② 实测会话
    z5 = np.load(TEMP / "palm5" / "out" / "streams.npz")
    t5, disp5 = z5["t"], z5["tot_seg"]
    sm5_old = disp5.copy()  # 实测显示=已写入参数的嵌入式结果
    # 新参数在"实测显示的合成恒压形状"上的表现（输入不可得，用恒压合成展示下坠上限）
    tf = np.arange(0, 15.5, 1.0 / 100.7)
    yf = s100.synth(tf)
    Vf = np.repeat(yf[:, None] * 1000.0 / 71.0, 71, axis=1)
    ro = obs.run(tf, Vf, obs.default_with(OLD))
    rn = obs.run(tf, Vf, obs.default_with(NEW))
    yo = s93.medfilt1s(ro["out_tot"] / 1000.0, 100.7)
    yn = s93.medfilt1s(rn["out_tot"] / 1000.0, 100.7)
    # ③④ 真实输入会话 101043
    z4 = np.load(OUT4 / "streams.npz")
    t4, pre4, tin4 = z4["t"], z4["pre"], z4["tot_pre"]
    fps4 = (len(t4) - 1) / (t4[-1] - t4[0])
    r4o = obs.run(t4, pre4, obs.default_with(OLD))
    r4n = obs.run(t4, pre4, obs.default_with(NEW))
    y4o = s93.medfilt1s(r4o["out_tot"], fps4) / 1000.0
    y4n = s93.medfilt1s(r4n["out_tot"], fps4) / 1000.0
    panels = [
        PanelSnapshot("① 实测（102924，嵌入式+已写入参数）",
                      "峰 14.72 → 末 13.55 N，回调 ~1.17 N；raw/1000≡seg（文件里没有算法前输入）",
                      "经过时间 (s)", "总力值 (N)",
                      [CurveSnapshot("实测显示（已写入 rf.10）", t5, disp5, "#d62728", 1.2)]),
        PanelSnapshot("② 恒压合成输入：算法自身回调上限",
                      "输入压到 14.97 N 后完全恒定。红=已写入参数自己就下坠 0.82 N（x1 目标 1.35 N）；"
                      "绿=新参数仅 0.09 N",
                      "经过时间 (s)", "总力值 (N)",
                      [CurveSnapshot("输入(恒压)", tf, yf, "#7f7f7f", 1.0),
                       CurveSnapshot("已写入参数", tf, yo, "#d62728", 1.2),
                       CurveSnapshot("新参数 rf.01 conf5 cap.003", tf, yn, "#2ca02c", 1.8)]),
        PanelSnapshot("③ 真实输入会话（101043）：旧 vs 新",
                      "红=旧参数 12 s 内被拽低 0.56 N；绿=新参数几乎贴着输入（上漂 +1.8 N，方向合规）",
                      "经过时间 (s)", "总力值 (N)",
                      [CurveSnapshot("输入", t4, tin4 / 1000.0, "#7f7f7f", 1.0),
                       CurveSnapshot("旧参数", t4, y4o, "#d62728", 1.2),
                       CurveSnapshot("新参数", t4, y4n, "#2ca02c", 1.8)]),
        PanelSnapshot("④ 旧参数的扣除量构成（恒压合成）",
                      "x1（快态稳态 rf/(1+rf)·电平）是回调主源；新参数 x1 目标只有 0.13 N",
                      "经过时间 (s)", "扣除量 (N)",
                      [CurveSnapshot("旧参数 x1", tf, ro["x1"] / 1000.0, "#d62728", 1.2),
                       CurveSnapshot("旧参数 x1+x2", tf, (ro["x1"] + ro["x2"]) / 1000.0,
                                     "#d62728", 1.0, "dash"),
                       CurveSnapshot("新参数 x1", tf, rn["x1"] / 1000.0, "#2ca02c", 1.8)]),
    ]
    snap = FigureSnapshot(
        suptitle="手掌 · 力值尺度 · 回调归因与修复：rf.10 的 x1 稳态目标≈1.35N 是回调主源 → rf.01 后回调 ≤0.09N",
        panels=panels, ncol=2, width_in=15.0, panel_height_in=3.2, dpi=110,
        legend_panel=0, max_points_per_curve=9000)
    out = TEMP / "figures" / "手掌_力值尺度_回调修复终版.png"
    r = export_small_multiples(snap, str(out), check=True)
    print("check:", r.get("check"), "bytes:", r.get("bytes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
