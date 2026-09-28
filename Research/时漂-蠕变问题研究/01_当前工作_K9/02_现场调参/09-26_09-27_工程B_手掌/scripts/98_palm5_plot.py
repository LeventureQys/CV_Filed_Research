# -*- coding: utf-8 -*-
"""98_palm5_plot：力值会话（算法关）输入天然回调 vs 算法附加回调的分解图。"""
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
OUT = TEMP / "palm5" / "out"
WRITTEN = {"r_fast": 0.10, "tau_c_fast_s": 15, "slow_confirm_s": 1,
           "soft_unfreeze_s": 2, "slope_cap_frac": 0.008, "r_slow_max": 0.10,
           "tau_r_fast_s": 6, "tau_r_slow_idle_s": 0.5}
FIX = {**WRITTEN, "r_fast": 0.03, "tau_c_fast_s": 15, "slope_cap_frac": 0.003,
       "r_slow_max": 0.05}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, V, tin = z["t"], z["seg"], z["tot_seg"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    rw = obs.run(t, V, obs.default_with(WRITTEN))
    rf_ = obs.run(t, V, obs.default_with(FIX))
    yw = s93.medfilt1s(rw["out_tot"], fps)
    yf = s93.medfilt1s(rf_["out_tot"], fps)
    tin_s = s93.medfilt1s(tin, fps)
    m = t >= 1.0
    panels = [
        PanelSnapshot("① 输入（算法关的显示）：手压过程",
                      "1.35~2.2 s 压到峰 14.97 N，随后自然松弛到 13.68 N——天然回调 1.29 N",
                      "经过时间 (s)", "总力值 (N)",
                      [CurveSnapshot("输入", t, tin_s, "#7f7f7f", 1.0)]),
        PanelSnapshot("② 显示对比：已写入参数 vs 修正档",
                      "红=已写入（rf.10 cap.008）：算法再拽 0.57 N，总回调 1.86 N；"
                      "绿=修正档（rf.03 cap.003）：仅附加 ~0.06 N",
                      "经过时间 (s)", "总力值 (N)",
                      [CurveSnapshot("输入（算法关）", t, tin_s, "#7f7f7f", 1.0),
                       CurveSnapshot("已写入参数", t, yw, "#d62728", 1.2),
                       CurveSnapshot("修正档 rf.03 τc1=15 cap.003", t, yf, "#2ca02c", 1.8)]),
        PanelSnapshot("③ 算法扣除量（输入 − 显示）",
                      "红=已写入：压上后扣除冲到 ~2.2 N 且不随松弛释放；绿=修正档扣除小而平",
                      "经过时间 (s)", "扣除量 (N)",
                      [CurveSnapshot("已写入参数", t, tin - rw["out_tot"], "#d62728", 1.2),
                       CurveSnapshot("修正档", t, tin - rf_["out_tot"], "#2ca02c", 1.8)]),
        PanelSnapshot("④ 附加回调（显示下坠 − 输入下坠）视角",
                      "同一时刻显示与输入之差（相对 3 s 处）：红持续下弯 = 算法在拽；绿近似贴 0",
                      "经过时间 (s)", "相对 3s 的差值 (N)",
                      [CurveSnapshot("已写入参数", t, (yw - yw[t >= 3.0][0]), "#d62728", 1.2),
                       CurveSnapshot("修正档", t, (yf - yf[t >= 3.0][0]), "#2ca02c", 1.8),
                       CurveSnapshot("输入", t, (tin_s - tin_s[t >= 3.0][0]), "#7f7f7f", 1.0)]),
    ]
    snap = FigureSnapshot(
        suptitle="手掌 · 力值尺度（1 N=1000 ADC）· 回调分解：输入天然 1.29 N + 算法附加（已写入 0.57 → 修正 0.06 N）",
        panels=panels, ncol=2, width_in=15.0, panel_height_in=3.2, dpi=110,
        legend_panel=0, max_points_per_curve=9000)
    out = TEMP / "figures" / "手掌_力值尺度_回调分解.png"
    r = export_small_multiples(snap, str(out), check=True)
    print("check:", r.get("check"), "bytes:", r.get("bytes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
