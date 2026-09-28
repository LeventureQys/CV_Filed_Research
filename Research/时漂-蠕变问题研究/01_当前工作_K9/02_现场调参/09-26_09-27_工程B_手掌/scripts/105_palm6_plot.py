# -*- coding: utf-8 -*-
"""105_palm6_plot：本会话（152928）现参数 vs 推荐档 + x2 轨迹。"""
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
OUT = TEMP / "palm6" / "out"
CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
       "tau_r_slow_idle_s": 0.5}
NEW = {**CUR, "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0,
       "slope_cap_frac": 0.005, "r_slow_max": 0.15}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, tin, seg = z["t"], z["tot_pre"], z["tot_seg"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    rc = obs.run(t, z["pre"], obs.default_with(CUR))
    rn = obs.run(t, z["pre"], obs.default_with(NEW))
    yc = s93.medfilt1s(rc["out_tot"], fps) / 1000.0
    yn = s93.medfilt1s(rn["out_tot"], fps) / 1000.0
    E = 16481.0 / 1000.0
    panels = [
        PanelSnapshot("① 输入（pre 流）：总力值 N",
                      "1.05 s 加载，E≈16.48 N，蠕变 +2.01 N（+12%），快相 +1.16 N（+7%）",
                      "经过时间 (s)", "总力值 (N)",
                      [CurveSnapshot("输入", t, tin / 1000.0, "#7f7f7f", 1.0)]),
        PanelSnapshot("② 显示：现参数 vs 推荐档",
                      "红=现参数：x2 被 rsm=0.03 封顶后显示随蠕变上浮（末段 +0.0033 N/s）；"
                      "绿=推荐档：末段斜率≈0，不再上浮",
                      "经过时间 (s)", "总力值 (N)",
                      [CurveSnapshot("E≈16.48", np.array([1.2, 41.5]), np.array([E, E]),
                                     "#1f77b4", 1.0, "dash"),
                       CurveSnapshot("现参数（录制）", t, seg / 1000.0, "#7f7f7f", 1.0),
                       CurveSnapshot("现参数(复算)", t, yc, "#d62728", 1.2),
                       CurveSnapshot("推荐档 rf.06 τc1=2 conf2 cap.005 rsm.15", t, yn,
                                     "#2ca02c", 1.8)]),
        PanelSnapshot("③ 慢态 x2 轨迹",
                      "红=现参数：x2 在 ~8 s 冲到 ~390 后被 rsm·e 封顶不再长；"
                      "绿=推荐档：x2 持续跟随蠕变",
                      "经过时间 (s)", "x2 总值 (ADC)",
                      [CurveSnapshot("现参数 x2", t, rc["x2"], "#d62728", 1.2),
                       CurveSnapshot("推荐档 x2", t, rn["x2"], "#2ca02c", 1.8)]),
        PanelSnapshot("④ 总扣除量（x1+x2）",
                      "现参数扣除封在 ~980 ADC（≈1 N）追不动 +2010 的蠕变；推荐档持续增长",
                      "经过时间 (s)", "applied (ADC)",
                      [CurveSnapshot("现参数", t, rc["applied"], "#d62728", 1.2),
                       CurveSnapshot("推荐档", t, rn["applied"], "#2ca02c", 1.8),
                       CurveSnapshot("输入蠕变 +2010", np.array([1.2, 41.5]),
                                     np.array([0, 2011]), "#1f77b4", 1.0, "dash")]),
    ]
    snap = FigureSnapshot(
        suptitle="手掌 152928 · 上浮定因（rsm=0.03 封顶慢态）与修复：末段 +0.0033→≈0 N/s（v3.4 · K9 复算）",
        panels=panels, ncol=2, width_in=15.0, panel_height_in=3.2, dpi=110,
        legend_panel=0, max_points_per_curve=9000)
    out = TEMP / "figures" / "手掌_152928_上浮修复对比.png"
    r = export_small_multiples(snap, str(out), check=True)
    print("check:", r.get("check"), "bytes:", r.get("bytes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
