# -*- coding: utf-8 -*-
"""109_paramspace_plot：参数结构图——旋钮杠杆跨度 + rf/cap 响应曲线 + 可达边界散点。"""
from __future__ import annotations

import json
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

rows = json.loads((TEMP / "palm6" / "out" / "108_paramspace.json").read_text(encoding="utf-8"))
KN = {"r_fast": "rf 快态幅度比", "slope_cap_frac": "cap 慢态速率上限",
      "slow_confirm_s": "conf 受载确认", "soft_unfreeze_s": "soft 软冻结",
      "r_slow_max": "rsm 慢态幅度上限", "tau_c_fast_s": "τc1 快态收敛",
      "tau_r_fast_s": "τr1 空载快恢复", "tau_r_slow_idle_s": "τrsi 空载慢泄放"}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    spans = []
    for k, name in KN.items():
        vals = [r["real"]["dev"] for r in rows if r["tag"] == f"ofat_{k}"]
        spans.append((name, min(vals), max(vals)))
    spans.sort(key=lambda x: x[1] - x[2])

    def ofat(key):
        d = sorted((r["params"][key], r["real"]["dev"], r["flat"]["drop"])
                   for r in rows if r["tag"] == f"ofat_{key}")
        return ([a for a, _, _ in d], [b for _, b, _ in d], [c for _, _, c in d])

    rf_x, rf_dev, rf_fl = ofat("r_fast")
    cap_x, cap_dev, cap_fl = ofat("slope_cap_frac")
    g = [r for r in rows if r["tag"] == "grid"]

    panels = [
        PanelSnapshot("① 各旋钮对「落点」的杠杆跨度（单旋钮全量程扫）",
                      "只有 rf 是强旋钮（跨 4.75 N）；cap/conf/soft 弱；τc1 近死；τr1/τrsi 在单次保压中恒为 0",
                      "落点跨度 (N)", "",
                      [CurveSnapshot(f"{n}（跨 {hi-lo:.2f} N）",
                                     np.array([lo, hi]), np.array([i, i]), "#1f77b4", 2.4)
                       for i, (n, lo, hi) in enumerate(spans)]),
        PanelSnapshot("② rf：唯一的平台旋钮（对数横轴）",
                      "落点随 rf 单调下降；每压 1 N 落点，恒压下坠涨 ~1.15 N——1:1.15 的单向兑换",
                      "r_fast", "数值 (N)",
                      [CurveSnapshot("落点（真实台）", np.array(rf_x), np.array(rf_dev), "#d62728", 1.8),
                       CurveSnapshot("恒压下坠（合成台）", np.array(rf_x), np.array(rf_fl),
                                     "#1f77b4", 1.5, "dash")]),
        PanelSnapshot("③ cap：平滑/跟踪旋钮（对数横轴）",
                      "cap 大→补得快但回拽大；cap 小→残 余上漂（<0.002 时末斜率转正）；>0.025 饱和",
                      "slope_cap_frac", "数值 (N)",
                      [CurveSnapshot("落点（真实台）", np.array(cap_x), np.array(cap_dev), "#d62728", 1.8),
                       CurveSnapshot("下坠（真实台）", np.array(cap_x),
                                     np.array([r["real"]["drop"] for r in rows
                                               if r["tag"] == "ofat_slope_cap_frac"]),
                                     "#ff7f0e", 1.5, "dash")]),
        PanelSnapshot("④ rf×cap 网格的可达边界（真实台）",
                      "红框=下坠≤0.2N 且过减≤0.05N 的合规区；合规区内 |落点| 最小 = +0.44 N（rf.10 cap.002）——这就是 8 参数的天花板",
                      "下坠 (N)", "落点 (N)",
                      [CurveSnapshot(f"rf={r['params']['r_fast']:g}",
                                     np.array([r["real"]["drop"]]), np.array([r["real"]["dev"]]),
                                     "#7f7f7f", 1.0, "dot") for r in g[::7]]
                      + [CurveSnapshot("合规区", np.array([0, 0.2, 0.2, 0]),
                                      np.array([0.4, 0.4, 1.85, 1.85]), "#d62728", 1.5, "dash")]),
    ]
    snap = FigureSnapshot(
        suptitle="v3.4 观测器 8 参数的纯结构研究（152928 真实输入 + 恒压合成双台）",
        panels=panels, ncol=2, width_in=15.0, panel_height_in=3.4, dpi=110,
        legend_panel=0, max_points_per_curve=9000)
    out = TEMP / "figures" / "参数空间_结构研究.png"
    r = export_small_multiples(snap, str(out), check=True)
    print("check:", r.get("check"), "bytes:", r.get("bytes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
