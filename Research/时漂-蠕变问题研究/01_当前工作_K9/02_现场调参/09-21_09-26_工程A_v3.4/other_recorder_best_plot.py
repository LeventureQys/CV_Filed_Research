# -*- coding: utf-8 -*-
"""本数据（other_recorder）推荐参数对比图：输入 / 现役 / 方案A（仅 7 项可调）/ 方案A+沿阈。

用 toolbox\\数据解析工具 的 dptool 出图，单面板 overlay、signal=sum（总值）。
总值写进「单元0」、其余通道补 0 ⇒ dptool 求和即该变体的 52 通道总值。

用法：python other_recorder_best_plot.py [--out out]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv, CSV_PATH  # noqa: E402

DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")

# 方案 A：只用「参数...」对话框里那 7 项（搜索出的本数据最优）
PLAN_A = dict(r_fast=0.08, tau_c_fast_s=6.0, slow_confirm_s=3.0, soft_unfreeze_s=2.0,
              slope_cap_frac=0.012, tau_r_slow_idle_s=8.0, tau_r_fast_s=0.25)
# 方案 A+：A 之上再启用 H3 沿前馈（edge_slope_thres 按力值量级重标；该参数目前不在对话框里）
PLAN_A_PLUS = {**PLAN_A, "edge_slope_thres": 0.1}


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def write_total_csv(path: Path, t, total, n_ch):
    zeros = ",".join(["0.000000"] * (n_ch - 1))
    ts = t + 171642.0
    lines = ["##Session", "数据阶段,processed_display", "值阶段,processed_display",
             f"行数,{len(t)}", f"列数,{n_ch}", "##Data",
             "timestamp,elapsed,frame_index," + ",".join(f"ch{j}" for j in range(n_ch))]
    for i in range(len(t)):
        lines.append(f"{ts[i]:.6f},{t[i]:.6f},{i},{total[i]:.6f},{zeros}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out_dir = HERE / args.out
    d = read_any_session_csv(CSV_PATH)
    t, V = d["t"], d["V"]
    tin = V.sum(axis=1)
    n_ch = V.shape[1]

    curves = [
        ("00_输入(算法关闭)", tin),
        ("01_现役默认", run(LIVE, t, V)),
        ("02_方案A_仅7项可调", run(replace(LIVE, **PLAN_A), t, V)),
        ("03_方案A+启用沿前馈", run(replace(LIVE, **PLAN_A_PLUS), t, V)),
    ]
    tmp = out_dir / "_dptool_best"
    if tmp.exists():
        shutil.rmtree(tmp)
    for label, y in curves:
        write_total_csv(tmp / label / "device_001_seg000.csv", t, y, n_ch)
        print(f"{label:22s} 末帧总值 {y[-1]:7.3f} N  段2平台 {y[t >= 40].mean():7.3f} N")

    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api, merge as M  # noqa: E402
    dirs = [str(tmp / c[0]) for c in curves]
    png = out_dir / "other_recorder_推荐参数对比.png"
    res = api.plot_to_file(dirs, str(png), mode="overlay", series_by="dir",
                           series_stream="out", signal="sum", dpi=130,
                           suptitle="other_recorder（20260924_130514）：推荐参数 vs 现役（52 通道总值 / N）")
    print("出图：", res.get("path") or res.get("error"))
    tbl, _ = M.merge(dirs, start=int(np.searchsorted(t, 24.5)))
    png2 = out_dir / "other_recorder_推荐参数对比_放大.png"
    M.plot_table(tbl, str(png2), mode="overlay", series_by="dir", series_stream="out",
                 signal="sum", ylim=(8.6, 12.6), dpi=130,
                 suptitle="同图放大（25 s 之后：重载阶跃 + 慢爬段）")
    print("放大图：", png2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
