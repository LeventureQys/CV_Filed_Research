# -*- coding: utf-8 -*-
"""长保压"持续下漂"的定位与抑制（v3.4 观测器）。

现象与机理（实测，长期数据 20260922_095849 末窗 110~141 s）：
  载荷已进入平台（输入仅 +2.5 ADC/s），但 1 s 低通斜率估计的均值仍有 +3.5 ADC/s、
  慢态 x2 实际以 +5.0 ADC/s 增长 ⇒ 扣除量以输入的约 2 倍速率继续长，显示以
  −3.5 ADC/s 持续下漂（载荷 35,000 ⇒ −0.01 %/s ≈ −0.6 %/min ⇒ 10 分钟 −6 %）。

抑制旋钮实测（两个 142 s 保压会话）：
  slope_cap_frac（对话框里有）：0.010→0.002 把下漂 −3.55→−2.19（另一会话 −1.39→+0.22），
      0.001 进一步到 −0.66（另一会话 +2.53）；代价是扣除量少 15~25 %（显示整体略高）。
  r_slow_max（不在对话框）：0.35→0.08 下漂 −3.55→−1.43（另一会话 +0.18），与上面同向、可叠加。
  无效或反向：slope_gate_frac 0.10/0.20（无效）、tau_slope_s 0.5/0.25（更糟）、slow_confirm=6（无效）、
      tau_c_fast_s=2（对下漂无影响，它只影响加载后前 ~10 s 的对齐）。

用法：python long_hold_drift.py [--plot]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402

DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
ROOT = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\01_当前工作_K9\02_现场调参\09-21_09-26_工程A_v3.4\长期数据")
SESSIONS = ["20260922_095849_single_device_6cca99", "20260922_100118_single_device_2113fb"]

ARMS = [
    ("现役默认 cap=0.010", LIVE),
    ("cap=0.005", replace(LIVE, slope_cap_frac=0.005)),
    ("cap=0.003", replace(LIVE, slope_cap_frac=0.003)),
    ("cap=0.002", replace(LIVE, slope_cap_frac=0.002)),
    ("cap=0.001", replace(LIVE, slope_cap_frac=0.001)),
    ("r_slow_max=0.08", replace(LIVE, r_slow_max=0.08)),
    ("cap=0.002 + r_slow_max=0.08", replace(LIVE, slope_cap_frac=0.002, r_slow_max=0.08)),
]
PLOT_ARMS = ["现役默认 cap=0.010", "cap=0.003", "cap=0.002", "cap=0.001", "r_slow_max=0.08"]


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def segments(tin, t, rel=0.3, min_len=200):
    base = float(np.percentile(tin, 3))
    thr = base + rel * (float(np.max(tin)) - base)
    load = tin > thr
    segs, i, n = [], 0, load.size
    while i < n:
        if load[i]:
            j = i
            while j + 1 < n and load[j + 1]:
                j += 1
            if j - i >= min_len:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


def cell(t, tin, out, i, j):
    ts, a, b = t[i:j + 1], tin[i:j + 1], out[i:j + 1]
    ded = a - b
    k = min(max(int(np.searchsorted(ts, ts[0] + 3.0)), 0), len(ts) - 1)
    m = ts >= ts[-1] - 30.0
    slope = float(np.polyfit(ts[m], b[m], 1)[0])
    return f"{ded[k]:+6.0f}/{ded[-1]:+6.0f}/{ded[-1] - ded[m][0]:+5.0f}/{slope:+6.2f}"


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
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out_dir = HERE / "out"
    for name in SESSIONS:
        d = read_any_session_csv(ROOT / name / "device_001_seg000.csv")
        t, V = d["t"], d["V"]
        tin = V.sum(axis=1)
        segs = segments(tin, t)
        print(f"\n===== {name}  峰值 {tin.max():.0f} ADC  受载段 {len(segs)} =====", flush=True)
        print("  列 = 段: ded_s/ded_e/d30/slope30（slope30 负 = 仍在向下漂，单位 ADC/s）")
        curves = {}
        for tag, p in ARMS:
            t0 = time.time()
            out = run(p, t, V)
            curves[tag] = out
            print(f"{tag:30s} " + " | ".join(cell(t, tin, out, i, j) for (i, j) in segs)
                  + f"  [{time.time() - t0:.0f}s]", flush=True)

        if args.plot:
            sys.path.insert(0, str(DPTOOL_ROOT))
            from dptool import api  # noqa: E402
            tmp = out_dir / f"_dptool_long_{name[-6:]}"
            if tmp.exists():
                shutil.rmtree(tmp)
            n_ch = V.shape[1]
            write_total_csv(tmp / "00_输入总ADC" / "device_001_seg000.csv", t, tin, n_ch)
            for tag in PLOT_ARMS:
                write_total_csv(tmp / f"{PLOT_ARMS.index(tag) + 1:02d}_{tag}".replace(" ", "")
                                / "device_001_seg000.csv", t, curves[tag], n_ch)
            dirs = [str(p) for p in sorted(tmp.iterdir())]
            png = out_dir / f"长保压_下漂抑制_{name[-6:]}.png"
            res = api.plot_to_file(dirs, str(png), mode="overlay", series_by="dir",
                                   series_stream="out", signal="sum", dpi=130,
                                   suptitle=f"{name}：长保压下的持续下漂与抑制（52 通道总 ADC）")
            print("出图：", res.get("path") or res.get("error"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
