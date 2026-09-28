# -*- coding: utf-8 -*-
"""最终候选参数组的跨数据集验证（把"对快分量的对齐"与"长保压防下漂"合到一起看）。

数据集：
  1) 20260924_155543（ADC、18 s、无快漂传感器）→ 以拟合弹性电平 E 为基准：末值−E / ride / dip / settle
  2) 长期数据 095849（ADC、142 s 保压）      → 末 30 s 下漂速率 slope30、扣除增长 d30、总扣除 ded_e
  3) 长期数据 100118（ADC、142 s、两段保压）  → 同上（取两段）
  4) other_recorder（力值 N、50 s）          → 仅作单位口径的旁证：末帧扣除 / 段2 显示漂移

用法：python final_params_check.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402
from new_session_analyze import fit_elastic  # noqa: E402

NEW = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法"
           r"\20260924_155543_single_device_1201c1\device_001_seg000.csv")
LONG = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\01_当前工作_K9\02_现场调参\09-21_09-26_工程A_v3.4\长期数据")
OTHER = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\other_recorder\device_001_seg000.csv")

SETS = [
    ("现役默认", LIVE),
    ("仅 τc1=2", replace(LIVE, tau_c_fast_s=2.0)),
    ("仅 cap=0.002", replace(LIVE, slope_cap_frac=0.002)),
    ("★ τc1=2 + cap=0.002", replace(LIVE, tau_c_fast_s=2.0, slope_cap_frac=0.002)),
    ("τc1=2 + cap=0.001", replace(LIVE, tau_c_fast_s=2.0, slope_cap_frac=0.001)),
    ("τc1=2 + cap=0.002 + r_slow_max=0.06",
     replace(LIVE, tau_c_fast_s=2.0, slope_cap_frac=0.002, r_slow_max=0.06)),
    ("τc1=2 + cap=0.001 + r_slow_max=0.06",
     replace(LIVE, tau_c_fast_s=2.0, slope_cap_frac=0.001, r_slow_max=0.06)),
]


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


def sink_cells(t, tin, out, segs):
    cells = []
    for (i, j) in segs:
        ts, a, b = t[i:j + 1], tin[i:j + 1], out[i:j + 1]
        ded = a - b
        k = min(max(int(np.searchsorted(ts, ts[0] + 3.0)), 0), len(ts) - 1)
        m = ts >= ts[-1] - 30.0
        slope = float(np.polyfit(ts[m], b[m], 1)[0])
        cells.append(f"ded_s={ded[k]:+.0f} ded_e={ded[-1]:+.0f} "
                     f"d30={ded[-1] - ded[m][0]:+.0f} slope30={slope:+.2f}")
    return cells


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # ── 1) 无快漂传感器（18 s）：以拟合 E 为基准 ──
    d = read_any_session_csv(NEW)
    t, V = d["t"], d["V"]
    tin = V.sum(axis=1)
    E = float(fit_elastic(t, tin)[0][0])
    print(f"=== 1) 20260924_155543（ADC、18 s、无快漂）  拟合 E = {E:.0f} ADC ===")
    print(f"{'方案':34s} {'末值−E':>9s} {'ride':>8s} {'dip':>8s} {'settle':>7s} {'|偏差|均':>9s}")
    for tag, p in SETS:
        out = run(p, t, V)
        ride = float(np.max(out[t >= 2.0] - E))
        dip = float(np.min(out[t >= 8.0] - E))
        integ = float(np.mean(np.abs(out[t >= 2.0] - E)))
        ok = np.abs(out - E) <= 100.0
        idx = np.nonzero(ok)[0]
        settle = float("nan")
        if idx.size:
            lb = np.nonzero(~ok[idx[0]:])[0]
            jj = min(idx[0] + (lb[-1] + 1 if lb.size else 0), len(t) - 1)
            settle = float(t[jj] - 1.10)
        print(f"{tag:34s} {out[-1] - E:+9.1f} {ride:+8.1f} {dip:+8.1f} {settle:7.1f} "
              f"{integ:9.1f}")

    # ── 2/3) 长保压（142 s）──
    for name in ("20260922_095849_single_device_6cca99", "20260922_100118_single_device_2113fb"):
        d = read_any_session_csv(LONG / name / "device_001_seg000.csv")
        t, V = d["t"], d["V"]
        tin = V.sum(axis=1)
        segs = segments(tin, t)
        print(f"\n=== {'2' if '095849' in name else '3'}) {name}  峰值 {tin.max():.0f} "
              f"段数 {len(segs)}（slope30 负 = 仍在向下漂，ADC/s）===")
        for tag, p in SETS:
            t0 = time.time()
            out = run(p, t, V)
            print(f"{tag:34s} " + " | ".join(sink_cells(t, tin, out, segs))
                  + f"  [{time.time() - t0:.0f}s]", flush=True)

    # ── 4) 力值单位旁证 ──
    d = read_any_session_csv(OTHER)
    t, V = d["t"], d["V"]
    tin = V.sum(axis=1)
    m1 = (t >= 10.0) & (t <= 21.0)
    m2 = t >= 35.0
    print("\n=== 4) other_recorder（力值 N、50 s；单位不同，仅旁证）===")
    for tag, p in SETS:
        out = run(p, t, V)
        print(f"{tag:34s} 末帧扣除={float((tin - out)[-1]):6.3f} N  "
              f"段1显示漂移={out[m1][-1] - out[m1][0]:+6.3f}  "
              f"段2显示漂移={out[m2][-1] - out[m2][0]:+6.3f}")


if __name__ == "__main__":
    main()
