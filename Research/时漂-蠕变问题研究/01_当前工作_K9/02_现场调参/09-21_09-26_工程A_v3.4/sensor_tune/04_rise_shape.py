# -*- coding: utf-8 -*-
"""04 上升形态：总读数的「相对起点的增量」在 log 时间轴上的分布，
   用来判断录制里有没有真实加载台阶、后续是蠕变还是持续慢加载。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import SENSORS, load  # noqa: E402

BINS = [0.0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for sensor, info in SENSORS.items():
        print(f"\n================ {sensor} ================")
        for tag, rel in info["sessions"].items():
            d = load(rel)
            t, V = d["t"], d["V"]
            t = t - t[0]
            tin = V.sum(axis=1)
            v0 = float(np.median(tin[t <= min(0.2, t[-1])]))
            dur = t[-1]
            print(f"\n-- {tag}  dur={dur:.0f}s  v(0)={v0:.0f}  v(end)={tin[-1]:.0f}  "
                  f"总增量={tin[-1]-v0:+.0f} ADC（{(tin[-1]-v0)/max(v0,1)*100:+.2f}% of v0）")
            edges = [b for b in BINS if b <= dur] + [dur]
            row = []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (t >= a) & (t < b)
                if not m.any():
                    continue
                seg = tin[m]
                dt = max(b - a, 1e-9)
                row.append(f"[{a:g}-{b:g}){seg[-1]-v0:+.0f}({(seg[-1]-seg[0])/dt:+.1f}/s)")
            print("   " + " ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
