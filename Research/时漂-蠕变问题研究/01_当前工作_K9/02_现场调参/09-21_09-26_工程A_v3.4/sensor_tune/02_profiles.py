# -*- coding: utf-8 -*-
"""02 概览：把每份会话的总读数曲线按等间隔抽点打印（判断工况形状，不出图）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import SENSORS, load  # noqa: E402


def shape(tin: np.ndarray, t: np.ndarray, k: int = 48) -> str:
    idx = np.linspace(0, len(tin) - 1, k).astype(int)
    y = tin[idx]
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo if hi > lo else 1.0
    rows = []
    for lvl in range(9, -1, -1):
        thr = lo + span * lvl / 10.0
        rows.append("".join("#" if v >= thr else "." for v in y))
    lbl = f"[{t[0]:.0f}s..{t[-1]:.0f}s]"
    return "\n".join("  " + r for r in rows) + f"   lo={lo:.0f} hi={hi:.0f} {lbl}"


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
            tin = V.sum(axis=1)
            print(f"\n-- {tag}  ({V.shape[1]} ch, {len(t)} 帧, {t[-1] - t[0]:.0f}s) --")
            print(shape(tin, t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
