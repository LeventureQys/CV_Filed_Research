# -*- coding: utf-8 -*-
"""01 盘点：data\\ 下 5 类传感器的会话形状、工况与量程（只读，不出图）。"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from sensor_common import DATA_ROOT, OUT_DIR, SENSORS, load, load_segments  # noqa: E402


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    print(f"{'传感器':<12s} {'会话':<10s} {'帧数':>7s} {'通道':>5s} {'时长s':>8s} {'Hz':>6s} "
          f"{'基线':>10s} {'峰值':>10s} {'受载段':>6s} {'显示模式':>8s} {'力单位':>6s}")
    for sensor, info in SENSORS.items():
        for tag, rel in info["sessions"].items():
            d = load(rel)
            t, V = d["t"], d["V"]
            tin = V.sum(axis=1)
            segs, base, peak, thr = load_segments(tin)
            h = d["header"]
            dur = float(t[-1] - t[0])
            print(f"{sensor:<12s} {tag:<10s} {len(t):7d} {V.shape[1]:5d} {dur:8.2f} "
                  f"{len(t) / dur:6.1f} {base:10.1f} {peak:10.1f} {len(segs):6d} "
                  f"{h.get('显示模式', '?'):>8s} {h.get('显示力值单位', '?'):>6s}")
            report[f"{sensor}/{tag}"] = {
                "path": str(DATA_ROOT / rel),
                "frames": int(len(t)), "channels": int(V.shape[1]),
                "duration_s": dur, "hz": len(t) / dur,
                "baseline": base, "peak": peak, "segments": [list(map(int, s)) for s in segs],
                "header": {k: h.get(k, "") for k in
                           ("会话ID", "数据阶段", "显示模式", "显示力值单位", "上报力值单位",
                            "显示压强单位", "含原始ADC", "录制频率")},
                "nan_ratio": float(np.isnan(V).mean()),
                "monotonic_t": bool(np.all(np.diff(t) >= 0)),
                "dup_t_ratio": float(np.mean(np.diff(t) == 0)) if len(t) > 1 else 0.0,
            }
    (OUT_DIR / "01_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '01_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
