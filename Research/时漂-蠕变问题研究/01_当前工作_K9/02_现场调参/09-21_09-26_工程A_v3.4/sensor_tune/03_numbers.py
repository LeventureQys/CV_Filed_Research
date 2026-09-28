# -*- coding: utf-8 -*-
"""03 数值概览：每份会话总读数在若干时刻的绝对值 + 全局斜率分位（判断加载方式与蠕变量级）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import SENSORS, load  # noqa: E402

PROBE_T = [0.0, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0]


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
            dur = t[-1] - t[0]
            print(f"\n-- {tag}  ch={V.shape[1]}  时长={dur:.0f}s  Hz={len(t)/dur:.1f}")
            vals = []
            for tt in PROBE_T:
                if tt <= dur:
                    i = int(np.searchsorted(t, t[0] + tt))
                    i = min(i, len(t) - 1)
                    vals.append(f"{tt:g}:{tin[i]:.0f}")
            print("   " + "  ".join(vals))
            # 加载段形态：从首帧到 95% 峰值用时
            lo = float(tin[0])
            hi = float(tin.max())
            thr = lo + 0.95 * (hi - lo)
            i95 = int(np.argmax(tin >= thr))
            print(f"   首帧={lo:.0f} 末帧={tin[-1]:.0f} 峰={hi:.0f} 到 95%峰用时={t[i95]-t[0]:.2f}s"
                  f"   末段总增量(峰→末)={tin[-1]-hi:+.0f}")
            # 逐通道斜率分位（用相邻帧差 / 实际 dt，重复时间戳跳过）
            dtt = np.diff(t)
            dvv = np.diff(V, axis=0)
            m = dtt > 1e-9
            rate = np.abs(dvv[m] / dtt[m, None])
            print(f"   逐通道 |dv/dt| 分位: p50={np.percentile(rate,50):.2f} "
                  f"p90={np.percentile(rate,90):.2f} p99={np.percentile(rate,99):.2f} "
                  f"p99.9={np.percentile(rate,99.9):.2f} max={rate.max():.1f}  (ADC/s)")
            # 每通道首末与峰
            load_span = tin.max() - tin[0]
            print(f"   总载荷步(峰-首帧)={load_span:.0f} ADC, 记录末总蠕变(末-95%峰区)="
                  f"{tin[int(len(t)*0.5):].max()-tin[int(len(t)*0.5):].min():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
