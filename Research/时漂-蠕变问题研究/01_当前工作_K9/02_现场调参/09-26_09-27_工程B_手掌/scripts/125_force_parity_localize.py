# -*- coding: utf-8 -*-
"""125_force_parity_localize：定位「复算 vs 录制」偏差的时间位置与幅值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
fl = import_module("122_force_lib")
OUT = TEMP / "palm7" / "out"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    t, pre, seg = z["t"], z["pre"], z["seg"]
    rec = fl.REC8
    r = obs.run(t, pre, obs.default_with(rec))
    mine, dev = r["out_tot"], r["out_tot"] - seg.sum(1)
    ad = np.abs(dev)
    print(f"全段 |偏差|：均={ad.mean():.4f} 中位={np.median(ad):.4f} p95={np.percentile(ad,95):.4f} "
          f"max={ad.max():.4f} N")
    # 偏差按时间分段统计
    print(f"\n{'时间窗':>16s} {'均差':>8s} {'最大':>8s}")
    for a in np.arange(0.0, 33.0, 2.0):
        m = (t >= a) & (t < a + 2.0)
        if m.sum() < 5:
            continue
        print(f"{a:6.1f}~{a+2.0:5.1f}s {ad[m].mean():8.4f} {ad[m].max():8.4f}")
    # 偏差最大的时刻
    idx = np.argsort(ad)[-12:][::-1]
    print("\n偏差最大的 12 帧：")
    for i in sorted(idx):
        print(f"  t={t[i]:6.3f}  输入={pre.sum(1)[i]:8.3f}  复算={mine[i]:8.3f}  "
              f"录制={seg.sum(1)[i]:8.3f}  偏差={dev[i]:+8.4f}  x1={r['x1'][i]:7.3f} "
              f"x2={r['x2'][i]:7.3f}  applied={r['applied'][i]:7.3f}")
    # 全卸载窗与再加载窗
    for lo, hi, lab in ((15.6, 19.0, "全卸载→再加载"),):
        m = (t >= lo) & (t <= hi)
        print(f"\n{lab}（{lo}~{hi}s）逐 0.1 s 偏差：")
        for i in range(int(np.searchsorted(t, lo)), int(np.searchsorted(t, hi)), 10):
            print(f"  t={t[i]:6.3f} 输入={pre.sum(1)[i]:8.3f} 复算={mine[i]:8.3f} "
                  f"录制={seg.sum(1)[i]:8.3f} 偏差={dev[i]:+8.4f} "
                  f"x1={r['x1'][i]:6.3f} x2={r['x2'][i]:6.3f} ap={r['applied'][i]:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
