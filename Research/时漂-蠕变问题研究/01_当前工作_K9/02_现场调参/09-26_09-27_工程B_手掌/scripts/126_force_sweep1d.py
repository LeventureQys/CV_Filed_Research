# -*- coding: utf-8 -*-
"""126_force_sweep1d：力值模式单旋钮扫描（OFAT）+ 计时。

基线 = 录制现役参数（rf.03/τc1 40/conf 2/soft 2/cap.011/rsm.35/τr1 6/τrsi 0.5）。
指标（两段保压）：落点=段末显示−E、下坠=段内峰−段末、过扣=E−段内最小、末斜率、稳定时间。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
fl = import_module("122_force_lib")
OUT = TEMP / "palm7" / "out"

GRID = {
    "r_fast": [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "tau_c_fast_s": [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 40.0, 120.0],
    "slow_confirm_s": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
    "soft_unfreeze_s": [0.1, 0.5, 2.0, 8.0, 20.0, 60.0],
    "slope_cap_frac": [0.001, 0.002, 0.004, 0.006, 0.008, 0.011, 0.015, 0.02, 0.03, 0.05],
    "r_slow_max": [0.02, 0.05, 0.10, 0.20, 0.35, 0.60],
    "tau_r_fast_s": [0.05, 0.5, 2.0, 6.0, 30.0, 300.0],
    "tau_r_slow_idle_s": [0.0, 0.1, 0.5, 2.0, 10.0, 100.0],
}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    t, pre = z["t"], z["pre"]
    tin = pre.sum(1)
    fps = (len(t) - 1) / (t[-1] - t[0])
    segs = fl.segments(t, tin)
    base = dict(fl.REC8)
    t0 = time.perf_counter()
    r = obs.run(t, pre, obs.default_with(base))
    dt = time.perf_counter() - t0
    print(f"Python 移植单次复算耗时 {dt*1000:.0f} ms（{len(t)} 帧 × {pre.shape[1]} 通道）")
    print(f"段：{[(round(s['t0'],2), round(s['t1'],2), round(s['E'],3)) for s in segs]}")

    def line(tag: str, over: dict) -> dict:
        p = {**base, **over}
        rr = obs.run(t, pre, obs.default_with(p))
        rows = fl.evaluate(rr["out_tot"], t, segs, fps)
        s = " | ".join(f"落点{r['dev']:+6.3f} 下坠{r['drop']:5.3f} 过扣{r['over']:5.3f} "
                       f"稳{r['tsettle']:5.2f}s 末斜{r['slope_end']:+.4f}" for r in rows)
        print(f"{tag:<28s} {s}")
        return {"tag": tag, "over": over, "rows": rows}

    print(f"\n== 基线（录制现役参数）==")
    b = line("基线", {})
    res = [b]
    for key, vals in GRID.items():
        print(f"\n== {key} ==")
        for v in vals:
            res.append(line(f"{key}={v:g}", {key: v}))
    (OUT / "126_sweep1d.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
