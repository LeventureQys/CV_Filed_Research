# -*- coding: utf-8 -*-
"""133_palm8_cross：把"最小参数档"放到 152928（41.8 s 短保压）上交叉验证。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
Z6 = np.load(TEMP / "palm6" / "out" / "streams.npz")
T0, E1 = 1.05, 16481.0

CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
       "tau_r_slow_idle_s": 0.5}
CANDS = [
    ("cur", CUR),
    ("rsm.1_cap.005", {**CUR, "r_slow_max": 0.1, "slope_cap_frac": 0.005}),
    ("rsm.15_cap.005", {**CUR, "r_slow_max": 0.15, "slope_cap_frac": 0.005}),
    ("rf.02_rsm.15_cap.005", {**CUR, "r_fast": 0.02, "r_slow_max": 0.15, "slope_cap_frac": 0.005}),
    ("rec", {**CUR, "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0,
             "slope_cap_frac": 0.005, "r_slow_max": 0.15}),
]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, pre = Z6["t"], Z6["pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"{'参数集':<24s}{'落点(N)':>9s}{'10s→末(N)':>10s}{'下坠(N)':>9s}{'过扣(N)':>9s}{'末斜率':>8s}")
    for name, p in CANDS:
        r = obs.run(t, pre, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"], fps)
        m = t >= T0 + 0.6
        i10 = int(np.searchsorted(t, T0 + 10))
        end = float(y[-1])
        mm = t[m] >= t[m][-1] - 5.0
        print(f"{name:<24s}{end-E1:+9.3f}{end-y[i10]:+10.3f}{y[m].max()-end:9.3f}"
              f"{max(0.0, E1-y[m].min()):9.3f}{np.polyfit(t[m][mm], y[m][mm], 1)[0]:+8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
