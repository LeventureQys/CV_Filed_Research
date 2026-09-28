# -*- coding: utf-8 -*-
"""112_slowonly：只考虑慢漂（放开过扣/下坠）的极限参数。

三台验证：152928 真实输入 / 恒压合成 / 1800 s 长蠕变合成。
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
s93 = import_module("93_palm4_sweep")
s110 = import_module("110_longhold")
s111 = import_module("111_longhold2")
OUT = TEMP / "palm6" / "out"
E1 = 16481.0

SLOW = {"r_fast": 0.08, "tau_c_fast_s": 2.0, "slow_confirm_s": 0.0,
        "soft_unfreeze_s": 0.1, "slope_cap_frac": 0.05, "r_slow_max": 0.6,
        "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, pre = z["t"], z["pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    print("== 台1：152928 真实输入（E=16481）==")
    for label, p in (("BEST(旧: rf.1 cap.002 rsm.15)", {"r_fast": 0.10, "tau_c_fast_s": 10.0,
                     "slow_confirm_s": 2.0, "soft_unfreeze_s": 2.0, "slope_cap_frac": 0.002,
                     "r_slow_max": 0.15, "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}),
                     ("SLOW(新: rf.08 τc1=2 conf0 soft.1 cap.05 rsm.6)", SLOW)):
        r = obs.run(t, pre, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"], fps)
        m = t >= 1.6
        end = y[m][-1]
        mm = t[m] >= t[m][-1] - 5.0
        print(f"  {label:<44s} 落点={end-E1:+7.1f}({(end-E1)*0.001:+.3f}N) "
              f"下坠={(y[m].max()-end)*0.001:6.3f}N 过减={max(0.0,E1-y[m].min())*0.001:6.3f}N "
              f"末斜率={np.polyfit(t[m][mm], y[m][mm], 1)[0]*0.001:+.4f}N/s")

    print("\n== 台2：恒压 17 N（无蠕变，看过扣代价）==")
    t2, V2 = s110.synth_long(creep_nps=0.0, dur=60.0)
    for label, p in (("BEST", {"r_fast": 0.10, "tau_c_fast_s": 10.0, "slow_confirm_s": 2.0,
                               "soft_unfreeze_s": 2.0, "slope_cap_frac": 0.002,
                               "r_slow_max": 0.15, "tau_r_fast_s": 6.0,
                               "tau_r_slow_idle_s": 0.5}), ("SLOW", SLOW)):
        r = obs.run(t2, V2, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"] / 1000.0, 100.0)
        m = t2 >= 2.2
        print(f"  {label:<6s} 稳态={y[m][-1]:7.3f} N（过扣 {17.0-y[m][-1]:+.3f} N）")

    print("\n== 台3：1800 s 长蠕变（+4.5 N）==")
    t3, V3 = s111.synth(dur=1800.0)
    for label, p in (("BEST(rsm.15)", {"r_fast": 0.10, "tau_c_fast_s": 10.0,
                      "slow_confirm_s": 2.0, "soft_unfreeze_s": 2.0,
                      "slope_cap_frac": 0.002, "r_slow_max": 0.15,
                      "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}), ("SLOW", SLOW)):
        r = obs.run(t3, V3, obs.default_with(p))
        y = r["out_tot"] / 1000.0
        row = " ".join(f"{y[min(int(np.searchsorted(t3, x)), len(t3)-1)] - 16.5:+7.3f}"
                       for x in (60, 300, 600, 1200, 1800))
        print(f"  {label:<12s} 显示−E @60/300/600/1200/1800s: {row} N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
