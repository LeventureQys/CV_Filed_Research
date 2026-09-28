# -*- coding: utf-8 -*-
"""113_compromise：慢漂优先 + 过扣压到 ~0.5N 的折中档。

四台：152928 真实 / 恒压 60s（过扣）/ 1800s 长蠕变（尾漂）/ 恒压长时（过扣是否稳定）。
扫 rf × cap（conf=0/soft=0.1/rsm=0.6/τc1=2 固定）。
"""
from __future__ import annotations

import itertools
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
BASE = {"r_fast": 0.03, "tau_c_fast_s": 2.0, "slow_confirm_s": 0.0,
        "soft_unfreeze_s": 0.1, "slope_cap_frac": 0.005, "r_slow_max": 0.6,
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
    tf, Vf = s110.synth_long(creep_nps=0.0, dur=60.0)
    t3, V3 = s111.synth(dur=1800.0)
    print(f"{'参数':<26s} | 真实: 落点N 末斜率N/s | 恒压过扣N | 1800s 显示−E (600/1800)")
    for rf, cap in itertools.product([0.02, 0.03, 0.05], [0.003, 0.005, 0.008]):
        p = {**BASE, "r_fast": rf, "slope_cap_frac": cap}
        r = obs.run(t, pre, obs.default_with(p))
        y = s93.medfilt1s(r["out_tot"], fps)
        m = t >= 1.6
        end = y[m][-1]
        mm = t[m] >= t[m][-1] - 5.0
        rf_ = obs.run(tf, Vf, obs.default_with(p))
        yf = s93.medfilt1s(rf_["out_tot"] / 1000.0, 100.0)
        over = 17.0 - yf[tf >= 2.2][-1]
        r3 = obs.run(t3, V3, obs.default_with(p))
        y3 = r3["out_tot"] / 1000.0
        d600 = y3[min(int(np.searchsorted(t3, 600)), len(t3) - 1)] - 16.5
        d1800 = y3[min(int(np.searchsorted(t3, 1800)) - 1, len(t3) - 1)] - 16.5
        print(f"rf={rf:<5g} cap={cap:<6g} | {end-E1:+8.1f}({(end-E1)*0.001:+.3f}N) "
              f"{np.polyfit(t[m][mm], y[m][mm], 1)[0]*0.001:+8.4f} | {over:+8.3f} | "
              f"{d600:+7.3f} {d1800:+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
