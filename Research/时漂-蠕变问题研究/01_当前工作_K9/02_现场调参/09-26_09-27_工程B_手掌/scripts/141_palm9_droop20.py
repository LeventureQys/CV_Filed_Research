# -*- coding: utf-8 -*-
"""141_palm9_droop20：约束=回调≤0.2N（两份数据取 max），慢漂须收敛（末段斜率小）。
扫 rf×cap×τc1，找满足约束、爬升最小且收敛的组合。"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT9 = TEMP / "palm9" / "out"
OUT8 = TEMP / "palm8" / "out"
HOLDS9 = [(0.9, 77.9, 13222.0), (84.5, 149.6, 16752.0)]
E8, T08 = 16502.0, 1.2


def mets(t, y, tt_all, fps, holds):
    out = []
    for t0, t1, E in holds:
        i0 = int(np.searchsorted(t, t0 + 0.6))
        i1 = int(np.searchsorted(t, t1 - 0.6))
        yy, tt = y[i0:i1 + 1], t[i0:i1 + 1]
        end = float(yy[-1])
        i10 = min(int(10 * fps), len(yy) - 1)
        mm = tt >= tt[-1] - max(10.0, 0.2 * (tt[-1] - tt[0]))
        out.append({"climb": end - yy[i10], "drop": float(yy.max() - end),
                    "slope_end": float(np.polyfit(tt[mm], yy[mm], 1)[0]) * 1000.0})
    return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z9 = np.load(OUT9 / "streams.npz")
    t9, pre9 = z9["t"], z9["pre"]
    fps9 = (len(t9) - 1) / (t9[-1] - t9[0])
    z8 = np.load(OUT8 / "streams.npz")
    t8, V8 = z8["t"], z8["V"]
    fps8 = (len(t8) - 1) / (t8[-1] - t8[0])
    print(f"{'rf':>5s}{'τc1':>5s}{'cap':>7s} | {'右1爬升':>8s}{'右2爬升':>8s}{'右末斜率':>9s}"
          f"{'右下坠':>7s} | {'16爬升':>7s}{'16末斜率':>9s}{'16下坠':>7s} | 达标")
    rows = []
    for rf, tc1, cap in itertools.product([0.02, 0.03, 0.04], [1.0, 2.0],
                                          [0.007, 0.009, 0.011]):
        p = {"r_fast": rf, "tau_c_fast_s": tc1, "slow_confirm_s": 1.5,
             "soft_unfreeze_s": 1.5, "slope_cap_frac": cap, "r_slow_max": 0.20,
             "tau_r_fast_s": 2.0, "tau_r_slow_idle_s": 0.5}
        m9 = mets(t9, s93.medfilt1s(obs.run(t9, pre9, obs.default_with(p))["out_tot"], fps9) / 1000.0,
                  t9, fps9, HOLDS9)
        m8 = mets(t8, s93.medfilt1s(obs.run(t8, V8, obs.default_with(p))["out_tot"], fps8) / 1000.0,
                  t8, fps8, [(T08, 419.0, E8)])[0]
        drop_max = max(m["drop"] for m in m9 + [m8])
        conv = max(abs(m["slope_end"]) for m in m9 + [m8])
        ok = drop_max <= 0.2 and conv <= 2.0
        print(f"{rf:5.2f}{tc1:5.1f}{cap:7.3f} | {m9[0]['climb']:+8.2f}{m9[1]['climb']:+8.2f}"
              f"{max(abs(m['slope_end']) for m in m9):9.2f}{max(m['drop'] for m in m9):7.2f} | "
              f"{m8['climb']:+7.2f}{m8['slope_end']:9.2f}{m8['drop']:7.2f} | "
              f"{'✓' if ok else '（' + ('下坠' if drop_max > 0.2 else '斜率') + '超）'}")
        rows.append({"rf": rf, "tc1": tc1, "cap": cap, "m9": m9, "m8": m8, "ok": ok})
    (OUT9 / "141_droop20.json").write_text(json.dumps(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
