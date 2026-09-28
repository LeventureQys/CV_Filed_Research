# -*- coding: utf-8 -*-
"""139_palm9_2d：cap×rsm 二维（右手掌冻结 vs 161747 下坠），τr1=2 固定。"""
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


def mets(t, y, fps, holds):
    out = []
    for t0, t1, E in holds:
        i0 = int(np.searchsorted(t, t0 + 0.6))
        i1 = int(np.searchsorted(t, t1 - 0.6))
        yy = y[i0:i1 + 1]
        E_n = E / 1000.0
        end = float(yy[-1])
        i10 = min(int(10 * fps), len(yy) - 1)
        out.append({"climb": end - yy[i10], "dev": end - E_n,
                    "drop": float(yy.max() - end)})
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
    print(f"{'cap/rsm':<18s}{'段1爬升':>8s}{'段2爬升':>8s}{'下坠max':>8s}{'落点1/2':>14s}"
          f"{'161747爬升':>10s}{'161747下坠':>10s}")
    rows = []
    for cap, rsm in itertools.product([0.008, 0.01, 0.012, 0.015], [0.15, 0.25, 0.35]):
        p = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5,
             "soft_unfreeze_s": 1.5, "slope_cap_frac": cap, "r_slow_max": rsm,
             "tau_r_fast_s": 2.0, "tau_r_slow_idle_s": 0.5}
        m9 = mets(t9, s93.medfilt1s(obs.run(t9, pre9, obs.default_with(p))["out_tot"], fps9) / 1000.0,
                  fps9, HOLDS9)
        m8 = mets(t8, s93.medfilt1s(obs.run(t8, V8, obs.default_with(p))["out_tot"], fps8) / 1000.0,
                  fps8, [(T08, 419.0, E8)])[0]
        print(f"cap{cap:g}/rsm{rsm:g}{'':<6}{m9[0]['climb']:+8.2f}{m9[1]['climb']:+8.2f}"
              f"{max(m['drop'] for m in m9):8.2f}{m9[0]['dev']:+7.2f}/{m9[1]['dev']:+5.2f}"
              f"{m8['climb']:+10.2f}{m8['drop']:10.2f}")
        rows.append({"cap": cap, "rsm": rsm, "m9": m9, "m8": m8})
    (OUT9 / "139_2d.json").write_text(json.dumps(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
