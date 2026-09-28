# -*- coding: utf-8 -*-
"""137_palm9_sweep：cap×conf×soft 联合扫（右手掌 164332），回代 161747 查下坠，查卸载负冲。"""
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
IDLE9 = (78.5, 84.0)  # 全卸载窗口

BASE = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
        "slope_cap_frac": 0.005, "r_slow_max": 0.15, "tau_r_fast_s": 6.0,
        "tau_r_slow_idle_s": 0.5}


def mets(t, y, fps, holds):
    out = []
    for t0, t1, E in holds:
        i0 = int(np.searchsorted(t, t0 + 0.6))
        i1 = int(np.searchsorted(t, t1 - 0.6))
        yy, tt = y[i0:i1 + 1], t[i0:i1 + 1]
        E_n = E / 1000.0
        end = float(yy[-1])
        i10 = min(int(10 * fps), len(yy) - 1)
        i50 = min(int(50 * fps), len(yy) - 1)
        mm = tt >= tt[-1] - 5.0
        out.append({"dev": end - E_n, "climb": end - yy[i10],
                    "span": float(yy[i50:].max() - yy[i50:].min()),
                    "drop": float(yy.max() - end), "over": float(max(0.0, E_n - yy.min()))})
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

    print(f"{'参数(cap,conf,soft,rsm)':<30s}{'段1爬升':>8s}{'段2爬升':>8s}{'段1落点':>8s}{'段2落点':>8s}"
          f"{'下坠max':>8s}{'过扣max':>8s}{'卸载负冲':>8s}{'161747爬升':>10s}{'161747下坠':>10s}")
    rows = []
    for cap, conf, soft in itertools.product([0.005, 0.01, 0.015, 0.025], [0.5, 1.5], [0.5, 1.5]):
        p = {**BASE, "slope_cap_frac": cap, "slow_confirm_s": conf, "soft_unfreeze_s": soft}
        r9 = obs.run(t9, pre9, obs.default_with(p))
        y9 = s93.medfilt1s(r9["out_tot"], fps9) / 1000.0
        m9 = mets(t9, y9, fps9, HOLDS9)
        ia, ib = int(np.searchsorted(t9, IDLE9[0])), int(np.searchsorted(t9, IDLE9[1]))
        dip = float(y9[ia:ib + 1].min())
        r8 = obs.run(t8, V8, obs.default_with(p))
        y8 = s93.medfilt1s(r8["out_tot"], fps8) / 1000.0
        m8 = mets(t8, y8, fps8, [(T08, 419.0, E8)])[0]
        tag = f"cap{cap:g} conf{conf:g} soft{soft:g}"
        print(f"{tag:<30s}{m9[0]['climb']:+8.2f}{m9[1]['climb']:+8.2f}{m9[0]['dev']:+8.2f}"
              f"{m9[1]['dev']:+8.2f}{max(m['drop'] for m in m9):8.2f}"
              f"{max(m['over'] for m in m9):8.2f}{dip:+8.2f}{m8['climb']:+10.2f}{m8['drop']:10.2f}")
        rows.append({"tag": tag, "params": p, "m9": m9, "dip": dip, "m8": m8})
    (OUT9 / "137_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
