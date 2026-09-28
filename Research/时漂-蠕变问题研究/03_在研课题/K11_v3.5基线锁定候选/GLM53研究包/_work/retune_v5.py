# -*- coding: utf-8 -*-
"""v5 定稿：K11 纯锁定小网格（拇指/手掌四类），过减口径=持续型（1s 均值 < −150）。"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
import sweep_lib  # noqa: E402
sweep_lib.EXE = ST / "build" / "v34_sweep_k11.exe"
sys.modules["sweep_lib"].EXE = sweep_lib.EXE
import retune_v4 as R  # noqa: E402
R.EXE = sweep_lib.EXE

from sensor_common import SENSORS  # noqa: E402

GRID = {"hold_lock_tau_s": [0.15, 0.2, 0.3, 0.5, 0.8],
        "hold_lock_freeze_s": [1.0, 2.0, 4.0],
        "edge_slope_thres": [10.0, 60.0, 150.0]}
BASE = {"r_fast": 0.0, "slope_gate_frac": 1e-6, "r_slow_max": 0.0, "hold_eps": 0.0,
        "hold_lock_tau_s": 0.3, "hold_lock_freeze_s": 2.0, "edge_slope_thres": 60.0}

out = {}
for sensor in ("右拇指指腹", "左拇指指腹", "右手掌", "左手掌"):
    print(f"\n########## {sensor}", flush=True)
    cands = []
    for tau in GRID["hold_lock_tau_s"]:
        for fz in GRID["hold_lock_freeze_s"]:
            for ed in GRID["edge_slope_thres"]:
                cands.append({**BASE, "hold_lock_tau_s": tau, "hold_lock_freeze_s": fz,
                              "edge_slope_thres": ed})
    per = R.evaluate(sensor, cands)
    order = np.argsort([m["score"] for m in per])
    b = per[order[0]]
    bp = cands[order[0]]
    m = bp
    print(f"  ★ tau={bp['hold_lock_tau_s']} freeze={bp['hold_lock_freeze_s']} "
          f"edge={bp['edge_slope_thres']}: score={b['score']:.1f} 持续过减={b['over_adc']:.0f} "
          f"带宽={b['band_adc']:.0f} 入带={b['settle_s']:.1f}s 末漂={b['up_adc']:.0f} "
          f"扣除={b['ded_pct']:.0f}%", flush=True)
    for i in order[1:4]:
        mm = per[i]
        print(f"    (次优 tau={cands[i]['hold_lock_tau_s']}/{cands[i]['hold_lock_freeze_s']}/"
              f"{cands[i]['edge_slope_thres']}: {mm['score']:.1f}, 带宽={mm['band_adc']:.0f}, "
              f"入带={mm['settle_s']:.1f}s)", flush=True)
    out[sensor] = {"best": {**b, "params": bp}}
(HERE / "retune5_all.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
print("\n写出 retune5_all.json")
