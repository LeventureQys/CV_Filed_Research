# -*- coding: utf-8 -*-
"""评估一个拇指/手掌四类通用预设（取四类新最优的并集折中）。"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retune_over import evaluate  # noqa: E402

COMMON = {"r_fast": 0.05, "tau_c_fast_s": 8.0, "r_slow_max": 30.0,
          "slope_cap_frac": 0.15, "slope_gate_frac": 1.0, "slow_confirm_s": 0.5,
          "soft_unfreeze_s": 0.5, "edge_slope_thres": 30.0, "tau_slope_s": 2.5,
          "tau_r_slow_idle_s": 8.0, "tau_r_fast_s": 1.5, "hold_eps": 2.0,
          "idle_frac": 0.05, "ramp_slope_min": 0.0, "tau_c_fast_boost_s": 1.0}

out = {}
for sensor in ("右拇指指腹", "左拇指指腹", "右手掌", "左手掌"):
    m = evaluate(sensor, [COMMON])[0]
    out[sensor] = m
    print(f"{sensor}: score={m['score']:.1f} 过减={m['over_adc']:.0f} ADC "
          f"扣除={m['ded_pct']:.0f}% 残漂={m['resid_pct']:.0f}%")
(HERE / "common_preset.json").write_text(
    json.dumps({"params": COMMON, "metrics": out}, ensure_ascii=False, indent=2),
    encoding="utf-8")
