# -*- coding: utf-8 -*-
"""25 定稿对比：现役默认 / 旧推荐预设（比例制寻优）/ 新寻优（绝对 ADC 口径 + 过减约束）
   三者在「过减 notch（ADC）/ 欠减峰值 / 末值偏差 / 绝对过减」上的逐会话对比，并试几组微调。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score_adc  # noqa: E402

OLD = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
NEW = json.loads((OUT_DIR / "14_search_adc.json").read_text(encoding="utf-8"))

# 手工微调：以新寻优为骨架，把「显示能压多低」的 slope_cap_frac 再加一档、
# 把预留池漏率 hold_eps / 速率估计滞后 hold_tau_s 收紧，用来削掉剩余的 notch 超限
TRIMMED: dict[str, dict] = {
    "四指指腹": {"edge_boost_s": 0.0, "edge_slope_thres": 60.0, "hold_eps": 0.0,
                 "idle_frac": 0.05, "r_fast": 0.02, "r_slow_max": 0.6, "ramp_slope_min": 0.0,
                 "slope_cap_frac": 6.0, "slope_gate_frac": 20.0, "slow_confirm_s": 0.0,
                 "soft_unfreeze_s": 4.0, "tau_c_fast_s": 1.0, "tau_r_fast_s": 0.2,
                 "tau_r_slow_idle_s": 2.0, "tau_slope_s": 0.2, "hold_tau_s": 0.1},
    "右拇指指腹": {"edge_boost_s": 0.0, "edge_slope_thres": 3000.0, "hold_eps": 0.0,
                   "hold_tau_s": 0.1, "idle_frac": 0.2, "r_fast": 0.30, "r_slow_max": 30.0,
                   "ramp_slope_min": 0.0, "slope_cap_frac": 3.0, "slope_gate_frac": 20.0,
                   "slow_confirm_s": 0.0, "soft_unfreeze_s": 0.1, "tau_c_fast_s": 2.0,
                   "tau_r_fast_s": 2.0, "tau_r_slow_idle_s": 0.0, "tau_slope_s": 0.3},
    "左拇指指腹": {"edge_boost_s": 0.0, "edge_slope_thres": 3000.0, "hold_eps": 0.0,
                   "hold_tau_s": 0.1, "idle_frac": 0.2, "r_fast": 0.30, "r_slow_max": 30.0,
                   "ramp_slope_min": 0.0, "slope_cap_frac": 3.0, "slope_gate_frac": 20.0,
                   "slow_confirm_s": 0.0, "soft_unfreeze_s": 0.1, "tau_c_fast_s": 2.0,
                   "tau_r_fast_s": 0.2, "tau_r_slow_idle_s": 0.5, "tau_slope_s": 0.3},
    "右手掌": {"edge_boost_s": 0.0, "edge_slope_thres": 60.0, "hold_eps": 0.0,
               "hold_tau_s": 0.1, "idle_frac": 0.2, "r_fast": 0.16, "r_slow_max": 30.0,
               "ramp_slope_min": 2.0, "slope_cap_frac": 3.0, "slope_gate_frac": 20.0,
               "slow_confirm_s": 0.0, "soft_unfreeze_s": 1.0, "tau_c_fast_s": 4.0,
               "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 2.0, "tau_slope_s": 0.3},
    "左手掌": {"edge_boost_s": 2.0, "edge_slope_thres": 3000.0, "hold_eps": 2.0,
               "hold_tau_s": 0.2, "idle_frac": 0.1, "r_fast": 0.30, "r_slow_max": 30.0,
               "ramp_slope_min": 2.0, "slope_cap_frac": 3.0, "slope_gate_frac": 5.0,
               "slow_confirm_s": 0.0, "soft_unfreeze_s": 0.1, "tau_c_fast_s": 2.0,
               "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 2.0, "tau_slope_s": 0.5},
}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    only = sys.argv[1:] or list(SENSORS)
    out = {}
    for sensor in only:
        cands = [("现役默认", {}), ("旧预设(比例制)", OLD[sensor]["params"]),
                 ("新寻优(绝对口径)", NEW[sensor]["params"]),
                 ("新+微调", TRIMMED[sensor])]
        sets = [Set(f"t{i}", c) for i, (_, c) in enumerate(cands)]
        res = run_sensor(sensor, sets, tag_name=f"pick_{sensor}")
        print(f"\n### {sensor}")
        print(f"  {'方案':<18s} {'会话':<8s} {'过减ADC':>8s} {'@t':>7s} {'欠减峰':>8s} "
              f"{'末值偏差':>9s} {'绝对过减':>9s} {'扣除%':>7s} {'J':>7s}")
        for i, (nm, c) in enumerate(cands):
            per = res[f"t{i}"]
            for tag, m in per.items():
                print(f"  {nm:<18s} {tag:<8s} {m['notch']:8.1f} {m['notch_t']:7.1f} "
                      f"{m['peak_exc']:8.1f} {m['end_exc']:9.1f} {m['low_exc']:9.1f} "
                      f"{m['ded_pct']:7.1f} {score_adc(m):7.1f}")
            v = list(per.values())
            print(f"  {'':<18s} {'平均':<8s} "
                  f"{sum(m['notch'] for m in v) / len(v):8.1f} {'':>7s} "
                  f"{sum(m['peak_exc'] for m in v) / len(v):8.1f} "
                  f"{sum(m['end_exc'] for m in v) / len(v):9.1f} "
                  f"{sum(m['low_exc'] for m in v) / len(v):9.1f} "
                  f"{sum(m['ded_pct'] for m in v) / len(v):7.1f} "
                  f"{sum(score_adc(m) for m in v) / len(v):7.1f}")
            out.setdefault(sensor, {})[nm] = {"params": c, "sessions": per}
        print()
    (OUT_DIR / "15_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"写出 {OUT_DIR / '15_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
