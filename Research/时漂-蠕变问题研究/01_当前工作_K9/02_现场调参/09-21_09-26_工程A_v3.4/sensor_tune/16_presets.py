# -*- coding: utf-8 -*-
"""16 定稿预设：把每类传感器的最终推荐参数固化成 out/09_presets.json，并再复算一遍确认。

选取依据（见 out/08_final.txt）
------------------------------
* 极值最优（放宽网格搜出来的 r_slow_max=30 等）分数最低，但 r_slow_max 等价于
  「允许把观测到的上升当成蠕变扣掉的比例上限 = r/(1+r)」，取 30 ⇒ 96.8%，现场太激进。
* 故最终预设取「实用档」：r_slow_max ≤ 5（拇指/手掌）/ 0.6（四指指腹），
  其余取搜索最优；实测仍能把扣除率从 11~19% 提到 60~89%，且过扣 ≤ 10% 蠕变。

产物：out/09_presets.json + out/09_presets.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score  # noqa: E402

PRESETS: dict[str, dict] = {
    "四指指腹": {
        "note": "有真实加载台阶；蠕变只占载荷 3~8%。现役默认严重过扣（扣掉 168~288% 蠕变）。",
        "params": {
            "r_fast": 0.01, "tau_c_fast_s": 20.0, "r_slow_max": 0.6, "slope_cap_frac": 0.08,
            "slope_gate_frac": 1.0, "slow_confirm_s": 5.0, "soft_unfreeze_s": 2.0,
            "edge_slope_thres": 60.0, "tau_slope_s": 0.5, "tau_r_slow_idle_s": 0.5,
            "tau_r_fast_s": 2.0, "hold_eps": 2.0, "idle_frac": 0.02, "ramp_slope_min": 2.0,
        },
    },
    "右拇指指腹": {
        "note": "无弹性台阶（录制开始即受载），读数整体上升都是蠕变；现役默认只扣 6~16%。",
        "params": {
            "r_fast": 0.10, "tau_c_fast_s": 4.0, "r_slow_max": 5.0, "slope_cap_frac": 0.08,
            "slope_gate_frac": 1.0, "slow_confirm_s": 0.0, "soft_unfreeze_s": 4.0,
            "edge_slope_thres": 150.0, "tau_slope_s": 3.0, "tau_r_slow_idle_s": 30.0,
            "tau_r_fast_s": 2.0, "hold_eps": 8.0, "idle_frac": 0.10, "ramp_slope_min": 2.0,
        },
    },
    "左拇指指腹": {
        "note": "同右拇指：无弹性台阶。现役默认只扣 18~19%。",
        "params": {
            "r_fast": 0.10, "tau_c_fast_s": 2.0, "r_slow_max": 5.0, "slope_cap_frac": 0.08,
            "slope_gate_frac": 2.0, "slow_confirm_s": 0.0, "soft_unfreeze_s": 2.0,
            "edge_slope_thres": 60.0, "tau_slope_s": 0.5, "tau_r_slow_idle_s": 30.0,
            "tau_r_fast_s": 2.0, "hold_eps": 0.0, "idle_frac": 0.10, "ramp_slope_min": 2.0,
        },
    },
    "右手掌": {
        "note": "同拇指（无台阶）。现役默认只扣 12~14%。",
        "params": {
            "r_fast": 0.10, "tau_c_fast_s": 2.0, "r_slow_max": 5.0, "slope_cap_frac": 0.08,
            "slope_gate_frac": 2.0, "slow_confirm_s": 0.0, "soft_unfreeze_s": 2.0,
            "edge_slope_thres": 20.0, "tau_slope_s": 1.0, "tau_r_slow_idle_s": 30.0,
            "tau_r_fast_s": 2.0, "hold_eps": 0.0, "idle_frac": 0.10, "ramp_slope_min": 0.0,
        },
    },
    "左手掌": {
        "note": "同拇指（无台阶）。现役默认只扣 9~12%。",
        "params": {
            "r_fast": 0.10, "tau_c_fast_s": 2.0, "r_slow_max": 5.0, "slope_cap_frac": 0.08,
            "slope_gate_frac": 2.0, "slow_confirm_s": 0.0, "soft_unfreeze_s": 2.0,
            "edge_slope_thres": 60.0, "tau_slope_s": 2.0, "tau_r_slow_idle_s": 30.0,
            "tau_r_fast_s": 2.0, "hold_eps": 0.5, "idle_frac": 0.10, "ramp_slope_min": 2.0,
        },
    },
}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    lines = ["# v3.4 观测器 · 五类传感器推荐预设", ""]
    out = {}
    for sensor, spec in PRESETS.items():
        best = spec["params"]
        res_u = run_sensor(sensor, [Set("live", {}), Set("preset", best)],
                           tag_name=f"pre_u_{sensor}")
        res_r = run_sensor(sensor, [Set("live", {}), Set("preset", best)], time_mode="raw",
                           tag_name=f"pre_r_{sensor}")
        rows = []
        for tag in SENSORS[sensor]["sessions"]:
            rows.append({"tag": tag, "live": res_u["live"][tag], "preset": res_u["preset"][tag],
                         "preset_raw": res_r["preset"][tag]})
        sl = sum(score(r["live"]) for r in rows) / len(rows)
        sp = sum(score(r["preset"]) for r in rows) / len(rows)
        sr = sum(score(r["preset_raw"]) for r in rows) / len(rows)
        ded_l = sum(r["live"]["ded_pct"] for r in rows) / len(rows)
        ded_p = sum(r["preset"]["ded_pct"] for r in rows) / len(rows)
        over_l = sum(r["live"]["over_pct"] for r in rows) / len(rows)
        over_p = sum(r["preset"]["over_pct"] for r in rows) / len(rows)
        lines.append(f"\n## {sensor}")
        lines.append(f"   {spec['note']}")
        lines.append(f"   参数：{json.dumps(best, ensure_ascii=False)}")
        lines.append(f"   现役默认 score={sl:.1f} 扣除率={ded_l:.1f}% 过扣={over_l:.1f}%")
        lines.append(f"   推荐预设 score={sp:.1f} 扣除率={ded_p:.1f}% 过扣={over_p:.1f}%"
                     f"  （raw 时间轴复算 score={sr:.1f}）")
        out[sensor] = {"note": spec["note"], "params": best,
                       "live_score": sl, "preset_score": sp, "preset_score_raw": sr,
                       "live_ded_pct": ded_l, "preset_ded_pct": ded_p,
                       "live_over_pct": over_l, "preset_over_pct": over_p,
                       "sessions": rows}
        print("\n".join(lines[-5:]))
    (OUT_DIR / "09_presets.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    (OUT_DIR / "09_presets.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '09_presets.json'} / 09_presets.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
