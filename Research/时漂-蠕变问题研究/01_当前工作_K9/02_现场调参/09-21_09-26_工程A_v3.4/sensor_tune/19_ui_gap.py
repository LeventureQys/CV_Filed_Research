# -*- coding: utf-8 -*-
"""19 可调面缺口量化：把推荐预设里"当前 UI 调不到"的项逐个退回现役值，看分数损失多少。
   用来回答「到底需不需要新增/放宽参数」——只看可调面缺口值多少分。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score  # noqa: E402

PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))

# 现状（creep_observer.h 默认 + UI kSpecs 范围）下"调不到"的项 → 退回值
UI_MAX = {"r_slow_max": 0.600, "slope_cap_frac": 0.050}   # kSpecs 上限
NOT_EXPOSED = {"slope_gate_frac": 0.05, "tau_slope_s": 1.0, "edge_slope_thres": 60.0,
               "idle_frac": 0.05, "ramp_slope_min": 0.5, "slow_confirm_s": None,
               "tau_r_slow_idle_s": None, "tau_r_fast_s": None, "tau_c_fast_s": None,
               "soft_unfreeze_s": None, "hold_eps": None, "r_fast": None}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"{'传感器':<12s} {'情形':<44s} {'score':>8s} {'扣除%':>7s} {'过扣%':>7s}")
    out = {}
    for sensor in SENSORS:
        p = PRESETS[sensor]["params"]
        cases = [("① 推荐预设（全量 14 项）", p),
                 ("② 把 UI 上限外的项夹回上限", {**p,
                    **{k: min(p[k], v) for k, v in UI_MAX.items() if k in p}}),
                 ("③ 再把「UI 未暴露」的项退回现役", {**p,
                    **{k: min(p[k], v) for k, v in UI_MAX.items() if k in p},
                    **{k: v for k, v in NOT_EXPOSED.items() if v is not None and k in p}}),
                 ("④ 退回现役默认", {})]
        sets = [Set(f"s{i}", c) for i, (_, c) in enumerate(cases)]
        res = run_sensor(sensor, sets, tag_name=f"gap_{sensor}")
        rows = []
        for i, (nm, c) in enumerate(cases):
            per = list(res[f"s{i}"].values())
            sc = sum(score(m) for m in per) / len(per)
            ded = sum(m["ded_pct"] for m in per) / len(per)
            over = sum(m["over_pct"] for m in per) / len(per)
            rows.append({"case": nm, "score": sc, "ded_pct": ded, "over_pct": over})
            print(f"{sensor:<12s} {nm:<44s} {sc:8.2f} {ded:7.1f} {over:7.1f}")
        out[sensor] = rows
        print()
    (OUT_DIR / "11_ui_gap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"写出 {OUT_DIR / '11_ui_gap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
