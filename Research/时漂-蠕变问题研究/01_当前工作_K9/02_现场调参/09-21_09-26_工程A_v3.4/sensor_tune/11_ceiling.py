# -*- coding: utf-8 -*-
"""11 结构性上限验证：x2 的钳位 x2 ≤ r_slow_max·max(e,1) 且 e = y − x1 − x2
⇒ 当「弹性电平≈0、整个上升都是蠕变」时，可扣除比例的理论上限 = r/(1+r)。
本脚本用实机数据把这个上限打出来，作为「是否需要放宽/新增参数」的证据。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor  # noqa: E402

RS = [0.1, 0.2, 0.35, 0.6, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
EXTRA = {"slope_gate_frac": 5.0, "slope_cap_frac": 0.3, "slow_confirm_s": 0.0,
         "soft_unfreeze_s": 0.5, "r_fast": 0.0, "edge_slope_thres": 400.0,
         "tau_r_slow_idle_s": 30.0, "hold_eps": 0.0, "tau_slope_s": 1.0}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sets = [Set("live", {})]
    for r in RS:
        sets.append(Set(f"r_slow_max={r:g}(宽松其余)", {**EXTRA, "r_slow_max": r}))
    out = {}
    print(f"{'传感器':<12s} {'参数集':<26s} {'扣除%':>7s} {'残漂%':>7s} {'过扣%':>7s} "
          f"{'理论上限 r/(1+r)':>16s}")
    for sensor in SENSORS:
        res = run_sensor(sensor, sets, tag_name=f"ceil_{sensor}")
        for nm in [s.name for s in sets]:
            per = res[nm]
            ded = sum(m["ded_pct"] for m in per.values()) / len(per)
            resid = sum(m["resid_pct"] for m in per.values()) / len(per)
            over = sum(m["over_pct"] for m in per.values()) / len(per)
            r = None
            if nm.startswith("r_slow_max="):
                r = float(nm.split("=")[1].split("(")[0])
            theo = f"{r / (1 + r) * 100:15.1f}%" if r else " " * 16
            print(f"{sensor:<12s} {nm:<26s} {ded:7.1f} {resid:7.1f} {over:7.1f} {theo}")
            out.setdefault(sensor, {})[nm] = {"ded_pct": ded, "resid_pct": resid, "over_pct": over}
        print()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    (OUT_DIR / "06_ceiling.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
