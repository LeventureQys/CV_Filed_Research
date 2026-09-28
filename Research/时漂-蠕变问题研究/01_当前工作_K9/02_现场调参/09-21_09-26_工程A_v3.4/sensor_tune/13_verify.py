# -*- coding: utf-8 -*-
"""13 复核：把 04_search*.json 里报出的「最优参数」重新独立复算一遍，逐会话打全指标。
   目的：确认寻优结果不是缓存/口径造成的假象。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import PREP, Set, run_sensor, score  # noqa: E402


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    files = sys.argv[1:] or ["04_search.json", "04_search_ext.json"]
    for fn in files:
        f = OUT_DIR / fn
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        print(f"\n{'#' * 90}\n### {fn}\n")
        for sensor in SENSORS:
            if sensor not in data:
                continue
            best = data[sensor]["runs"][0]["params"]
            res = run_sensor(sensor, [Set("live", {}), Set("best", best)],
                             tag_name=f"ver_{fn[:11]}_{sensor}", time_mode="raw")
            print(f"===== {sensor}  best={json.dumps(best, ensure_ascii=False)}")
            print(f"  {'会话':<12s} {'参数':<6s} {'残漂%':>8s} {'过扣%':>8s} {'欠扣%':>8s} "
                  f"{'末30s%':>8s} {'抖动%':>7s} {'扣除%':>7s} {'score':>7s} {'末残ADC':>9s}")
            sc = {"live": [], "best": []}
            for tag in SENSORS[sensor]["sessions"]:
                p = PREP[f"{sensor}/{tag}"]
                for nm, cn in (("live", "现役"), ("best", "最优")):
                    m = res[nm][tag]
                    sc[nm].append(score(m))
                    print(f"  {tag:<12s} {cn:<6s} {m['resid_pct']:8.1f} {m['over_pct']:8.1f} "
                          f"{m['under_pct']:8.1f} {m['tail_pct']:8.2f} {m['std_pct']:7.2f} "
                          f"{m['ded_pct']:7.1f} {score(m):7.2f} {m['err_end_adc']:9.1f}")
            print(f"  → 平均 score：现役 {sum(sc['live']) / len(sc['live']):.2f}  "
                  f"最优 {sum(sc['best']) / len(sc['best']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
