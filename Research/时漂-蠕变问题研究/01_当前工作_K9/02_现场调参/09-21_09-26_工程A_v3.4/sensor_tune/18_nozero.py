# -*- coding: utf-8 -*-
"""18 现场口径检查：「不调零直接开算法」时会发生什么（全局总值旁路把带载读数当成空载基线）。

对照组
------
 (a) 调零口径（本次所有结论的口径）：输入 = 原值 − 首帧值 → 算法 zero=0、旁路基线=0
 (b) 原值口径（现场「装好负载但不调零就开算法」）：输入 = 录制原值
两组的参数都取该传感器的推荐预设。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_common import OUT_DIR, SENSORS, load  # noqa: E402
from sweep_lib import PREP, Set, run_sensor  # noqa: E402

PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"{'传感器':<12s} {'会话':<8s} {'首帧原值':>10s} {'不调零·扣除ADC':>14s} "
          f"{'不调零·扣除%':>12s} {'调零·扣除ADC':>13s} {'调零·扣除%':>11s}")
    out = {}
    for sensor in SENSORS:
        p = PRESETS[sensor]["params"]
        a = run_sensor(sensor, [Set("preset", p)], tag_name=f"nz_{sensor}")          # 调零口径
        b = run_sensor(sensor, [Set("preset", p)], zero=False, tag_name=f"raw_{sensor}")  # 原值口径
        for tag in SENSORS[sensor]["sessions"]:
            key = f"{sensor}/{tag}"
            info = PREP[key]
            creep = max(info["creep_total"], 1.0)
            da = a["preset"][tag]["ded_end_adc"]
            db = b["preset"][tag]["ded_end_adc"]
            print(f"{sensor:<12s} {tag:<8s} {info['v0_total_raw']:10.0f} {db:14.1f} "
                  f"{db / creep * 100:12.1f} {da:13.1f} {da / creep * 100:11.1f}")
            out[key] = {"v0_raw": info["v0_total_raw"], "ded_raw_adc": db,
                        "ded_raw_pct": db / creep * 100, "ded_zero_adc": da,
                        "ded_zero_pct": da / creep * 100}
        print()
    (OUT_DIR / "10_nozero.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
