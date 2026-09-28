# -*- coding: utf-8 -*-
"""22 过减/漂移诊断与收敛：把「过减深度 / 欠减高度 / 基线漂移」都换算成绝对 ADC，
   先量出现役默认与推荐预设的实际水平，再在「过减深度 ≤ 上限」的硬约束下重新寻优。

用户验收口径（2026-09-26 追加）
-----------------------------
* 拇指指腹：x1 阶段有明显**过减**；可接受的过减量级是 **50~150 ADC（绝对值）**，
  不能接受现在这种程度。
* 右手掌：同样的问题 + 基线漂移太大。
定义（显示值口径，已调零、首帧=0）：
    过减深度 dip   = ref − min(显示)          （正 = 显示被压到参考电平下方多少 ADC）
    欠减高度 rise  = max(显示) − ref          （正 = 显示还浮在参考线上方多少 ADC）
    基线漂移 wander = max(显示) − min(显示)
    ref：有加载台阶的会话取 ΣE；无台阶的会话取 0（调零后的起始基线）
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import LIVE, OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import PREP, Set, run_sensor, score  # noqa: E402

PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))


def show(sensor: str, cases: list[tuple[str, dict]], tag: str) -> dict:
    sets = [Set(f"s{i}", c) for i, (_, c) in enumerate(cases)]
    res = run_sensor(sensor, [s for s in sets], tag_name=tag)
    print(f"\n### {sensor}")
    print(f"  {'方案':<26s} {'会话':<8s} {'过减ADC':>8s} {'@t':>7s} {'欠减ADC':>8s} "
          f"{'基线漂移':>9s} {'末值ADC':>8s} {'扣除%':>7s}")
    out = {}
    for i, (nm, c) in enumerate(cases):
        per = res[f"s{i}"]
        for tag_ in per:
            m = per[tag_]
            print(f"  {nm:<26s} {tag_:<8s} {m['dip_adc']:8.1f} {m['dip_t']:7.1f} "
                  f"{m['rise_adc']:8.1f} {m['wander_adc']:9.1f} {m['err_end_adc']:8.1f} "
                  f"{m['ded_pct']:7.1f}")
        vals = list(per.values())
        out[nm] = {k: sum(m[k] for m in vals) / len(vals)
                   for k in ("dip_adc", "rise_adc", "wander_adc", "ded_pct", "err_end_adc")}
    print(f"  ---- 平均 ----")
    for nm, v in out.items():
        print(f"  {nm:<26s} {'':<8s} {v['dip_adc']:8.1f} {'':>7s} {v['rise_adc']:8.1f} "
              f"{v['wander_adc']:9.1f} {v['err_end_adc']:8.1f} {v['ded_pct']:7.1f}")
    return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    only = [a for a in sys.argv[1:] if not a.startswith("--")] or \
           ["四指指腹", "右拇指指腹", "左拇指指腹", "右手掌", "左手掌"]
    allout = {}
    for sensor in only:
        cases = [("现役默认", {}), ("当前推荐预设", PRESETS[sensor]["params"])]
        allout[sensor] = show(sensor, cases, f"trim0_{sensor}")
    (OUT_DIR / "13_trim_before.json").write_text(json.dumps(allout, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '13_trim_before.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
