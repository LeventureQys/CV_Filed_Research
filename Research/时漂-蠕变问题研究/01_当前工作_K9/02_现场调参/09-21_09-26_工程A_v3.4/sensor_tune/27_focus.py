# -*- coding: utf-8 -*-
"""27 定点收敛：针对 notch 仍超限的会话，只在「预留池速率估计」相关的 4 个旋钮上做小网格。

机理：显示回落速率 ≤ (slope_a − dv/dt) + hold_eps，其中 slope_a 是用 hold_tau_s 低通的输入
导数。减速段低通导数必然高估当前真实爬升率 ⇒ 补偿继续增长 ⇒ 显示被拉回。
故 hold_tau_s 与 hold_eps 是「过减」的直接旋钮；r_fast（比例式 x1）能在不引入回落的
前提下压低整个显示水位；slope_cap_frac 决定慢态追不追得上早期快蠕变。
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score_adc  # noqa: E402

TOL = 120.0
AXES = {
    "hold_tau_s": [0.03, 0.05, 0.08, 0.12, 0.2],
    "hold_eps": [0.0, 0.05, 0.1, 0.3],
    "slope_cap_frac": [3.0, 6.0, 10.0, 20.0],
    "r_fast": [0.3, 0.5, 0.8, 1.2, 2.0],
}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    refine = json.loads((OUT_DIR / "16_refine.json").read_text(encoding="utf-8"))
    old = json.loads((OUT_DIR / "14_search_adc.json").read_text(encoding="utf-8"))
    only = sys.argv[1:] or ["右手掌", "左拇指指腹"]
    out = {}
    for sensor in only:
        base = refine.get(sensor, old[sensor])["params"]
        keys = list(AXES)
        combos = list(itertools.product(*(AXES[k] for k in keys)))
        cands = []
        for c in combos:
            p = dict(base)
            p.update(dict(zip(keys, c)))
            cands.append(p)
        print(f"\n########## {sensor}：{len(cands)} 组合 × "
              f"{len(SENSORS[sensor]['sessions'])} 会话", flush=True)
        sets = [Set(f"f{i:04d}", c) for i, c in enumerate(cands)]
        res = run_sensor(sensor, sets, tag_name=f"focus_{sensor}")
        rows = []
        for i, c in enumerate(cands):
            per = list(res[f"f{i:04d}"].values())
            if not per:
                continue
            rows.append((max(m["notch"] for m in per),
                         sum(score_adc(m, TOL) for m in per) / len(per),
                         sum(m["notch"] for m in per) / len(per),
                         sum(m["peak_exc"] for m in per) / len(per),
                         sum(m["end_exc"] for m in per) / len(per),
                         sum(m["ded_pct"] for m in per) / len(per), c, res[f"f{i:04d}"]))
        rows.sort(key=lambda r: (r[0] > 150, r[1]))
        print(f"  {'过减max':>8s} {'J':>7s} {'过减均':>7s} {'欠减峰':>7s} {'末值':>7s} {'扣除%':>7s} "
              f"│ hold_tau hold_eps cap r_fast")
        for r in rows[:14]:
            print(f"  {r[0]:8.1f} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f} {r[4]:7.1f} {r[5]:7.1f} │ "
                  f"{r[6]['hold_tau_s']:8g} {r[6]['hold_eps']:8g} {r[6]['slope_cap_frac']:4g} "
                  f"{r[6]['r_fast']:6g}")
        best = rows[0]
        print(f"  ★ 最优：过减max={best[0]:.1f} J={best[1]:.1f} 扣除={best[5]:.1f}%")
        for t, m in best[7].items():
            print(f"      {t:<8s} 过减 {m['notch']:7.1f} @{m['notch_t']:6.1f}s  "
                  f"欠减峰 {m['peak_exc']:7.1f}  末值 {m['end_exc']:8.1f}  "
                  f"绝对过减 {m['low_exc']:7.1f}  扣除 {m['ded_pct']:6.1f}%")
        out[sensor] = {"params": best[6], "max_notch": best[0], "J": best[1],
                       "sessions": best[7]}
        (OUT_DIR / "17_focus.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '17_focus.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
