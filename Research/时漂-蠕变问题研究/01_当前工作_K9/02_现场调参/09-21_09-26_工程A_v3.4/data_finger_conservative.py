# -*- coding: utf-8 -*-
"""在四份四指指腹会话上比较「保守档」（保证任何一份都不过扣）与推荐档。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from other_recorder_tune import LIVE  # noqa: E402
from data_finger_params import SESSIONS  # noqa: E402
from data_finger_pick import evaluate, prep  # noqa: E402

CANDS = [
    ("现役默认", {}),
    ("A  rf.020 rsm.030", dict(r_fast=0.02, r_slow_max=0.03, tau_c_fast_s=2.0)),
    ("G  rf.015 rsm.020", dict(r_fast=0.015, r_slow_max=0.02, tau_c_fast_s=2.0)),
    ("H  rf.010 rsm.020", dict(r_fast=0.01, r_slow_max=0.02, tau_c_fast_s=2.0)),
    ("I  rf.015 rsm.025", dict(r_fast=0.015, r_slow_max=0.025, tau_c_fast_s=2.0)),
    ("J  rf.020 rsm.020", dict(r_fast=0.02, r_slow_max=0.02, tau_c_fast_s=2.0)),
    ("K  rf.025 rsm.030", dict(r_fast=0.025, r_slow_max=0.03, tau_c_fast_s=2.0)),
]


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    keys = list(SESSIONS)
    for k in keys:
        prep(k)
    print(f"{'候选':20s}" + "".join(f"| {k[:16]:>17s} " for k in keys))
    print(f"{'':20s}" + "".join(f"| {'min%  end%  ded%':>17s} " for k in keys))
    for tag, kw in CANDS:
        p = replace(LIVE, **kw)
        line = f"{tag:20s}"
        for k in keys:
            r = evaluate(p, k)
            line += f"| {r['errmin']:+6.2f} {r['errend']:+5.1f} {r['ded']:5.0f} "
        print(line, flush=True)
    print("\n各候选的四份合计：最差过扣深度（越接近 0 越好）、平均蠕变扣除率")
    for tag, kw in CANDS:
        p = replace(LIVE, **kw)
        mins, deds = [], []
        for k in keys:
            r = evaluate(p, k)
            mins.append(r["errmin"])
            deds.append(r["ded"])
        print(f"  {tag:20s} 最差 errmin {min(mins):+6.2f}%   平均 ded {sum(deds) / len(deds):5.0f}%")


if __name__ == "__main__":
    main()
