# -*- coding: utf-8 -*-
"""152_b2d896_sweep：b2d896「清尾漂」的三条扫描。

A. r_slow_max 单旋钮（cap 固定 0.005，其余出厂默认）→ 找「尾漂消失」的最小幅度上限
B. slope_cap_frac 单旋钮（rsm 固定 0.15）→ 验证「尾端不需要大速率参数」
C. r_fast 单旋钮（cap.005/rsm.15）→ 用「电平扣除」换落点，量过扣何时出现
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm10" / "out"
E, T0, WIN = 16502.0, 1.20, 1.30


def evaluate(out_tot, t, fps):
    y = s93.medfilt1s(out_tot, fps)
    m = t >= WIN
    yy, tt = y[m], t[m]
    end = float(yy[-1])
    tail = tt >= tt[-1] - 60.0
    i10 = int(np.searchsorted(tt, 10.0))
    bad = np.abs(yy - end) > 100.0
    return {"dev": end - E, "over": max(0.0, E - float(yy.min())),
            "drop": float(yy.max()) - end,
            "slope_tail": float(np.polyfit(tt[tail], yy[tail], 1)[0]),
            "d10": float(yy[-1] - yy[i10]) if i10 < yy.size else 0.0,
            "freeze": float(tt[np.flatnonzero(bad)[-1]]) if bad.any() else 0.0}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "b2d896.npz")
    t, V = z["t"], z["V"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    hdr = (f"{'扫描':<10s} {'取值':>8s} | {'落点':>7s} {'过扣':>6s} {'下坠':>6s} "
           f"{'尾端斜率':>9s} {'10s→末':>8s} {'冻结@s':>7s}")
    out = []

    def run(tag, val, over):
        r = obs.run(t, V, obs.default_with(over))
        e = evaluate(r["out_tot"], t, fps)
        print(f"{tag:<10s} {val:>8g} | {e['dev']:+7.0f} {e['over']:6.0f} {e['drop']:6.0f} "
              f"{e['slope_tail']:+9.3f} {e['d10']:+8.0f} {e['freeze']:7.1f}")
        out.append({"sweep": tag, "value": val, "params": over, **e})

    print("A. r_slow_max 扫描（cap=0.005，其余出厂默认 rf.01/τc1 40/conf5/soft8）")
    print(hdr)
    for rsm in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.35):
        run("rsm", rsm, {"slope_cap_frac": 0.005, "r_slow_max": rsm})

    print("\nB. slope_cap_frac 扫描（rsm=0.15，其余出厂默认）")
    print(hdr)
    for cap in (0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.05):
        run("cap", cap, {"slope_cap_frac": cap, "r_slow_max": 0.15})

    print("\nC. r_fast 扫描（cap=0.005, rsm=0.15，τc1=2）")
    print(hdr)
    for rf in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12):
        run("rf", rf, {"r_fast": rf, "tau_c_fast_s": 2.0,
                       "slope_cap_frac": 0.005, "r_slow_max": 0.15})

    print("\nD. 参照：上一轮 161747 最小档原样（rf.04 τc1 1 conf1.5 soft1.5 cap.005 rsm.15）")
    print(hdr)
    run("prevmin", 0, {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5,
                       "soft_unfreeze_s": 1.5, "slope_cap_frac": 0.005, "r_slow_max": 0.15})
    (OUT / "152_sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
