# -*- coding: utf-8 -*-
"""26 精细收敛：把「过减 ≤ 120 ADC」当硬约束（8 倍罚），在更细/更宽的网格上再收一轮。

新增可动项（上一轮没覆盖、但对「显示能压多低」直接影响）：
  r_fast 到 2.0   —— x1 = r·e/(1+r) 是**比例式**状态，调大直接压低显示水位且不会过冲；
  ramp_full_s      —— ≤0 即关闭 K9 缓坡前馈；>0 时 ramp_dwell 累积会把 tc1 换成 tau_c_fast_boost_s；
  tau_c_fast_boost_s —— 上述情形下的实际 τc1；
  slope_gate_frac 到 100、slope_cap_frac 到 20 —— 让 x2 真正追得上早期快蠕变。

产物：out/16_refine.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score_adc  # noqa: E402

TOL = 100.0        # low_exc（显示沉到参考线下方）的容差，单位 ADC

GRID: dict[str, list[float]] = {
    "r_fast": [0.06, 0.1, 0.16, 0.3, 0.5, 0.8, 1.2, 2.0],
    "tau_c_fast_s": [1.0, 2.0, 4.0, 8.0],
    "tau_c_fast_boost_s": [0.5, 1.0, 2.0, 4.0],
    "r_slow_max": [1.0, 5.0, 30.0, 100.0],
    "slope_cap_frac": [0.4, 0.8, 1.5, 3.0, 6.0, 10.0, 20.0],
    "slope_gate_frac": [1.0, 2.0, 5.0, 20.0, 100.0],
    "slow_confirm_s": [0.0, 0.5],
    "soft_unfreeze_s": [0.05, 0.1, 0.2, 0.5],
    "edge_slope_thres": [400.0, 3000.0, 1.0e9],
    "edge_boost_s": [0.0],
    "tau_slope_s": [0.1, 0.2, 0.3, 0.5],
    "tau_r_slow_idle_s": [0.0, 0.5, 2.0, 8.0],
    "tau_r_fast_s": [0.2, 0.5, 1.0, 2.0, 6.0],
    "hold_eps": [0.0, 0.1, 0.2, 0.5],
    "hold_tau_s": [0.05, 0.1, 0.15, 0.2, 0.3],
    "idle_frac": [0.02, 0.1, 0.2, 0.5],
    "ramp_slope_min": [0.0, 0.5, 2.0],
    "ramp_full_s": [0.0, 4.0],
}
ORDER = ["r_fast", "slope_cap_frac", "slope_gate_frac", "hold_tau_s", "hold_eps",
         "tau_slope_s", "soft_unfreeze_s", "ramp_full_s", "tau_c_fast_boost_s", "tau_c_fast_s",
         "edge_slope_thres", "r_slow_max", "slow_confirm_s", "tau_r_slow_idle_s",
         "tau_r_fast_s", "idle_frac", "ramp_slope_min", "edge_boost_s"]

CACHE: dict[str, float] = {}
BATCH = 0


def key_of(over: dict, sensor: str) -> str:
    return f"{sensor}|{json.dumps(over, sort_keys=True)}"


def evaluate_batch(sensor: str, cands: list[dict]) -> dict[str, float]:
    global BATCH
    seen, uniq = set(), []
    for c in cands:
        k = key_of(c, sensor)
        if k not in CACHE and k not in seen:
            seen.add(k)
            uniq.append(c)
    for i in range(0, len(uniq), 250):
        chunk = uniq[i:i + 250]
        BATCH += 1
        sets = [Set(f"e{BATCH:04d}_{j}", c) for j, c in enumerate(chunk)]
        res = run_sensor(sensor, sets, tag_name=f"e_{sensor}_{BATCH:04d}")
        for j, c in enumerate(chunk):
            per = res.get(f"e{BATCH:04d}_{j}", {})
            CACHE[key_of(c, sensor)] = (sum(score_adc(m, TOL) for m in per.values())
                                        / max(len(per), 1))
    return {key_of(c, sensor): CACHE[key_of(c, sensor)] for c in cands}


def coord_descent(sensor: str, start: dict, passes: int = 6):
    cur = dict(start)
    evaluate_batch(sensor, [cur])
    best = CACHE[key_of(cur, sensor)]
    for p in range(passes):
        improved = False
        for k in ORDER:
            cands = []
            for v in GRID[k]:
                if abs(v - cur.get(k, float("nan"))) < 1e-12:
                    continue
                c = dict(cur)
                c[k] = v
                cands.append(c)
            sc = evaluate_batch(sensor, cands)
            bk, bv = None, best
            for c in cands:
                if sc[key_of(c, sensor)] < bv - 1e-9:
                    bk, bv = c, sc[key_of(c, sensor)]
            if bk is not None:
                cur, best = bk, bv
                improved = True
        print(f"    pass{p + 1}: J={best:.1f}", flush=True)
        if not improved:
            break
    return cur, best


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng = random.Random(11)
    only = sys.argv[1:] or list(SENSORS)
    prev = json.loads((OUT_DIR / "14_search_adc.json").read_text(encoding="utf-8"))
    out = {}
    for sensor in only:
        print(f"\n########## {sensor} ##########", flush=True)
        starts = [prev[sensor]["params"]]
        for _ in range(10):
            starts.append({k: rng.choice(v) for k, v in GRID.items()})
        sc = evaluate_batch(sensor, starts)
        starts.sort(key=lambda s: sc[key_of(s, sensor)])
        print(f"  起点最好 J={sc[key_of(starts[0], sensor)]:.1f}", flush=True)
        best, bj = None, 1e18
        for i, st in enumerate(starts[:3]):
            print(f"  起点{i}（J={sc[key_of(st, sensor)]:.1f}）→ 坐标下降", flush=True)
            p, j = coord_descent(sensor, st)
            if j < bj:
                best, bj = p, j
        print(f"  ★ J={bj:.1f}  {json.dumps({k: v for k, v in sorted(best.items())}, ensure_ascii=False)}")
        res = run_sensor(sensor, [Set("cand", best)], tag_name=f"ref_rep_{sensor}")["cand"]
        print(f"    {'会话':<8s} {'过减ADC':>8s} {'@t':>7s} {'欠减峰':>8s} {'末值偏差':>9s} "
              f"{'绝对过减':>9s} {'扣除%':>7s} {'J':>7s}")
        for t, m in res.items():
            print(f"    {t:<8s} {m['notch']:8.1f} {m['notch_t']:7.1f} {m['peak_exc']:8.1f} "
                  f"{m['end_exc']:9.1f} {m['low_exc']:9.1f} {m['ded_pct']:7.1f} "
                  f"{score_adc(m, TOL):7.1f}")
        out[sensor] = {"J": bj, "params": best, "sessions": res}
        (OUT_DIR / "16_refine.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '16_refine.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
