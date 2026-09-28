# -*- coding: utf-8 -*-
"""29 四指指腹专项：这类会话的载荷在 7.9k~33.7k ADC 之间差 4 倍，
   绝对 ADC 指标横比会被大会话主导，故改用「相对 ΣE 的百分比」做目标函数：
       J = peak% + 0.5·|end%| + 4·max(0, 过减% − 1.0) + max(0, 回落% − 1.5)
   即「过减不超过弹性电平的 1 %、回落不超过 1.5 %」。"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor  # noqa: E402

SENSOR = "四指指腹"
GRID: dict[str, list[float]] = {
    "r_fast": [0.0, 0.005, 0.01, 0.02, 0.04, 0.06],
    "tau_c_fast_s": [1.0, 2.0, 4.0, 8.0, 12.0, 20.0],
    "tau_c_fast_boost_s": [1.0, 2.0, 4.0],
    "r_slow_max": [0.05, 0.1, 0.2, 0.35, 0.6, 1.0],
    "slope_cap_frac": [0.02, 0.05, 0.1, 0.2, 0.5, 1.5, 4.0],
    "slope_gate_frac": [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0],
    "slow_confirm_s": [0.0, 0.5, 2.0, 5.0, 10.0],
    "soft_unfreeze_s": [0.1, 0.5, 1.0, 2.0, 4.0],
    "edge_slope_thres": [20.0, 60.0, 150.0, 400.0, 3000.0],
    "edge_boost_s": [0.0, 2.0],
    "tau_slope_s": [0.2, 0.5, 1.0, 2.0],
    "tau_r_slow_idle_s": [0.0, 0.5, 2.0, 8.0, 30.0],
    "tau_r_fast_s": [0.2, 0.5, 1.0, 2.0, 6.0],
    "hold_eps": [0.0, 0.2, 0.5, 2.0],
    "hold_tau_s": [0.1, 0.2, 0.5],
    "idle_frac": [0.02, 0.05, 0.1, 0.2],
    "ramp_slope_min": [0.0, 0.5, 2.0],
    "ramp_full_s": [0.0, 4.0],
}
ORDER = ["r_fast", "r_slow_max", "slope_cap_frac", "slope_gate_frac", "slow_confirm_s",
         "soft_unfreeze_s", "edge_slope_thres", "tau_c_fast_s", "tau_slope_s", "hold_eps",
         "hold_tau_s", "tau_r_slow_idle_s", "tau_r_fast_s", "idle_frac", "ramp_full_s",
         "ramp_slope_min", "edge_boost_s", "tau_c_fast_boost_s"]
CACHE: dict[str, float] = {}
BATCH = 0


def key_of(o: dict) -> str:
    return json.dumps(o, sort_keys=True)


def score_pct(m: dict) -> float:
    e = m["Esum"]
    return ((m["peak_exc"] / e * 100) + 0.5 * abs(m["end_exc"] / e * 100)
            + 4.0 * max(0.0, m["low_exc"] / e * 100 - 1.0)
            + 1.0 * max(0.0, m["notch"] / e * 100 - 1.5))


def evaluate(cands: list[dict]) -> dict[str, float]:
    global BATCH
    seen, uniq = set(), []
    for c in cands:
        k = key_of(c)
        if k not in CACHE and k not in seen:
            seen.add(k)
            uniq.append(c)
    for i in range(0, len(uniq), 250):
        chunk = uniq[i:i + 250]
        BATCH += 1
        sets = [Set(f"g{BATCH:04d}_{j}", c) for j, c in enumerate(chunk)]
        res = run_sensor(SENSOR, sets, tag_name=f"g_{BATCH:04d}")
        for j, c in enumerate(chunk):
            per = res.get(f"g{BATCH:04d}_{j}", {})
            CACHE[key_of(c)] = sum(score_pct(m) for m in per.values()) / max(len(per), 1)
    return {key_of(c): CACHE[key_of(c)] for c in cands}


def descent(start: dict, passes: int = 5):
    cur = dict(start)
    evaluate([cur])
    best = CACHE[key_of(cur)]
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
            sc = evaluate(cands)
            bk, bv = None, best
            for c in cands:
                if sc[key_of(c)] < bv - 1e-9:
                    bk, bv = c, sc[key_of(c)]
            if bk is not None:
                cur, best = bk, bv
                improved = True
        print(f"    pass{p + 1}: J%={best:.3f}", flush=True)
        if not improved:
            break
    return cur, best


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng = random.Random(3)
    prev = json.loads((OUT_DIR / "16_refine.json").read_text(encoding="utf-8"))[SENSOR]["params"]
    starts = [prev, {}]
    for _ in range(8):
        starts.append({k: rng.choice(v) for k, v in GRID.items()})
    sc = evaluate(starts)
    starts.sort(key=lambda s: sc[key_of(s)])
    print(f"起点最好 J%={sc[key_of(starts[0])]:.3f}")
    best, bj = None, 1e18
    for i, st in enumerate(starts[:3]):
        print(f"  起点{i}（J%={sc[key_of(st)]:.3f}）→ 坐标下降", flush=True)
        p, j = descent(st)
        if j < bj:
            best, bj = p, j
    print(f"★ J%={bj:.3f}  {json.dumps({k: v for k, v in sorted(best.items())}, ensure_ascii=False)}")
    res = run_sensor(SENSOR, [Set("cand", best)], tag_name="g4f_rep")["cand"]
    print(f"  {'会话':<12s} {'过减ADC':>8s} {'过减%ΣE':>8s} {'回落ADC':>8s} {'欠减峰':>8s} "
          f"{'末值':>8s} {'扣除%':>7s} {'J%':>6s}")
    for t, m in res.items():
        print(f"  {t:<12s} {m['low_exc']:8.1f} {m['low_exc'] / m['Esum'] * 100:8.2f} "
              f"{m['notch']:8.1f} {m['peak_exc']:8.1f} {m['end_exc']:8.1f} "
              f"{m['ded_pct']:7.1f} {score_pct(m):6.2f}")
    (OUT_DIR / "18_search_4f.json").write_text(json.dumps(
        {SENSOR: {"J": bj, "params": best, "sessions": res}}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"写出 {OUT_DIR / '18_search_4f.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
