# -*- coding: utf-8 -*-
"""24 按用户验收口径重新寻优（绝对 ADC + 过减硬约束）。

用户口径（2026-09-26）
--------------------
* 拇指指腹：x1 阶段有一个**过减**（显示被后续过度补偿拉回去），可接受量级 **50~150 ADC**，
  不接受现在的程度；右手掌同问题 + **基线漂移太大**。
* 定义：过减 notch = 显示（0.2 s 中值滤波后）相对**历史最高点**的最大回落，单位 ADC；
  欠减 peak_exc = 显示峰值 − 参考线；末值偏差 end_exc = 末值 − 参考线。
  参考线：有加载台阶的会话 = ΣE；无台阶的会话 = 0（调零后的起始基线）。

目标函数：J = 0.5·peak_exc + 0.5·|end_exc| + 2·max(0, notch − 120)
   —— 先保证"不出现超过 120 ADC 的回落"，再让显示尽量贴着参考线。
网格：放宽档（r_slow_max 到 30、slope_cap_frac 到 0.30、slope_gate_frac 到 20），
      因为上一轮已确认 UI 现范围对这几类不够。

产物：out/14_search_adc.json
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

GRID: dict[str, list[float]] = {
    "r_fast": [0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.16, 0.30],
    "tau_c_fast_s": [1.0, 2.0, 4.0, 8.0, 12.0, 20.0],
    "r_slow_max": [0.2, 0.6, 1.0, 2.0, 5.0, 10.0, 30.0],
    # slope_cap_frac 是「显示能压多低」的硬约束：cap = frac·max(e,1) 必须 ≥ 输入爬升率，
    # 否则显示被迫抬到 ≥ 爬升率/frac 才追得上 —— 故本轮到 6.0
    "slope_cap_frac": [0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0],
    "slope_gate_frac": [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0],
    "slow_confirm_s": [0.0, 0.5, 2.0],
    "soft_unfreeze_s": [0.1, 0.2, 0.5, 1.0, 2.0, 4.0],
    # 这几类传感器没有「加载沿」，把沿阈放到极大 = 关掉沿机制，让 x1/x2 全程一致工作
    "edge_slope_thres": [60.0, 150.0, 400.0, 1000.0, 3000.0, 1.0e9],
    "edge_boost_s": [0.0, 2.0],
    "tau_slope_s": [0.2, 0.3, 0.5, 1.0, 2.0],
    "tau_r_slow_idle_s": [0.0, 0.5, 2.0, 8.0, 30.0],
    "tau_r_fast_s": [0.2, 0.5, 1.0, 2.0, 6.0],
    "hold_eps": [0.0, 0.2, 0.5, 2.0, 8.0],
    "hold_tau_s": [0.1, 0.2, 0.3, 0.5, 1.0],
    "idle_frac": [0.02, 0.05, 0.10, 0.20],
    "ramp_slope_min": [0.0, 0.5, 2.0],
}
ORDER = ["slope_cap_frac", "hold_tau_s", "hold_eps", "tau_slope_s", "soft_unfreeze_s",
         "edge_slope_thres", "r_slow_max", "slope_gate_frac", "r_fast", "slow_confirm_s",
         "tau_c_fast_s", "tau_r_slow_idle_s", "tau_r_fast_s", "idle_frac", "edge_boost_s",
         "ramp_slope_min"]

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
    for i in range(0, len(uniq), 300):
        chunk = uniq[i:i + 300]
        BATCH += 1
        sets = [Set(f"d{BATCH:04d}_{j}", c) for j, c in enumerate(chunk)]
        res = run_sensor(sensor, sets, tag_name=f"d_{sensor}_{BATCH:04d}")
        for j, c in enumerate(chunk):
            per = res.get(f"d{BATCH:04d}_{j}", {})
            CACHE[key_of(c, sensor)] = (sum(score_adc(m) for m in per.values())
                                        / max(len(per), 1))
    return {key_of(c, sensor): CACHE[key_of(c, sensor)] for c in cands}


def coord_descent(sensor: str, start: dict, passes: int = 5):
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
        print(f"    pass{p + 1}: J={best:.1f}  { {k: v for k, v in sorted(cur.items())} }",
              flush=True)
        if not improved:
            break
    return cur, best


def report(sensor: str, params: dict, tag: str) -> dict:
    res = run_sensor(sensor, [Set("cand", params)], tag_name=tag)["cand"]
    rows = {}
    print(f"    {'会话':<8s} {'过减ADC':>8s} {'@t':>7s} {'欠减峰':>8s} {'末值偏差':>9s} "
          f"{'绝对过减':>9s} {'扣除%':>7s} {'J':>7s}")
    for t, m in res.items():
        print(f"    {t:<8s} {m['notch']:8.1f} {m['notch_t']:7.1f} {m['peak_exc']:8.1f} "
              f"{m['end_exc']:9.1f} {m['low_exc']:9.1f} {m['ded_pct']:7.1f} "
              f"{score_adc(m):7.1f}")
        rows[t] = m
    return rows


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng = random.Random(7)
    only = sys.argv[1:] or list(SENSORS)
    out = {}
    for sensor in only:
        print(f"\n########## {sensor} ##########", flush=True)
        base = {}
        starts = [base] + [{k: rng.choice(v) for k, v in GRID.items()} for _ in range(14)]
        # 把上一轮的比例制最优也作为一个起点
        prev = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
        starts.append(prev[sensor]["params"])
        sc = evaluate_batch(sensor, starts)
        starts.sort(key=lambda s: sc[key_of(s, sensor)])
        print(f"  起点最好 J={sc[key_of(starts[0], sensor)]:.1f} / "
              f"最差 {sc[key_of(starts[-1], sensor)]:.1f}", flush=True)
        best, bj = None, 1e18
        for i, st in enumerate(starts[:3]):
            print(f"  起点{i}（J={sc[key_of(st, sensor)]:.1f}）→ 坐标下降", flush=True)
            p, j = coord_descent(sensor, st)
            if j < bj:
                best, bj = p, j
        print(f"  ★ J={bj:.1f}  {json.dumps({k: v for k, v in sorted(best.items())}, ensure_ascii=False)}")
        rows = report(sensor, best, f"adc_rep_{sensor}")
        out[sensor] = {"J": bj, "params": best, "sessions": rows}
        (OUT_DIR / "14_search_adc.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '14_search_adc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
