# -*- coding: utf-8 -*-
"""08 寻优：对每类传感器做「随机起点 + 坐标下降」的参数搜索（候选批量喂给 C++ 复算器）。

目标函数（越小越好，逐会话平均）：
    score = |末值残漂%| + |末 30 s 残余漂移%| + 2·max(0, 过扣% − 10)
百分比一律以「本该被扣掉的累计蠕变 creep」为分母，故 5 类传感器可横比；
过扣（显示低于弹性电平）按 2 倍罚，且超过蠕变的 10% 才开始罚。

产物：out/04_search.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score  # noqa: E402

GRID: dict[str, list[float]] = {
    "r_fast": [0.0, 0.01, 0.02, 0.04, 0.06, 0.10],
    "tau_c_fast_s": [2.0, 4.0, 8.0, 12.0, 20.0],
    "r_slow_max": [0.02, 0.04, 0.06, 0.10, 0.20, 0.35, 0.60],
    "slope_cap_frac": [0.002, 0.005, 0.01, 0.02, 0.04],
    "slope_gate_frac": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
    "slow_confirm_s": [0.0, 0.5, 2.0, 5.0],
    "soft_unfreeze_s": [0.5, 1.0, 2.0, 4.0],
    "edge_slope_thres": [10.0, 20.0, 60.0, 150.0, 400.0],
    "tau_slope_s": [0.5, 1.0, 2.0, 3.0],
    "tau_r_slow_idle_s": [0.5, 2.0, 8.0, 30.0],
    "tau_r_fast_s": [0.2, 0.5, 1.0, 2.0],
    "hold_eps": [0.0, 0.5, 2.0, 8.0],
    "idle_frac": [0.02, 0.05, 0.10],
    "ramp_slope_min": [0.0, 0.5, 2.0],
}
ORDER = ["r_fast", "r_slow_max", "slope_gate_frac", "slope_cap_frac", "tau_slope_s",
         "slow_confirm_s", "soft_unfreeze_s", "edge_slope_thres", "tau_c_fast_s",
         "tau_r_slow_idle_s", "tau_r_fast_s", "hold_eps", "idle_frac", "ramp_slope_min"]

# --ext：放宽到「超出当前 UI 可调范围」的取值，用来判断是否需要新增/放宽参数
import os  # noqa: E402

EXT = "--ext" in sys.argv
if EXT:
    sys.argv = [a for a in sys.argv if a != "--ext"]
    GRID["r_slow_max"] = [0.02, 0.04, 0.06, 0.10, 0.20, 0.35, 0.60, 1.0, 2.0, 5.0, 10.0, 30.0]
    GRID["slope_cap_frac"] = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30]
    GRID["slope_gate_frac"] = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0]
    GRID["hold_eps"] = [0.0, 0.5, 2.0, 8.0, 30.0]
# --ui：只允许动「对话框 kSpecs 里那 8 项」，且每项只取该控件 min~max/step 内的值
#       —— 用来回答「今天不改任何东西、只靠现有 8 个旋钮能调到什么程度」
_argv = [a for a in sys.argv if not a.startswith("--") or a == "--ext"]
UI_ONLY = "--ui" in sys.argv
if UI_ONLY:
    sys.argv = [a for a in sys.argv if a != "--ui"]
    GRID = {
        "r_fast": [0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.16, 0.30],
        "tau_c_fast_s": [2.0, 4.0, 8.0, 12.0, 20.0, 40.0],
        "slow_confirm_s": [0.0, 0.5, 2.0, 5.0, 10.0, 20.0],
        "soft_unfreeze_s": [0.5, 1.0, 2.0, 4.0, 8.0],
        "slope_cap_frac": [0.002, 0.005, 0.01, 0.02, 0.04, 0.05],
        "r_slow_max": [0.02, 0.04, 0.06, 0.10, 0.20, 0.35, 0.60],
        "tau_r_slow_idle_s": [0.0, 0.5, 2.0, 8.0, 30.0, 100.0],
        "tau_r_fast_s": [0.2, 0.5, 1.0, 2.0, 6.0],
    }
    ORDER = list(GRID)
OUTFILE = "04_search_ui.json" if UI_ONLY else ("04_search_ext.json" if EXT else "04_search.json")

CACHE: dict[str, float] = {}
BATCH = 0
DETAIL: dict[str, dict] = {}


def key_of(over: dict, sensor: str = "") -> str:
    return f"{sensor}|{json.dumps(over, sort_keys=True)}"


def evaluate_batch(sensor: str, cands: list[dict]) -> dict[str, float]:
    """批量评估（一次 exec 跑完所有候选）。返回 {key: score}。注意缓存键必须带传感器名。"""
    global BATCH
    todo = [c for c in cands if key_of(c, sensor) not in CACHE]
    seen, uniq = set(), []
    for c in todo:
        k = key_of(c, sensor)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    for i in range(0, len(uniq), 400):
        chunk = uniq[i:i + 400]
        BATCH += 1
        sets = [Set(f"c{BATCH:04d}_{j}", c) for j, c in enumerate(chunk)]
        res = run_sensor(sensor, sets, tag_name=f"b_{sensor}_{BATCH:04d}")
        for j, c in enumerate(chunk):
            per = res.get(f"c{BATCH:04d}_{j}", {})
            CACHE[key_of(c, sensor)] = sum(score(m) for m in per.values()) / max(len(per), 1)
            DETAIL[key_of(c, sensor)] = per
    return {key_of(c, sensor): CACHE[key_of(c, sensor)] for c in cands}


def coord_descent(sensor: str, start: dict, passes: int = 4):
    cur = dict(start)
    evaluate_batch(sensor, [cur])
    best = CACHE[key_of(cur, sensor)]
    trace = [{"params": dict(cur), "score": best}]
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
                trace.append({"params": dict(cur), "score": best, "changed": k})
        print(f"    pass{p + 1}: score={best:.3f}  "
              f"{ {k: v for k, v in sorted(cur.items())} }", flush=True)
        if not improved:
            break
    return cur, best, trace


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng = random.Random(20260926)
    only = sys.argv[1:] or list(SENSORS)
    all_out = {}
    for sensor in only:
        print(f"\n########## {sensor} ##########", flush=True)
        base = {}
        v0 = evaluate_batch(sensor, [base])[key_of(base, sensor)]
        print(f"  现役默认 score={v0:.3f}", flush=True)

        starts = [base] + [{k: rng.choice(vals) for k, vals in GRID.items()} for _ in range(20)]
        sc = evaluate_batch(sensor, starts)
        starts.sort(key=lambda s: sc[key_of(s, sensor)])
        print(f"  随机 {len(starts)} 个起点：最好 {sc[key_of(starts[0], sensor)]:.3f} / "
              f"最差 {sc[key_of(starts[-1], sensor)]:.3f}", flush=True)

        results = []
        seeds = starts[:3] + [base]
        for i, st in enumerate(seeds):
            print(f"  起点{i}（{sc[key_of(st, sensor)]:.3f}）→ 坐标下降", flush=True)
            p, val, tr = coord_descent(sensor, st)
            results.append({"start_score": sc[key_of(st, sensor)], "score": val,
                            "params": p, "trace": tr})
        results.sort(key=lambda r: r["score"])
        all_out[sensor] = {"live_score": v0, "runs": results}
        best = results[0]
        print(f"  ★ 最优 score={best['score']:.3f}（现役 {v0:.3f}）", flush=True)
        print(f"    {json.dumps({k: v for k, v in sorted(best['params'].items())}, ensure_ascii=False)}",
              flush=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / OUTFILE).write_text(
            json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {OUT_DIR / OUTFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
