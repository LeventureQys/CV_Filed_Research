# -*- coding: utf-8 -*-
"""129_force_search：力值模式三维搜索（r_fast × tau_c_fast_s × slope_cap_frac）。

口径（见 122_force_lib）：
  分析窗 = 台阶后 0.35 s（加载斜坡结束）~ 段末留 0.6 s；显示先做 1 s 中值。
  E  = 输入在 [t0+0.35, t0+1.0] 的最小值（加载完成时的弹性平台，物理上的「刚压下去的读数」）
  落点 = 段末显示 − E；漂移 = 段末显示 − 显示(t0+0.35)；下坠 = 段内峰 − 段末；
  过扣 = E − 段内最小；稳定 = 1 s 中值末次偏离落点 > 0.3 N。
第一阶段只在主会话 hold_1d925c 上跑粗网格，产出 Pareto 前沿；
第二阶段对前沿候选跑全测试台（含合成恒压 → 「纯电平扣除」失真代价）。
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
fl = import_module("122_force_lib")
c128 = import_module("128_force_cases")
OUT = TEMP / "palm7" / "out"

RF = [0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]
TC1 = [0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 40.0]
CAP = [0.002, 0.004, 0.006, 0.008, 0.011, 0.015, 0.02, 0.03]
CONF = [0.0, 2.0]
RSM = [0.10]


def eval_case(case: dict, p: dict) -> list[dict]:
    t, V = case["t"], case["V"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    r = obs.run(t, V, obs.default_with(p))
    return fl.evaluate(r["out_tot"], t, case["segs"], fps)


def score(rows: list[dict]) -> tuple[float, float, float]:
    """返回 (最大|落点|, 最大下坠, 最大过扣)。"""
    return (max(abs(r["dev"]) for r in rows),
            max(r["drop"] for r in rows),
            max(r["over"] for r in rows))


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cases = {c["name"]: c for c in c128.build()}
    main_case = cases["hold_1d925c"]
    t0 = time.perf_counter()
    rows_all = []
    combos = list(itertools.product(RF, TC1, CAP, CONF, RSM))
    print(f"粗网格 {len(combos)} 组（主会话 hold_1d925c）", flush=True)
    for k, (rf, tc1, cap, conf, rsm) in enumerate(combos):
        p = {**fl.REC8, "r_fast": rf, "tau_c_fast_s": tc1, "slope_cap_frac": cap,
             "slow_confirm_s": conf, "r_slow_max": rsm}
        rows = eval_case(main_case, p)
        dev, drop, over = score(rows)
        rows_all.append({"rf": rf, "tc1": tc1, "cap": cap, "conf": conf, "rsm": rsm,
                         "dev": dev, "drop": drop, "over": over,
                         "rows": [{kk: r[kk] for kk in ("dev", "drop", "over", "drift",
                                                        "tsettle", "slope_end")} for r in rows]})
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(combos)}  用时 {time.perf_counter()-t0:.0f}s", flush=True)
    (OUT / "129_stage1.json").write_text(json.dumps(rows_all, ensure_ascii=False), encoding="utf-8")
    print(f"阶段1 完成，用时 {time.perf_counter()-t0:.0f}s", flush=True)

    # Pareto 前沿：|落点| 与 下坠 同时最小，且过扣 ≤0.2
    ok = [r for r in rows_all if r["over"] <= 0.20]
    print(f"\n合规（过扣≤0.2）{len(ok)}/{len(rows_all)} 组；Pareto 前沿（|落点| / 下坠）：")
    pareto = []
    for r in sorted(ok, key=lambda r: r["dev"]):
        if not pareto or r["drop"] < min(q["drop"] for q in pareto) - 1e-9:
            pareto.append(r)
    for r in pareto:
        print(f"  rf={r['rf']:<5g} τc1={r['tc1']:<5g} cap={r['cap']:<6g} conf={r['conf']:<4g} "
              f"|落点|={r['dev']:.3f} 下坠={r['drop']:.3f} 过扣={r['over']:.3f}")

    # 阶段2：前沿 + 若干均衡候选跑全台
    picks = {id(r): r for r in pareto}
    for r in sorted(ok, key=lambda r: r["dev"] + 1.5 * r["drop"])[:8]:
        picks[id(r)] = r
    print(f"\n阶段2：{len(picks)} 个候选跑全测试台", flush=True)
    full = []
    for r in picks.values():
        p = {**fl.REC8, "r_fast": r["rf"], "tau_c_fast_s": r["tc1"],
             "slope_cap_frac": r["cap"], "slow_confirm_s": r["conf"], "r_slow_max": r["rsm"]}
        rec = {"params": {k: r[k] for k in ("rf", "tc1", "cap", "conf", "rsm")}, "cases": {}}
        for name, case in cases.items():
            rr = eval_case(case, p)
            rec["cases"][name] = [{kk: x[kk] for kk in
                                   ("dev", "dev_fast", "drop", "over", "drift", "tsettle",
                                    "slope_end", "E")} for x in rr]
        full.append(rec)
    (OUT / "129_stage2.json").write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
    for rec in full:
        pr = rec["params"]
        print(f"\nrf={pr['rf']:g} τc1={pr['tc1']:g} cap={pr['cap']:g} conf={pr['conf']:g} "
              f"rsm={pr['rsm']:g}")
        for name, rr in rec["cases"].items():
            for j, x in enumerate(rr):
                print(f"   {name:<14s} 段{j+1} E={x['E']:7.3f} 落点={x['dev']:+7.3f} "
                      f"下坠={x['drop']:6.3f} 过扣={x['over']:6.3f} 漂移={x['drift']:+7.3f} "
                      f"稳={x['tsettle']:5.2f}s")
    print(f"\n总用时 {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
