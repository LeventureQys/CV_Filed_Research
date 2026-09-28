# -*- coding: utf-8 -*-
"""131_force_analyze：读 130_fine.json，出 Pareto 前沿 + 若干判据下的最优 + 现役基线对照。

判据（全部取「越接近 0 越好」）：
  ① 稳定优先：max|下坠| 最小（在 落点 不劣于现役 +0.3 N 的前提下）
  ② 准确优先：max|落点| 最小（在下坠 ≤0.3 N 的前提下）
  ③ 综合：max|落点| + 1.2·max|下坠| + 25·max|末斜率|
并打印恒压合成（纯阶跃、零蠕变）的过扣作为「失真代价」。
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
fl = import_module("122_force_lib")
c128 = import_module("128_force_cases")
OUT = TEMP / "palm7" / "out"
REAL = ("hold_1d925c", "in_03225d")


def head() -> str:
    return (f"{'rf':>5s} {'τc1':>5s} {'cap':>6s} {'conf':>4s} {'rsm':>5s} | "
            f"{'落点1':>7s} {'下坠1':>6s} {'落点2':>7s} {'下坠2':>6s} {'落点3':>7s} {'下坠3':>6s} | "
            f"{'过扣max':>7s} {'末斜max':>8s} | {'恒压落点':>8s} {'恒压下坠':>8s}")


def line(r: dict) -> str:
    c = r["c"]
    a1, a2 = c["hold_1d925c"][0], c["hold_1d925c"][1]
    a3 = c["in_03225d"][0]
    sf = c["synth_flat17"][0]
    rows = [a1, a2, a3]
    return (f"{r['rf']:5g} {r['tc1']:5g} {r['cap']:6g} {r['conf']:4g} {r['rsm']:5g} | "
            f"{a1['dev']:+7.3f} {a1['drop']:6.3f} {a2['dev']:+7.3f} {a2['drop']:6.3f} "
            f"{a3['dev']:+7.3f} {a3['drop']:6.3f} | "
            f"{max(x['over'] for x in rows):7.3f} {max(abs(x['slope_end']) for x in rows):8.5f} | "
            f"{sf['dev']:+8.3f} {sf['drop']:8.3f}")


def base_rec() -> dict:
    cases = {c["name"]: c for c in c128.build()}
    rec = {"rf": 0.03, "tc1": 40.0, "cap": 0.011, "conf": 2.0, "rsm": 0.35, "c": {}}
    for k in ("hold_1d925c", "in_03225d", "synth_flat17"):
        case = cases[k]
        t, V = case["t"], case["V"]
        fps = (len(t) - 1) / (t[-1] - t[0])
        r = obs.run(t, V, obs.default_with(fl.REC8))
        rec["c"][k] = [{kk: x[kk] for kk in ("dev", "drop", "over", "drift",
                                             "slope_end", "tsettle", "E")}
                       for x in fl.evaluate(r["out_tot"], t, case["segs"], fps)]
    return rec


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    data = json.loads((OUT / "130_fine.json").read_text(encoding="utf-8"))
    base = base_rec()
    print("== 现役基线（rf.03/τc1 40/conf 2/soft 2/cap.011/rsm.35/τr1 6/τrsi .5）==")
    print(head())
    print(line(base))
    print(f"\n共 {len(data)} 组；落点/下坠 散点见下方统计")

    # 现役指标（E_pl 口径）
    b1, b2 = base["c"]["hold_1d925c"]
    b3 = base["c"]["in_03225d"][0]
    bdev = max(abs(b1["dev"]), abs(b2["dev"]), abs(b3["dev"]))
    bdrop = max(b1["drop"], b2["drop"], b3["drop"])
    print(f"现役：max|落点|={bdev:.3f}  max下坠={bdrop:.3f}")

    for r in data:
        rows = [r["c"]["hold_1d925c"][0], r["c"]["hold_1d925c"][1], r["c"]["in_03225d"][0]]
        r["_dev"] = max(abs(x["dev"]) for x in rows)
        r["_drop"] = max(x["drop"] for x in rows)
        r["_over"] = max(x["over"] for x in rows)
        r["_slope"] = max(abs(x["slope_end"]) for x in rows)
        r["_flat"] = r["c"]["synth_flat17"][0]["dev"]
        r["_score"] = r["_dev"] + 1.2 * r["_drop"] + 25 * r["_slope"] + 8 * max(0.0, r["_over"] - 0.05)

    print("\n== 判据① 稳定优先：下坠最小，且 落点 ≤ 现役+0.3 且 过扣≤0.15 ==")
    ok = [r for r in data if r["_dev"] <= bdev + 0.3 and r["_over"] <= 0.15]
    print(head())
    for r in sorted(ok, key=lambda r: r["_drop"])[:10]:
        print(line(r))

    print("\n== 判据② 准确优先：|落点| 最小，且 下坠 ≤ 0.30 且 过扣≤0.15 ==")
    ok2 = [r for r in data if r["_drop"] <= 0.30 and r["_over"] <= 0.15]
    print(head())
    for r in sorted(ok2, key=lambda r: r["_dev"])[:10]:
        print(line(r))

    print("\n== 判据③ 综合评分最小（落点+1.2下坠+25末斜率+过扣罚） ==")
    print(head())
    for r in sorted(data, key=lambda r: r["_score"])[:15]:
        print(line(r))

    print("\n== Pareto 前沿（|落点| 与 下坠 同时最小） ==")
    print(head())
    pareto = []
    for r in sorted(data, key=lambda r: r["_dev"]):
        if not pareto or r["_drop"] < min(q["_drop"] for q in pareto) - 1e-9:
            pareto.append(r)
    for r in pareto:
        print(line(r))
    (OUT / "131_analysis.json").write_text(json.dumps(
        {"base": base, "pareto": [{"rf": r["rf"], "tc1": r["tc1"], "cap": r["cap"],
                                   "conf": r["conf"], "rsm": r["rsm"], "dev": r["_dev"],
                                   "drop": r["_drop"], "over": r["_over"]} for r in pareto]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
