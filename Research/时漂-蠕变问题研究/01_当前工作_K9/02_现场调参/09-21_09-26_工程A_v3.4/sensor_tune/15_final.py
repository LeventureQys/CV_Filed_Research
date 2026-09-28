# -*- coding: utf-8 -*-
"""15 定稿：把「搜索出来的极值解」收成「现场可落地的推荐预设」，逐类传感器比较候选。

原则
----
* 极值解（r_slow_max=30 之类）虽然分数最低，但 r_slow_max 直接等价于「允许把读数的多少
  当成蠕变扣掉」，取 30 意味着几乎无条件把慢上升全扣掉；现场要留安全余量。
* 故对每类传感器同时评估：
    - 现役默认
    - UI 范围内搜索最优（04_search.json）
    - 放宽范围搜索最优（04_search_ext.json）
    - 若干「收敛后的实用候选」（把极值拉回合理区间）
  再按 score 与安全边界共同拍板。

产物：out/08_final.json / 08_final.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import Set, run_sensor, score  # noqa: E402

GRID_MAX = 5.0


def practical(base: dict, r_slow_max: float, gate: float, cap: float) -> dict:
    d = dict(base)
    d.update({"r_slow_max": r_slow_max, "slope_gate_frac": gate, "slope_cap_frac": cap})
    return d


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ui = json.loads((OUT_DIR / "04_search.json").read_text(encoding="utf-8"))
    ext = json.loads((OUT_DIR / "04_search_ext.json").read_text(encoding="utf-8"))
    lines, report = [], {}

    def w(s: str = "") -> None:
        print(s)
        lines.append(s)

    for sensor in SENSORS:
        ui_best = ui[sensor]["runs"][0]["params"]
        ext_best = ext[sensor]["runs"][0]["params"]
        cands: list[tuple[str, dict]] = [("现役默认", {}), ("UI最优", ui_best), ("放宽最优", ext_best)]
        # 实用候选：以 UI 最优为骨架，把 r_slow_max / gate / cap 拉回合理区间
        for r in (0.6, 1.0, 2.0, 5.0):
            for gate in (0.5, 1.0, 2.0):
                for cap in (0.04, 0.08):
                    cands.append((f"实用 r{r:g}·gate{gate:g}·cap{cap:g}",
                                  practical(ui_best, r, gate, cap)))
        # 以放宽最优为骨架的收敛版
        for r in (1.0, 2.0, 5.0):
            cands.append((f"收敛(放宽骨架) r{r:g}·gate{ext_best.get('slope_gate_frac', 1.0):g}"
                          f"·cap{min(ext_best.get('slope_cap_frac', 0.08), 0.15):g}",
                          practical(ext_best, r, ext_best.get("slope_gate_frac", 1.0),
                                    min(ext_best.get("slope_cap_frac", 0.08), 0.15))))

        sets = [Set(f"s{i:03d}", c) for i, (_, c) in enumerate(cands)]
        res_u = run_sensor(sensor, sets, tag_name=f"fin_u_{sensor}")
        res_r = run_sensor(sensor, sets, time_mode="raw", tag_name=f"fin_r_{sensor}")
        w(f"\n{'=' * 104}\n### {sensor}")
        w(f"  {'候选':<26s} {'score':>7s} {'扣除%':>7s} {'残漂%':>7s} {'过扣%':>7s} "
          f"{'末30s%':>8s} {'抖动%':>7s} │ {'raw score':>9s} {'raw扣除%':>8s}")
        rows = []
        for i, (nm, c) in enumerate(cands):
            k = f"s{i:03d}"
            mu, mr = list(res_u[k].values()), list(res_r[k].values())
            a = {kk: sum(m[kk] for m in mu) / len(mu) for kk in
                 ("ded_pct", "resid_pct", "over_pct", "tail_pct", "std_pct")}
            a["score"] = sum(score(m) for m in mu) / len(mu)
            b = {"score": sum(score(m) for m in mr) / len(mr),
                 "ded_pct": sum(m["ded_pct"] for m in mr) / len(mr)}
            rows.append((a["score"], nm, c, a, b))
        rows.sort(key=lambda r: r[0])
        for sc, nm, c, a, b in rows[:26]:
            w(f"  {nm:<26s} {a['score']:7.2f} {a['ded_pct']:7.1f} {a['resid_pct']:7.1f} "
              f"{a['over_pct']:7.1f} {a['tail_pct']:8.2f} {a['std_pct']:7.2f} │ "
              f"{b['score']:9.2f} {b['ded_pct']:8.1f}")
        report[sensor] = {"candidates": [
            {"name": nm, "params": c, "uniform": a, "raw": b} for sc, nm, c, a, b in rows]}
    (OUT_DIR / "08_final.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    (OUT_DIR / "08_final.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '08_final.json'} / 08_final.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
