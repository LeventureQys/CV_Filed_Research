# -*- coding: utf-8 -*-
"""141_final_table：终版对照表（现役 / 推荐B / 备选A 稳显 / 备选C 贴平），全测试台。

输出直接可抄进报告的 markdown 表。
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
s132 = import_module("132_adc_cross")
OUT7 = TEMP / "palm7" / "out"

SETS = {
    "现役": dict(fl.REC8),
    "推荐B": {**fl.REC8, "r_fast": 0.05, "tau_c_fast_s": 2.0, "slope_cap_frac": 0.009},
    "备选A稳显": {**fl.REC8, "r_fast": 0.04, "tau_c_fast_s": 2.0, "slope_cap_frac": 0.009},
    "备选C贴平": {**fl.REC8, "r_fast": 0.05, "tau_c_fast_s": 1.0, "slope_cap_frac": 0.010},
}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cases = {c["name"]: c for c in c128.build()}
    adc = np.load(TEMP / "palm6" / "out" / "streams.npz")
    ta, pa = adc["t"], adc["pre"]
    fa = (len(ta) - 1) / (ta[-1] - ta[0])
    sga = s132.segs_adc(ta, pa.sum(1))
    result = {}
    for name, p in SETS.items():
        rec = {"params": p, "rows": []}
        for k in ("hold_1d925c", "in_03225d", "in_102924", "synth_flat17",
                  "synth_creep10", "synth_tap2s"):
            c = cases[k]
            if not c["segs"]:
                continue
            t, V = c["t"], c["V"]
            fps = (len(t) - 1) / (t[-1] - t[0])
            r = obs.run(t, V, obs.default_with(p))
            for j, x in enumerate(fl.evaluate(r["out_tot"], t, c["segs"], fps)):
                rec["rows"].append({"case": k, "seg": j + 1, **{kk: x[kk] for kk in
                                    ("E", "dev", "drop", "over", "drift", "tsettle",
                                     "slope_end")}})
        ra = obs.run(ta, pa, obs.default_with(p))
        for x in s132.eval_adc(ra["out_tot"], ta, sga, fa):
            rec["rows"].append({"case": "ADC_3efae9", "seg": 1, **{kk: x[kk] for kk in
                                ("E", "dev", "drop", "over", "tsettle", "slope_end")}})
        result[name] = rec
    (OUT7 / "141_final_table.json").write_text(json.dumps(result, ensure_ascii=False,
                                                          default=float), encoding="utf-8")
    cases_order = ["hold_1d925c", "in_03225d", "in_102924", "synth_flat17",
                   "synth_creep10", "synth_tap2s", "ADC_3efae9"]
    for name, rec in result.items():
        p = rec["params"]
        print(f"\n### {name}  r_fast={p['r_fast']:g} tau_c_fast_s={p['tau_c_fast_s']:g} "
              f"slow_confirm_s={p['slow_confirm_s']:g} soft_unfreeze_s={p['soft_unfreeze_s']:g} "
              f"slope_cap_frac={p['slope_cap_frac']:g} r_slow_max={p['r_slow_max']:g} "
              f"tau_r_fast_s={p['tau_r_fast_s']:g} tau_r_slow_idle_s={p['tau_r_slow_idle_s']:g}")
        print(f"| 台 | 段 | E | 落点 | 下坠 | 过扣 | 漂移 | 稳定(s) | 末斜率 |")
        print(f"|---|---|---|---|---|---|---|---|---|")
        for r in rec["rows"]:
            print(f"| {r['case']} | {r['seg']} | {r['E']:.3f} | {r['dev']:+.3f} | "
                  f"{r['drop']:.3f} | {r['over']:.3f} | {r.get('drift', float('nan')):+.3f} | "
                  f"{r['tsettle']:.2f} | {r['slope_end']:+.5f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
