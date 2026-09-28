# -*- coding: utf-8 -*-
"""137_force_micro：按界面步长（r_fast 0.01 / τc1 0.5 / cap 0.001）做最后一轮微网格。

只保留可写入设备的取值组合，输出「最大偏离」与全部代价，供最终选档。
"""
from __future__ import annotations

import itertools
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

RF = [0.03, 0.04, 0.05]
TC1 = [1.0, 1.5, 2.0]
CAP = [0.010, 0.011, 0.012]


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
    rows = []
    for rf, tc1, cap in itertools.product(RF, TC1, CAP):
        p = {**fl.REC8, "r_fast": rf, "tau_c_fast_s": tc1, "slow_confirm_s": 0.0,
             "slope_cap_frac": cap}
        rec = {"rf": rf, "tc1": tc1, "cap": cap}
        for k in ("hold_1d925c", "in_03225d", "synth_flat17", "synth_tap2s", "synth_creep10"):
            c = cases[k]
            t, V = c["t"], c["V"]
            fps = (len(t) - 1) / (t[-1] - t[0])
            rr = obs.run(t, V, obs.default_with(p))
            rec[k] = fl.evaluate(rr["out_tot"], t, c["segs"], fps)
        ra = obs.run(ta, pa, obs.default_with(p))
        rec["adc"] = s132.eval_adc(ra["out_tot"], ta, sga, fa)
        real = rec["hold_1d925c"] + rec["in_03225d"]
        rec["_dev"] = max(max(abs(x["dev"]), x["drop"]) for x in real)
        rec["_dip"] = max(x["over"] for x in real)
        rec["_up"] = max(abs(x["dev"]) for x in real)
        rec["_drop"] = max(x["drop"] for x in real)
        rec["_slope"] = max(abs(x["slope_end"]) for x in real)
        rows.append(rec)
    hdr = (f"{'rf':>5s} {'τc1':>4s} {'cap':>6s} | {'最大偏离':>7s} {'落点':>6s} {'下坠':>6s} "
           f"{'过扣':>6s} {'末斜':>7s} | {'段1上/下':>12s} {'段2上/下':>12s} {'3上/下':>12s} | "
           f"{'2s短按':>7s} {'恒压':>7s} | {'ADC落点':>8s}")
    print(hdr)
    for r in sorted(rows, key=lambda r: r["_dev"]):
        a1, a2 = r["hold_1d925c"]
        a3 = r["in_03225d"][0]
        print(f"{r['rf']:5g} {r['tc1']:4g} {r['cap']:6g} | {r['_dev']:7.3f} {r['_up']:6.3f} "
              f"{r['_drop']:6.3f} {r['_dip']:6.3f} {r['_slope']:7.5f} | "
              f"{a1['dev']:+5.2f}/{a1['drop']:5.2f} {a2['dev']:+5.2f}/{a2['drop']:5.2f} "
              f"{a3['dev']:+5.2f}/{a3['drop']:5.2f} | "
              f"{r['synth_tap2s'][0]['dev']:+7.3f} {r['synth_flat17'][0]['dev']:+7.3f} | "
              f"{r['adc'][0]['dev']:+8.0f}")
    (OUT7 / "137_micro.json").write_text(json.dumps(rows, ensure_ascii=False, default=float),
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
