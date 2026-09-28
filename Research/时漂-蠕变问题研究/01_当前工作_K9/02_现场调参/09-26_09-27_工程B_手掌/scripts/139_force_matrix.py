# -*- coding: utf-8 -*-
"""139_force_matrix：rf × τc1 × cap 的小矩阵（步长合规），为最终档位选择给全数据。

重点看 cap 降到 0.008/0.009 时（治下坠）配上 rf .04/.05 + τc1 2（治上漂）能否两头都收。
"""
from __future__ import annotations

import itertools
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
    print(f"{'rf':>5s} {'τc1':>4s} {'cap':>6s} | {'段1上':>6s} {'段1下':>6s} {'段2上':>6s} "
          f"{'段2下':>6s} {'3上':>6s} {'3下':>6s} | {'最大偏离':>7s} {'过扣':>6s} | "
          f"{'2s短按':>7s} {'恒压':>7s} | {'ADC落点':>8s} {'ADC下坠':>7s}")
    for rf, tc1, cap in itertools.product([0.03, 0.04, 0.05], [1.0, 2.0],
                                          [0.008, 0.009, 0.010, 0.011]):
        p = {**fl.REC8, "r_fast": rf, "tau_c_fast_s": tc1, "slow_confirm_s": 2.0,
             "slope_cap_frac": cap}
        rows = []
        for k in ("hold_1d925c", "in_03225d", "synth_flat17", "synth_tap2s"):
            c = cases[k]
            t, V = c["t"], c["V"]
            fps = (len(t) - 1) / (t[-1] - t[0])
            rr = obs.run(t, V, obs.default_with(p))
            rows.append(fl.evaluate(rr["out_tot"], t, c["segs"], fps))
        a1, a2 = rows[0]
        a3 = rows[1][0]
        real = [a1, a2, a3]
        ra = obs.run(ta, pa, obs.default_with(p))
        aa = s132.eval_adc(ra["out_tot"], ta, sga, fa)[0]
        print(f"{rf:5g} {tc1:4g} {cap:6g} | {a1['dev']:+6.2f} {a1['drop']:6.2f} "
              f"{a2['dev']:+6.2f} {a2['drop']:6.2f} {a3['dev']:+6.2f} {a3['drop']:6.2f} | "
              f"{max(max(abs(x['dev']), x['drop']) for x in real):7.3f} "
              f"{max(x['over'] for x in real):6.3f} | {rows[3][0]['dev']:+7.3f} "
              f"{rows[2][0]['dev']:+7.3f} | {aa['dev']:+8.0f} {aa['drop']:7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
