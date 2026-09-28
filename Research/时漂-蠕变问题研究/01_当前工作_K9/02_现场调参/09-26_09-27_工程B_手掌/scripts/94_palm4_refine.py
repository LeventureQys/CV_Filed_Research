# -*- coding: utf-8 -*-
"""94_palm4_refine：细化 cap/rsm 压末段上漂；并定位过减谷值时刻。"""
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
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm4" / "out"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, pre, tin = z["t"], z["pre"], z["tot_pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    REC = s93.REC

    cands = [("rec", REC, "录制参数 rf.3")]
    for rf, tc1, cap, rsm in itertools.product(
            [0.08, 0.10], [8.0, 15.0], [0.006, 0.008, 0.012, 0.020],
            [0.10, 0.20]):
        cands.append((f"b_{rf}_{tc1:g}_{cap}_{rsm}",
                      {**REC, "r_fast": rf, "tau_c_fast_s": tc1,
                       "slope_cap_frac": cap, "r_slow_max": rsm},
                      f"rf={rf:g} τc1={tc1:g} cap={cap:g} rsm={rsm:g}"))
    rows = []
    T0 = s93.T0
    for name, over_p, label in cands:
        r = obs.run(t, pre, obs.default_with(over_p))
        m = s93.evaluate(r, t, tin, fps)
        y = s93.medfilt1s(r["out_tot"], fps)
        m2 = t >= T0 + 0.1
        i_min = int(np.argmin(y[m2]))
        m["t_min"] = float(t[m2][i_min] - T0)
        rows.append({"name": name, "label": label, "params": over_p, "m": m})
        print(f"{label:<36s} 过减={m['over']:6.1f}@{m['t_min']:5.2f}s 落点={m['dev']:+7.1f} "
              f"下坠={m['drop']:6.1f} 稳定={m['tsettle']:4.2f}s 末斜率={m['slope_end']:+5.1f}")
    (OUT / "94_refine.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("\n== 按（过减≤65、下坠≤120、稳定≤0.6s）内 |落点| 最小 ==")
    ok = [r for r in rows if r["m"]["over"] <= 65 and r["m"]["drop"] <= 120
          and r["m"]["tsettle"] <= 0.6]
    for r in sorted(ok, key=lambda r: abs(r["m"]["dev"]))[:10]:
        a = r["m"]
        print(f"  {r['label']:<36s} 过减={a['over']:6.1f} 落点={a['dev']:+7.1f} "
              f"下坠={a['drop']:6.1f} 稳定={a['tsettle']:4.2f}s 末斜率={a['slope_end']:+5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
