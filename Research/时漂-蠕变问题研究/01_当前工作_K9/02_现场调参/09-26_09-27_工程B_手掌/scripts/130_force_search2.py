# -*- coding: utf-8 -*-
"""130_force_search2：力值模式细网格（两个真实施力会话同时评，避免过拟合单会话）。

保存 temp/palm7/out/130_fine.json：每组参数的
  hold_1d925c 段1/段2 与 in_03225d 段1 的 落点/下坠/过扣/漂移/末斜率/稳定，
外加合成恒压 17 N（纯阶跃、零蠕变）的 落点/下坠 作为「失真代价」。
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

RF = [0.02, 0.03, 0.05, 0.08, 0.12]
TC1 = [0.5, 1.0, 2.0, 4.0, 8.0]
CAP = [0.005, 0.006, 0.008, 0.010, 0.012, 0.015]
CONF = [0.0, 2.0, 3.0]
RSM = [0.10, 0.35]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cases = {c["name"]: c for c in c128.build()}
    keys = ("hold_1d925c", "in_03225d", "synth_flat17")
    meta = {}
    for k in keys:
        t = cases[k]["t"]
        meta[k] = {"fps": (len(t) - 1) / (t[-1] - t[0])}
    out = []
    combos = list(itertools.product(RF, TC1, CAP, CONF, RSM))
    print(f"细网格 {len(combos)} 组 × {len(keys)} 台", flush=True)
    t0 = time.perf_counter()
    for i, (rf, tc1, cap, conf, rsm) in enumerate(combos):
        p = {**fl.REC8, "r_fast": rf, "tau_c_fast_s": tc1, "slope_cap_frac": cap,
             "slow_confirm_s": conf, "r_slow_max": rsm}
        rec = {"rf": rf, "tc1": tc1, "cap": cap, "conf": conf, "rsm": rsm, "c": {}}
        for k in keys:
            case = cases[k]
            t, V = case["t"], case["V"]
            r = obs.run(t, V, obs.default_with(p))
            rows = fl.evaluate(r["out_tot"], t, case["segs"], meta[k]["fps"])
            rec["c"][k] = [{kk: x[kk] for kk in ("dev", "drop", "over", "drift",
                                                 "slope_end", "tsettle", "E")} for x in rows]
        out.append(rec)
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(combos)}  {time.perf_counter()-t0:.0f}s", flush=True)
    (OUT / "130_fine.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"完成 {time.perf_counter()-t0:.0f}s → 130_fine.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
