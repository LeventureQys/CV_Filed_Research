# -*- coding: utf-8 -*-
"""138_final_check：最终候选档的完整验证表（力值 3 段 + 合成 3 台 + ADC 会话 + 松弛会话）。

所有取值都在界面步长上（r_fast 0.01 / τc1 0.5 / conf 0.5 / soft 0.5 / cap 0.001 / rsm 0.01）。
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

CANDS = {
    "现役(设备现有)": {"r_fast": 0.03, "tau_c_fast_s": 40.0, "slow_confirm_s": 2.0,
                       "soft_unfreeze_s": 2.0, "slope_cap_frac": 0.011, "r_slow_max": 0.35,
                       "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5},
    "R1 推荐 rf.04 τc1 2 conf0 cap.011": {"r_fast": 0.04, "tau_c_fast_s": 2.0,
                                          "slow_confirm_s": 0.0, "slope_cap_frac": 0.011},
    "R1b 同上但 conf2（少改一项）": {"r_fast": 0.04, "tau_c_fast_s": 2.0,
                                    "slow_confirm_s": 2.0, "slope_cap_frac": 0.011},
    "R2 更稳 rf.05 τc1 2 conf0 cap.010": {"r_fast": 0.05, "tau_c_fast_s": 2.0,
                                          "slow_confirm_s": 0.0, "slope_cap_frac": 0.010},
    "R3 更准 rf.04 τc1 1 conf0 cap.011": {"r_fast": 0.04, "tau_c_fast_s": 1.0,
                                          "slow_confirm_s": 0.0, "slope_cap_frac": 0.011},
    "对照 ADC 专用档 rf.06 cap.005 rsm.15": {"r_fast": 0.06, "tau_c_fast_s": 2.0,
                                             "slow_confirm_s": 2.0, "soft_unfreeze_s": 1.5,
                                             "slope_cap_frac": 0.005, "r_slow_max": 0.15},
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
    out = {}
    for name, p in CANDS.items():
        pp = {**fl.REC8, **p}
        rec = {"params": pp, "force": {}, "adc": None}
        for k in ("hold_1d925c", "in_03225d", "in_102924", "synth_flat17",
                  "synth_creep10", "synth_tap2s"):
            c = cases[k]
            if not c["segs"]:
                continue
            t, V = c["t"], c["V"]
            fps = (len(t) - 1) / (t[-1] - t[0])
            rr = obs.run(t, V, obs.default_with(pp))
            rec["force"][k] = fl.evaluate(rr["out_tot"], t, c["segs"], fps)
        ra = obs.run(ta, pa, obs.default_with(pp))
        rec["adc"] = s132.eval_adc(ra["out_tot"], ta, sga, fa)
        out[name] = rec
    (OUT7 / "138_final_check.json").write_text(
        json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")

    keys = ["hold_1d925c", "in_03225d", "in_102924", "synth_flat17", "synth_creep10",
            "synth_tap2s"]
    for name, rec in out.items():
        print(f"\n===== {name} =====")
        pp = rec["params"]
        print("   " + " ".join(f"{k}={pp[k]:g}" for k in
                               ("r_fast", "tau_c_fast_s", "slow_confirm_s", "soft_unfreeze_s",
                                "slope_cap_frac", "r_slow_max", "tau_r_fast_s",
                                "tau_r_slow_idle_s")))
        for k in keys:
            for j, x in enumerate(rec["force"].get(k, [])):
                print(f"   {k:<14s} 段{j+1} E={x['E']:8.3f} 落点={x['dev']:+7.3f} "
                      f"下坠={x['drop']:6.3f} 过扣={x['over']:6.3f} 漂移={x['drift']:+7.3f} "
                      f"稳定={x['tsettle']:5.2f}s 末斜率={x['slope_end']:+7.4f}")
        for x in rec["adc"]:
            print(f"   ADC 3efae9      E={x['E']:8.0f} 过减={x['over']:6.0f} "
                  f"落点={x['dev']:+7.0f} 下坠={x['drop']:6.0f} 稳定={x['tsettle']:5.2f}s "
                  f"末斜率={x['slope_end']:+7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
