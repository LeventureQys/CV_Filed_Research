# -*- coding: utf-8 -*-
"""136_force_final：按「对理想平直线的最大偏离」选最终档，并给出各候选的代价。

对每个受载段：
  up   = 落点 = 段末显示 − E_pl           （读数最终偏高多少）
  down = 下坠 = 段内峰 − 段末              （读数中途往下掉多少）
  dev  = max(up, down)                    （对「理想平直读数」的最大偏离，越小越好）
  dip  = 过扣 = E_pl − 段内最小            （是否穿到目标电平以下）
代价台：合成 2 s 短按（x1 电平扣除对短按的影响）、合成恒压 17 N（纯阶跃失真）、
        ADC 会话 3efae9（同寄存器在 ADC 显示下的表现）。
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

RF = [0.02, 0.03, 0.05]
TC1 = [1.0, 2.0, 4.0]
CAP = [0.010, 0.012, 0.014]


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
        r = obs.run(cases["hold_1d925c"]["t"], cases["hold_1d925c"]["V"],
                    obs.default_with(p))
        for k in ("hold_1d925c", "in_03225d", "synth_flat17", "synth_tap2s"):
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
        rows.append(rec)

    hdr = (f"{'rf':>5s} {'τc1':>4s} {'cap':>6s} | {'最大偏离':>7s} {'|落点|':>7s} {'下坠':>6s} "
           f"{'过扣':>6s} | {'段1 上/下':>13s} {'段2 上/下':>13s} {'3 上/下':>13s} | "
           f"{'2s短按':>7s} {'恒压':>7s} | {'ADC落点':>8s} {'ADC下坠':>7s}")
    print(hdr)
    for r in sorted(rows, key=lambda r: r["_dev"]):
        a1, a2 = r["hold_1d925c"]
        a3 = r["in_03225d"][0]
        sf = r["synth_flat17"][0]
        st = r["synth_tap2s"][0]
        aa = r["adc"][0]
        print(f"{r['rf']:5g} {r['tc1']:4g} {r['cap']:6g} | {r['_dev']:7.3f} {r['_up']:7.3f} "
              f"{r['_drop']:6.3f} {r['_dip']:6.3f} | "
              f"{a1['dev']:+6.2f}/{a1['drop']:5.2f} {a2['dev']:+6.2f}/{a2['drop']:5.2f} "
              f"{a3['dev']:+6.2f}/{a3['drop']:5.2f} | "
              f"{st['dev']:+7.3f} {sf['dev']:+7.3f} | {aa['dev']:+8.0f} {aa['drop']:7.0f}")
    (OUT7 / "136_final.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_") or True} for r in rows],
        ensure_ascii=False, default=float), encoding="utf-8")

    print("\n== 现役对照（cap.011 rf.03 τc1 40 conf2 rsm.35）==")
    print(hdr)
    p = fl.REC8
    rec = {"rf": 0.03, "tc1": 40.0, "cap": 0.011}
    for k in ("hold_1d925c", "in_03225d", "synth_flat17", "synth_tap2s"):
        c = cases[k]
        t, V = c["t"], c["V"]
        fps = (len(t) - 1) / (t[-1] - t[0])
        rr = obs.run(t, V, obs.default_with(p))
        rec[k] = fl.evaluate(rr["out_tot"], t, c["segs"], fps)
    ra = obs.run(ta, pa, obs.default_with(p))
    rec["adc"] = s132.eval_adc(ra["out_tot"], ta, sga, fa)
    real = rec["hold_1d925c"] + rec["in_03225d"]
    a1, a2 = rec["hold_1d925c"]
    a3 = rec["in_03225d"][0]
    print(f"{rec['rf']:5g} {rec['tc1']:4g} {rec['cap']:6g} | "
          f"{max(max(abs(x['dev']), x['drop']) for x in real):7.3f} "
          f"{max(abs(x['dev']) for x in real):7.3f} "
          f"{max(x['drop'] for x in real):6.3f} {max(x['over'] for x in real):6.3f} | "
          f"{a1['dev']:+6.2f}/{a1['drop']:5.2f} {a2['dev']:+6.2f}/{a2['drop']:5.2f} "
          f"{a3['dev']:+6.2f}/{a3['drop']:5.2f} | "
          f"{rec['synth_tap2s'][0]['dev']:+7.3f} {rec['synth_flat17'][0]['dev']:+7.3f} | "
          f"{rec['adc'][0]['dev']:+8.0f} {rec['adc'][0]['drop']:7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
