# -*- coding: utf-8 -*-
"""104_palm6_tune：联合选参——本会话（蠕变12%/快相7%）压平上浮 + 恒压合成测下坠上限。"""
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
s100 = import_module("100_palm5_synth")
OUT = TEMP / "palm6" / "out"
T0, E1 = 1.05, 16481.0


def ev_real(p):
    z = np.load(OUT / "streams.npz")
    t, pre, tin = z["t"], z["pre"], z["tot_pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    r = obs.run(t, pre, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"], fps)
    m = t >= T0 + 0.6
    end = float(y[m][-1])
    mm = t[m] >= t[m][-1] - 5.0
    return {"drop": float(y[m].max() - end), "dev": end - E1,
            "slope_end": float(np.polyfit(t[m][mm], y[m][mm], 1)[0]),
            "over": float(max(0.0, E1 - y[m].min())), "x2": float(r["x2"][-1]),
            "applied": float(r["applied"][-1])}


def ev_flat(p):
    t = np.arange(0, 30.0, 1.0 / 100.7)
    ytot = s100.synth(t, level_n=17.0)
    V = np.repeat(ytot[:, None] * 1000.0 / 71.0, 71, axis=1)
    r = obs.run(t, V, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"] / 1000.0, 100.7)
    m = t >= 2.2
    return {"drop": float(y[m].max() - y[m][-1])}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5,
           "soft_unfreeze_s": 1.5, "slope_cap_frac": 0.025, "r_slow_max": 0.03,
           "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}
    cands = [("cur", CUR, "现参数")]
    for rf, conf, cap, rsm in itertools.product(
            [0.04, 0.06], [2.0, 3.0], [0.004, 0.006], [0.12, 0.20]):
        cands.append((f"d_{rf}_{conf:g}_{cap}_{rsm}",
                      {**CUR, "r_fast": rf, "slow_confirm_s": conf,
                       "slope_cap_frac": cap, "r_slow_max": rsm,
                       "tau_c_fast_s": 2.0},
                      f"rf={rf:g} τc1=2 conf={conf:g} cap={cap:g} rsm={rsm:g}"))
    print(f"{'参数集':<38s} | 本会话: 落点(N) 末斜率(N/s) 下坠(N) 过减(N) x2末 | 恒压: 下坠(N)")
    rows = []
    for name, p, label in cands:
        a, f = ev_real(p), ev_flat(p)
        rows.append({"name": name, "label": label, "params": p, "real": a, "flat": f})
        print(f"{label:<38s} | {a['dev']*0.001:+8.3f} {a['slope_end']*0.001:+9.4f} "
              f"{a['drop']*0.001:7.3f} {a['over']*0.001:6.3f} {a['x2']:7.0f} | {f['drop']:7.3f}")
    (OUT / "104_tune.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\n== 按本会话 |落点|+|末斜率|*1000 排序 ==")
    for r in sorted(rows, key=lambda r: abs(r["real"]["dev"]) + abs(r["real"]["slope_end"]) * 1000)[:8]:
        a, f = r["real"], r["flat"]
        print(f"  {r['label']:<38s} 落点={a['dev']*0.001:+.3f}N 末斜率={a['slope_end']*0.001:+.4f}N/s "
              f"下坠={a['drop']*0.001:.3f}N 恒压下坠={f['drop']:.3f}N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
