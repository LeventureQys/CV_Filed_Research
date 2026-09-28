# -*- coding: utf-8 -*-
"""129_palm8_tune2：161747 长保压——轨迹分解、rf/cap 细扫、恒压合成代价、饱和检查。"""
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
OUT = TEMP / "palm8" / "out"
E1 = 16502.0
T0 = 1.2

CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
       "tau_r_slow_idle_s": 0.5}
REC = {**CUR, "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0,
       "slope_cap_frac": 0.005, "r_slow_max": 0.15}


def ev(p):
    z = np.load(OUT / "streams.npz")
    t, V = z["t"], z["V"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    r = obs.run(t, V, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"], fps)
    m = t >= T0 + 0.6
    end = float(y[m][-1])
    mm = t[m] >= t[m][-1] - 5.0
    # 平台漂移：显示 10s 处 vs 末
    i10 = int(np.searchsorted(t, T0 + 10))
    return {"dev": end - E1, "slope": float(np.polyfit(t[m][mm], y[m][mm], 1)[0]),
            "drop": float(y[m].max() - end), "over": float(max(0.0, E1 - y[m].min())),
            "climb": float(end - y[i10]), "x2": float(r["x2"][-1])}


def ev_flat(p, secs=60.0):
    t = np.arange(0, secs, 1.0 / 100.7)
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
    # 轨迹表
    z = np.load(OUT / "streams.npz")
    t, V = z["t"], z["V"]
    tin = V.sum(axis=1)
    rows = []
    print(f"{'t':>6s} {'输入-E':>8s} {'爬升%':>6s}")
    for tt in [2, 5, 10, 30, 60, 120, 240, 360, 419]:
        i = int(np.searchsorted(t, tt))
        if i >= len(t):
            break
        print(f"{t[i]:6.0f} {tin[i]-E1:+8.0f} {(tin[i]-E1)/E1*100:5.1f}%")
        rows.append([float(t[i]), float(tin[i] - E1)])
    # 细扫
    print(f"\n{'参数集':<34s} | 落点(N) 末斜率 爬升(10s→末,N) 下坠(N) 过减(N) x2末 | 恒压60s下坠(N)")
    out = []
    cands = [("cur", CUR), ("rec", REC), ("rec_rsm6", {**REC, "r_slow_max": 0.6})]
    for rf, cap in itertools.product([0.06, 0.075, 0.09], [0.005, 0.01]):
        cands.append((f"rf{rf:g}_cap{cap:g}",
                      {**REC, "r_fast": rf, "slope_cap_frac": cap}))
    for name, p in cands:
        a, f = ev(p), ev_flat(p)
        out.append({"name": name, "params": p, "real": a, "flat": f})
        print(f"{name:<34s} | {a['dev']/1000:+7.3f} {a['slope']:+6.1f} {a['climb']/1000:+8.3f} "
              f"{a['drop']/1000:7.3f} {a['over']/1000:6.3f} {a['x2']:6.0f} | {f['drop']:7.3f}")
    (OUT / "129_tune2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    # rsm 饱和检查（rec 档）
    r = obs.run(t, V, obs.default_with(REC))
    hi = 0.15 * np.maximum(tin - r["x1"], 1.0)
    ratio = r["x2"] / np.maximum(hi, 1e-9)
    m = t > 10
    print(f"\nrec 档 x2/(rsm·e)：t>10s 均值={ratio[m].mean():.3f} p95={np.percentile(ratio[m],95):.3f} "
          f"max={ratio[m].max():.3f}（≈1 即撞幅度顶）")
    np.savez(OUT / "run_rec.npz", out_tot=r["out_tot"], x1=r["x1"], x2=r["x2"],
             applied=r["applied"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
