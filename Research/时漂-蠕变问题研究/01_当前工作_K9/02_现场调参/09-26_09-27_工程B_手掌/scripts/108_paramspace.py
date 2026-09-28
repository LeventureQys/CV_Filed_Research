# -*- coding: utf-8 -*-
"""108_paramspace：纯参数结构研究（抛开任何"推荐值"）。

双测试台：
- real：152928 真实输入（快相 7% + 蠕变 12%，E=16481）
- flat：恒压合成（压到 17 N 后恒定，无快相无蠕变）——测参数自身的下坠上限

产出：OFAT（其余参数取中性基座）+ rf×cap 全网格 → 可达边界（帕累托）。
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
s93 = import_module("93_palm4_sweep")
s100 = import_module("100_palm5_synth")
OUT = TEMP / "palm6" / "out"
E1 = 16481.0

RF = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25, 0.40, 0.50]
TC1 = [0.5, 2, 5, 10, 20, 40, 80]
CONF = [0.0, 0.5, 1, 2, 5, 10, 20, 60]
SOFT = [0.1, 0.5, 1, 2, 8, 30, 60]
CAP = [0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, 0.025, 0.040, 0.050]
RSM = [0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.40, 0.60]
TR1 = [0.05, 0.5, 2, 6, 20, 60, 300]
TRSI = [0.5, 2, 5, 20, 60, 300]

NEUTRAL = {"r_fast": 0.05, "tau_c_fast_s": 10.0, "slow_confirm_s": 2.0,
           "soft_unfreeze_s": 2.0, "slope_cap_frac": 0.005, "r_slow_max": 0.15,
           "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}


def benches(p):
    z = np.load(OUT / "streams.npz")
    t, pre = z["t"], z["pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    r = obs.run(t, pre, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"], fps)
    m = t >= 1.6
    end = float(y[m][-1])
    mm = t[m] >= t[m][-1] - 5.0
    real = {"dev": (end - E1) / 1000.0, "drop": float(y[m].max() - end) / 1000.0,
            "over": float(max(0.0, E1 - y[m].min())) / 1000.0,
            "slope": float(np.polyfit(t[m][mm], y[m][mm], 1)[0]) / 1000.0}
    t2 = np.arange(0, 30.0, 1.0 / 100.7)
    Vf = np.repeat(s100.synth(t2, 17.0)[:, None] * 1000.0 / 71.0, 71, axis=1)
    r2 = obs.run(t2, Vf, obs.default_with(p))
    y2 = s93.medfilt1s(r2["out_tot"] / 1000.0, 100.7)
    m2 = t2 >= 2.2
    flat = {"drop": float(y2[m2].max() - y2[m2][-1]),
            "dev": float(y2[m2][-1] - 17.0)}
    return real, flat


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rows = []

    def rec(p, tag):
        real, flat = benches(p)
        rows.append({"tag": tag, "params": dict(p), "real": real, "flat": flat})
        return real, flat

    print("== OFAT：单旋钮扫描（其余=中性基座 rf.05 τc1=10 conf2 soft2 cap.005 rsm.15）==")
    print(f"{'旋钮':<6s} {'值':>6s} | real: 落点N 下坠N 末斜率N/s | flat: 下坠N 落点N")
    for key, vals in (("r_fast", RF), ("tau_c_fast_s", TC1), ("slow_confirm_s", CONF),
                      ("soft_unfreeze_s", SOFT), ("slope_cap_frac", CAP),
                      ("r_slow_max", RSM), ("tau_r_fast_s", TR1),
                      ("tau_r_slow_idle_s", TRSI)):
        for v in vals:
            real, flat = rec({**NEUTRAL, key: v}, f"ofat_{key}")
            print(f"{key:<16s} {v:>6g} | {real['dev']:+8.3f} {real['drop']:7.3f} "
                  f"{real['slope']:+9.4f} | {flat['drop']:7.3f} {flat['dev']:+8.3f}")

    print("\n== rf × cap 全网格（其余=中性基座）==")
    for rf, cap in itertools.product(RF, CAP):
        rec({**NEUTRAL, "r_fast": rf, "slope_cap_frac": cap}, "grid")

    (TEMP / "palm6" / "out" / "108_paramspace.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    g = [r for r in rows if r["tag"] == "grid"]
    print(f"\n网格 {len(g)} 点的可达边界（real 台）：")
    print("  下坠≤0.2N 且 过减≤0.05N 的点里，|落点| 最小前 8：")
    ok = [r for r in g if r["real"]["drop"] <= 0.2 and r["real"]["over"] <= 0.05]
    for r in sorted(ok, key=lambda r: abs(r["real"]["dev"]))[:8]:
        print(f"    rf={r['params']['r_fast']:g} cap={r['params']['slope_cap_frac']:g}: "
              f"落点={r['real']['dev']:+.3f} 下坠={r['real']['drop']:.3f} "
              f"flat下坠={r['flat']['drop']:.3f}")
    if not ok:
        print("    （无）")
    print("  全网格 落点 范围：["
          f"{min(r['real']['dev'] for r in g):+.3f}, "
          f"{max(r['real']['dev'] for r in g):+.3f}] N；"
          f"下坠范围：[{min(r['real']['drop'] for r in g):.3f}, "
          f"{max(r['real']['drop'] for r in g):.3f}] N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
