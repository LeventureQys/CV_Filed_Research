# -*- coding: utf-8 -*-
"""111_longhold2：尾漂的两个真实机制——空载门漏通道 + rsm 幅度撞顶。

台A（异质通道）：51 个满载 + 20 个轻触通道（电平仅 2%，落进 idle_frac=5% 空载带），
  全部以同相对速率蠕变 600 s → 检验空载门漏通道的蠕变是否变成显示尾漂。
台B（超长蠕变）：71 通道同载，蠕变持续 1800 s（+4.5 N ≈ 27%）→ 检验 rsm 撞顶时刻。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")


def synth(level_n=16.5, creep_nps=0.0025, dur=600.0, nch=71, light=0, light_frac=0.02):
    t = np.arange(0, dur, 1.0 / 100.0)
    y = np.where(t < 1.35, 0.0,
                 np.where(t < 2.2, level_n * (t - 1.35) / 0.85,
                          level_n + creep_nps * np.maximum(t - 2.2, 0.0)))
    shares = np.full(nch, (1.0 - light * light_frac) / (nch - light))
    if light:
        shares[(nch - light):] = light_frac / light * np.ones(light) * 0 + light_frac / light
        shares[(nch - light):] *= 0 + 1.0  # 20 个轻触通道合计 2% 电平
        tot = shares.sum()
        shares *= 1.0 / tot
    V = (y[:, None] / shares.sum()) * shares[None, :] * 1000.0
    return t, V


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    BEST = {"r_fast": 0.10, "tau_c_fast_s": 10.0, "slow_confirm_s": 2.0,
            "soft_unfreeze_s": 2.0, "slope_cap_frac": 0.002, "r_slow_max": 0.15,
            "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}

    print("台A：51 满载 + 20 轻触（2% 电平）通道，同速蠕变 600 s")
    t, V = synth(light=20)
    for label, p in (("BEST(rsm.15)", BEST),
                     ("rsm=0.6", {**BEST, "r_slow_max": 0.6})):
        r = obs.run(t, V, obs.default_with(p))
        y = r["out_tot"] / 1000.0
        row = " ".join(f"{y[min(int(np.searchsorted(t, x)), len(t)-1)] - 16.5:+7.3f}"
                       for x in (60, 180, 360, 600))
        print(f"  {label:<14s} 显示−E @60/180/360/600s: {row} N")

    print("\n台B：同载 71 通道，蠕变 1800 s（+4.5 N ≈ 27%），看 rsm 撞顶")
    t2, V2 = synth(dur=1800.0)
    for label, p in (("BEST(rsm.15)", BEST),
                     ("rsm=0.3", {**BEST, "r_slow_max": 0.3}),
                     ("rsm=0.6", {**BEST, "r_slow_max": 0.6})):
        r = obs.run(t2, V2, obs.default_with(p))
        y = r["out_tot"] / 1000.0
        x2 = r["x2"] / 1000.0
        row = " ".join(f"{y[min(int(np.searchsorted(t2, x)), len(t2)-1)] - 16.5:+7.3f}"
                       for x in (120, 300, 600, 1200, 1800))
        x2r = " ".join(f"{x2[min(int(np.searchsorted(t2, x)), len(t2)-1)]:6.3f}"
                       for x in (600, 1200, 1800))
        print(f"  {label:<14s} 显示−E @120/300/600/1200/1800s: {row} N   x2@600/1200/1800: {x2r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
