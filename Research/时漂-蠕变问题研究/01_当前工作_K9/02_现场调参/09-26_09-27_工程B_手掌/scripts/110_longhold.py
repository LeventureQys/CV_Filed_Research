# -*- coding: utf-8 -*-
"""110_longhold：600 s 长保压合成——尾漂为什么会一直涨。

合成输入：0.85 s 压到 16.5 N，随后以恒定 0.0025 N/s（=2.5 ADC/s 总值，模拟长时间
粘弹性蠕变，10 分钟 +1.5 N ≈ +9%）持续蠕变。对比参数组，量显示的长时间漂移，
并拆分 x1/x2/未开门通道。
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


def synth_long(level_n=16.5, creep_nps=0.0025, dur=600.0, nch=71):
    t = np.arange(0, dur, 1.0 / 100.0)
    y = np.where(t < 1.35, 0.0,
                 np.where(t < 2.2, level_n * (t - 1.35) / 0.85,
                          level_n + creep_nps * np.maximum(t - 2.2, 0.0)))
    V = np.repeat(y[:, None] * 1000.0 / nch, nch, axis=1)
    return t, V


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, V = synth_long()
    tin = V.sum(axis=1) / 1000.0
    E = 16.5
    BEST = {"r_fast": 0.10, "tau_c_fast_s": 10.0, "slow_confirm_s": 2.0,
            "soft_unfreeze_s": 2.0, "slope_cap_frac": 0.002, "r_slow_max": 0.15,
            "tau_r_fast_s": 6.0, "tau_r_slow_idle_s": 0.5}
    cands = [
        ("best", BEST, "上轮最佳 rf.10 cap.002 rsm.15"),
        ("rsm6", {**BEST, "r_slow_max": 0.6}, "rsm=0.6（幅度上限放开）"),
        ("cap4", {**BEST, "slope_cap_frac": 0.004}, "cap=0.004（速率上限放开）"),
        ("rsm6cap4", {**BEST, "r_slow_max": 0.6, "slope_cap_frac": 0.004}, "rsm.6 + cap.004"),
        ("rsm6cap25", {**BEST, "r_slow_max": 0.6, "slope_cap_frac": 0.025}, "rsm.6 + cap.025"),
    ]
    print(f"输入：恒速蠕变 +0.0025 N/s，600 s 共 +1.5 N（+9%）\n")
    print(f"{'参数集':<28s} 显示−E @60s/@180s/@600s   600s漂移   x2末   x2上限(rsm·e)")
    for name, p, label in cands:
        r = obs.run(t, V, obs.default_with(p))
        y = r["out_tot"] / 1000.0
        i60, i180, i600 = (min(int(np.searchsorted(t, x)), len(t) - 1)
                           for x in (60, 180, 600))
        # x2 幅度上限（总值近似：Σ rsm·e_k ≈ rsm·Σe）
        print(f"{label:<28s} {y[i60]-E:+7.3f} {y[i180]-E:+7.3f} {y[i600]-E:+7.3f} N   "
              f"{y[i600]-y[i60]:+7.3f} N   {r['x2'][i600]/1000:.3f} N   "
              f"{p['r_slow_max']*(tin[i600]-r['x2'][i600]/1000-r['x1'][i600]/1000):.3f} N")
    # 细看 best 的 x2 是否撞幅度上限
    r = obs.run(t, V, obs.default_with(BEST))
    y = r["out_tot"] / 1000.0
    print(f"\nbest 逐时轨迹（显示−E、x1、x2，N）：")
    for tt in (10, 60, 120, 240, 360, 480, 599):
        i = int(np.searchsorted(t, tt))
        e_total = tin[i] - r["x1"][i] / 1000 - r["x2"][i] / 1000
        print(f"  t={tt:4d}s 显示−E={y[i]-E:+7.3f}  x1={r['x1'][i]/1000:6.3f}  "
              f"x2={r['x2'][i]/1000:6.3f}  x2上限≈{0.15*e_total:6.3f}"
              f"{'  ← 撞上限' if r['x2'][i]/1000 > 0.95*0.15*e_total else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
