# -*- coding: utf-8 -*-
"""旁路死区宽度扫描：release_frac 收窄能否让轻载进入补偿且不破坏空载保护。

用法：python deadband_sweep.py <session_name> ...
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
src = open(os.path.join(HERE, "k8_eval.py"), encoding="utf-8").read().split("def main()")[0]
exec(compile(src, "k8_eval", "exec"), ns)

ROOT = (
    r"D:\workshop\Processing\multi-device-cascade-host-cpp\Document\Update\Dev-Version"
    r"\v2.7 - 抗蠕变补偿算法\算法数据&原始数据\working"
)


def seg_rates(t, sums, min_hold=20.0, skip=15.0):
    on = sums > 0.5 * np.percentile(sums[sums > 3], 80)
    segs, in_on = [], False
    for i in range(len(t)):
        if on[i] and not in_on:
            start, in_on = i, True
        elif not on[i] and in_on:
            segs.append((start, i))
            in_on = False
    if in_on:
        segs.append((start, len(t)))
    out = []
    for a, b in segs:
        ta, tb = t[a], t[min(b, len(t) - 1)]
        if tb - ta < min_hold + skip:
            continue
        m = (t >= ta + skip) & (t <= tb)
        if m.sum() < 10:
            continue
        out.append(np.polyfit(t[m], sums[m], 1)[0])
    return out


def idle_bias(t, sums, out_sums):
    hi = np.percentile(sums[sums > 3], 80) if (sums > 3).any() else 1.0
    idle = sums < 0.5 * hi
    return float(np.mean((out_sums - sums)[idle])) if idle.any() else 0.0


CONFIGS = {
    "release 1.15 (K7)": dict(bypass_release_frac=1.15),
    "release 1.10": dict(bypass_release_frac=1.10),
    "release 1.05": dict(bypass_release_frac=1.05),
    "release 1.02": dict(bypass_release_frac=1.02),
    "release 1.05 + zc2": dict(bypass_release_frac=1.05, zero_confirm_s=2.0,
                               discharge_confirm_s=2.0),
}

for sid in (sys.argv[1:] or ["20260920_175017_single_device_efd432"]):
    hits = glob.glob(os.path.join(ROOT, "**", sid), recursive=True)
    if not hits:
        continue
    t, pre = ns["load_csv"](os.path.join(hits[0], "device_001_pre_seg0.csv"))
    pre_sums = pre.sum(axis=1)
    print(f"===== {sid}")
    print(f"  pre 平台蠕变率: " + ", ".join(f"{r:+.2f}" for r in seg_rates(t, pre_sums)[:4]))
    for name, over in CONFIGS.items():
        p = dict(ns["P7"], **over)
        sums = ns["observer"](t, pre, p, k7=True).sum(axis=1)
        rates = ", ".join(f"{r:+.2f}" for r in seg_rates(t, sums)[:4])
        print(f"  {name:20s} 残余漂移 [{rates}]  空载={idle_bias(t, pre_sums, sums):+.2f}")
