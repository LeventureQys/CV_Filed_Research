# -*- coding: utf-8 -*-
"""K8 vs K7 对比：平台残余漂移率 + 循环偏差/空载偏差。

用法：python k8_probe.py <session_name> [<session_name> ...]
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


def cycle_idle(t, sums, out_sums):
    hi = np.percentile(sums[sums > 3], 80) if (sums > 3).any() else 1.0
    on = sums > 0.6 * hi
    idle = sums < 0.5 * hi
    segs, in_on = [], False
    for i in range(len(t)):
        if on[i] and not in_on:
            start, in_on = i, True
        elif not on[i] and in_on:
            segs.append((start, i))
            in_on = False
    if in_on:
        segs.append((start, len(t)))
    segs = [s for s in segs if s[1] - s[0] >= 20]
    bias = np.array([np.mean(out_sums[a:b] - sums[a:b]) for a, b in segs])
    ib = float(np.mean((out_sums - sums)[idle])) if idle.any() else 0.0
    return bias, ib


for sid in (sys.argv[1:] or ["20260920_175017_single_device_efd432"]):
    hits = glob.glob(os.path.join(ROOT, "**", sid), recursive=True)
    if not hits:
        print(f"===== {sid} 未找到")
        continue
    t, pre = ns["load_csv"](os.path.join(hits[0], "device_001_pre_seg0.csv"))
    pre_sums = pre.sum(axis=1)
    print(f"===== {sid}")
    print(f"  pre 平台蠕变率: " + ", ".join(f"{r:+.2f}" for r in seg_rates(t, pre_sums)[:4]))
    for tag, p in (("K7", ns["P7"]), ("K8", ns["P8"])):
        sums = ns["observer"](t, pre, p, k7=True).sum(axis=1)
        rates = ", ".join(f"{r:+.2f}" for r in seg_rates(t, sums)[:4])
        bias, idle = cycle_idle(t, pre_sums, sums)
        seq = ", ".join(f"{b:+.0f}" for b in bias[:6])
        growth = f"{bias[-1] - bias[0]:+.0f}" if len(bias) > 1 else "n/a"
        print(f"  {tag} 残余漂移: [{rates}]  循环偏差[{seq}] growth={growth}  空载={idle:+.2f}")
