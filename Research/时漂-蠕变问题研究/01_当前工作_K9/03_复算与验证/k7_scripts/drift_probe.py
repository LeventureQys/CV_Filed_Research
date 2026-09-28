# -*- coding: utf-8 -*-
"""K7 平台期残余漂移诊断：x1 段之后显示是否缓慢上漂。

指标：对每个保压平台（pre 上升沿后 ~3s 到下降沿），取 x1 段结束（沿后 15s）之后的部分，
线性拟合输出斜率（ADC/s）与漂移幅度（末−初）。斜率>0 即用户观察到的"缓慢上漂"。
变体：x1 更激进的几种口径。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
src = open(os.path.join(HERE, "k7_eval.py"), encoding="utf-8").read().split("def main()")[0]
exec(compile(src, "k7_eval", "exec"), ns)

ROOT = (
    r"D:\workshop\Processing\multi-device-cascade-host-cpp\Document\Update\Dev-Version"
    r"\v2.7 - 抗蠕变补偿算法\算法数据&原始数据\working"
)


def plateau_drift(t, sums, min_hold=20.0, skip=15.0):
    """每个平台（上升沿后 skip 秒起）的输出漂移率与幅度；只统计保压时长够的段。"""
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
        # x1 段之后：从 a 对应时间 + skip 起拟合
        m = (t >= ta + skip) & (t <= tb)
        if m.sum() < 10:
            continue
        x, y = t[m], sums[m]
        rate = np.polyfit(x, y, 1)[0]
        out.append((rate, y[-1] - y[0], tb - ta))
    return out


def pre_rate(t, pre_sums, min_hold=20.0, skip=15.0):
    on = pre_sums > 0.5 * np.percentile(pre_sums[pre_sums > 3], 80)
    segs, in_on = [], False
    for i in range(len(t)):
        if on[i] and not in_on:
            start, in_on = i, True
        elif not on[i] and in_on:
            segs.append((start, i))
            in_on = False
    if in_on:
        segs.append((start, len(t)))
    rates = []
    for a, b in segs:
        ta, tb = t[a], t[min(b, len(t) - 1)]
        if tb - ta < min_hold + skip:
            continue
        m = (t >= ta + skip) & (t <= tb)
        if m.sum() < 10:
            continue
        rates.append(np.polyfit(t[m], pre_sums[m], 1)[0])
    return rates


def run(session_name, overrides):
    import glob
    hits = glob.glob(os.path.join(ROOT, "**", session_name), recursive=True)
    if not hits:
        return None
    t, pre = ns["load_csv"](os.path.join(hits[0], "device_001_pre_seg0.csv"))
    pre_sums = pre.sum(axis=1)
    p = dict(ns["P7"], **overrides)
    sums = ns["observer"](t, pre, p, k7=True).sum(axis=1)
    segs = plateau_drift(t, sums)
    prates = pre_rate(t, pre_sums)
    return segs, prates


CONFIGS = {
    "K7 当前": {},
    "zero_confirm 2s": dict(zero_confirm_s=2.0),
    "zero_confirm 2s + r1 0.15": dict(zero_confirm_s=2.0, r_fast=0.15),
    "r1 0.15 + eps 6": dict(r_fast=0.15, hold_eps=6.0),
}

sessions = sys.argv[1:] or ["20260920_175017_single_device_efd432",
                            "20260919_193320_single_device_3f32c5",
                            "20260919_192141_single_device_73032d"]
for sid in sessions:
    print(f"===== {sid}")
    first = True
    for name, over in CONFIGS.items():
        res = run(sid, over)
        if res is None:
            print(f"  {name:28s} 会话未找到")
            continue
        segs, prates = res
        if not segs:
            print(f"  {name:28s} 无够长平台")
            continue
        if first:
            print(f"  {'[pre 蠕变率]':28s} " + ", ".join(f"{r:+.2f}" for r in prates[:5]))
            first = False
        rates = ", ".join(f"{r:+.2f}" for r, _, _ in segs[:5])
        drifts = ", ".join(f"{d:+.0f}" for _, d, _ in segs[:5])
        print(f"  {name:28s} 漂移率(ADC/s) [{rates}]  漂移幅 [{drifts}]")
