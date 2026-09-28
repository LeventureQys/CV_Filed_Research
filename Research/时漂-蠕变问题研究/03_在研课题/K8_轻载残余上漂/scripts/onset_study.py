# -*- coding: utf-8 -*-
"""起步过充 vs 收敛速度的相关性检索。

定义：
  开路过充 = 会话开头 max(total) 超过其后空载基线的倍数（或首峰出现在加载前）；
  收敛速度 = 各加载平台起点到显示进入平台终值 ±5% 带宽的耗时（s）。
用法：python onset_study.py  （跑全部 10 会话）
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


def analyze(sid):
    hits = glob.glob(os.path.join(ROOT, "**", sid), recursive=True)
    if not hits:
        return None
    t, pre = ns["load_csv"](os.path.join(hits[0], "device_001_pre_seg0.csv"))
    pre_s = pre.sum(axis=1)
    hi = np.percentile(pre_s[pre_s > 3], 90) if (pre_s > 3).any() else 1.0
    on = pre_s > 0.5 * hi

    # 开路过充：首 10s 内的峰值 vs 其后的低水位
    head_max = float(pre_s[t < 10].max()) if (t < 10).any() else 0.0
    tail_base = float(np.percentile(pre_s[pre_s < 0.5 * hi], 80)) if (pre_s < 0.5 * hi).any() else 0.0
    overshoot = head_max > max(2.0 * max(tail_base, 1e-6), 0.1 * hi)

    # 收敛速度：每个平台，加载起点(首次过阈)到 out 进入终值±5%带宽的耗时
    k7 = ns["observer"](t, pre, ns["P7"], k7=True).sum(axis=1)
    settle = []
    segs, in_on = [], False
    for i in range(len(t)):
        if on[i] and not in_on:
            start, in_on = i, True
        elif not on[i] and in_on:
            segs.append((start, i))
            in_on = False
    if in_on:
        segs.append((start, len(t)))
    for a, b in segs:
        if b - a < 30:
            continue
        final = np.mean(k7[b - 5:b])
        band = 0.05 * max(abs(final - k7[a]), 1.0) + 0.02 * hi
        below = np.abs(k7[a:b] - final) <= band
        idx = np.argmax(below) if below.any() else -1
        settle.append(float(t[a + idx] - t[a]) if idx >= 0 else float("nan"))
    settle = [s for s in settle if not np.isnan(s)]
    return dict(sid=sid, overshoot=overshoot, head_max=head_max, tail_base=tail_base,
                n=len(settle), settle_med=float(np.median(settle)) if settle else None,
                settle_max=float(np.max(settle)) if settle else None)


rows = []
for sid in sorted({os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(ROOT, "**", "session.json"), recursive=True)}):
    r = analyze(sid)
    if r:
        rows.append(r)

print(f"{'会话':44s} {'过充':>4s} {'首10s峰':>9s} {'低水位':>9s} {'平台数':>4s} {'收敛中位(s)':>10s} {'收敛最慢(s)':>10s}")
for r in rows:
    print(f"{r['sid']:44s} {'有' if r['overshoot'] else '无':>4s} {r['head_max']:9.1f} {r['tail_base']:9.1f} "
          f"{r['n']:4d} {r['settle_med']:10.1f} {r['settle_max']:10.1f}")
