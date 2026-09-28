# -*- coding: utf-8 -*-
"""只读实验（三）：定位"振荡把基线搞坏"的那个内部环节。

已知：同一份输入，
  · 只改时间轴 ⇒ 输出零差（不是时间戳问题）
  · 逐帧原始（4 帧/包不同值）vs 包内平均 ⇒ 偏移差可达 1400 ADC
⇒ 说明"包内那 4 个不同值"被算法当成了真实变化。

本实验按 v6 的 **检测器公式** 复算，看它在哪里把振荡误判为负载变化：
  lv_now  = mean(total, [t−0.20, t])
  lv_ref  = mean(total, [t−0.65, t−0.35])
  d       = lv_now − lv_ref
  thr_d   = max(5σ_d, 0.05·|lv_ref|, 0.01·max_tot)
  命中     = |d| 连续 ≥3 帧 且 armed
再统计：命中率、以及"命中时 d 与输入自身的包内极差"的关系。
"""
import os
import sys
from itertools import groupby

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

DS = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                  "20260919_134056_single_device_f40a1b")
TAU, LVL = 0.30, 10.0
FAST, GAP, LAG = 0.20, 0.15, 0.30
K, REL, ABSF, PERSIST = 5.0, 0.05, 0.01, 3


def detect(el, tot):
    """按 v6 公式逐帧复算检测器（时间窗用给定 elapsed）。"""
    n = len(el)
    ts_s = np.empty(n)
    lvl = np.empty(n)
    ts_s[0] = lvl[0] = tot[0]
    last = el[0]
    for i in range(1, n):
        dt = el[i] - last
        last = el[i]
        dt = min(dt, 0.1) if dt > 0 else 0.0
        if dt > 0:
            ts_s[i] = ts_s[i - 1] + (dt / TAU) * (tot[i] - ts_s[i - 1])
            lvl[i] = lvl[i - 1] + (dt / LVL) * (ts_s[i] - lvl[i - 1])
        else:
            ts_s[i], lvl[i] = ts_s[i - 1], lvl[i - 1]
    d = np.zeros(n)
    for i in range(n):
        a = (el > el[i] - FAST) & (el <= el[i])
        b = (el > el[i] - FAST - GAP - LAG) & (el <= el[i] - FAST - GAP)
        if a.any() and b.any():
            d[i] = tot[a].mean() - tot[b].mean()
    sig = np.zeros(n)
    run = 0
    hit = np.zeros(n, dtype=bool)
    for i in range(n):
        h = max(0, i - 1024)
        w = d[h:i + 1]
        if len(w) > 40:
            sig[i] = 1.4826 * np.median(np.abs(w - np.median(w)))
        thr = max(K * sig[i], REL * abs(tot[max(0, i - 1)]), ABSF * tot[:i + 1].max())
        raw = abs(d[i]) > thr
        run = run + 1 if raw else 0
        hit[i] = run >= PERSIST
    return d, sig, hit


def main():
    pre = L.load_stream(DS, "device_001_pre_seg0.csv")
    el, V = pre["el"], pre["V"]
    tot = V.sum(1)
    n = len(el)
    keys = np.round(el, 3)
    bounds = np.flatnonzero(np.concatenate([[True], np.diff(keys) != 0]))
    ends = np.append(bounds[1:], n)

    # 包内极差（总量）
    pack_rng = np.zeros(n)
    for a, b in zip(bounds, ends):
        pack_rng[a:b] = tot[a:b].max() - tot[a:b].min()

    d, sig, hit = detect(el, tot)
    print("=== 检测器在整段上的行为（逐帧原始口径）===")
    print("  d 的分布：中位=%.0f  p90|d|=%.0f  max|d|=%.0f ADC"
          % (np.median(d), np.percentile(np.abs(d), 90), np.abs(d).max()))
    print("  σ_d 中位=%.1f  max=%.1f ADC  ⇒ 5σ_d 门限中位=%.0f"
          % (np.median(sig), sig.max(), 5 * np.median(sig)))
    print("  命中帧数=%d / %d（%.1f%%）" % (int(hit.sum()), n, 100 * hit.mean()))

    thr_rel = 0.05 * tot  # 相对门（主导项）
    print("  相对门 0.05·|lv_ref| ≈ 0.05·总量：中位=%.0f ADC" % np.median(thr_rel))

    print("\n=== 按区间：命中率 vs 输入抖动强度 ===")
    print("%-16s %10s %12s %12s %12s" %
          ("区间", "命中帧%", "输入std", "包内极差中位", "|d|中位"))
    for a, b in ((5, 8), (27, 36), (55, 58), (69.7, 71.7), (72, 74),
                 (79.6, 89.6), (108, 118), (120, 125)):
        m = (el >= a) & (el < b)
        if not m.any():
            continue
        pr = pack_rng[m]
        print("%-16s %9.1f%% %12.0f %12.0f %12.0f" %
              ("%g~%gs" % (a, b), 100 * hit[m].mean(), tot[m].std(),
               np.median(pr), np.median(np.abs(d[m]))))

    print("\n=== 关键：d 是不是被包内抖动直接喂出来的？ ===")
    m = (el >= 69.7) & (el < 71.7)
    print("  振荡段：|d| 与 包内极差 的相关系数 = %+.3f"
          % np.corrcoef(np.abs(d[m]), pack_rng[m])[0, 1])
    mq = (el >= 108) & (el < 118)
    print("  安静段：|d| 中位=%.0f（包内极差中位=%.0f）⇒ d 基本为 0"
          % (np.median(np.abs(d[mq])), np.median(pack_rng[mq])))
    mo = (el >= 2) & (el < 126)
    hi = pack_rng[mo] > 1000
    lo = pack_rng[mo] <= 100
    print("  全录分组：包内极差>1000 的帧命中率=%.1f%%（n=%d）；包内极差≤100 的帧命中率=%.1f%%（n=%d）"
          % (100 * hit[mo][hi].mean(), int(hi.sum()),
             100 * hit[mo][lo].mean(), int(lo.sum())))


if __name__ == "__main__":
    main()
