# -*- coding: utf-8 -*-
"""GLM53 探针 2：定位负载切换瞬间"大跳变"的逐帧来源

输入 results/g_trace_A.csv / g_trace_B.csv（由 g_probe_varying.py 生成）
输出：切换窗内逐帧状态表 + 关键通道 A 捕获对比
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

CASES = [("B", 23.83, "中途切换负载B 真实 step +1314"), ("A", 40.98, "数据A 段内 restep 真实 step +1150")]
COLS = ["ts", "total", "out_total", "level", "fast", "slow", "ts_sm", "A_mean", "n_loaded",
        "g", "creep_total", "b_total", "in_load", "a_captured", "armed", "pend"]

for tag, t0, note in CASES:
    tr = pd.read_csv(os.path.join(RES, f"g_trace_{tag}.csv"))
    w = tr[(tr.ts >= t0 - 1.0) & (tr.ts <= t0 + 6.0)].copy()
    print(f"\n{'='*130}\n[{tag}] {note}  event t≈{t0}s")
    print(f"  {'':2s} pre-step 显示={w[w.ts<t0].out_total.median():>8,.0f}  原始={w[w.ts<t0].total.median():>8,.0f}")
    sub = w[[c for c in COLS if c in w.columns]]
    # 只打印 unique-timestamp 帧，避免 3/4 重复帧淹没
    keep = np.r_[True, np.diff(sub.ts.values) > 1e-6]
    print(sub[keep].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # 逐帧增量定位最大跳变
    u = w[np.r_[True, np.diff(w.ts.values) > 1e-6]]
    d = np.diff(u.out_total.values)
    i = int(np.argmax(np.abs(d)))
    print(f"\n  >>> 显示总量最大单帧跃变 {d[i]:+,.0f} ADC 在 t={u.ts.values[i+1]:.3f}s "
          f"(dt={u.ts.values[i+1]-u.ts.values[i]:.3f}s, 原始同帧变化 {u.total.values[i+1]-u.total.values[i]:+,.0f})")
    print(f"      creep_total 同帧变化 {u.creep_total.values[i+1]-u.creep_total.values[i]:+,.0f}  "
          f"b_total 同帧变化 {u.b_total.values[i+1]-u.b_total.values[i]:+,.0f}  "
          f"g: {u.g.values[i]:.3f} -> {u.g.values[i+1]:.3f}  a_captured: {u.a_captured.values[i]} -> {u.a_captured.values[i+1]}")

    # 前后 1s / 3s 的净变化
    for dtw in (0.2, 0.5, 1.0, 3.0):
        pre = w[(w.ts > t0 - dtw) & (w.ts < t0)]
        post = w[(w.ts > t0) & (w.ts < t0 + dtw)]
        if len(pre) and len(post):
            print(f"      ±{dtw:>4.1f}s: 显示 {post.out_total.median()-pre.out_total.median():+10,.0f}   "
                  f"原始 {post.total.median()-pre.total.median():+10,.0f}   "
                  f"creep {post.creep_total.median()-pre.creep_total.median():+10,.0f}")

print("\n说明: out_total=显示总量, total=原始总量, creep_total=被扣除的蠕变补偿总量, b_total=基线补偿总量")
