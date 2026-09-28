# -*- coding: utf-8 -*-
"""GLM53 分析第五步：混合方案算力实测（Python 原型每帧耗时）

计时内容 = 完整在线流水线（因果）：卸载检测 + 门控基线 + 幅度估计 + g/γ 更新 + 补偿
Python/numpy 原型耗时为 C++ 标量实现的上界参考（通常差 1~2 个数量级）。
输出: results/compute_cost.txt
"""
import os
import time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # temp


def find_segment(total):
    thr = 0.15 * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if loaded[0]:
        s = np.r_[0, s]
    if loaded[-1]:
        e = np.r_[e, len(loaded)]
    segs = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)
    return segs[0]


def run_hybrid(X, t, s0, s1):
    """与 d_latency.py 相同的完整因果流水线"""
    n, m = X.shape
    dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.1)
    dtm = np.median(dt)
    total = X.sum(axis=1)
    a_s = np.clip(dtm / 0.3, 0, 1)
    ts = np.empty(n)
    ts[0] = total[0]
    for i in range(1, n):
        ts[i] = ts[i - 1] + a_s * (total[i] - ts[i - 1])
    min_ts = np.minimum.accumulate(ts)
    max_ts = np.maximum.accumulate(ts)
    unloaded = (ts < 1.5 * min_ts) | (ts < 0.15 * max_ts)
    b0 = X[0].copy()
    a_b = np.clip(dt / 2.0, 0, 1)
    baseline = np.empty((n, m))
    for i in range(n):
        if unloaded[i]:
            b0 = b0 + a_b[i] * (X[i] - b0)
        baseline[i] = b0
    Z = X - baseline
    i_on1 = s0 + int(1.0 / dtm)
    i_on2 = s0 + int(3.0 / dtm)
    A = Z[i_on1:i_on2].mean(axis=0)
    loaded_ch = A > 0.10 * A.max()
    Y = Z.copy()
    g_smooth = 0.0
    a_g = np.clip(dtm / 3.0, 0, 1)
    gamma = np.ones(m)
    g2_acc = 0.0
    g_rel_acc = np.zeros(m)
    A_safe = np.where(np.abs(A) > 1e-9, A, 1.0)
    A_ld = A[loaded_ch]
    u_prev_h = t[s0]
    for i in range(s0, s1):
        du = max(0.0, t[i] - u_prev_h)
        u_prev_h = t[i]
        rel_ld = (Z[i, loaded_ch] - A_ld) / A_ld
        g_raw = np.median(rel_ld)
        g_smooth = g_smooth + a_g * (g_raw - g_smooth)
        if g_smooth > 0.02:
            rel_full = np.where(loaded_ch, (Z[i] - A) / A_safe, 0.0)
            g2_acc += du * g_smooth * g_smooth
            g_rel_acc += du * g_smooth * rel_full
            gamma = np.where(g2_acc > 1e-8,
                             np.clip(g_rel_acc / max(g2_acc, 1e-8), 0.3, 2.0), 1.0)
        creep = np.clip(gamma * A * g_smooth, -0.5 * A, 1.5 * A)
        Y[i] = Z[i] - creep
    return Y


lines = ["混合方案（含自动归零基座）算力实测 — Python/numpy 原型"]
for loc, name in [("右拇指指尖", "数据2"), ("左拇指指尖", "数据2"), ("四指指尖", "数据2")]:
    ddir = os.path.join(BASE, loc, name)
    df = pd.read_csv(os.path.join(ddir, "device_001_seg000.csv"), skiprows=24)
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    s0, s1 = find_segment(X.sum(axis=1))
    # 预热（排除首次 numpy 开销）
    run_hybrid(X[:2000], t[:2000], 10, 1000)
    reps = 3
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_hybrid(X, t, s0, s1)
        times.append(time.perf_counter() - t0)
    tm = min(times)
    lines.append(f"[{loc}/{name}] m={X.shape[1]}通道 帧数={len(X)}  "
                 f"整段流水线 {tm*1000:.0f} ms → **{tm/len(X)*1e6:.1f} µs/帧**（{reps}次取最小）")

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "compute_cost.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
