# -*- coding: utf-8 -*-
"""GLM53 分析第四步：混合方案的因果性与延迟量化

1. 因果性审计：逐组件确认只使用当前/历史采样
2. 信号通路延迟：补偿不减波、不滤波 → 阶跃无群延迟（用阶跃响应验证）
3. 估计收敛延迟（"补偿暖机"）：
   R(u) = 1 - |补偿后瞬时偏离A| / |原始瞬时偏离A|  （受载通道中位，1s平滑）
   输出 R 首次越过 0.5 / 0.9 的时刻
4. onset 检测延迟（因果检测器 vs 分段真值）与卸载后基线重置时间
输出: figures/d1_latency.png, results/latency.txt
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # temp
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")
TAU_CREEP = 5.0

CASES = [("右拇指指尖", "数据2"), ("左拇指指尖", "数据2"), ("四指指尖", "数据2")]


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
    """混合方案（与 b_compare/c_validate 相同），同时返回逐帧补偿量与门控状态"""
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
    # 混合
    Y = Z.copy()
    creep_amt = np.zeros((n, m))
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
        creep_amt[i] = creep
    return Y, Z, A, loaded_ch, unloaded, ts, min_ts


def smooth(v, dtm, tau=1.0):
    a = np.clip(dtm / tau, 0, 1)
    out = np.empty_like(v)
    s = v[0].copy()
    for i in range(len(v)):
        if i:
            s = s + a * (v[i] - s)
        out[i] = s
    return out


report = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
for k, (loc, name) in enumerate(CASES):
    ddir = os.path.join(BASE, loc, name)
    df = pd.read_csv(os.path.join(ddir, "device_001_seg000.csv"), skiprows=24)
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    dtm = np.median(np.diff(t))
    s0, s1 = find_segment(X.sum(axis=1))
    main = int(np.argmax(X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)))
    Y, Z, A, loaded_ch, unloaded, ts, min_ts = run_hybrid(X, t, s0, s1)

    # --- onset 检测延迟（因果检测器 = unloaded 翻 False） ---
    idx_after = np.where(~unloaded[s0:])[0]
    onset_lag = t[s0 + idx_after[0]] - t[s0] if len(idx_after) else np.nan

    # --- 补偿收敛曲线 R(u) ---
    # 理想水平：原始域 = 前空载均值 + A；补偿域 = A（各自零点约定）
    tt = t[s0:s1] - t[s0]
    lvl_raw = X[:s0][:, loaded_ch].mean(axis=0) + A[loaded_ch]
    dev_raw = np.abs(X[s0:s1][:, loaded_ch] - lvl_raw)
    dev_cmp = np.abs(Y[s0:s1][:, loaded_ch] - A[loaded_ch])
    R = 1 - np.median(dev_cmp, axis=1) / np.maximum(np.median(dev_raw, axis=1), 1e-9)
    R_s = smooth(R, dtm, tau=2.0)
    u50 = tt[np.argmax(R_s >= 0.5)] if (R_s >= 0.5).any() else np.nan
    u90 = tt[np.argmax(R_s >= 0.9)] if (R_s >= 0.9).any() else np.nan

    # --- 阶跃响应延迟：onset 后 Y 与 X 的上升是否同步（互相关/首达阈值时刻） ---
    n2 = int(2.0 / dtm)
    x_step = X[s0:s0 + n2, main] - X[:s0, main].mean()
    y_step = Y[s0:s0 + n2, main] - Y[:s0, main].mean()
    thr_x = 0.5 * x_step.max()
    i_x = np.argmax(x_step >= thr_x)
    i_y = np.argmax(y_step >= 0.5 * y_step.max()) if y_step.max() > 0 else 0

    # --- 卸载检测延迟（unloaded 翻 True）与基线重置（τ=2s，95%≈3τ=6s 解析值） ---
    idx_un = np.where(unloaded[s1:])[0]
    unload_detect = t[s1 + idx_un[0]] - t[s1] if len(idx_un) else np.nan

    report.append(f"[{loc}/{name}] 恒载{t[s1]-t[s0]:.0f}s  onset检测延迟={onset_lag*1000:.0f}ms  "
                  f"补偿收敛: 50%@{u50:.1f}s  90%@{u90:.1f}s  "
                  f"阶跃半幅时刻 X={t[s0+i_x]-t[s0]:.3f}s Y={t[s0+i_y]-t[s0]:.3f}s (Δ={t[s0+i_y]-t[s0+i_x]:+.3f}s)  "
                  f"卸载检测延迟={unload_detect*1000:.0f}ms + 基线τ=2s(95%≈6s)")

    ax = axes[k]
    ax.plot(tt, R_s, lw=1.2, color="tab:red")
    ax.axhline(0.9, color="gray", ls="--", lw=0.8)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.axvspan(0, 10, color="orange", alpha=0.12)
    ax.set_title(f"{loc} · {name}\n补偿收敛 R(u)（50%@{u50:.1f}s, 90%@{u90:.1f}s）", fontsize=10)
    ax.set_xlabel("负载持续时间 (s)")
    ax.set_ylabel("瞬时蠕变被补偿比例 R")
    ax.set_ylim(-0.2, 1.05)
    ax.text(5, -0.1, "暖机区", ha="center", fontsize=8, color="darkorange")
fig.savefig(os.path.join(FIG, "d1_latency.png"), dpi=140)
plt.close(fig)

text = "\n".join(report)
text = "混合方案延迟量化（受载通道中位，2s 平滑）\n" + text
with open(os.path.join(RES, "latency.txt"), "w", encoding="utf-8") as f:
    f.write(text)
print(text)
print("\nsaved: figures/d1_latency.png, results/latency.txt")
