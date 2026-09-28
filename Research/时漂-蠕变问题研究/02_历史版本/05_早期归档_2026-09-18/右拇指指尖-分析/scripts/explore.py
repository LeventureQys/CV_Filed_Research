# -*- coding: utf-8 -*-
"""探索数据：通道总览、负载段识别、时漂量化"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")
DATASETS = ["数据1", "数据2", "数据3"]


def load_csv(path):
    """跳过 ##Session 头（##Data 之后是列名行）"""
    df = pd.read_csv(path, skiprows=24)
    return df


for name in DATASETS:
    path = os.path.join(BASE, name, "device_001_seg000.csv")
    df = load_csv(path)
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)

    print(f"\n=== {name} ===  帧数={len(df)}  时长={t[-1]:.1f}s  "
          f"采样≈{1/np.median(np.diff(t)):.1f}Hz")
    # 各通道均值 / 最大值
    means = X.mean(axis=0)
    maxs = X.max(axis=0)
    order = np.argsort(maxs)[::-1]
    print("通道  均值     最大值")
    for i in order[:8]:
        print(f"{ch_cols[i]:>5}  {means[i]:.4f}  {maxs[i]:.4f}")

    # 用总量信号粗略分段：空载-负载-空载
    thr = 0.15 * total.max()
    loaded = total > thr
    # 找最长连续负载段
    d = np.diff(loaded.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if loaded[0]:
        starts = np.r_[0, starts]
    if loaded[-1]:
        ends = np.r_[ends, len(loaded)]
    segs = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)
    s0, s1 = segs[0]
    print(f"主负载段: {t[s0]:.1f}s ~ {t[s1]:.1f}s  (持续 {t[s1]-t[s0]:.1f}s)")
    print(f"前空载: 0 ~ {t[s0]:.1f}s   后空载: {t[s1]:.1f}s ~ {t[-1]:.1f}s")

    # 主受载通道
    main_ch = int(order[0])
    sig = X[:, main_ch]
    pre = sig[:s0]; load = sig[s0:s1]; post = sig[s1:]
    print(f"主通道 {ch_cols[main_ch]}: 前空载均值={pre.mean():.4f}±{pre.std():.4f}  "
          f"负载首10%均值={load[:len(load)//10].mean():.4f}  负载末10%均值={load[-len(load)//10:].mean():.4f}  "
          f"后空载均值={post.mean():.4f}±{post.std():.4f}")
    # 负载段线性漂移斜率
    tt = t[s0:s1] - t[s0]
    k, b = np.polyfit(tt, load, 1)
    print(f"负载段线性斜率={k*1000:.3f}e-3 /s   漂移总量≈{k*(tt[-1]-tt[0]):.4f}  "
          f"占负载幅度 {100*k*(tt[-1]-tt[0])/(load[:len(load)//10].mean()-pre.mean()):.1f}%")
    # 前/后空载段自身斜率（看基线是否也在漂）
    for lbl, seg, tt_seg in [("前空载", pre, t[:s0]), ("后空载", post, t[s1:])]:
        if len(seg) > 100:
            kk, _ = np.polyfit(tt_seg - tt_seg[0], seg, 1)
            print(f"  {lbl}段斜率={kk*1000:.4f}e-3 /s  (范围 {seg.min():.4f}~{seg.max():.4f})")

# 画总览图：每组数据 总量+主通道
fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13, 8), constrained_layout=True)
for r, name in enumerate(DATASETS):
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    axes[r, 0].plot(t, total, lw=0.6)
    axes[r, 0].set_title(f"{name} 全部31通道总和")
    axes[r, 0].set_xlabel("时间 (s)")
    main = int(np.argmax(X.max(axis=0)))
    axes[r, 1].plot(t, X[:, main], lw=0.6, color="tab:red")
    axes[r, 1].set_title(f"{name} 主通道 {ch_cols[main]}")
    axes[r, 1].set_xlabel("时间 (s)")
fig.savefig(os.path.join(FIG, "overview.png"), dpi=140)
print("\nsaved overview.png")
