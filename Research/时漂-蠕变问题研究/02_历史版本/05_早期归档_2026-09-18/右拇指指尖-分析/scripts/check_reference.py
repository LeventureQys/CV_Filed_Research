# -*- coding: utf-8 -*-
"""检查未受载通道是否漂移（决定参考通道法可行性）"""
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
    return pd.read_csv(path, skiprows=24)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    thr = 0.15 * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    segs = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)
    s0, s1 = segs[0]

    # 受载组：负载段均值 > 0.15；未受载组：< 0.05
    load_mean = X[s0:s1].mean(axis=0)
    loaded_mask = load_mean > 0.15
    idle_mask = load_mean < 0.05
    print(f"{name}: 受载通道 {loaded_mask.sum()} 个, 未受载通道 {idle_mask.sum()} 个")
    print(f"  未受载通道: {[ch_cols[i] for i in np.where(idle_mask)[0]]}")

    ref = X[:, idle_mask].mean(axis=1)
    k_all, _ = np.polyfit(t[s0:s1] - t[s0], X[s0:s1][:, loaded_mask].mean(axis=1), 1)
    k_ref, _ = np.polyfit(t[s0:s1] - t[s0], ref[s0:s1], 1)
    k_ref_pre, _ = np.polyfit(t[:s0] - t[0], ref[:s0], 1)
    print(f"  受载组均值负载段斜率={k_all*1000:.3f}e-3/s   未受载参考负载段斜率={k_ref*1000:.3f}e-3/s  前空载={k_ref_pre*1000:.3f}e-3/s")

    ax.plot(t, X[:, loaded_mask].mean(axis=1), lw=0.7, label=f"受载组均值({loaded_mask.sum()}ch)")
    ax.plot(t, ref, lw=0.9, label=f"未受载参考({idle_mask.sum()}ch)")
    ax.set_title(f"{name}")
    ax.set_xlabel("时间 (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "reference_channels.png"), dpi=140)
print("saved reference_channels.png")
