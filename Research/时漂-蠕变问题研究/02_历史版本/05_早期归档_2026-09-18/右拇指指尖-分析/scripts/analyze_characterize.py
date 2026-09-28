# -*- coding: utf-8 -*-
"""数据特征分析：时漂形态、零漂、噪声、空间分布、共模性"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")
DATASETS = ["数据1", "数据2", "数据3"]
LAYOUT_ROWS, LAYOUT_COLS = 9, 7  # session.json: rows=9 cols=7, 31 active


def load_csv(path):
    return pd.read_csv(path, skiprows=24)


def segment(t, total, thr_ratio=0.15):
    """返回 (前空载slice, 负载slice, 后空载slice, 主负载段边界s0,s1)"""
    thr = thr_ratio * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if loaded[0]:
        starts = np.r_[0, starts]
    if loaded[-1]:
        ends = np.r_[ends, len(loaded)]
    segs = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)
    s0, s1 = segs[0]
    return s0, s1


def exp_model(t, a, tau, c):
    return a * (1 - np.exp(-t / tau)) + c


def log_model(t, a, b):
    return a * np.log1p(t) + b


report = {}

for name in DATASETS:
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    s0, s1 = segment(t, total)

    # 主受载通道 = 负载段均值最大
    load_mean_per_ch = X[s0:s1].mean(axis=0)
    main_ch = int(np.argmax(load_mean_per_ch))
    sig = X[:, main_ch]

    pre, load, post = sig[:s0], sig[s0:s1], sig[s1:]
    tl = t[s0:s1] - t[s0]

    # ---- 时漂形态拟合（负载段）----
    # 线性
    kl, bl = np.polyfit(tl, load, 1)
    r_lin = 1 - np.var(load - (kl * tl + bl)) / np.var(load)
    # 指数 a(1-exp(-t/tau))+c
    try:
        p0 = [load[-1] - load[0], max(tl[-1] / 3, 1.0), load[0]]
        popt, _ = curve_fit(exp_model, tl, load, p0=p0, maxfev=20000)
        r_exp = 1 - np.var(load - exp_model(tl, *popt)) / np.var(load)
    except Exception:
        popt, r_exp = None, np.nan
    # 对数 a*log1p(t)+b
    A = np.vstack([np.log1p(tl), np.ones_like(tl)]).T
    (al, bl2), *_ = np.linalg.lstsq(A, load, rcond=None)
    r_log = 1 - np.var(load - (al * np.log1p(tl) + bl2)) / np.var(load)

    # ---- 零漂 ----
    pre_mean, post_mean = pre.mean(), post.mean()
    zero_shift = post_mean - pre_mean
    k_pre, _ = np.polyfit(t[:s0] - t[0], pre, 1) if s0 > 50 else (np.nan, None)
    k_post, _ = np.polyfit(t[s1:] - t[s1], post, 1) if len(post) > 50 else (np.nan, None)
    step = load[: len(load) // 10].mean() - pre_mean  # 初始阶跃幅度
    drift_total = load[-len(load) // 10:].mean() - load[: len(load) // 10].mean()

    # ---- 噪声 ----
    noise_std = X[:s0].std(axis=0)
    # 前空载各通道基线偏移（相对0点）
    baseline_offset = X[:s0].mean(axis=0)

    # ---- 受载/未受载分组 ----
    loaded_mask = load_mean_per_ch > 0.3 * load_mean_per_ch.max()
    idle_mask = load_mean_per_ch < 0.05 * load_mean_per_ch.max()

    # 负载段各通道漂移斜率
    slopes = np.array([np.polyfit(tl, X[s0:s1, j], 1)[0] for j in range(X.shape[1])])
    # 未受载通道平均信号的负载段斜率（共模检验）
    ref = X[:, idle_mask].mean(axis=1) if idle_mask.sum() > 0 else np.zeros_like(total)
    k_ref_load = np.polyfit(tl, ref[s0:s1], 1)[0] if idle_mask.sum() > 0 else np.nan
    # 受载组斜率与未受载组斜率相关性
    if idle_mask.sum() > 1:
        corr_idle = np.corrcoef(slopes[idle_mask], load_mean_per_ch[idle_mask])[0, 1]
    else:
        corr_idle = np.nan

    report[name] = dict(
        n=len(df), dur=t[-1], fs=1 / np.median(np.diff(t)),
        s0=s0, s1=s1, t=t, X=X, ch_cols=ch_cols, total=total,
        main_ch=main_ch, pre_mean=pre_mean, post_mean=post_mean,
        zero_shift=zero_shift, k_pre=k_pre, k_post=k_post,
        step=step, drift_total=drift_total,
        k_lin=kl, r_lin=r_lin, r_exp=r_exp, r_log=r_log,
        exp_popt=popt, log_ab=(al, bl2),
        loaded_mask=loaded_mask, idle_mask=idle_mask, slopes=slopes,
        k_ref_load=k_ref_load, noise_std=noise_std,
        baseline_offset=baseline_offset, load_mean_per_ch=load_mean_per_ch,
    )

    print(f"\n======== {name} ========")
    print(f"帧数={len(df)} 时长={t[-1]:.1f}s 采样≈{report[name]['fs']:.1f}Hz")
    print(f"分段: 前空载 0~{t[s0]:.1f}s | 负载 {t[s0]:.1f}~{t[s1]:.1f}s ({t[s1]-t[s0]:.1f}s) | 后空载 {t[s1]:.1f}~{t[-1]:.1f}s")
    print(f"主通道={ch_cols[main_ch]} 初始阶跃={step:.4f}N 负载段漂移量={drift_total:.4f}N ({100*drift_total/step:.1f}% of step)")
    print(f"时漂拟合R²: 线性={r_lin:.4f}  指数(τ={popt[1]:.1f}s)={r_exp:.4f}  对数={r_log:.4f}")
    print(f"线性斜率={kl*1000:.3f}e-3 N/s")
    print(f"零漂: 前空载均值={pre_mean:.4f} 后空载均值={post_mean:.4f} 差(零漂)={zero_shift:.4f}N ({100*zero_shift/step:.1f}% of step)")
    print(f"空载段斜率: 前={k_pre*1000:.4f}e-3/s 后={k_post*1000:.4f}e-3/s")
    print(f"受载通道={loaded_mask.sum()} 未受载={idle_mask.sum()} 未受载组负载段斜率={k_ref_load*1000:.4f}e-3/s")
    print(f"基线偏移: min={baseline_offset.min():.4f} max={baseline_offset.max():.4f} 非零均值通道数={np.sum(np.abs(baseline_offset)>0.005)}")
    print(f"噪声std(前空载): 中位={np.median(noise_std):.4f} max={noise_std.max():.4f}")

# ================= 图1: 总览（总量+主通道） =================
fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    axes[r, 0].plot(R["t"], R["total"], lw=0.5, color="tab:blue")
    axes[r, 0].axvspan(R["t"][R["s0"]], R["t"][R["s1"]], color="orange", alpha=0.15)
    axes[r, 0].set_title(f"{name} 31通道总和（橙区=恒定负载段）")
    axes[r, 0].set_xlabel("时间 (s)"); axes[r, 0].set_ylabel("ΣForce (N)"); axes[r, 0].grid(alpha=0.3)
    sig = R["X"][:, R["main_ch"]]
    axes[r, 1].plot(R["t"], sig, lw=0.5, color="tab:red")
    axes[r, 1].axvspan(R["t"][R["s0"]], R["t"][R["s1"]], color="orange", alpha=0.15)
    axes[r, 1].set_title(f"{name} 主通道 {R['ch_cols'][R['main_ch']]}")
    axes[r, 1].set_xlabel("时间 (s)"); axes[r, 1].set_ylabel("Force (N)"); axes[r, 1].grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "c1_overview.png"), dpi=140)
plt.close(fig)

# ================= 图2: 负载段时漂形态拟合 =================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    R = report[name]
    tl = R["t"][R["s0"]:R["s1"]] - R["t"][R["s0"]]
    sig = R["X"][R["s0"]:R["s1"], R["main_ch"]]
    ax.plot(tl, sig, lw=0.4, color="gray", label="原始")
    ax.plot(tl, R["k_lin"] * tl + np.polyfit(tl, sig, 1)[1], lw=1.4,
            label=f"线性 R²={R['r_lin']:.3f}")
    if R["exp_popt"] is not None:
        ax.plot(tl, exp_model(tl, *R["exp_popt"]), lw=1.4,
                label=f"指数(τ={R['exp_popt'][1]:.0f}s) R²={R['r_exp']:.3f}")
    al, bl2 = R["log_ab"]
    ax.plot(tl, al * np.log1p(tl) + bl2, lw=1.4, ls="--",
            label=f"对数 R²={R['r_log']:.3f}")
    ax.set_title(f"{name} 主通道负载段时漂形态")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "c2_drift_models.png"), dpi=140)
plt.close(fig)

# ================= 图3: 零漂（前/后空载基线） =================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    R = report[name]
    sig = R["X"][:, R["main_ch"]]
    pre_idx = np.arange(0, R["s0"]); post_idx = np.arange(R["s1"], len(sig))
    ax.plot(R["t"][pre_idx], sig[pre_idx], lw=0.5, color="tab:blue", label="前空载")
    ax.plot(R["t"][post_idx], sig[post_idx], lw=0.5, color="tab:green", label="后空载")
    ax.axhline(R["pre_mean"], color="tab:blue", ls="--", lw=1, alpha=0.7)
    ax.axhline(R["post_mean"], color="tab:green", ls="--", lw=1, alpha=0.7)
    ax.set_title(f"{name} 零漂={R['zero_shift']:+.4f}N ({100*R['zero_shift']/R['step']:.1f}% of step)")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "c3_zero_drift.png"), dpi=140)
plt.close(fig)

# ================= 图4: 空间分布（响应热图 + 漂移斜率热图） =================
# layout_mask: 63 bits for 9x7; CSV 31 ch 按 mask 中 1 的顺序填入
mask_str = "110110011011101111110111111011111100000001000000100000010000001"
fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    resp_map = np.full(LAYOUT_ROWS * LAYOUT_COLS, np.nan)
    slope_map = np.full(LAYOUT_ROWS * LAYOUT_COLS, np.nan)
    ones = [i for i, b in enumerate(mask_str) if b == "1"]
    for j, pos in enumerate(ones):
        resp_map[pos] = R["load_mean_per_ch"][j] - R["X"][: R["s0"], j].mean()
        slope_map[pos] = R["slopes"][j] * 1000  # e-3 N/s
    resp_map = resp_map.reshape(LAYOUT_ROWS, LAYOUT_COLS)
    slope_map = slope_map.reshape(LAYOUT_ROWS, LAYOUT_COLS)
    im0 = axes[r, 0].imshow(resp_map, cmap="hot", aspect="auto")
    axes[r, 0].set_title(f"{name} 负载响应空间分布 (N)")
    plt.colorbar(im0, ax=axes[r, 0])
    im1 = axes[r, 1].imshow(slope_map, cmap="coolwarm", aspect="auto")
    axes[r, 1].set_title(f"{name} 负载段漂移斜率 (e-3 N/s)")
    plt.colorbar(im1, ax=axes[r, 1])
fig.savefig(os.path.join(FIG, "c4_spatial_maps.png"), dpi=140)
plt.close(fig)

# ================= 图5: 未受载通道共模检验 =================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    R = report[name]
    ref = R["X"][:, R["idle_mask"]].mean(axis=1)
    loaded_avg = R["X"][:, R["loaded_mask"]].mean(axis=1)
    ax.plot(R["t"], loaded_avg, lw=0.5, color="tab:red",
            label=f"受载组均值({R['loaded_mask'].sum()}ch)")
    ax2 = ax.twinx()
    ax2.plot(R["t"], ref, lw=0.7, color="tab:blue",
             label=f"未受载组均值({R['idle_mask'].sum()}ch)")
    ax2.set_ylabel("未受载组 Force (N)", color="tab:blue")
    ax.axvspan(R["t"][R["s0"]], R["t"][R["s1"]], color="orange", alpha=0.12)
    ax.set_title(f"{name} 未受载组负载段斜率={R['k_ref_load']*1000:.3f}e-3/s")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("受载组 Force (N)", color="tab:red")
    ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "c5_common_mode.png"), dpi=140)
plt.close(fig)

print("\nsaved c1~c5 figures")
