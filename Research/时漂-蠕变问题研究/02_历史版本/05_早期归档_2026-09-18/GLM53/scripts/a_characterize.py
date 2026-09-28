# -*- coding: utf-8 -*-
"""GLM53 分析第一步：数据特征刻画
- 三组 空载-恒载-空载 数据的分段识别
- 时漂量化：负载段各通道漂移斜率 / 幅度占比 / 漂移模型拟合（线性/对数/指数）
- 零漂量化：前/后空载基线差、空载段自身游走
- 共模分析：空载参考通道是否与受载通道同步漂移（决定阵列级共模扣除是否可行）
输出: figures/a1_*.png ... a4_*.png, results/characterization.txt / characterization.csv
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

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "右拇指指尖")  # temp/右拇指指尖
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # GLM53
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")
DATASETS = ["数据1", "数据2", "数据3"]

LAYOUT_MASK = "110110011011101111110111111011111100000001000000100000010000001"
ROWS, COLS = 9, 7


def load_csv(path):
    df = pd.read_csv(path, skiprows=24)
    return df


def segment(total, t):
    """按总量信号识别主负载段，返回 (s0, s1) 索引"""
    thr = 0.15 * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if loaded[0]:
        starts = np.r_[0, starts]
    if loaded[-1]:
        ends = np.r_[ends, len(loaded)]
    segs = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)
    return segs[0]


def fit_models(tt, y):
    """拟合线性 / 对数蠕变 / 指数饱和三种漂移模型，返回 R2"""
    out = {}
    # 线性
    p = np.polyfit(tt, y, 1)
    r = y - np.polyval(p, tt)
    out["linear"] = 1 - r.var() / y.var()
    # 对数蠕变 a + b*ln(1+t/tau)
    best = None
    for tau in [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]:
        x = np.log1p(tt / tau)
        if x.std() < 1e-9:
            continue
        b, a = np.polyfit(x, y, 1)
        r = y - (a + b * x)
        r2 = 1 - r.var() / y.var()
        if best is None or r2 > best[0]:
            best = (r2, a, b, tau)
    out["log"] = best[0]
    # 指数饱和 a - b*exp(-t/tau)
    best = None
    y0, yinf = y[: max(1, len(y) // 50)].mean(), y[-max(1, len(y) // 50):].mean()
    for tau in [5, 10, 20, 50, 100, 200, 500, 1000]:
        x = np.exp(-tt / tau)
        b, a = np.polyfit(x, y, 1)
        r = y - (a + b * x)
        r2 = 1 - r.var() / y.var()
        if best is None or r2 > best[0]:
            best = (r2, a, b, tau)
    out["exp"] = best[0]
    return out


all_rows = []
report_lines = []

# ============ 逐数据集刻画 ============
for name in DATASETS:
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    s0, s1 = segment(total, t)
    fs = 1.0 / np.median(np.diff(t))

    # 分段
    pre = slice(0, s0)
    load = slice(s0, s1)
    post = slice(s1, len(t))
    pre_dur, load_dur, post_dur = t[s0], t[s1] - t[s0], t[-1] - t[s1]

    report_lines.append(f"\n{'='*70}\n[{name}]  帧数={len(df)}  fs≈{fs:.1f}Hz  时长={t[-1]:.1f}s")
    report_lines.append(f"分段: 前空载 0~{t[s0]:.1f}s | 恒载 {t[s0]:.1f}~{t[s1]:.1f}s ({load_dur:.1f}s) | 后空载 {t[s1]:.1f}~{t[-1]:.1f}s")

    # 各通道统计
    n_ch = X.shape[1]
    amp = X[load].mean(axis=0) - X[pre].mean(axis=0)          # 负载幅度（相对前空载基线）
    loaded_chs = np.where(amp > 0.05 * np.max(amp))[0]         # 受载通道
    ref_chs = np.array([i for i in range(n_ch) if i not in set(loaded_chs)])

    # 负载段漂移（末10%均值 - 首10%均值）
    L = X[load]
    nL = len(L)
    drift_seg = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)

    # 零漂：后空载基线 - 前空载基线
    zpre = X[pre].mean(axis=0)
    zpost = X[post].mean(axis=0)
    zero_drift = zpost - zpre
    # 后空载段内部游走（自身斜率）
    post_slope = np.array([
        np.polyfit(t[post] - t[s1], X[post, i], 1)[0] if post_dur > 5 else np.nan
        for i in range(n_ch)
    ])

    main = int(np.argmax(amp))
    tt = t[load] - t[s0]
    r2 = fit_models(tt, X[load, main])

    report_lines.append(
        f"受载通道数={len(loaded_chs)} (amp>{0.05*np.max(amp):.3f})  "
        f"主通道={ch_cols[main]} 负载幅度={amp[main]:.3f}")
    report_lines.append(
        f"[时漂] 主通道负载段漂移={drift_seg[main]:+.4f} ({100*drift_seg[main]/amp[main]:+.1f}% 满幅)  "
        f"斜率={np.polyfit(tt, X[load, main], 1)[0]*1e3:+.3f}e-3/s")
    report_lines.append(
        f"[时漂] 全体受载通道漂移占满幅: 均值={100*np.mean(drift_seg[loaded_chs]/amp[loaded_chs]):+.1f}%  "
        f"中位={100*np.median(drift_seg[loaded_chs]/amp[loaded_chs]):+.1f}%")
    report_lines.append(
        f"[时漂模型R2] 主通道: linear={r2['linear']:.3f}  log-creep={r2['log']:.3f}  exp={r2['exp']:.3f}")
    report_lines.append(
        f"[零漂] 主通道: 前空载={zpre[main]:.4f}  后空载={zpost[main]:.4f}  "
        f"残差零漂={zero_drift[main]:+.4f} ({100*zero_drift[main]/amp[main]:+.1f}% 满幅)")
    report_lines.append(
        f"[零漂] 全体通道零漂: 均值={zero_drift.mean():+.4f}  最大={zero_drift[np.argmax(np.abs(zero_drift))]:+.4f}")
    report_lines.append(
        f"[后空载游走] 全体通道斜率均值={np.nanmean(post_slope)*1e3:+.4f}e-3/s")

    # 共模分析：参考通道在负载段的漂移（它们不受载，漂了就是真·基线漂）
    ref_drift = drift_seg[ref_chs] if len(ref_chs) else np.array([np.nan])
    ref_zero = zero_drift[ref_chs] if len(ref_chs) else np.array([np.nan])
    # 受载通道漂移之间的相关性（剔除幅度差异后）
    Dr = drift_seg[loaded_chs]
    Ar = amp[loaded_chs]
    if len(loaded_chs) > 2:
        rel = Dr / Ar
        corr = np.corrcoef(Dr, Ar)[0, 1] if Dr.std() > 1e-12 and Ar.std() > 1e-12 else np.nan
    else:
        corr = np.nan
    report_lines.append(
        f"[共模] 空载参考通道({len(ref_chs)}个) 负载段漂移: 均值={np.nanmean(ref_drift):+.5f}  "
        f"最大={np.nanmax(np.abs(ref_drift)) if len(ref_chs) else float('nan'):.5f}")
    report_lines.append(
        f"[共模] 参考通道零漂均值={np.nanmean(ref_zero):+.5f}  "
        f"受载通道漂移-幅度相关系数={corr if corr==corr else float('nan'):.3f}")

    for i, ch in enumerate(ch_cols):
        all_rows.append(dict(
            dataset=name, channel=ch, idx=i,
            load_amp=amp[i], drift=drift_seg[i],
            drift_pct=100 * drift_seg[i] / amp[i] if abs(amp[i]) > 1e-6 else np.nan,
            is_loaded=i in set(loaded_chs.tolist()),
            zero_pre=zpre[i], zero_post=zpost[i], zero_drift=zero_drift[i],
            post_slope=abs(post_slope[i]),
        ))

# 汇总表
ddf = pd.DataFrame(all_rows)
ddf.to_csv(os.path.join(RES, "characterization.csv"), index=False, encoding="utf-8-sig")

report_lines.append(f"\n{'='*70}\n[三组汇总]")
ld = ddf[ddf.is_loaded]
rc = ddf[~ddf.is_loaded]
report_lines.append(f"受载通道样本 n={len(ld)}: 漂移占满幅 均值={ld.drift_pct.mean():+.1f}%  "
                    f"中位={ld.drift_pct.median():+.1f}%  P90={ld.drift_pct.quantile(0.9):+.1f}%")
report_lines.append(f"全体通道零漂绝对值: 中位={ddf.zero_drift.abs().median():.4f}  "
                    f"P90={ddf.zero_drift.abs().quantile(0.9):.4f}")
report_lines.append(f"空载参考通道零漂绝对值: 中位={rc.zero_drift.abs().median():.4f}")
report_lines.append(f"空载参考通道负载段漂移绝对值: 中位={rc.drift.abs().median():.5f}")

text = "\n".join(report_lines)
with open(os.path.join(RES, "characterization.txt"), "w", encoding="utf-8") as f:
    f.write(text)
print(text)

# ============ 图 a1: 三组数据总览（总量 + 主通道 + 分段标注） ============
fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True, sharex=False)
for r, name in enumerate(DATASETS):
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    s0, s1 = segment(total, t)
    main = int(np.argmax(X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)))
    ax = axes[r]
    ax.plot(t, X[:, main], lw=0.5, color="tab:red", label=f"主通道 {ch_cols[main]}")
    ax.plot(t, total * 0 + X[:, main].min(), lw=0)  # 保持范围
    ax2 = ax.twinx()
    ax2.plot(t, total, lw=0.5, color="tab:blue", alpha=0.6, label="31通道总和")
    for x in (t[s0], t[s1]):
        ax.axvline(x, color="gray", ls="--", lw=0.8)
    ax.axvspan(t[s0], t[s1], color="orange", alpha=0.08)
    ax.text((t[s0] + t[s1]) / 2, ax.get_ylim()[1] * 0.95, "恒定负载段", ha="center", fontsize=9, color="darkorange")
    ax.set_title(f"{name}（恒载 {t[s1]-t[s0]:.0f}s）主通道 vs 通道总和", fontsize=11)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("主通道", color="tab:red")
    ax2.set_ylabel("通道总和", color="tab:blue")
fig.savefig(os.path.join(FIG, "a1_overview.png"), dpi=140)
plt.close(fig)

# ============ 图 a2: 主通道负载段漂移 + 三模型拟合 ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for r, name in enumerate(DATASETS):
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    s0, s1 = segment(total, t)
    main = int(np.argmax(X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)))
    tt = t[load := slice(s0, s1)] - t[s0]
    y = X[load, main]
    ax = axes[r]
    ax.plot(tt, y, lw=0.5, color="tab:red", label="原始")
    ax.plot(tt, np.polyval(np.polyfit(tt, y, 1), tt), "k--", lw=1, label="线性拟合")
    # log 模型
    best = None
    for tau in [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]:
        x = np.log1p(tt / tau)
        b, a = np.polyfit(x, y, 1)
        rr = y - (a + b * x)
        r2 = 1 - rr.var() / y.var()
        if best is None or r2 > best[0]:
            best = (r2, a, b, tau)
    r2, a, b, tau = best
    ax.plot(tt, a + b * np.log1p(tt / tau), "g-.", lw=1.2,
            label=f"对数蠕变 R²={r2:.3f}")
    ax.set_title(f"{name} 主通道负载段", fontsize=11)
    ax.set_xlabel("负载持续时间 (s)")
    ax.set_ylabel("读数")
    ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "a2_drift_models.png"), dpi=140)
plt.close(fig)

# ============ 图 a3: 零漂（前/后空载基线对比，全部通道） ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for r, name in enumerate(DATASETS):
    sub = ddf[ddf.dataset == name]
    ax = axes[r]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.zero_pre, width=0.4, label="前空载基线", color="tab:blue", alpha=0.7)
    ax.bar(x + 0.2, sub.zero_post, width=0.4, label="后空载基线", color="tab:red", alpha=0.7)
    ld_idx = sub[sub.is_loaded].index.tolist()
    for i in ld_idx:
        ax.axvspan(x[sub.index.get_loc(i)] - 0.5, x[sub.index.get_loc(i)] + 0.5, color="orange", alpha=0.15)
    ax.set_title(f"{name} 前/后空载基线（橙色=受载通道）", fontsize=11)
    ax.set_xlabel("通道索引")
    ax.set_ylabel("空载读数")
    ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "a3_zero_drift.png"), dpi=140)
plt.close(fig)

# ============ 图 a4: 空载参考通道 vs 受载通道漂移对比（共模检验） ============
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
ax = axes[0]
colors = {"数据1": "tab:blue", "数据2": "tab:orange", "数据3": "tab:green"}
for name in DATASETS:
    sub = ddf[ddf.dataset == name]
    ax.scatter(sub[sub.is_loaded].load_amp, sub[sub.is_loaded].drift,
               s=22, alpha=0.7, color=colors[name], label=f"{name} 受载通道")
    ax.scatter(sub[~sub.is_loaded].load_amp, sub[~sub.is_loaded].drift,
               s=22, alpha=0.7, color=colors[name], marker="x")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("负载幅度")
ax.set_ylabel("负载段漂移量")
ax.set_title("通道漂移 vs 负载幅度（×=空载参考通道）", fontsize=11)
ax.legend(fontsize=8)
ax = axes[1]
for name in DATASETS:
    sub = ddf[ddf.dataset == name]
    ax.scatter(sub[sub.is_loaded].load_amp, sub[sub.is_loaded].zero_drift,
               s=22, alpha=0.7, color=colors[name], label=f"{name} 受载通道")
    ax.scatter(sub[~sub.is_loaded].load_amp, sub[~sub.is_loaded].zero_drift,
               s=22, alpha=0.7, color=colors[name], marker="x")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("负载幅度")
ax.set_ylabel("零漂（后-前空载基线）")
ax.set_title("零漂 vs 负载幅度（×=空载参考通道）", fontsize=11)
ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "a4_common_mode_check.png"), dpi=140)
plt.close(fig)

print("\nsaved: a1_overview / a2_drift_models / a3_zero_drift / a4_common_mode_check")
