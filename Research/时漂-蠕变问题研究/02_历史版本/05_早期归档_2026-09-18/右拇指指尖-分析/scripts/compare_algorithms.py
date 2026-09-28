# -*- coding: utf-8 -*-
"""漂移抑制算法实装与横向对比
在线/盲方法: 固定基线 / EMA基线 / 卡尔曼平滑 / 自动零点跟踪 / 参考通道共模扣除 / PCA去共模
离线/模型方法: 高通滤波 / 空载锚点基线 / 指数蠕变扣除 / 对数蠕变扣除 / 组合(基线+指数蠕变)
另: 应用内 kalman_compensated 导出作为参照
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
DATASETS = ["数据1", "数据2", "数据3"]


def load_csv(path):
    return pd.read_csv(path, skiprows=24)


def segment(t, total, thr_ratio=0.15):
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
    return segs[0]


# ---------------- 算法实现（输入 X: T×C, t, s0, s1；输出同形状 Y） ----------------

def algo_raw(X, t, s0, s1, **kw):
    return X.copy()


def algo_fixed_baseline(X, t, s0, s1, **kw):
    """固定基线扣除：减去前空载均值（每次实验前校准零点）"""
    return X - X[:s0].mean(axis=0)


def algo_ema_baseline(X, t, s0, s1, tau=30.0, **kw):
    """慢EMA基线跟踪扣除（在线盲方法）"""
    T = X.shape[0]
    Y = np.empty_like(X)
    b = X[0].copy()
    for i in range(T):
        dt = t[i] - t[i - 1] if i > 0 else 0.015
        a = 1 - np.exp(-max(dt, 1e-3) / tau)
        b = b + a * (X[i] - b)
        Y[i] = X[i] - b
    return Y


def algo_kalman(X, t, s0, s1, q=1e-6, r=1e-4, **kw):
    """随机游走卡尔曼平滑（在线，平滑噪声，不去漂）"""
    T, C = X.shape
    Y = np.empty_like(X)
    x = X[0].copy()
    p = np.ones(C) * 1e-3
    for i in range(T):
        p = p + q
        k = p / (p + r)
        x = x + k * (X[i] - x)
        p = (1 - k) * p
        Y[i] = x
    return Y


def algo_auto_zero(X, t, s0, s1, total_raw, hold_after_load=True, tau=3.0, **kw):
    """自动零点跟踪：检测空载(总量低于阈值)时用EMA更新零点，负载时保持零点不变"""
    T, C = X.shape
    thr = 0.15 * total_raw.max()
    zero = X[0].copy()
    Y = np.empty_like(X)
    for i in range(T):
        idle = total_raw[i] < thr
        if idle:
            dt = t[i] - t[i - 1] if i > 0 else 0.015
            a = 1 - np.exp(-max(dt, 1e-3) / tau)
            zero = zero + a * (X[i] - zero)
        Y[i] = X[i] - zero
    return Y


def algo_ref_common(X, t, s0, s1, idle_mask=None, **kw):
    """参考通道共模扣除：减去未受载通道均值（阵列方法）"""
    ref = X[:, idle_mask].mean(axis=1, keepdims=True)
    return X - ref


def algo_pca_remove_pc1(X, t, s0, s1, **kw):
    """PCA去第一主成分（阵列方法，离线）"""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = np.outer(U[:, 0] * S[0], Vt[0])
    return Xc - pc1 + mu


def algo_highpass(X, t, s0, s1, fc=0.005, **kw):
    """一阶Butterworth高通（离线filtfilt，零相位）"""
    fs = 1 / np.median(np.diff(t))
    b, a = butter(1, fc / (fs / 2), btype="high")
    return filtfilt(b, a, X, axis=0)


def algo_anchor_linear(X, t, s0, s1, **kw):
    """前后空载锚点线性基线（离线零漂校正）"""
    T, C = X.shape
    t_pre = t[:s0].mean(); t_post = t[s1:].mean()
    pre_m = X[:s0].mean(axis=0); post_m = X[s1:].mean(axis=0)
    slope = (post_m - pre_m) / max(t_post - t_pre, 1e-6)
    baseline = pre_m[None, :] + slope[None, :] * (t - t_pre)[:, None]
    return X - baseline


def _creep_fit_subtract(X, t, s0, s1, model="exp"):
    """负载段蠕变拟合扣除（离线oracle，只对受载通道）"""
    Y = X - X[:s0].mean(axis=0)  # 先扣固定基线
    tl = t[s0:s1] - t[s0]
    load_mean = Y[s0:s1].mean(axis=0)
    loaded = load_mean > 0.3 * load_mean.max()
    for j in np.where(loaded)[0]:
        sig = Y[s0:s1, j]
        try:
            if model == "exp":
                f = lambda tt, a, tau, c: a * (1 - np.exp(-tt / tau)) + c
                p0 = [sig[-1] - sig[0], max(tl[-1] / 3, 1.0), sig[0]]
                popt, _ = curve_fit(f, tl, sig, p0=p0, maxfev=20000)
                drift = f(tl, *popt) - f(0.0, *popt)
            else:  # log
                A = np.vstack([np.log1p(tl), np.ones_like(tl)]).T
                (a, c), *_ = np.linalg.lstsq(A, sig, rcond=None)
                drift = a * np.log1p(tl)
            Y[s0:s1, j] = sig - drift
        except Exception:
            pass
    return Y


def algo_creep_exp(X, t, s0, s1, **kw):
    return _creep_fit_subtract(X, t, s0, s1, model="exp")


def algo_creep_log(X, t, s0, s1, **kw):
    return _creep_fit_subtract(X, t, s0, s1, model="log")


ALGOS = [
    ("原始", algo_raw, "online"),
    ("固定基线扣除", algo_fixed_baseline, "online"),
    ("EMA基线跟踪(τ=30s)", algo_ema_baseline, "online"),
    ("卡尔曼平滑", algo_kalman, "online"),
    ("自动零点跟踪", algo_auto_zero, "online"),
    ("参考通道共模扣除", algo_ref_common, "online"),
    ("PCA去第一主成分", algo_pca_remove_pc1, "online"),
    ("一阶高通(0.005Hz)", algo_highpass, "offline"),
    ("空载锚点线性基线", algo_anchor_linear, "offline"),
    ("指数蠕变拟合扣除", algo_creep_exp, "offline"),
    ("对数蠕变拟合扣除", algo_creep_log, "offline"),
]


def evaluate(Y, t, s0, s1, main_ch, raw_step, raw_pre_std):
    """指标: 残余漂移%/阶跃保真度%/零漂残余%/噪声比"""
    sig = Y[:, main_ch]
    pre = sig[:s0]; load = sig[s0:s1]; post = sig[s1:]
    n10 = max(len(load) // 10, 10)
    resid_drift = abs(load[-n10:].mean() - load[:n10].mean()) / raw_step * 100
    fidelity = (load[:n10].mean() - pre.mean()) / raw_step * 100
    zero_resid = abs(post.mean() - pre.mean()) / raw_step * 100
    noise_ratio = pre.std() / max(raw_pre_std, 1e-9)
    return resid_drift, fidelity, zero_resid, noise_ratio


all_metrics = []
store = {}

for name in DATASETS:
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    s0, s1 = segment(t, total)
    load_mean = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    main_ch = int(np.argmax(load_mean))
    idle_mask = load_mean < 0.05 * load_mean.max()

    raw_sig = X[:, main_ch]
    n10 = max((s1 - s0) // 10, 10)
    raw_step = raw_sig[s0:s0 + n10].mean() - raw_sig[:s0].mean()
    raw_pre_std = raw_sig[:s0].std()

    store[name] = dict(t=t, X=X, s0=s0, s1=s1, main_ch=main_ch,
                       ch_cols=ch_cols, results={})

    for aname, fn, mode in ALGOS:
        if fn is algo_auto_zero:
            Y = fn(X, t, s0, s1, total_raw=total)
        elif fn is algo_ref_common:
            Y = fn(X, t, s0, s1, idle_mask=idle_mask)
        else:
            Y = fn(X, t, s0, s1)
        store[name]["results"][aname] = Y
        m = evaluate(Y, t, s0, s1, main_ch, raw_step, raw_pre_std)
        all_metrics.append([name, aname, mode, *m])

    # 应用内 kalman_compensated 参照
    kp = os.path.join(BASE, name, "device_001_seg000_kalman_compensated.csv")
    if os.path.exists(kp):
        dfk = load_csv(kp)
        Xk = dfk[ch_cols].to_numpy()
        store[name]["results"]["应用内Kalman补偿"] = Xk
        m = evaluate(Xk, t, s0, s1, main_ch, raw_step, raw_pre_std)
        all_metrics.append([name, "应用内Kalman补偿", "online", *m])

M = pd.DataFrame(all_metrics, columns=[
    "数据集", "算法", "类型", "负载段残余漂移%", "阶跃保真度%", "零漂残余%", "噪声比"])
M.to_csv(os.path.join(OUT, "metrics.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.1f}")
print(M.to_string(index=False))

# 汇总（3数据集均值）
summary = M.groupby(["算法", "类型"], as_index=False)[
    ["负载段残余漂移%", "阶跃保真度%", "零漂残余%", "噪声比"]].mean()
summary = summary.sort_values("负载段残余漂移%")
summary.to_csv(os.path.join(OUT, "metrics_summary.csv"), index=False, encoding="utf-8-sig")
print("\n==== 3数据集平均 ====")
print(summary.to_string(index=False))

# ---------------- 图d1: 主通道全时长对比（数据1为代表，分两组） ----------------
repr_ds = "数据1"
R = store[repr_ds]
t, s0, s1, mc = R["t"], R["s0"], R["s1"], R["main_ch"]
online_list = ["原始", "固定基线扣除", "EMA基线跟踪(τ=30s)", "卡尔曼平滑",
               "自动零点跟踪", "参考通道共模扣除", "PCA去第一主成分", "应用内Kalman补偿"]
offline_list = ["原始", "固定基线扣除", "一阶高通(0.005Hz)", "空载锚点线性基线",
                "指数蠕变拟合扣除", "对数蠕变拟合扣除"]

fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
for ax, group, title in zip(axes, [online_list, offline_list],
                            ["在线/盲方法", "离线/模型方法"]):
    for aname in group:
        if aname not in R["results"]:
            continue
        sig = R["results"][aname][:, mc]
        lw = 1.6 if aname == "原始" else 0.9
        ax.plot(t, sig, lw=lw, label=aname,
                color="black" if aname == "原始" else None, alpha=0.85)
    ax.axvspan(t[s0], t[s1], color="orange", alpha=0.10)
    ax.set_title(f"{repr_ds} 主通道 ch{mc} — {title}")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "d1_main_channel_compare.png"), dpi=140)
plt.close(fig)

# ---------------- 图d2: 负载段放大对比（3数据集 × 在线/离线） ----------------
fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = store[name]
    t, s0, s1, mc = R["t"], R["s0"], R["s1"], R["main_ch"]
    for c, (group, title) in enumerate([(online_list, "在线/盲方法"),
                                        (offline_list, "离线/模型方法")]):
        ax = axes[r, c]
        for aname in group:
            if aname not in R["results"]:
                continue
            sig = R["results"][aname][s0:s1, mc]
            ax.plot(t[s0:s1] - t[s0], sig, lw=1.5 if aname == "原始" else 0.9,
                    label=aname, color="black" if aname == "原始" else None, alpha=0.85)
        ax.set_title(f"{name} 负载段放大 — {title}")
        ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force (N)")
        ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "d2_load_segment_zoom.png"), dpi=140)
plt.close(fig)

# ---------------- 图d3: 指标条形图 ----------------
metrics_show = ["负载段残余漂移%", "阶跃保真度%", "零漂残余%"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
algos_order = summary["算法"].tolist()
colors = {"online": "tab:blue", "offline": "tab:orange"}
for ax, met in zip(axes, metrics_show):
    vals = [summary.loc[summary["算法"] == a, met].iloc[0] for a in algos_order]
    types = [summary.loc[summary["算法"] == a, "类型"].iloc[0] for a in algos_order]
    ax.barh(range(len(algos_order)), vals,
            color=[colors[tp] for tp in types], alpha=0.85)
    ax.set_yticks(range(len(algos_order)))
    ax.set_yticklabels(algos_order, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"{met}（3数据集平均，越小越好）" if met != "阶跃保真度%"
                 else f"{met}（越接近100%越好）")
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=7.5)
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="在线/盲"),
                        plt.Rectangle((0, 0), 1, 1, color="tab:orange", label="离线/模型")],
               fontsize=8, loc="lower right")
fig.savefig(os.path.join(FIG, "d3_metrics_bar.png"), dpi=140)
plt.close(fig)

# ---------------- 图d4: 残余漂移热图（算法×数据集） ----------------
pivot = M.pivot_table(index="算法", columns="数据集", values="负载段残余漂移%")
pivot = pivot.loc[algos_order]
fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
ax.set_title("负载段残余漂移% （算法×数据集，越小越好）")
plt.colorbar(im, ax=ax)
fig.savefig(os.path.join(FIG, "d4_drift_heatmap.png"), dpi=140)
plt.close(fig)

# ---------------- 图d5: 最优组合 vs 原始（全阵列热图+主通道） ----------------
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
mask_str = "110110011011101111110111111011111100000001000000100000010000001"
ones = [i for i, b in enumerate(mask_str) if b == "1"]
for c, name in enumerate(DATASETS):
    R = store[name]
    X, s0, s1, t = R["X"], R["s0"], R["s1"], R["t"]
    mc = R["main_ch"]
    Ybest = R["results"]["指数蠕变拟合扣除"]
    # 负载末段响应热图（原始 vs 处理后）：末10%均值 - 首10%均值 = 残余漂移图
    n10 = max((s1 - s0) // 10, 10)
    for rr, (Mat, ttl) in enumerate([(X, "原始"), (Ybest, "指数蠕变扣除后")]):
        m = np.full(63, np.nan)
        drift_arr = Mat[s1 - n10:s1].mean(axis=0) - Mat[s0:s0 + n10].mean(axis=0)
        for j, pos in enumerate(ones):
            m[pos] = drift_arr[j]
        im = axes[rr, c].imshow(m.reshape(9, 7), cmap="coolwarm", aspect="auto",
                                vmin=-0.5, vmax=0.5)
        axes[rr, c].set_title(f"{name} {ttl}\n负载段漂移量(N)")
        plt.colorbar(im, ax=axes[rr, c])
fig.savefig(os.path.join(FIG, "d5_array_drift_before_after.png"), dpi=140)
plt.close(fig)

print("\nsaved d1~d5 figures + metrics csv")
