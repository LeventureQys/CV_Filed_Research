# -*- coding: utf-8 -*-
"""GLM53 分析第三步：新数据（左拇指指尖 / 四指指尖）上验证已推荐算法

目的：检验在右拇指指尖数据上开发/调参的 7 种因果算法，泛化到新传感器
（不同 layout、不同蠕变强度：左拇指 ~+5%、四指 ~+10-15%，右拇指 ~+32-45%）
是否依然有效、是否过补偿。

参评算法（7 种，全部因果，与 b_compare.py 相同实现）：
  raw / ema_hp / kalman2 / log_creep / ref_common / creep_field / creep_hybrid
（新数据无 kalman_compensated.csv，故不含内置Kalman）

输出:
  figures/c1_left_timeseries.png, c1_four_timeseries.png
  figures/c2_left_zoom.png,      c2_four_zoom.png
  figures/c3_validation_bars.png
  figures/c4_left_heatmap.png,   c4_four_heatmap.png
  results/c_validation_metrics.csv, c_validation_summary.csv
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
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                    # GLM53
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")
LOCATIONS = ["左拇指指尖", "四指指尖"]
LOCS_ZH = {"左拇指指尖": "左拇指指尖(9×7)", "四指指尖": "四指指尖(8×5)"}
DATASETS = ["数据1", "数据2", "数据3"]
TAU_CREEP = 5.0
TAU_EMA_HP = 60.0
TAU_KF_C = 40.0

ALGO_LABELS = {
    "raw": "原始(无补偿)",
    "ema_hp": "EMA高通 τ=60s",
    "kalman2": "双状态Kalman τc=40s",
    "log_creep": "在线对数蠕变模型",
    "ref_common": "参考通道共模扣除",
    "creep_field": "负载比例蠕变场(阵列)",
    "creep_hybrid": "混合:场形状+逐通道增益",
}
ALGO_ORDER = list(ALGO_LABELS.keys())


# ---------- 工具（与 b_compare.py 一致） ----------
def load_csv(path):
    return pd.read_csv(path, skiprows=24)


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


def ema_track(x, dt, tau):
    a = np.clip(dt / tau, 0, 1)
    y = np.empty_like(x)
    b = x[0].copy()
    for i in range(len(x)):
        if i:
            b = b + a[i] * (x[i] - b)
        y[i] = b
    return y


def kalman2(X, dt, R, tau_f=0.3, tau_c=TAU_KF_C):
    n, m = X.shape
    f = X[0].copy()
    c = np.zeros(m)
    Pf = Pc = R
    c_hist = np.empty((n, m))
    c_hist[0] = c
    for i in range(n):
        if i:
            d = dt[i]
            qf = 4 * R * d / tau_f
            qc = 4 * R * d / tau_c
            S = (Pf + qf) + (Pc + qc) + R
            Kf = (Pf + qf) / S
            Kc = (Pc + qc) / S
            inn = X[i] - (f + c)
            f = f + Kf * inn
            c = c + Kc * inn
            Pf = (Pf + qf) * (1 - Kf)
            Pc = (Pc + qc) * (1 - Kc)
        c_hist[i] = c
    return c_hist


def run_algorithms(X, t, s0, s1):
    """返回 {algo: Y}, A, loaded_ch, baseline。全部因果（与 b_compare.py 相同）。"""
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
    Ys = {}

    Ys["raw"] = X.copy()
    b = ema_track(X, dt, TAU_EMA_HP)
    Ys["ema_hp"] = X - b
    R = max(X[: s0].std(axis=0).mean() ** 2, 1e-6)
    Ys["kalman2"] = X - kalman2(X, dt, R)

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
    ref_ch = ~loaded_ch

    # 在线对数蠕变
    Y = Z.copy()
    phi2_acc = np.zeros(m)
    phid_acc = np.zeros(m)
    u_prev = 0.0
    for i in range(s0, s1):
        u = t[i] - t[s0]
        du = u - u_prev
        if u >= 1.0:
            phi = np.log1p(u / TAU_CREEP)
            d = Z[i] - A
            phi2_acc += du * phi * phi
            phid_acc += du * phi * d
            c_hat_ch = np.where(phi2_acc > 1e-8, phid_acc / np.maximum(phi2_acc, 1e-8), 0.0)
            if u > 10.0:
                creep = np.clip(c_hat_ch * phi, -0.1 * A, 0.9 * A)
                Y[i] = Z[i] - creep
        u_prev = u
    Ys["log_creep"] = Y

    # 参考通道共模扣除
    if ref_ch.any():
        common = np.median(Z[:, ref_ch], axis=1)
    else:
        common = np.zeros(n)
    Ys["ref_common"] = Z - common[:, None]

    # 蠕变场
    Y = Z.copy()
    g_smooth = 0.0
    a_g = np.clip(dtm / 3.0, 0, 1)
    A_ld = A[loaded_ch]
    for i in range(s0, s1):
        rel = (Z[i, loaded_ch] - A_ld) / A_ld
        g_raw = np.median(rel)
        g_smooth = g_smooth + a_g * (g_raw - g_smooth)
        creep = np.clip(A * g_smooth, -0.5 * A, 1.5 * A)
        Y[i] = Z[i] - creep
    Ys["creep_field"] = Y

    # 混合：场形状 + 逐通道增益
    Y = Z.copy()
    g_smooth = 0.0
    a_g = np.clip(dtm / 3.0, 0, 1)
    gamma = np.ones(m)
    g2_acc = 0.0
    g_rel_acc = np.zeros(m)
    A_safe = np.where(np.abs(A) > 1e-9, A, 1.0)
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
    Ys["creep_hybrid"] = Y

    return Ys, A, loaded_ch, baseline


def evaluate(Y, X, t, s0, s1, A, loaded_ch, main):
    nL = s1 - s0
    L = Y[s0:s1]
    amp = A[main]
    dtm = np.median(np.diff(t))
    drift = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
    drift_main = 100 * drift[main] / amp
    drift_loaded = 100 * np.median(drift[loaded_ch] / A[loaded_ch])
    base_y = Y[:s0, main].mean()
    base_x = X[:s0, main].mean()
    i1 = s0 + int(0.5 / dtm)
    i2 = s0 + int(2.5 / dtm)
    step_ratio = ((Y[i1:i2, main].mean() - base_y) / (X[i1:i2, main].mean() - base_x)
                  if abs(X[i1:i2, main].mean() - base_x) > 1e-9 else np.nan)
    tt = t[s0:s1] - t[s0]

    def dstd(sig, tts):
        k, b0 = np.polyfit(tts, sig, 1)
        return (sig - (k * tts + b0)).std()

    noise_y = dstd(L[10:, main], tt[10:])
    noise_x = dstd(X[s0 + 10: s1, main], tt[10:])
    noise_ratio = noise_y / noise_x if noise_x > 1e-12 else np.nan
    j1 = s1 + int(5.0 / dtm)
    j2 = min(len(t), s1 + int(30.0 / dtm))
    zero_resid = 100 * (Y[j1:j2, main].mean() - base_y) / amp
    return dict(drift_main=drift_main, drift_loaded=drift_loaded,
                step_ratio=step_ratio, noise_ratio=noise_ratio,
                flat_main=100 * noise_y / amp, zero_resid=zero_resid)


# ---------- 主流程 ----------
all_metrics = []
store = {}
layouts = {}
for loc in LOCATIONS:
    for name in DATASETS:
        ddir = os.path.join(BASE, loc, name)
        df = load_csv(os.path.join(ddir, "device_001_seg000.csv"))
        meta = json.load(open(os.path.join(ddir, "session.json"), encoding="utf-8"))["devices"][0]
        layouts[loc] = (meta["rows"], meta["cols"], meta["layout_mask"])
        t = df["elapsed"].to_numpy()
        ch_cols = [c for c in df.columns if c.startswith("ch")]
        X = df[ch_cols].to_numpy()
        s0, s1 = find_segment(X.sum(axis=1))
        # 主通道自动选择（受载幅度最大者）
        main = int(np.argmax(X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)))
        Ys, A, loaded_ch, baseline = run_algorithms(X, t, s0, s1)
        store[(loc, name)] = dict(t=t, X=X, Ys=Ys, s0=s0, s1=s1, A=A,
                                  loaded_ch=loaded_ch, main=main)
        for algo, Y in Ys.items():
            met = evaluate(Y, X, t, s0, s1, A, loaded_ch, main)
            met.update(location=loc, dataset=name, algo=algo, main_ch=main)
            all_metrics.append(met)
        print(f"[{loc}/{name}] main=ch{main}  " + "  ".join(
            f"{r['algo']}={r['drift_main']:+.1f}%" for r in all_metrics
            if r["location"] == loc and r["dataset"] == name))

mdf = pd.DataFrame(all_metrics)
mdf["algo_label"] = mdf["algo"].map(ALGO_LABELS)
mdf.to_csv(os.path.join(RES, "c_validation_metrics.csv"), index=False, encoding="utf-8-sig")

agg = mdf.groupby(["location", "algo_label"]).agg(
    时漂残余_主通道=("drift_main", lambda s: f"{s.abs().mean():.1f}%"),
    时漂残余_受载中位=("drift_loaded", lambda s: f"{s.abs().mean():.1f}%"),
    阶跃保真=("step_ratio", lambda s: f"{s.mean():.2f}"),
    噪声比=("noise_ratio", lambda s: f"{s.mean():.2f}"),
    平坦度=("flat_main", lambda s: f"{s.mean():.1f}%"),
    零漂残余=("zero_resid", lambda s: f"{s.abs().mean():.1f}%"),
).reset_index()
agg.to_csv(os.path.join(RES, "c_validation_summary.csv"), index=False, encoding="utf-8-sig")
print("\n===== 新数据验证汇总（各位置三组平均） =====")
print(agg.to_string(index=False))

colors = plt.cm.tab10.colors
ds_alpha = {"数据1": 1.0, "数据2": 0.65, "数据3": 0.4}
loc_color = {"左拇指指尖": "tab:blue", "四指指尖": "tab:red"}

# ---------- 图 c1: 各位置主通道全时序 ----------
for loc in LOCATIONS:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    for r, name in enumerate(DATASETS):
        d = store[(loc, name)]
        main = d["main"]
        ax = axes[r]
        for k, algo in enumerate(ALGO_ORDER):
            ax.plot(d["t"], d["Ys"][algo][:, main], lw=0.7, alpha=0.85,
                    color=colors[k], label=ALGO_LABELS[algo])
        ax.axvspan(d["t"][d["s0"]], d["t"][d["s1"]], color="orange", alpha=0.06)
        ax.set_title(f"{LOCS_ZH[loc]} · {name} 主通道 ch{main}（恒载 {d['t'][d['s1']]-d['t'][d['s0']]:.0f}s）", fontsize=11)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("读数")
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.savefig(os.path.join(FIG, f"c1_{'left' if '左' in loc else 'four'}_timeseries.png"), dpi=140)
    plt.close(fig)

# ---------- 图 c2: 负载段放大（各位置数据2） ----------
for loc in LOCATIONS:
    d = store[(loc, "数据2")]
    main = d["main"]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    tt = d["t"][d["s0"]:d["s1"]] - d["t"][d["s0"]]
    for k, algo in enumerate(ALGO_ORDER):
        ax.plot(tt, d["Ys"][algo][d["s0"]:d["s1"], main], lw=0.8, alpha=0.85,
                color=colors[k], label=ALGO_LABELS[algo])
    ax.set_title(f"{LOCS_ZH[loc]} · 数据2 主通道 ch{main} 负载段放大", fontsize=11)
    ax.set_xlabel("负载持续时间 (s)")
    ax.set_ylabel("读数")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(os.path.join(FIG, f"c2_{'left' if '左' in loc else 'four'}_zoom.png"), dpi=140)
    plt.close(fig)

# ---------- 图 c3: 指标横向对比 ----------
fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
met_defs = [
    ("drift_main", "时漂残余 |末-首/幅|（主通道，%）", True),
    ("drift_loaded", "时漂残余（受载通道中位，%）", True),
    ("step_ratio", "阶跃保真（1.0=完美）", False),
    ("noise_ratio", "噪声放大比", False),
    ("flat_main", "负载段平坦度（%）", True),
    ("zero_resid", "零漂残余 |残差/幅|（%）", True),
]
xs = np.arange(len(ALGO_ORDER))
for ax, (met, title, use_abs) in zip(axes.flat, met_defs):
    w = 0.13
    gi = 0
    for loc in LOCATIONS:
        for name in DATASETS:
            vals = []
            for algo in ALGO_ORDER:
                row = mdf[(mdf.location == loc) & (mdf.dataset == name) & (mdf.algo == algo)][met].iloc[0]
                vals.append(abs(row) if use_abs else row)
            ax.bar(xs + (gi - 2.5) * w, vals, width=w * 0.9,
                   color=loc_color[loc], alpha=ds_alpha[name],
                   label=f"{loc}/{name}" if gi < 2 or gi == 3 else None)
            gi += 1
    ax.set_xticks(xs)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGO_ORDER], rotation=38, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    if met in ("step_ratio", "noise_ratio"):
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
# 图例（去重）
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color="tab:blue", label="左拇指指尖"),
           mpatches.Patch(color="tab:red", label="四指指尖")]
axes.flat[0].legend(handles=handles + [mpatches.Patch(color="gray", alpha=1.0, label="数据1"),
                                       mpatches.Patch(color="gray", alpha=0.65, label="数据2"),
                                       mpatches.Patch(color="gray", alpha=0.4, label="数据3")], fontsize=7)
fig.savefig(os.path.join(FIG, "c3_validation_bars.png"), dpi=140)
plt.close(fig)

# ---------- 图 c4: 阵列漂移热图 raw vs hybrid ----------
for loc in LOCATIONS:
    rows, cols, mask = layouts[loc]
    mask_arr = np.array([c == "1" for c in mask]).reshape(rows, cols)
    pos = {}
    ci = 0
    for r_ in range(rows):
        for c_ in range(cols):
            if mask_arr[r_, c_]:
                pos[(r_, c_)] = ci
                ci += 1
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for di, name in enumerate(DATASETS):
        d = store[(loc, name)]
        nL = d["s1"] - d["s0"]
        for row, algo in enumerate(["raw", "creep_hybrid"]):
            L = d["Ys"][algo][d["s0"]:d["s1"]]
            drift = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
            pct = np.full((rows, cols), np.nan)
            for (r_, c_), idx in pos.items():
                pct[r_, c_] = 100 * drift[idx] / d["A"][idx] if abs(d["A"][idx]) > 0.05 else np.nan
            ax = axes[row, di]
            vmax = 40
            im = ax.imshow(np.clip(pct, -vmax, vmax), cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
            for (r_, c_), idx in pos.items():
                if abs(d["A"][idx]) > 0.05:
                    ax.text(c_, r_, f"{pct[r_, c_]:.0f}", ha="center", va="center", fontsize=6.5)
            ax.set_title(f"{name} · {ALGO_LABELS[algo]}", fontsize=10)
            ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    fig.colorbar(im, ax=axes, shrink=0.6, label="负载段漂移占满幅 (%)")
    fig.savefig(os.path.join(FIG, f"c4_{'left' if '左' in loc else 'four'}_heatmap.png"), dpi=140)
    plt.close(fig)

print("\nsaved: c1/c2 (timeseries, zoom), c3_validation_bars, c4 heatmaps")
print("results: c_validation_metrics.csv / c_validation_summary.csv")
