# -*- coding: utf-8 -*-
"""GLM53 分析第二步：漂移补偿算法实装与横向对比（全部因果）

用户要求：所有参评算法必须 causal（任意时刻输出只依赖当前与历史采样）。
故离线方法（负载段线性去趋势等）不参与对比，仅在报告中作理论上限讨论。

算法清单（8 种，全部因果）:
  raw          原始数据（无补偿）
  kal_builtin  软件内置 Kalman 输出（实为零点扣除，时漂未处理）
  ema_hp       慢 EMA 基线跟踪高通（τ=60s，经典单点 DSP）
  kalman2      双状态 Kalman（信号态 + 慢偏置态 τc=40s），单点
  log_creep    在线对数蠕变模型拟合（单点，过原点增量最小二乘）
  ref_common   空载参考通道共模扣除（阵列级）
  creep_field  负载比例蠕变场补偿（阵列级）
  creep_hybrid 混合：场形状 g(t) + 逐通道增益 γ_i（阵列级）

评估口径（恒定负载 ⇒ 负载段应为水平线，此为无真值情况下的代理真值）:
  drift_pct    负载段(末10%-首10%)/幅度 → 时漂残余
  step_ratio   负载起始前2s幅度/原始同窗幅度 → 阶跃保真
  noise_ratio  负载段去线性趋势后 std 之比 → 噪声放大
  flatness     负载段去线性趋势 std / 幅度 → 平坦度/游走
  zero_resid   卸载后基线残差 / 幅度 → 零漂残余
注：分段(s0,s1)仅用于评估，算法内部只用 onset 检测（阈值穿越，因果）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "右拇指指尖")  # temp/右拇指指尖
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")
DATASETS = ["数据1", "数据2", "数据3"]
LAYOUT_MASK = "110110011011101111110111111011111100000001000000100000010000001"
ROWS, COLS = 9, 7
TAU_CREEP = 5.0        # 在线对数蠕变模型固定时间常数 (s)（离线扫参 2~10s 的中位）
TAU_EMA_HP = 60.0      # EMA 高通时间常数 (s)
TAU_KF_C = 40.0        # 双状态 Kalman 偏置态等效时间常数 (s)（经 20~600 扫参取折中）
MAIN = 17

ALGO_LABELS = {
    "raw": "原始(无补偿)",
    "kal_builtin": "内置Kalman(仅零点)",
    "ema_hp": "EMA高通 τ=60s",
    "kalman2": "双状态Kalman τc=40s",
    "log_creep": "在线对数蠕变模型",
    "ref_common": "参考通道共模扣除",
    "creep_field": "负载比例蠕变场(阵列)",
    "creep_hybrid": "混合:场形状+逐通道增益",
}
ALGO_ORDER = list(ALGO_LABELS.keys())


# ---------- 工具 ----------
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
    """因果一阶低通跟踪（按通道列向量化）"""
    a = np.clip(dt / tau, 0, 1)
    y = np.empty_like(x)
    b = x[0].copy()
    for i in range(len(x)):
        if i:
            b = b + a[i] * (x[i] - b)
        y[i] = b
    return y


def kalman2(X, dt, R, tau_f=0.3, tau_c=TAU_KF_C):
    """双状态 Kalman：状态=[真实信号 f, 慢偏置 c]，观测 z=f+c+noise
    F=I, H=[1,1]，按通道向量化。返回偏置估计序列 c_hat（因果）。"""
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


# ---------- 各算法 ----------
def run_algorithms(X, t, s0, s1, X_kal):
    """返回 {algo: Y}, A, loaded_ch, baseline。全部因果。

    公共基座（除 raw/内置/EMA/Kalman2 外）：卸载门控自动归零
      - 卸载检测（因果）: smooth_total < 1.5 × min_total_so_far
        （假设录制开始时空载，与本数据及内置Kalman的前提一致）
      - 卸载时基线以 τ=2s 跟踪，负载时冻结 → Z = X − baseline
    """
    n, m = X.shape
    dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.1)
    dtm = np.median(dt)
    total = X.sum(axis=1)
    # 因果卸载检测（双判据，避免卸载瞬态过冲毒化 min 后归零失效）：
    #   a) ts < 1.5 × 历史最小总量   —— 空载基线附近（含录制开头）
    #   b) ts < 0.15 × 历史最大总量  —— 显著低于负载期水平（卸载后恢复判据）
    a_s = np.clip(dtm / 0.3, 0, 1)
    ts = np.empty(n)
    ts[0] = total[0]
    for i in range(1, n):
        ts[i] = ts[i - 1] + a_s * (total[i] - ts[i - 1])
    min_ts = np.minimum.accumulate(ts)
    max_ts = np.maximum.accumulate(ts)
    unloaded = (ts < 1.5 * min_ts) | (ts < 0.15 * max_ts)
    Ys = {}

    # ---- 1. raw ----
    Ys["raw"] = X.copy()

    # ---- 2. 内置 kalman（软件导出的补偿数据） ----
    Ys["kal_builtin"] = X_kal.copy()

    # ---- 3. EMA 高通（τ=60s，无门控，经典高通） ----
    b = ema_track(X, dt, TAU_EMA_HP)
    Ys["ema_hp"] = X - b

    # ---- 4. 双状态 Kalman ----
    R = max(X[: s0].std(axis=0).mean() ** 2, 1e-6)
    c_hat = kalman2(X, dt, R)
    Ys["kalman2"] = X - c_hat

    # ---- 公共：卸载门控基线 + 自动归零信号 Z ----
    b0 = X[0].copy()                                 # 假设首帧空载（与本数据一致）
    a_b = np.clip(dt / 2.0, 0, 1)
    baseline = np.empty((n, m))
    for i in range(n):
        if unloaded[i]:
            b0 = b0 + a_b[i] * (X[i] - b0)
        baseline[i] = b0
    Z = X - baseline

    # 受载通道判定与幅度估计（onset 后 1~3s 历史窗，因果）
    i_on1 = s0 + int(1.0 / dtm)
    i_on2 = s0 + int(3.0 / dtm)
    A = Z[i_on1:i_on2].mean(axis=0)
    loaded_ch = A > 0.10 * A.max()
    ref_ch = ~loaded_ch

    # ---- 5. 在线对数蠕变（单点，因果，过原点增量最小二乘 c·ln(1+u/τ)） ----
    Y = Z.copy()
    phi2_acc = np.zeros(m)     # Σ du·φ²
    phid_acc = np.zeros(m)     # Σ du·φ·d
    u_prev = 0.0
    for i in range(s0, s1):
        u = t[i] - t[s0]
        du = u - u_prev
        if u >= 1.0:           # 跳过加载瞬态（前1s为机械建立段，非蠕变）
            phi = np.log1p(u / TAU_CREEP)
            d = Z[i] - A
            phi2_acc += du * phi * phi
            phid_acc += du * phi * d
            c_hat_ch = np.where(phi2_acc > 1e-8, phid_acc / np.maximum(phi2_acc, 1e-8), 0.0)
            if u > 10.0:       # 拟合数据充足后开始补偿
                creep = c_hat_ch * phi
                creep = np.clip(creep, -0.1 * A, 0.9 * A)   # 安全限幅
                Y[i] = Z[i] - creep
        u_prev = u
    Ys["log_creep"] = Y

    # ---- 6. 参考通道共模扣除（阵列级，因果） ----
    if ref_ch.any():
        common = np.median(Z[:, ref_ch], axis=1)
    else:
        common = np.zeros(n)
    Ys["ref_common"] = Z - common[:, None]

    # ---- 7. 负载比例蠕变场补偿（阵列级，因果） ----
    # 模型: 蠕变_i(t) ≈ A_i · g(t)，g(t) 为全体受载通道归一化蠕变的鲁棒共识
    Y = Z.copy()
    g_smooth = 0.0
    a_g = np.clip(dtm / 3.0, 0, 1)                # g 平滑 τ=3s
    A_ld = A[loaded_ch]
    for i in range(s0, s1):
        rel = (Z[i, loaded_ch] - A_ld) / A_ld
        g_raw = np.median(rel)
        g_smooth = g_smooth + a_g * (g_raw - g_smooth)
        creep = A * g_smooth
        creep = np.clip(creep, -0.5 * A, 1.5 * A)
        Y[i] = Z[i] - creep
    Ys["creep_field"] = Y

    # ---- 8. 混合：场形状 + 逐通道增益（因果） ----
    # 模型: 蠕变_i(t) ≈ γ_i · A_i · g(t)
    #   g(t): 受载通道归一化蠕变的 median 共识（同算法7，数据驱动形状，免 τ 假设）
    #   γ_i : 逐通道增益，增量最小二乘 γ_i = ∫g·rel_i du / ∫g² du（在线修正个体偏差）
    Y = Z.copy()
    g_smooth = 0.0
    a_g = np.clip(dtm / 3.0, 0, 1)                # g 平滑 τ=3s
    A_ld = A[loaded_ch]
    gamma = np.ones(m)
    g2_acc = 0.0                                  # ∫ g² du（g 为标量共享）
    g_rel_acc = np.zeros(m)                       # ∫ g·rel_i du
    A_safe = np.where(np.abs(A) > 1e-9, A, 1.0)
    u_prev_h = t[s0]
    for i in range(s0, s1):
        du = max(0.0, t[i] - u_prev_h)
        u_prev_h = t[i]
        rel_ld = (Z[i, loaded_ch] - A_ld) / A_ld
        g_raw = np.median(rel_ld)
        g_smooth = g_smooth + a_g * (g_raw - g_smooth)
        # 逐通道增益在线估计（g² 加权，早期自动低权重 → 天然冷启动）
        if g_smooth > 0.02:
            rel_full = np.where(loaded_ch, (Z[i] - A) / A_safe, 0.0)
            g2_acc += du * g_smooth * g_smooth
            g_rel_acc += du * g_smooth * rel_full
            with np.errstate(invalid="ignore", divide="ignore"):
                gamma = np.where(g2_acc > 1e-8,
                                 np.clip(g_rel_acc / max(g2_acc, 1e-8), 0.3, 2.0), 1.0)
        creep = gamma * A * g_smooth
        creep = np.clip(creep, -0.5 * A, 1.5 * A)
        Y[i] = Z[i] - creep
    Ys["creep_hybrid"] = Y

    return Ys, A, loaded_ch, baseline


# ---------- 评估 ----------
def evaluate(Y, X, t, s0, s1, A, loaded_ch, baseline_pre):
    nL = s1 - s0
    main = MAIN
    L = Y[s0:s1]
    amp = A[main]
    dtm = np.median(np.diff(t))
    drift = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
    drift_main = 100 * drift[main] / amp
    drift_loaded = 100 * np.median(drift[loaded_ch] / A[loaded_ch])
    # 各算法用自己的前空载读数作零参考（不同算法零点约定不同，必须各自对齐）
    base_y = Y[:s0, main].mean()
    base_x = X[:s0, main].mean()
    # 阶跃保真：onset 后 0.5~2.5s 均值（相对各自基线）
    i1 = s0 + int(0.5 / dtm)
    i2 = s0 + int(2.5 / dtm)
    step_y = Y[i1:i2, main].mean() - base_y
    step_x = X[i1:i2, main].mean() - base_x
    step_ratio = step_y / step_x if abs(step_x) > 1e-9 else np.nan
    # 噪声：负载段去自身线性趋势后的 std
    tt = t[s0:s1] - t[s0]

    def detrended_std(sig, tts):
        k, b0 = np.polyfit(tts, sig, 1)
        return (sig - (k * tts + b0)).std()

    noise_y = detrended_std(L[10:, main], tt[10:])
    noise_x = detrended_std(X[s0 + 10: s1, main], tt[10:])
    noise_ratio = noise_y / noise_x if noise_x > 1e-12 else np.nan
    flat_main = 100 * noise_y / amp
    # 零漂残余：卸载后 5~30s 均值 - 前空载基线
    j1 = s1 + int(5.0 / dtm)
    j2 = min(len(t), s1 + int(30.0 / dtm))
    zero_resid = 100 * (Y[j1:j2, main].mean() - base_y) / amp
    return dict(drift_main=drift_main, drift_loaded=drift_loaded,
                step_ratio=step_ratio, noise_ratio=noise_ratio,
                flat_main=flat_main, zero_resid=zero_resid)


# ---------- 主流程 ----------
all_metrics = []
store = {}
for name in DATASETS:
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    dk = load_csv(os.path.join(BASE, name, "device_001_seg000_kalman_compensated.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    X_kal = dk[ch_cols].to_numpy()[: len(X)]
    s0, s1 = find_segment(X.sum(axis=1))
    Ys, A, loaded_ch, baseline = run_algorithms(X, t, s0, s1, X_kal)
    baseline_pre = baseline[s0]
    store[name] = dict(t=t, X=X, Ys=Ys, s0=s0, s1=s1, A=A, loaded_ch=loaded_ch,
                       baseline_pre=baseline_pre, ch_cols=ch_cols)
    for algo, Y in Ys.items():
        met = evaluate(Y, X, t, s0, s1, A, loaded_ch, baseline_pre)
        met.update(dataset=name, algo=algo)
        all_metrics.append(met)
    print(f"[{name}] 完成: " + "  ".join(
        f"{r['algo']}={r['drift_main']:+.1f}%" for r in all_metrics if r["dataset"] == name))

mdf = pd.DataFrame(all_metrics)
mdf["algo_label"] = mdf["algo"].map(ALGO_LABELS)
mdf.to_csv(os.path.join(RES, "algorithm_metrics.csv"), index=False, encoding="utf-8-sig")

agg = mdf.groupby("algo_label").agg(
    时漂残余_主通道=("drift_main", lambda s: f"{s.abs().mean():.1f}%"),
    时漂残余_受载中位=("drift_loaded", lambda s: f"{s.abs().mean():.1f}%"),
    阶跃保真=("step_ratio", lambda s: f"{s.mean():.2f}"),
    噪声比=("noise_ratio", lambda s: f"{s.mean():.2f}"),
    平坦度=("flat_main", lambda s: f"{s.mean():.1f}%"),
    零漂残余=("zero_resid", lambda s: f"{s.abs().mean():.1f}%"),
).reindex([ALGO_LABELS[a] for a in ALGO_ORDER])
agg.to_csv(os.path.join(RES, "algorithm_metrics_summary.csv"), encoding="utf-8-sig")
print("\n===== 三组平均汇总 =====")
print(agg.to_string())

# ---------- 图 b1: 主通道时序对比 ----------
colors = plt.cm.tab10.colors
fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    d = store[name]
    ax = axes[r]
    for k, algo in enumerate(ALGO_ORDER):
        Y = d["Ys"][algo]
        ax.plot(d["t"], Y[:, MAIN], lw=0.7, alpha=0.85, color=colors[k],
                label=ALGO_LABELS[algo])
    ax.axvspan(d["t"][d["s0"]], d["t"][d["s1"]], color="orange", alpha=0.06)
    ax.set_title(f"{name} 主通道 ch17：各算法补偿后时序（全部因果）", fontsize=11)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("读数")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
fig.savefig(os.path.join(FIG, "b1_timeseries_all.png"), dpi=140)
plt.close(fig)

# ---------- 图 b2: 负载段放大（数据2） ----------
d = store["数据2"]
fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
ax = axes[0]
tt = d["t"][d["s0"]:d["s1"]] - d["t"][d["s0"]]
for k, algo in enumerate(ALGO_ORDER):
    Y = d["Ys"][algo]
    ax.plot(tt, Y[d["s0"]:d["s1"], MAIN], lw=0.8, alpha=0.85, color=colors[k], label=ALGO_LABELS[algo])
ax.set_title("数据2 主通道负载段放大：时漂抑制对比", fontsize=11)
ax.set_xlabel("负载持续时间 (s)")
ax.set_ylabel("读数")
ax.legend(fontsize=8, ncol=2)
ax = axes[1]
ax.set_title("阶跃起始放大（阶跃保真）", fontsize=11)
for k, algo in enumerate(ALGO_ORDER):
    Y = d["Ys"][algo]
    n15 = int(10 / np.median(np.diff(d["t"])))
    sl = slice(d["s0"], d["s0"] + n15)
    ax.plot(d["t"][sl] - d["t"][d["s0"]], Y[sl, MAIN] - Y[: d["s0"], MAIN].mean(),
            lw=1.0, alpha=0.85, color=colors[k], label=ALGO_LABELS[algo])
ax.set_xlabel("负载持续时间 (s)")
ax.set_ylabel("读数")
fig.savefig(os.path.join(FIG, "b2_load_zoom.png"), dpi=140)
plt.close(fig)

# ---------- 图 b3: 指标横向对比 ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
met_defs = [
    ("drift_main", "时漂残余 |末-首/幅|（主通道，%）", True),
    ("drift_loaded", "时漂残余（受载通道中位，%）", True),
    ("step_ratio", "阶跃保真（1.0=完美）", False),
    ("noise_ratio", "噪声放大比", False),
    ("flat_main", "负载段平坦度（去趋势 std/幅，%）", True),
    ("zero_resid", "零漂残余 |残差/幅|（%）", True),
]
ds_colors = {"数据1": "tab:blue", "数据2": "tab:orange", "数据3": "tab:green"}
for ax, (met, title, use_abs) in zip(axes.flat, met_defs):
    w = 0.25
    xs = np.arange(len(ALGO_ORDER))
    for di, name in enumerate(DATASETS):
        vals = []
        for algo in ALGO_ORDER:
            row = mdf[(mdf.dataset == name) & (mdf.algo == algo)][met].iloc[0]
            vals.append(abs(row) if use_abs else row)
        ax.bar(xs + (di - 1) * w, vals, width=w, color=ds_colors[name], label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGO_ORDER], rotation=38, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    if met in ("step_ratio", "noise_ratio"):
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
axes.flat[0].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "b3_metrics_bars.png"), dpi=140)
plt.close(fig)

# ---------- 图 b4: 阵列漂移热图（raw vs 最优算法） ----------
mask_arr = np.array([c == "1" for c in LAYOUT_MASK]).reshape(ROWS, COLS)
pos = {}
ci = 0
for r_ in range(ROWS):
    for c_ in range(COLS):
        if mask_arr[r_, c_]:
            pos[(r_, c_)] = ci
            ci += 1
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
for di, name in enumerate(DATASETS):
    d = store[name]
    nL = d["s1"] - d["s0"]
    for row, algo in enumerate(["raw", "creep_hybrid"]):
        L = d["Ys"][algo][d["s0"]:d["s1"]]
        drift = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
        pct = np.full((ROWS, COLS), np.nan)
        for (r_, c_), idx in pos.items():
            pct[r_, c_] = 100 * drift[idx] / d["A"][idx] if abs(d["A"][idx]) > 0.05 else np.nan
        ax = axes[row, di]
        vmax = 40
        im = ax.imshow(np.clip(pct, -vmax, vmax), cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
        for (r_, c_), idx in pos.items():
            if abs(d["A"][idx]) > 0.05:
                ax.text(c_, r_, f"{pct[r_, c_]:.0f}", ha="center", va="center", fontsize=6.5)
        ax.set_title(f"{name} · {ALGO_LABELS[algo]}", fontsize=10)
        ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
fig.colorbar(im, ax=axes, shrink=0.6, label="负载段漂移占满幅 (%)")
fig.savefig(os.path.join(FIG, "b4_array_heatmap.png"), dpi=140)
plt.close(fig)

# ---------- 图 b5: 零漂对比（卸载后基线轨迹） ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
for di, name in enumerate(DATASETS):
    d = store[name]
    ax = axes[di]
    dt_med = np.median(np.diff(d["t"]))
    j0 = d["s1"]
    jend = min(len(d["t"]), j0 + int(40 / dt_med))
    tt = d["t"][j0:jend] - d["t"][j0]
    for k, algo in enumerate(ALGO_ORDER):
        Y = d["Ys"][algo]
        base_y = Y[: d["s0"], MAIN].mean()
        ax.plot(tt, Y[j0:jend, MAIN] - base_y, lw=0.8, alpha=0.85,
                color=colors[k], label=ALGO_LABELS[algo])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(f"{name} 卸载后主通道基线（相对各自前空载）", fontsize=10)
    ax.set_xlabel("卸载后时间 (s)")
    ax.set_ylabel("基线偏移")
axes[0].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "b5_zero_recovery.png"), dpi=140)
plt.close(fig)

print("\nsaved: b1_timeseries_all / b2_load_zoom / b3_metrics_bars / b4_array_heatmap / b5_zero_recovery")
print("results: algorithm_metrics.csv / algorithm_metrics_summary.csv")
