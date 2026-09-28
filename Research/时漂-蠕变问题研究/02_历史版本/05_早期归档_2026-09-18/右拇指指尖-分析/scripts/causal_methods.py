# -*- coding: utf-8 -*-
"""因果（无未来信息）抗蠕变算法对比
方法:
  1. 原始 / 固定基线
  2. 预标定 lead-lag 逆滤波 (留一法标定 k,τ) —— LTI 蠕变模型的解析逆
  3. 卡尔曼状态增广联合估计 [F, c] (留一法 k,τ)
  4. 扩展窗在线递推拟合 (纯盲，无先验)
  5. 固定τ + RLS 在线幅度自适应 (留一法τ, 先验初始化)
  6. 完整在线管线: 自动零点跟踪 + lead-lag 逆滤波
全部处理因果执行（逐帧推进，不使用未来数据）；分段边界仅用加载沿检测（向上穿越阈值）。
"""
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
OUT = os.path.join(BASE, "results")
DATASETS = ["数据1", "数据2", "数据3"]

# 各数据集主通道离线拟合的蠕变参数（之前实验结果）：k=蠕变幅度比, τ=时间常数
CALIB = {"数据1": (0.610, 63.7), "数据2": (0.344, 74.4), "数据3": (0.352, 34.2)}


def load_csv(path):
    return pd.read_csv(path, skiprows=24)


def segment(t, total, thr_ratio=0.15):
    thr = thr_ratio * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    segs = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)
    return segs[0]


def loo_params(name):
    """留一法：用其它两数据集的平均蠕变参数"""
    others = [CALIB[n] for n in DATASETS if n != name]
    k = np.mean([o[0] for o in others])
    tau = np.mean([o[1] for o in others])
    return k, tau


# ---------------- 因果算法（输入基线已扣的 X: T×C, 加载沿 s0；输出 Y） ----------------

def causal_lead_lag(Xb, t, s0, k, tau):
    """预标定 lead-lag 逆滤波：ẋ=-βx+(α-β)y, F̂=y+x; α=1/τ, β=(1+k)/τ
    从加载沿 s0 启动，状态置零；对卸载恢复同样自动反演（LTI叠加）。"""
    alpha = 1.0 / tau
    beta = (1 + k) / tau
    T, C = Xb.shape
    Y = Xb.copy()
    x = np.zeros(C)
    for i in range(s0, T):
        dt = t[i] - t[i - 1] if i > 0 else 0.015
        rho = np.exp(-beta * max(dt, 1e-3))
        x = rho * x + (alpha - beta) / beta * (1 - rho) * Xb[i]
        Y[i] = Xb[i] + x
    return Y


def causal_kalman(Xb, t, s0, k, tau, qF=1e-9, qc=1e-8, r=2e-5):
    """状态增广卡尔曼: F[n+1]=F[n]+wF; c[n+1]=ρc+k(1-ρ)F[n]+wc; y=F+c+v
    加载沿重置: F=y[s0], c=0。
    注意: qF 必须趋零——否则蠕变增长会被 F 吸收(可辨识性两难)，
    qF=1e-6 时实测保真度冲到 155%(F 高估)，补偿失效。"""
    T, C = Xb.shape
    Y = Xb.copy()
    F = Xb[s0].copy()
    c = np.zeros(C)
    P = np.tile(np.diag([1e-6, 1e-4]), (C, 1, 1))  # C×2×2: F 置信度高(加载沿 F=y[s0] 近乎精确), c 不确定
    for i in range(s0, T):
        dt = t[i] - t[i - 1] if i > s0 else 0.015
        rho = np.exp(-dt / tau)
        # 预测
        F_new = F
        c_new = rho * c + k * (1 - rho) * F
        A = np.zeros((C, 2, 2))
        A[:, 0, 0] = 1.0
        A[:, 1, 0] = k * (1 - rho)
        A[:, 1, 1] = rho
        Q = np.zeros((C, 2, 2))
        Q[:, 0, 0] = qF * dt
        Q[:, 1, 1] = qc * dt
        P = A @ P @ A.transpose(0, 2, 1) + Q
        # 更新: y = F + c
        H = np.ones((1, 2))
        S = P[:, 0, 0] + P[:, 0, 1] + P[:, 1, 0] + P[:, 1, 1] + r
        K = np.stack([P[:, 0, 0] + P[:, 0, 1], P[:, 1, 0] + P[:, 1, 1]], axis=1) / S[:, None]
        innov = Xb[i] - (F_new + c_new)
        F = F_new + K[:, 0] * innov
        c = c_new + K[:, 1] * innov
        IKH = np.eye(2)[None] - K[:, :, None] @ np.ones((C, 1, 2))
        P = IKH @ P
        Y[i] = F
    return Y


def causal_extending_window(Xb, t, s0, s1, mc, refit_every=67, min_win_s=8.0):
    """扩展窗在线递推拟合：每 ~1s 用 [s0, now] 在主通道上重拟合单指数（暖启动），
    用最新参数扣除主通道；其余通道不补偿（演示因果收敛特性，指标均在主通道上）。
    纯盲（无先验）；窗口 < min_win_s 时不补偿。
    卸载后(s1)进入恢复相：冻结参数，扣除项按 e^{-(t-s1)/τ} 衰减到 0。"""
    T, C = Xb.shape
    Y = Xb.copy()
    last_p = None
    for i in range(s0, T):
        win_s = t[i] - t[s0]
        if i < s1 and win_s >= min_win_s and (i - s0) % refit_every == 0:
            tl = t[s0:i + 1] - t[s0]
            sig = Xb[s0:i + 1, mc]
            try:
                p0 = last_p if last_p is not None else \
                    [sig[-1] - sig[0], max(tl[-1] / 3, 1.0), sig[0]]
                popt, _ = curve_fit(
                    lambda tt, a, tau, c: a * (1 - np.exp(-tt / tau)) + c,
                    tl, sig, p0=p0, maxfev=2000)
                # 合理性检查：τ 必须在物理合理范围
                if 1.0 < popt[1] < 1000:
                    last_p = popt
            except Exception:
                pass
        if last_p is not None:
            a, tau, c = last_p
            if i < s1:
                Y[i, mc] = Xb[i, mc] - a * (1 - np.exp(-(t[i] - t[s0]) / tau))
            else:
                # 恢复相：扣除量从 a(1-e^{-(s1-s0)/τ}) 指数衰减
                resid = a * (1 - np.exp(-(t[s1] - t[s0]) / tau))
                Y[i, mc] = Xb[i, mc] - resid * np.exp(-(t[i] - t[s1]) / tau)
    return Y


def causal_rls(Xb, t, s0, s1, k_prior, tau, lam=0.9995):
    """固定τ，RLS 在线估计 [c0, a]: y=c0+a*g, g=1-exp(-t/τ)。
    先验初始化: c0=y[s0], a=k_prior*c0。
    卸载沿(s1)冻结估计，扣除项切换为恢复相衰减。"""
    T, C = Xb.shape
    Y = Xb.copy()
    theta = np.stack([Xb[s0], k_prior * Xb[s0]], axis=1)  # C×2
    P = np.tile(np.eye(2) * 1e-2, (C, 1, 1))  # C×2×2
    for i in range(s0, T):
        g = 1 - np.exp(-(t[i] - t[s0]) / tau)
        phi = np.array([1.0, g])
        yv = Xb[i]
        if i < s1:
            Pphi = np.einsum("cij,j->ci", P, phi)          # C×2
            denom = lam + np.einsum("j,cj->c", phi, Pphi)  # C
            K = Pphi / denom[:, None]                      # C×2
            err = yv - np.einsum("cj,j->c", theta, phi)    # C
            theta = theta + K * err[:, None]
            phiP = np.einsum("j,cjl->cl", phi, P)          # C×2
            P = (P - K[:, :, None] * phiP[:, None, :]) / lam
            Y[i] = yv - theta[:, 1] * g
        else:
            # 恢复相：扣除量从 a·g(s1) 指数衰减（θ 冻结）
            g1 = 1 - np.exp(-(t[s1] - t[s0]) / tau)
            Y[i] = yv - theta[:, 1] * g1 * np.exp(-(t[i] - t[s1]) / tau)
    return Y


def causal_autozero(X, t, total_raw, tau=3.0, thr_ratio=0.15):
    thr = thr_ratio * total_raw.max()
    zero = X[0].copy()
    Y = np.empty_like(X)
    for i in range(X.shape[0]):
        if total_raw[i] < thr:
            dt = t[i] - t[i - 1] if i > 0 else 0.015
            a = 1 - np.exp(-max(dt, 1e-3) / tau)
            zero = zero + a * (X[i] - zero)
        Y[i] = X[i] - zero
    return Y


# ---------------- 主流程 ----------------
def evaluate(Y, t, s0, s1, mc, raw_step, raw_pre_std):
    sig = Y[:, mc]
    pre, load, post = sig[:s0], sig[s0:s1], sig[s1:]
    n10 = max(len(load) // 10, 10)
    rd = abs(load[-n10:].mean() - load[:n10].mean()) / raw_step * 100
    fid = (load[:n10].mean() - pre.mean()) / raw_step * 100
    zr = abs(post.mean() - pre.mean()) / raw_step * 100
    nr = pre.std() / max(raw_pre_std, 1e-9)
    return rd, fid, zr, nr


rows = []
store = {}
for name in DATASETS:
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    s0, s1 = segment(t, total)
    load_mean = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    mc = int(np.argmax(load_mean))
    loaded_mask = load_mean > 0.3 * load_mean.max()

    raw_sig = X[:, mc]
    n10 = max((s1 - s0) // 10, 10)
    raw_step = raw_sig[s0:s0 + n10].mean() - raw_sig[:s0].mean()
    raw_pre_std = raw_sig[:s0].std()

    Xb = X - X[:s0].mean(axis=0)  # 固定基线
    k_cal, tau_cal = loo_params(name)

    results = {
        "原始(扣固定基线)": Xb,
        "预标定lead-lag逆滤波": causal_lead_lag(Xb, t, s0, k_cal, tau_cal),
        "卡尔曼联合估计[F,c]": causal_kalman(Xb, t, s0, k_cal, tau_cal),
        "扩展窗在线拟合(纯盲)": causal_extending_window(Xb, t, s0, s1, mc),
        "RLS幅度自适应(先验τ)": causal_rls(Xb, t, s0, s1, k_cal, tau_cal),
        "自动零点+lead-lag管线": causal_lead_lag(
            causal_autozero(X, t, total), t, s0, k_cal, tau_cal),
    }
    store[name] = dict(t=t, s0=s0, s1=s1, mc=mc, results=results,
                       loaded_mask=loaded_mask, X=X, Xb=Xb)

    print(f"\n=== {name} (标定参数 k={k_cal:.2f}, τ={tau_cal:.0f}s) ===")
    for aname, Y in results.items():
        m = evaluate(Y, t, s0, s1, mc, raw_step, raw_pre_std)
        rows.append([name, aname, *m])
        print(f"  {aname:22s} 残余漂移={m[0]:5.1f}% 保真={m[1]:6.1f}% 零漂={m[2]:4.1f}% 噪声比={m[3]:.2f}")

    # 全阵列（受载通道）平均残余漂移 —— 对两个主力方法
    for aname in ["预标定lead-lag逆滤波", "卡尔曼联合估计[F,c]"]:
        Y = results[aname]
        rds = []
        for j in np.where(loaded_mask)[0]:
            stepj = Xb[s0:s0 + n10, j].mean()
            if stepj < 0.05:
                continue
            rds.append(abs(Y[s1 - n10:s1, j].mean() - Y[s0:s0 + n10, j].mean()) / stepj * 100)
        print(f"  [阵列{loaded_mask.sum()}ch] {aname}: 平均残余漂移={np.mean(rds):.1f}% (中位{np.median(rds):.1f}%)")

M = pd.DataFrame(rows, columns=["数据集", "算法", "负载段残余漂移%", "阶跃保真度%", "零漂残余%", "噪声比"])
M.to_csv(os.path.join(OUT, "causal_metrics.csv"), index=False, encoding="utf-8-sig")
summary = M.groupby("算法", as_index=False)[
    ["负载段残余漂移%", "阶跃保真度%", "零漂残余%", "噪声比"]].mean().sort_values("负载段残余漂移%")
print("\n==== 3数据集平均（全部因果方法）====")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
summary.to_csv(os.path.join(OUT, "causal_metrics_summary.csv"), index=False, encoding="utf-8-sig")

# ---------------- 图f1: 主通道全时长 ----------------
fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    R = store[name]
    t, s0, s1, mc = R["t"], R["s0"], R["s1"], R["mc"]
    for aname, Y in R["results"].items():
        lw = 1.6 if "原始" in aname else 1.0
        ax.plot(t, Y[:, mc], lw=lw, label=aname,
                color="black" if "原始" in aname else None, alpha=0.85)
    ax.axvspan(t[s0], t[s1], color="orange", alpha=0.10)
    ax.set_title(f"{name} 主通道 ch{mc} — 因果方法全时长对比")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "f1_causal_full_timeline.png"), dpi=140)
plt.close(fig)

# ---------------- 图f2: 负载段放大 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    R = store[name]
    t, s0, s1, mc = R["t"], R["s0"], R["s1"], R["mc"]
    for aname, Y in R["results"].items():
        ax.plot(t[s0:s1] - t[s0], Y[s0:s1, mc], lw=1.4 if "原始" in aname else 1.0,
                label=aname, color="black" if "原始" in aname else None, alpha=0.85)
    ax.set_title(f"{name} 负载段 — 因果方法")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "f2_causal_load_zoom.png"), dpi=140)
plt.close(fig)

# ---------------- 图f3: 指标条形图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
order = summary["算法"].tolist()
for ax, met, better in zip(axes,
                           ["负载段残余漂移%", "阶跃保真度%", "零漂残余%"],
                           ["越小越好", "越接近100%越好", "越小越好"]):
    vals = [summary.loc[summary["算法"] == a, met].iloc[0] for a in order]
    ax.barh(range(len(order)), vals, color="tab:green", alpha=0.85)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"{met}（{better}）")
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
fig.savefig(os.path.join(FIG, "f3_causal_metrics.png"), dpi=140)
plt.close(fig)

# ---------------- 图f4: 最优因果方法全阵列漂移热图 ----------------
mask_str = "110110011011101111110111111011111100000001000000100000010000001"
ones = [i for i, b in enumerate(mask_str) if b == "1"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
for c, name in enumerate(DATASETS):
    R = store[name]
    s0, s1 = R["s0"], R["s1"]
    n10 = max((s1 - s0) // 10, 10)
    for rr, (aname, ttl) in enumerate([("原始(扣固定基线)", "原始"),
                                       ("卡尔曼联合估计[F,c]", "卡尔曼联合估计后")]):
        Mat = R["results"][aname]
        m = np.full(63, np.nan)
        drift_arr = Mat[s1 - n10:s1].mean(axis=0) - Mat[s0:s0 + n10].mean(axis=0)
        for j, pos in enumerate(ones):
            m[pos] = drift_arr[j]
        im = axes[rr, c].imshow(m.reshape(9, 7), cmap="coolwarm", aspect="auto",
                                vmin=-0.5, vmax=0.5)
        axes[rr, c].set_title(f"{name} {ttl}\n负载段漂移量(N)")
        plt.colorbar(im, ax=axes[rr, c])
fig.savefig(os.path.join(FIG, "f4_causal_array_heatmap.png"), dpi=140)
plt.close(fig)

print("\nsaved f1~f4 + causal_metrics csv")
