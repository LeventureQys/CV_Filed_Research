# -*- coding: utf-8 -*-
"""v4.1flash 步骤5：评审报告用的图。

图 1 f_review_1.png  §6.1 逆滤波器的阶跃响应：文档声称「无延迟、完全保留」，实际是 N0 倍过冲慢衰减
图 2 f_review_2.png  实测 vs dsp.md 模型的形状：加载后 0.5~3s 均值 / 负载末段均值（9 组）
图 3 f_review_3.png  §4 空间反卷积的失败：拉普拉斯核把总力打成负数
图 4 f_review_4.png  三种算法在实测主通道上的时序对比（§4 结论的可视化）
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, optimize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "docs", "figures")
sys.path.insert(0, HERE)
from glm53_v3 import run_glm53_v3  # noqa: E402

os.makedirs(FIG, exist_ok=True)
FS = 100.5
DT = 1.0 / FS


def iir(a1, t1, a2, t2, fs):
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0]


# ==================== 图 1：阶跃响应 ====================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
a1, t1, a2, t2 = 0.15, 1.2, 0.20, 45.0
b, a = iir(a1, t1, a2, t2, FS)
n = int(60 * FS)
tt = np.arange(n) * DT
st = signal.lfilter(b, a, np.ones(n))
axes[0].plot(tt, st, color="tab:red", lw=1.6, label="§6.1 逆滤波器的阶跃响应")
axes[0].axhline(1.0, color="k", ls="--", lw=1.0, label="真值（弹性值，应为恒定 1.0）")
axes[0].axhline(1 + a1 + a2, color="tab:orange", ls=":", lw=1.2, label=f"N0 = 1+a1+a2 = {1+a1+a2:.2f}")
axes[0].fill_between(tt, 1.0, st, color="tab:red", alpha=0.15)
axes[0].set_xlim(0, 60)
axes[0].set_ylim(0.95, 1.45)
axes[0].set_xlabel("阶跃后时间 (s)")
axes[0].set_ylabel("补偿输出 / 真值")
axes[0].set_title(f"§6.1 逆滤波器的阶跃响应（τ1={t1}s, τ2={t2}s）\n"
                  f"文档称「无延迟、完全保留」，实际过冲 +{100*(a1+a2):.0f}% 后缓慢回落", fontsize=10)
axes[0].legend(fontsize=8)

x = np.linspace(0, 200, 2000)
for lab, par, col in [("τ1=3.5s τ2=1504s（右拇指实测拟合）", (0.192, 3.46, 6.0, 1504.5), "tab:red"),
                      ("τ1=1.2s τ2=45s（dsp.md 标称）", (0.15, 1.2, 0.20, 45.0), "tab:blue")]:
    aa1, at1, aa2, at2 = par
    N = np.array([at1 * at2, (1 + aa1) * at2 + (1 + aa2) * at1, 1 + aa1 + aa2])
    D = np.array([at1 * at2, at1 + at2, 1.0])
    s_ = (2 / DT) * (1 - np.exp(-1j * 2 * np.pi * x / 1000))  # 近似：用连续域频率轴
    bb, aa = iir(aa1, at1, aa2, at2, FS)
    w, h = signal.freqz(bb, aa, worN=4000, fs=FS)
    axes[1].semilogx(w[1:], np.abs(h[1:]), color=col, lw=1.6, label=lab)
axes[1].set_xlabel("频率 (Hz)")
axes[1].set_ylabel("|G(f)|")
axes[1].set_title("§6.1 逆滤波器的幅频响应\n直流增益 1 → 奈奎斯特增益 N0（高频被放大，与 §3.1 承诺相反）", fontsize=10)
axes[1].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "f_review_1.png"), dpi=140)
plt.close(fig)

# ==================== 图 2：形状检验 ====================
chk = pd.read_csv(os.path.join(RES, "model_shape_check.csv"))
fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
lbl = [f"{r.location[:2]}/{r.dataset}" for r in chk.itertuples()]
xs = np.arange(len(chk))
ax.bar(xs, chk.ratio, color="tab:blue", alpha=0.85)
ax.axhline(1.0, color="k", ls="--", lw=1.0, label="1.0 = 加载后不再上升（dsp.md 模型预言 > 1，即先高后落）")
ax.set_xticks(xs)
ax.set_xticklabels(lbl, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("加载后 0.5~3s 均值 / 负载末段均值")
ax.set_title("实测形状检验（9 组）：全部 < 1，即加载后持续单调上升，没有 dsp.md 模型预言的回落", fontsize=10)
ax.legend(fontsize=8)
for i, v in enumerate(chk.ratio):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
fig.savefig(os.path.join(FIG, "f_review_2.png"), dpi=140)
plt.close(fig)

# ==================== 图 3：空间反卷积 ====================
sp = pd.read_csv(os.path.join(RES, "spatial_deconv.csv"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
xs = np.arange(len(sp))
axes[0].bar(xs, sp.total_ratio, color="tab:red", alpha=0.85)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].axhline(1, color="g", ls="--", lw=1.0, label="1.0 = 总量守恒（应达到）")
axes[0].set_xticks(xs)
axes[0].set_xticklabels([f"{r.location[:2]}/{r.dataset}" for r in sp.itertuples()],
                        rotation=35, ha="right", fontsize=8)
axes[0].set_ylabel("∑反卷积 / ∑原始")
axes[0].set_title("§4 拉普拉斯核作用在实测载荷图上\n总力被打成负数（直流增益为 0）", fontsize=10)
axes[0].legend(fontsize=8)

alpha = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5])
rmse = np.array([0.1964, 0.2205, 0.2383, 0.2655, 0.2858, 0.3016, 0.3144])
axes[1].plot(alpha, rmse, "o-", color="tab:red")
axes[1].axhline(rmse[0], color="g", ls="--", lw=1.0, label=f"不处理基线 RMSE={rmse[0]:.4f}")
axes[1].set_xlabel(r"反卷积强度 $\alpha$（$(I-\alpha L)^{-1}$）")
axes[1].set_ylabel("重建 RMSE")
axes[1].set_title("已知真值重建（5×5 方形载荷 + 高斯 PSF σ=1）\n即使写成正确形式，反卷积也让误差变大", fontsize=10)
axes[1].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "f_review_3.png"), dpi=140)
plt.close(fig)

# ==================== 图 4：实测时序对比 ====================
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), constrained_layout=True)
for ax, (loc, name) in zip(axes, [("右拇指指尖", "数据1"), ("左拇指指尖", "数据2")]):
    df = pd.read_csv(os.path.join(TEMP, loc, name, "device_001_seg000.csv"), skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    t = tr - tr[0]
    X = df[ch].to_numpy(float)
    tot = X.sum(axis=1)
    thr = 0.15 * tot.max()
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    s0, s1 = int(s[0]) if len(s) else 0, int(e[0]) if len(e) else len(t)
    amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    base = X[:s0].mean(axis=0)
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    Yg = run_glm53_v3(tu, Xu)
    # dsp.md：用无约束标定参数（按 §5 流程能拿到的东西）
    s0u, s1u = int(np.searchsorted(tu, t[s0])), int(np.searchsorted(tu, t[s1]))
    n_on = s0u
    uu = tu[n_on:] - tu[n_on]
    y = Xu[n_on:, m] - base[m]
    stride = max(1, len(uu) // 800)

    def resid(p):
        aa1, at1, aa2, at2, sc = p
        N = np.array([at1 * at2, (1 + aa1) * at2 + (1 + aa2) * at1, 1 + aa1 + aa2])
        D = np.array([at1 * at2, at1 + at2, 1.0])
        bb, aa = signal.bilinear(N / N[2], D, fs=1 / dt)
        bb, aa = bb / bb[0], aa / aa[0]
        return sc * signal.lfilter(bb, aa, np.ones(len(uu[::stride]))) - y[::stride]

    rng = np.random.default_rng(7)
    best = None
    for _ in range(8):
        p0 = np.r_[rng.uniform(0.01, 0.4, 1), rng.uniform(0.5, 10, 1),
                   rng.uniform(0.01, 0.5, 1), rng.uniform(20, 400, 1), y[:10].mean()]
        r = optimize.least_squares(resid, p0, bounds=([0, 0.1, 0, 2, 1e-9], [4, 120, 6, 3000, 1e6]))
        if best is None or r.cost < best.cost:
            best = r
    bb, aa = iir(best.x[0], best.x[1], best.x[2], best.x[3], 1 / dt)
    Yd = np.vstack([signal.lfilter(bb, aa, Xu[:, c] - base[c]) + base[c]
                    for c in range(Xu.shape[1])]).T

    ax.plot(tu, Xu[:, m], color="0.6", lw=0.7, label="原始(无补偿)")
    ax.plot(tu, Yg[:, m], color="tab:blue", lw=0.9, label="GLM53 v3（现有实现）")
    ax.plot(tu, Yd[:, m], color="tab:red", lw=0.9, label="dsp.md §6.1（§5 流程标定）")
    ax.axvspan(tu[s0u], tu[min(s1u, len(tu) - 1)], color="orange", alpha=0.07)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("显示读数")
    ax.set_title(f"{loc}/{name} 主通道 ch{m}", fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle("9 组恒载实测：dsp.md 逆滤波方案 vs 现有 GLM53 v3（同口径同参数区间）", fontsize=12)
fig.savefig(os.path.join(FIG, "f_review_4.png"), dpi=140)
plt.close(fig)

print("saved: figures/f_review_1.png ~ f_review_4.png")
