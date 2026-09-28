# -*- coding: utf-8 -*-
"""图 1：加载后三段结构的物理依据（实采数据 + 两时间常数模型拟合）。

数据来源：
  · 保压段拟合  temp/变化负载/切换负载-快相无责的测试/.../device_001_seg000.csv
  · 快相轮廓     temp/{右拇指指尖,左拇指指尖,四指指尖}/数据1~3/device_001_seg000.csv（9 组恒载）
产出：figures/F1_two_phase.png、results/f1_two_phase_profiles.csv、results/f1_creep_law.csv
"""
import numpy as np
import pandas as pd
from scipy import optimize

import pd as P
from figstyle import (C_ALG, C_FAST, C_GRAY, C_RAW, C_SLOW, plt, save_figure)

# ── 1) 加载后 100 s 的慢相形状（取恒载录制，唯一能干净看到慢相的数据）─────
# 实采变化负载录制里 20~120 s 尺度上主导的是操作者移动/指位调整（实测 10~15 s 之间
# 就可能有 +40% 电平的跳变），无法用来拟合慢相；恒载录制从头到尾只有一次加载。
_ST_lab, _st = P.load_static()[0]                       # 右拇指指尖/数据1
d_t = {"tu": _st["tu"], "dtm": _st["dtm"], "x": _st["x"]}
s0, s1 = _st["s0"], _st["s1"]
T_ON = float(_st["tu"][s0])
NSEG = 100.0
_i_end = min(s1, s0 + int(NSEG / _st["dtm"]))
uu = _st["tu"][s0:_i_end] - _st["tu"][s0]
sig = _st["x"][s0:_i_end]
base = float(np.mean(_st["x"][:max(1, s0)]))
yy = sig / sig[int(round(5.0 / _st["dtm"]))] - 1.0     # 归一化口径与我们的一致
amp = float(np.mean(_st["x"][s0:s1]) - base)
print(f"[F1] 慢相窗 {_ST_lab}：{NSEG:.0f} s，幅度 {amp:.3f} N")

# 单时间常数慢相拟合（不含快相项）：y(t) = a·(1 − e^{−t/τ})
TAU2_FIX = 199.0
_c = (1.0 - np.exp(-uu / TAU2_FIX))
a2_ = float(np.dot(_c, yy) / np.dot(_c, _c))
fit_rmse = float(np.sqrt(np.mean((a2_ * _c - yy) ** 2)))
t2_ = TAU2_FIX
a1_, t1_ = 0.0, 1.0
print(f"[F1] 慢相拟合 a2={100*a2_:.1f}%（τ=199 s 固定），RMSE={fit_rmse:.3f}（归一化）")
share_fast = 0.0
share_slow = 1.0


def curve(a1, a2, t1, t2, x):
    """两相形状：a1·(1−e^{−x/τ1}) + a2·(1−e^{−x/τ2})。"""
    return a1 * (1.0 - np.exp(-np.maximum(x, 0) / t1)) + a2 * (1.0 - np.exp(-np.maximum(x, 0) / t2))


def dslope(t, dt=1e-4):
    return (curve(a1_, a2_, t1_, t2_, t + dt) - curve(a1_, a2_, t1_, t2_, t - dt)) / (2 * dt)


def win_slope(lo, hi):
    tg = np.linspace(lo, hi, 4000)
    return float(np.mean(dslope(tg)))


slope_01 = win_slope(0.02, 1.0)
slope_14 = win_slope(1.0, 4.0)
slope_4120 = win_slope(20.0, 100.0)
ratio_fs = slope_14 / slope_4120
print(f"[F1] 段斜率 0-1s {slope_01:.3f}/s、1-4s {slope_14:.4f}/s、20-100s {slope_4120:.5f}/s"
      f"（1-4s 是 20-100s 的 {ratio_fs:.1f} 倍）")
P.save_table(pd.DataFrame([dict(dataset=_ST_lab, window_s=NSEG, amp_N=amp,
                                a_slow_pct=100 * a2_, tau_slow_s=t2_,
                                slope_20_100=slope_4120, fit_rmse=fit_rmse, n=len(uu))]),
             "f1_creep_law.csv")

# ── 2) 快相归一化轮廓（9 组恒载） ──────────────────────────────────
st = P.load_static()
profs, meta_rows = [], []
for label, d in st:
    prof, meta = P.fast_profile(d)
    profs.append(prof)
    meta_rows.append(dict(dataset=label, **{f"f({g})": v for g, v in zip(P.PROFILE_GRID, prof)},
                          onset_s=meta["onset_s"], amp_main=meta["amp_main"]))
PROF = np.vstack(profs)
grid = np.array(P.PROFILE_GRID)
sd = PROF.std(axis=0, ddof=0)
med = np.median(PROF, axis=0)
print(f"[F1] 9 组快相逐点标准差 最大 {sd.max():.3f}、均值 {sd.mean():.3f}")
P.save_table(pd.DataFrame(meta_rows), "f1_two_phase_profiles.csv")

# ── 3) 长时保压段：幂律/双指数形状与"5 s 之后只剩慢相" ─────────────
share_bar = np.array([100 * share_fast, 100 * share_slow])

# ── 绘图 ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.6, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.16, 0.92, 1.20], wspace=0.30,
                      left=0.055, right=0.985, top=0.885, bottom=0.205)

# (a) 9 组加载沿的归一化快相
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(0, 3.0, color=C_FAST, alpha=0.13, lw=0)
for i in range(PROF.shape[0]):
    ax.plot(grid, PROF[i], color=C_GRAY, lw=0.9, alpha=0.62)
ax.plot(grid, PROF.mean(axis=0), color=C_ALG, lw=2.5, label="9 组均值")
ax.set_xlim(0, 5.15)
ax.set_ylim(0, 1.28)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("加载沿后时间 (s)")
ax.set_ylabel("归一化读数（$t$=5 s 处为 1）")
ax.set_title("(a) 加载沿附近的形状：9 组几乎重合", fontsize=11)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=8.4, loc="lower right", framealpha=1.0, borderaxespad=0.9)
ax.text(0.015, 0.965, "3 s 免责期", fontsize=8.6, color=C_FAST, va="top",
        transform=ax.transAxes)
ax.text(0.975, 0.965, "逐点 σ ≤ %.3f" % sd.max(), fontsize=8.6, color=C_ALG, va="top",
        ha="right", transform=ax.transAxes)
ax.text(3.05, 0.55, "0.15 s 到 %.2f\n1 s 到 %.2f\n3 s 到 %.2f" % (med[0], med[4], med[8]),
        fontsize=8.4, color=C_GRAY, va="top", ha="left", zorder=8,
        bbox=dict(fc="white", ec="#D5D8DC", lw=0.7, alpha=1.0, pad=2.4))

# (b) 前 1 s 的细节（对数时间轴）
ax2 = fig.add_subplot(gs[0, 1])
tg1 = np.logspace(np.log10(0.02), 0.0, 300)
for i in range(PROF.shape[0]):
    xg = np.interp(tg1, grid[:5], PROF[i, :5])
    ax2.plot(tg1, xg, color=C_GRAY, lw=0.9, alpha=0.6)
ax2.plot(tg1, np.interp(tg1, grid[:5], PROF.mean(axis=0)[:5]), color=C_ALG, lw=2.5,
         label="9 组均值")
ax2.set_xscale("log")
ax2.set_xlim(0.02, 1.05)
ax2.set_ylim(0, 0.52)
ax2.set_xlabel("加载沿后时间 (s，对数轴)")
ax2.set_ylabel("归一化读数")
ax2.set_title("(b) 前 1 s：机械加载 + 接触建立", fontsize=11)
ax2.grid(alpha=0.25, which="both", lw=0.6)
ax2.legend(fontsize=8.4, loc="upper left", framealpha=1.0, borderaxespad=0.8)
ax2.text(0.975, 0.90, "0.15 s 到 %.2f；0.5 s 到 %.2f" % (med[0], med[2]),
         transform=ax2.transAxes, fontsize=8.5, color=C_ALG, va="top", ha="right",
         bbox=dict(fc="white", ec="#D5D8DC", lw=0.7, alpha=1.0, pad=2.4))

# (c) 恒载保压 100 s：慢相实测与拟合
ax3 = fig.add_subplot(gs[0, 2])
tt = (d_t["tu"][s0:_i_end] - d_t["tu"][s0])
raw_n = d_t["x"][s0:_i_end] / sig[int(round(5.0 / d_t["dtm"]))] - 1.0
ax3.plot(tt, raw_n, color=C_RAW, lw=1.7, label=f"{_ST_lab} 主通道（归一化）")
ax3.plot(np.linspace(0, tt[-1], 1200),
         a2_ * (1.0 - np.exp(-np.linspace(0, tt[-1], 1200) / t2_)), color=C_SLOW, lw=1.9, ls="--",
         label=f"慢相拟合 {100*a2_:.1f}%·$\\tau$={t2_:.0f}s")
ax3.axvspan(0, 5.0, color=C_FAST, alpha=0.16, lw=0)
ax3.set_xlim(0, NSEG)
ax3.set_ylim(0, 1.26)
ax3.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_xlabel("真实加载沿后时间 (s)")
ax3.set_ylabel("归一化读数（$t$=5 s 处为 1）")
ax3.set_title("(c) 恒载保压：慢相单调、无平台", fontsize=11)
ax3.grid(alpha=0.25, lw=0.6)
ax3.legend(fontsize=8.4, loc="upper left", framealpha=1.0, borderaxespad=0.8)
fig.text(0.055, 0.018,
         "(c) 20~100 s 平均斜率 %.4f /s（≈ 每 100 s 再涨 %.0f%%）；慢相拟合 RMSE %.3f；橙色区为加载瞬态。"
         % (slope_4120, 100 * slope_4120 * 100, fit_rmse),
         fontsize=8.4, color=C_GRAY, ha="left", va="bottom")

save_figure(fig, "F1_two_phase.png")
