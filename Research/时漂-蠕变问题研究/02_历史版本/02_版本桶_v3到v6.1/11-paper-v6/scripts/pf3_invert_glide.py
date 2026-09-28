# -*- coding: utf-8 -*-
"""论文图 F3：形状约束反演与滑行器（v6 的前端，抗蠕变的第一道工序）。

以 切换负载-快相无责 的一份真实 onset 事件为例，画：
(a) 事件窗内的原始总量、v6 输出、反演目标 Â（含载前基线），并标出滑行窗与交接时刻
(b) 前 6 s 放大：原始继续上爬 vs 显示 0.6~0.8 s 到平台
(c) Â 随时间的滚动重估轨迹（含限幅带 κ·Δ_obs）与理论首扣时刻对比
(d) 实测的检测时延分布（65 个事件）+ 形状先验误差（留一法）

数据：paper_v6/results/cache/切换负载-快相无责.npz（含 v6 事件内部轨迹）
      temp/v4.1flash/results/v6_detect_latency.csv、_v6_singlepoint.log
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                            # noqa: E402
import pv_style as S                                             # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402

TAG = "切换负载-快相无责"
z = C.load_npz(TAG)
tu = z["tu"]
tot = z["tot_s"]
Yv6 = z["Y_v6"]
Y3 = z["Y_e3s"]
Y1 = z["Y_e1s"]
traces = json.loads(str(z["traces"][0]))

# ── 选事件：首个 onset（总跳变最大的早期事件） ────────────────────
keys = sorted(float(k) for k in traces)
cand = []
for k in keys:
    i = int(np.searchsorted(tu, k))
    pre = np.median(tot[max(0, i - int(2.0 / float(z["dtm"][0]))):i])
    post = np.median(tot[i + int(3 / float(z["dtm"][0])):i + int(5 / float(z["dtm"][0]))])
    cand.append((post - pre, k, pre, post))
cand.sort(reverse=True)
jump, t0, pre, post = cand[0]
print(f"选中事件 t0={t0:.2f}s  跳变={jump:.0f} ADC  前级={pre:.0f} 后级={post:.0f}")
tr = [r for r in traces[str(t0)] or traces[t0]]
tr = sorted(tr, key=lambda r: r["tau"])
tau = np.array([r["tau"] for r in tr])
Ahat = np.array([r["A_hat"] for r in tr])
out = np.array([r["out"] for r in tr])
stall = np.array([r["stalled"] for r in tr])
tgt = np.array([r["target"] for r in tr])
base_y = float(z["Y_v6"][int(np.searchsorted(tu, t0))]) if True else 0.0

# 交接时刻：轨迹最后一帧的 tau
t_ho = float(tau[-1])
t_glide_end = float(tau[min(np.argmax(Ahat > 0.99 * Ahat.max()) if (Ahat > 0.99 * Ahat.max()).any() else 0,
                            len(tau) - 1)])

fig, axs = plt.subplots(2, 2, figsize=(14.2, 9.4))
fig.suptitle(f"图 F3  形状约束反演与滑行器：一次真实加载事件（{TAG}，台阶 {jump:.0f} ADC）", fontsize=14)

# ── (a) 事件窗全景 ────────────────────────────────────────────────
win = (t0 - 2.0, t0 + 14.0)
m = (tu >= win[0]) & (tu <= win[1])
ax = axs[0, 0]
ax.plot(tu[m], tot[m], color=S.C_RAW, lw=1.4, label="原始（无补偿）")
ax.plot(tu[m], Y3[m], color=S.C_E3, lw=1.2, ls="-.", label="无责 3 s（v5.1，现役）")
ax.plot(tu[m], Yv6[m], color=S.C_V6, lw=2.0, label="v6（滑行 + 慢相）")
tm = t0 + tau
ax.plot(tm, out, color="#7e57c2", lw=1.2, ls=":", label="v6 事件期逐帧输出")
ax.axhline(float(np.median(tot[(tu > t0 + 4.6) & (tu < t0 + 5.4)])), color=S.C_IDEAL,
           ls="--", lw=1.3, label="真值（加载沿 +5 s 电平）")
ax.axvline(t0, color="0.4", lw=1.0, ls=":")
ax.axvline(t0 + t_ho, color=S.C_GLIDE, lw=1.1, ls=":")
ax.annotate("事件原点 $t_0$\n（回溯到真实加载沿）", xy=(t0, tot[m].max() * 0.72),
            xytext=(t0 - 1.9, tot[m].max() * 0.80), fontsize=8.8, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
ax.annotate(f"交接 τ={t_ho:.1f} s", xy=(t0 + t_ho, out.min()), xytext=(t0 + t_ho + 0.6, out.min() * 0.96),
            fontsize=8.8, color=S.C_GLIDE, arrowprops=dict(arrowstyle="->", color=S.C_GLIDE, lw=0.9))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("阵列总量 (ADC)")
ax.set_title("(a) 事件窗全景：台阶瞬间跟随，此后只做慢相扣除")
ax.legend(loc="lower right", ncol=2)

# ── (b) 前 6 s 放大 ──────────────────────────────────────────────
ax = axs[0, 1]
win2 = (t0 - 0.5, t0 + 6.0)
m2 = (tu >= win2[0]) & (tu <= win2[1])
ax.plot(tu[m2], tot[m2], color=S.C_RAW, lw=1.8, label="原始")
ax.plot(tu[m2], Y1[m2], color=S.C_E1, lw=1.2, ls="--", label="无责 1 s")
ax.plot(tu[m2], Y3[m2], color=S.C_E3, lw=1.2, ls="-.", label="无责 3 s")
ax.plot(tu[m2], Yv6[m2], color=S.C_V6, lw=2.2, label="v6")
ax.plot(tm, out, color="#7e57c2", lw=1.1, ls=":", label="v6 事件期")
kd = np.argmax(Ahat > 0)
ax.axvspan(t0 + tau[kd], t0 + max(t_ho, tau[kd] + 0.8), color=S.C_GLIDE, alpha=0.18, lw=0,
           label="滑行窗")
ax.axvline(t0 + t_ho, color=S.C_GLIDE, lw=1.0, ls=":")
ax.axhline(float(np.median(tot[(tu > t0 + 4.6) & (tu < t0 + 5.4)])), color=S.C_IDEAL, ls="--", lw=1.2)
for lab, arr, col in (("原始", tot, S.C_RAW), ("v6", Yv6, S.C_V6)):
    i = int(np.searchsorted(tu, t0 + 1.0))
    ax.plot([t0 + 1.0], [arr[i]], "o", color=col, ms=5)
    ax.annotate(f"{lab} {arr[i]:.0f}", xy=(t0 + 1.0, arr[i]),
                xytext=(t0 + 2.95, arr[i] + (0.10 if lab == "原始" else -0.14) * jump),
                fontsize=9.2, color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("阵列总量 (ADC)")
ax.set_title("(b) 加载后 1 s：v6 已在平台上，原始仍在爬")
ax.legend(loc="lower right", ncol=2)

# ── (c) Â 的滚动重估 ─────────────────────────────────────────────
ax = axs[1, 0]
d_obs = np.array([r["inc"] for r in tr])
ax.plot(tau, Ahat, color=S.C_V6, lw=2.0, label="反演幅值 $\\hat A$（每帧重估）")
ax.plot(tau, d_obs, color=S.C_RAW, lw=1.3, label="实测增量 $\\Delta_{obs}$")
ax.plot(tau, 1.30 * d_obs, color="#c62828", lw=1.0, ls=":", label="限幅上界 $\\kappa\\Delta_{obs}$（1.30）")
ax.plot(tau, 0.50 * d_obs, color="#c62828", lw=1.0, ls=":", label="限幅下界 0.50$\\Delta_{obs}$")
if stall.any():
    ax.plot(tau[stall], Ahat[stall], ".", color="#e65100", ms=6, label="停滞检测生效段（停止预判）")
ax.axvline(t_ho, color=S.C_GLIDE, lw=1.2, ls="--")
ax.text(0.03, 0.10, f"τ=5 s 交接慢相模块\n（形状走完，g_raw≈0）", transform=ax.transAxes,
        fontsize=9, color=S.C_GLIDE, va="bottom", ha="left",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
ax.set_xlim(0, min(8.0, tau[-1]))
ax.set_xlabel("事件内时间 $\\tau$ (s)")
ax.set_ylabel("总量口径 (ADC)")
ax.set_title("(c) $\\hat A$ 的电平域最小二乘滚动重估")
ax.legend(loc="lower right", fontsize=8.4)

# ── (d) 检测时延 + 形状先验误差 ───────────────────────────────────
ax = axs[1, 1]
import pandas as pd                                             # noqa: E402
lat = pd.read_csv(os.path.join(C.FLASH, "progress", "07-v6", "results", "v6_detect_latency.csv"))
v5 = lat["lat5.0"].to_numpy(float)
ax.hist(v5, bins=np.arange(0, 1.25, 0.05), color=S.C_V6, alpha=0.85, edgecolor="white")
ax.axvline(np.median(v5), color="#c62828", lw=1.6)
ax.text(np.median(v5) + 0.03, ax.get_ylim()[1] * 0.92, f"中位 {np.median(v5):.2f} s",
        color="#c62828", fontsize=9.4)
n_hit = int((v5 <= 0.06).sum())
ax.annotate(f"{n_hit}/{len(v5)} 个事件在判据网格下界 0.05 s 命中\n（检测不是瓶颈）",
            xy=(0.06, ax.get_ylim()[1] * 0.55), xytext=(0.30, ax.get_ylim()[1] * 0.62),
            fontsize=9, color="0.2", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
ax.set_xlabel("检测时延（真实加载沿 → 判据命中，s）")
ax.set_ylabel("事件数")
ax.set_title(f"(d) 65 个事件的检测时延（恒载 + 实录）")

fig.subplots_adjust(left=0.058, right=0.985, top=0.925, bottom=0.068, wspace=0.215, hspace=0.30)
outp = os.path.join(C.FIG, "F3_invert_glide.png")
fig.savefig(outp)
S.figcheck(fig, outp)
plt.close(fig)
print("saved", outp)

# 供正文引用的数字
i_glide = int(np.searchsorted(tu, t0 + 1.0))
i_pre = int(np.searchsorted(tu, t0 - 0.2))
print(f"\n[F3 数字] 事件 t0={t0:.2f}s 台阶 {jump:.0f} ADC（空载 {pre:.0f} → 受载 {post:.0f}）")
print(f"  加载后 1 s：原始 {tot[i_glide]:.0f}，v6 {Yv6[i_glide]:.0f}，差 {tot[i_glide]-Yv6[i_glide]:.0f}")
print(f"  Â 首次可用 τ={tau[kd]:.2f}s，Â={Ahat[kd]:.0f}；交接 τ={t_ho:.2f}s，Â={Ahat[-1]:.0f}")
print(f"  停滞触发帧数 {int(stall.sum())} / {len(stall)}")
