# -*- coding: utf-8 -*-
"""论文图 F5：慢相取舍 —— 「平台绝对平」与「平台绝对准」不可兼得。

(a) 平台静态偏置：切换负载 @133.84 s 的 onset，形状先验把 Â 钉高了 4.9%
(b) A 慢修正的机制与死区：偏差以 0.2%/s 限速向「交接时刻实测电平」收敛，±2.5% 死区内不动
(c) trim 三档消融：恒载慢相段时漂 / 平坦度 / 平台偏置 / T_stable
(d) 取舍平面：横轴「慢相段时漂」（绝对平），纵轴「实采全程最大偏差」（绝对准），
    每个配置一个点，右上角为不可达区

数据：results/trim_ablation.csv、metrics_all.csv、cache/切换负载-快相无责.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                            # noqa: E402
import pv_style as S                                             # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402

ta = pd.read_csv(os.path.join(C.RES, "trim_ablation.csv"))
m = pd.read_csv(os.path.join(C.RES, "metrics_all.csv"))
h, v = ta[ta.kind == "恒载"], ta[ta.kind == "实采"]
ARM = ["T0_pin", "T1_trim_dead", "T2_trim_nodead"]
ALB = {"T0_pin": "T0 纯 pin（trim 关）", "T1_trim_dead": "T1 trim + 2.5% 死区（采用）",
       "T2_trim_nodead": "T2 trim 无死区"}
ACOL = {"T0_pin": S.C_V6, "T1_trim_dead": S.C_TRIM, "T2_trim_nodead": "#8e24aa"}

fig, axs = plt.subplots(2, 2, figsize=(14.6, 9.6))
fig.suptitle("图 F5  慢相取舍：形状先验带来的「平台静态偏置」，与消掉它的代价", fontsize=14)

# ── (a) 平台静态偏置的来源 ──────────────────────────────────────
ax = axs[0, 0]
z = C.load_npz("切换负载-快相无责")
tu, tot = z["tu"], z["tot_s"]
ds = int(z["ds"][0])
i0 = int(z["i0"][0]) // ds
dtm = float(z["dtm"][0]) * ds
seat = 133.0
a, b = int(np.searchsorted(tu, seat)), int(np.searchsorted(tu, seat + 45))
tt = tu[a:b] - tu[a]
ax.plot(tt, tot[a:b], color=S.C_RAW, lw=1.4, label="原始（无补偿）")
ax.plot(tt, z["Y_v6"][a:b], color=S.C_V6, lw=1.8, label="v6（纯 pin：显示稳态 = Â）")
ax.plot(tt, z["Y_v6trim"][a:b], color=S.C_TRIM, lw=1.8, label="v6 + A 慢修正（trim）")
i5 = int(np.searchsorted(tu, seat + 5))
lvl5 = float(np.median(tot[i5:i5 + int(1 / dtm)]))
ax.axhline(lvl5, color=S.C_IDEAL, ls="--", lw=1.4, label="τ=5 s 实测电平（真值）")
pin = float(np.median(z["Y_v6"][i5:i5 + int(10 / dtm)]))
ax.annotate(f"Â 钉住 {pin:.0f}，比实测电平高 {100*(pin/lvl5-1):+.1f}%",
            xy=(8, pin), xytext=(9.5, pin + 0.055 * (pin - lvl5) * 8),
            fontsize=9.2, color=S.C_V6,
            arrowprops=dict(arrowstyle="->", color=S.C_V6, lw=1.0))
ax.annotate("trim 以 0.2%/s 把 A 拉回实测电平", xy=(30, float(np.median(z["Y_v6trim"][int(np.searchsorted(tu, seat + 30)):int(np.searchsorted(tu, seat + 32))]))),
            xytext=(14, lvl5 - 0.055 * (pin - lvl5) * 8), fontsize=9.2, color=S.C_TRIM,
            arrowprops=dict(arrowstyle="->", color=S.C_TRIM, lw=1.0))
ax.set_xlabel("时间（相对 133 s 的偏移，s）")
ax.set_ylabel("阵列总量 (ADC)")
ax.set_title("(a) 平台静态偏置：pin 模式下显示稳态 ≡ Â，先验偏多少就偏多少")
ax.legend(fontsize=8.6, loc="lower right", ncol=2)

# ── (b) 慢修正的死区机制 ────────────────────────────────────────
ax = axs[0, 1]
dev = np.linspace(-0.08, 0.08, 601) * 30000
dead = 0.025 * 30000
rate = 0.002 * 30000
eff = np.where(np.abs(dev) <= dead, 0.0, dev - np.sign(dev) * dead)
dlt = np.clip(eff, -rate, rate)
ax.plot(dev / 30000 * 100, dlt / 30000 * 100, color=S.C_TRIM, lw=2.2,
        label="实际修正速率（%/s）")
ax.plot(dev / 30000 * 100, dev / 30000 * 100 * 0 + 0.2, color="0.6", ls=":", lw=1.2,
        label="限速上限 0.2%/s")
ax.axvspan(-2.5, 2.5, color=S.C_V6, alpha=0.16, lw=0)
ax.text(0, -0.09, "死区 ±2.5%\n（偏差在此内不动）", ha="center", fontsize=9, color="#1b5e20")
for xv in (-2.5, 2.5):
    ax.axvline(xv, color="#1b5e20", ls="--", lw=1.2)
ax.set_xlabel("平台偏置（相对 ΣA，%）")
ax.set_ylabel("修正速率 (%/s)")
ax.set_ylim(-0.26, 0.26)
ax.set_title("(b) A 慢修正：死区内绝对平，死区外 0.2%/s 匀速回拉")
ax.legend(fontsize=8.6, loc="lower right")
S.note(ax, "目标在交接时一次性定下（= 交接时刻实测总电平），不再随帧变\n"
           "→ 不会来回抖，也不会把「本来就平的记录」拉出漂移", loc=(0.02, 0.97), va="top", ha="left",
       fs=8.6)

# ── (c) 三档消融 ────────────────────────────────────────────────
ax = axs[1, 0]
metrics = [("慢相段时漂\n(|·|均值 %)", h.groupby("arm").drift_slow.apply(lambda s: s.abs().mean())),
           ("平坦度\n(均值 %)", h.groupby("arm").flat.mean()),
           ("平台偏置\n(|·|均值 %)", ta.groupby("arm").platform_bias_pct.apply(lambda s: s.abs().mean())),
           ("实采全程偏差\n(中位 ADC)", v.groupby("arm").max_gap.median())]
xs = np.arange(len(metrics))
w = 0.26
for j, a in enumerate(ARM):
    vals = [mt[1].reindex(ARM)[a] for mt in metrics]
    b = ax.bar(xs + (j - 1) * w, vals, w, color=ACOL[a], label=ALB[a])
    for xi, vi in zip(xs + (j - 1) * w, vals):
        ax.text(xi, vi + 0.02 * max([max(mt[1].reindex(ARM)) for mt in metrics]), f"{vi:.2f}",
                ha="center", fontsize=8.4)
ax.set_xticks(xs)
ax.set_xticklabels([mt[0] for mt in metrics], fontsize=9)
ax.set_ylabel("指标值（四种量纲并列，仅看同组内相对高低）")
ax.set_ylim(0, max([max(mt[1].reindex(ARM)) for mt in metrics]) * 1.30)
ax.set_title("(c) trim 三档消融：慢相段时漂 1.20 → 1.42 → 2.40%，全程偏差 4583 → 3879 → 3808 ADC")
ax.legend(fontsize=8.4, loc="upper left")
S.note(ax, "死区只让「本来就不准」的三份记录被修正（其余 10 份逐帧不变）：\n"
           "收益仅多 71 ADC，却把慢相段时漂从 1.42% 推到 2.40%\n"
           "→ 取「死区 2.5% + 0.2%/s」", loc=(0.985, 0.97), va="top", fs=8.6)

# ── (d) 取舍平面 ────────────────────────────────────────────────
ax = axs[1, 1]
pts = []
for a in ARM:
    pts.append((ALB[a], h.groupby("arm").drift_slow.apply(lambda s: s.abs().mean())[a],
                v.groupby("arm").max_gap.median()[a], ACOL[a], "o"))
pts.append(("v5.1 无责 3 s（现役）", m[(m.kind == "恒载") & (m.algo == "e3s")].drift_slow.abs().mean(),
            m[(m.kind == "实采") & (m.algo == "e3s")].max_gap.median(), S.C_E3, "s"))
pts.append(("v5.1 无责 1 s", m[(m.kind == "恒载") & (m.algo == "e1s")].drift_slow.abs().mean(),
            m[(m.kind == "实采") & (m.algo == "e1s")].max_gap.median(), S.C_E1, "s"))
ys = [p[2] for p in pts]
span = max(ys) - min(ys)
shift = 0.085 * span
for i, (lab, x, y, col, mk) in enumerate(pts):
    ax.plot([x], [y], mk, color=col, ms=11, zorder=3)
    dy = shift if i % 2 == 0 else -0.95 * shift
    ha = "right" if i % 2 == 0 else "left"
    ax.annotate(lab, xy=(x, y), xytext=(x - 0.06 if ha == "right" else x + 0.06, y + dy),
                fontsize=9, color=col, ha=ha)
ax.set_xlabel("恒载 9 组慢相段时漂 |·| 均值 (%)　←「绝对平」越好")
ax.set_ylabel("实采 4 份全程最大偏差 中位 (ADC)　←「绝对准」越好")
ax.invert_xaxis()
ax.set_title("(d) 取舍平面：右下角（又平又准）在当前数据上不可达")
ax.grid(alpha=0.3)
S.note(ax, "v5 的 A 从数据里量出来 → 平；v6 的 Â 由先验反演 → 准与平由先验误差同时决定。\n"
           "trim 是把 v6 往 v5 那一侧拉的一根旋钮：拉多少，就用多少「平」去换「准」。",
       loc=(0.985, 0.045), fs=8.6)

fig.subplots_adjust(left=0.06, right=0.985, top=0.925, bottom=0.075, wspace=0.21, hspace=0.32)
outp = os.path.join(C.FIG, "F5_slow_tradeoff.png")
fig.savefig(outp)
S.figcheck(fig, outp)
plt.close(fig)
print("saved", outp)
print("\n[F5 数字]")
print(h.groupby("arm").agg(slow=("drift_slow", lambda s: s.abs().mean()), flat=("flat", "mean"),
                           tstab=("t_stable", "median")).round(3).to_string())
print(v.groupby("arm").agg(gap=("max_gap", "median"), tstab=("t_stable", "median")).round(3).to_string())
