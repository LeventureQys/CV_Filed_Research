# -*- coding: utf-8 -*-
"""论文图 F1：加载形状的实测事实（v6 的地基）。

(a) 9 组恒载 onset 的归一化形状（统一以 0.20 s 为 0、5 s 为 1）
(b) 原始读数 vs τ=2 s 因果平滑：既有「快相 0~4 s」轮廓是平滑伪影
(c) onset 与 restep 的爬升时间（t50/t80/t90/t95）：同一传感器，输入不同
(d) 保压慢相：5 s 电平之后还要再涨多少（10~45%），到段末仍未收敛

数据：results/v6_onset_profile9.csv、v6_raw_vs_smoothed_profile.csv、v6_rise_times.csv
      + 恒载 9 组录制（temp/右拇指指尖|左拇指指尖|四指指尖/数据1~3）
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
import ad_lib as L                                               # noqa: E402

OLD = os.path.join(C.FLASH, "progress", "07-v6", "results")                           # 既有分析产物
TCOL = ["f_0.02", "f_0.04", "f_0.06", "f_0.08", "f_0.10", "f_0.15", "f_0.20",
        "f_0.30", "f_0.40", "f_0.50", "f_0.75", "f_1.00", "f_1.50", "f_2.00",
        "f_3.00", "f_4.00", "f_5.00"]
TGRID = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                  0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00])

fig, axs = plt.subplots(2, 2, figsize=(13.4, 9.2))
fig.suptitle("图 F1  加载形状的实测事实：v6 的三条地基", fontsize=14)

# ── (a) 归一化 onset 形状 ─────────────────────────────────────────
ax = axs[0, 0]
prof = pd.read_csv(os.path.join(OLD, "v6_onset_profile9.csv"))
M = prof[TCOL].to_numpy(float)
ref = M[:, TCOL.index("f_0.20")][:, None]
sh = (M - ref) / (M[:, -1:] - ref)                 # 以 0.20 s 为 0、5 s 为 1
med = np.median(sh, axis=0)
lo, hi = np.percentile(sh, 10, axis=0), np.percentile(sh, 90, axis=0)
for k in range(sh.shape[0]):
    ax.plot(TGRID, sh[k], color="0.78", lw=0.9, zorder=1)
ax.fill_between(TGRID, lo, hi, color=S.C_V6, alpha=0.16, zorder=2, label="p10~p90（9 组）")
ax.plot(TGRID, med, color=S.C_V6, lw=2.2, zorder=3, label="9 组中位（= 形状 ROM）")
ax.axvline(0.20, color=S.C_ANNO, ls=":", lw=1.4)
ax.annotate("形状锚点 $\\tau_{ref}$=0.20 s\n（避开时间戳量化区）", xy=(0.20, 0.35),
            xytext=(0.026, 0.55), fontsize=9, color=S.C_ANNO,
            arrowprops=dict(arrowstyle="->", color=S.C_ANNO, lw=1.0))
for g, lab in ((0.5, "0.5 s"), (1.0, "1 s"), (5.0, "5 s")):
    i = int(np.argmin(np.abs(TGRID - g)))
    ax.plot([g], [med[i]], "o", ms=4.5, color=S.C_V6, zorder=4)
txt = "  ".join(f"{lab} {med[int(np.argmin(np.abs(TGRID - g)))]:.3f}"
                for g, lab in ((0.3, "0.3 s"), (0.5, "0.5 s"), (1.0, "1 s"),
                               (2.0, "2 s"), (3.0, "3 s")))
ax.text(0.02, 0.98, "中位形状：" + txt, transform=ax.transAxes, fontsize=8.7, va="top",
        color="0.15", bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=2.2))
ax.set_xscale("log")
ax.set_xlim(0.018, 5.4)
ax.set_ylim(-0.06, 1.06)
ax.set_xlabel("加载沿后时间 $\\tau$ (s，对数轴)")
ax.set_ylabel("归一化形状 $f(\\tau)$")
ax.set_title("(a) 9 组恒载 onset 形状：$f(0.2)$=0、$f(5\\,\\mathrm{s})$=1")
ax.legend(loc="lower right")
S.note(ax, "组间极差：0.20 s 处 0.094、1 s 处 0.056、3 s 处 0.039\n"
           "→ 形状可复现，可作先验；0.05 s 处 0.367 属时间戳量化", loc=(0.02, 0.02),
       ha="left")

# ── (b) 原始 vs τ=2 s 平滑 ────────────────────────────────────────
ax = axs[0, 1]
rr = pd.read_csv(os.path.join(OLD, "v6_raw_vs_smoothed_profile.csv"))
rawc = [c for c in rr.columns if c.startswith("raw_")]
smc = [c for c in rr.columns if c.startswith("sm2_")]
tg = np.array([float(c.split("_")[1]) for c in rawc])
raw_med = np.median(rr[rawc].to_numpy(float), axis=0)
sm_med = np.median(rr[smc].to_numpy(float), axis=0)
doc = 1 - np.exp(-tg / 2.0)
doc = doc / np.interp(5.0, tg, doc)
ax.plot(tg, raw_med, "o-", color=S.C_V6, lw=2.0, ms=4, label="原始读数（3 帧中值）")
ax.plot(tg, sm_med, "s--", color=S.C_E1, lw=1.8, ms=3.6, label="τ=2 s 因果指数平滑")
ax.plot(tg, doc, ":", color="0.45", lw=1.8, label="τ=2 s 一阶系统阶跃响应（理论）")
for g in (1.0,):
    i = int(np.argmin(np.abs(tg - g)))
    ax.annotate(f"1 s：原始 {raw_med[i]:.3f}\n      平滑 {sm_med[i]:.3f}",
                xy=(g, raw_med[i]), xytext=(1.35, 0.42), fontsize=9, color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
ax.axvspan(0.02, 0.2, color=S.C_EXEMPT, alpha=0.25, lw=0)
ax.text(0.06, 0.06, "量化区", fontsize=8.5, color="0.35", ha="center")
ax.set_xlim(0, 5.2)
ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("加载沿后时间 (s)")
ax.set_ylabel("按 5 s 归一化的读数")
ax.set_title("(b) 既有「快相 0~4 s」= τ=2 s 平滑伪影（13 份录制中位）")
ax.legend(loc="lower right")
S.note(ax, "比值 0.86~1.00 随 τ 单调趋 1，同一形状源", loc=(0.02, 0.94), va="top", ha="left")

# ── (c) onset vs restep 爬升时间 ──────────────────────────────────
ax = axs[1, 0]
rt = pd.read_csv(os.path.join(OLD, "v6_rise_times.csv"))
qs = ["t50", "t80", "t90", "t95"]
on = rt[rt.kind == "onset"]
rs = rt[rt.kind == "restep"]
x = np.arange(len(qs))
for j, (sub, col, lab, mk) in enumerate(((on, S.C_V6, f"onset（空载→负载，n={len(on)}）", "o"),
                                         (rs, S.C_E3, f"restep（负载内加重，n={len(rs)}）", "s"))):
    v = [sub[q].median() for q in qs]
    ax.bar(x + (j - 0.5) * 0.34, v, 0.32, color=col, alpha=0.9, label=lab)
    for xi, vi in zip(x + (j - 0.5) * 0.34, v):
        ax.text(xi, vi + 0.07, f"{vi:.2f}", ha="center", fontsize=8.8, color="0.15")
    ax.plot(x + (j - 0.5) * 0.34, v, mk, color="white", ms=4, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(["t50", "t80", "t90", "t95"])
ax.set_ylabel("达到该台阶比例的时刻 (s)")
ax.set_ylim(0, max(rs[q].median() for q in qs) * 1.30)
ax.set_title("(c) 同一传感器、不同输入：restep 的 t90 是 onset 的 6 倍")
ax.legend(loc="upper left")
S.note(ax, "onset 1 s 已走 92.5%；restep 1 s 只走 64.1%\n"
           "restep 的 1 s 目标不是「算法慢」，是输入还没发生", loc=(0.985, 0.03))

# ── (d) 保压慢相：5 s 之后还要涨多少 ──────────────────────────────
ax = axs[1, 1]
rows = []
for tag_, path in C.HOLD:
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    s0, s1 = L.find_periods(tot, dtm)[0]
    s0 = int(s0)
    s1 = min(int(s1), len(tu) - 1)
    base = float(np.median(tot[max(0, s0 - int(2 / dtm)):s0]))
    lvl5 = float(np.median(tot[s0 + int(4.9 / dtm):s0 + int(5.1 / dtm)]))
    end = float(np.median(tot[s1 - int(3 / dtm):s1]))
    rows.append((tag_, 100 * (end - lvl5) / lvl5, (s1 - s0) * dtm,
                 tu[s0:s1], 100 * (tot[s0:s1] - lvl5) / lvl5))
rows.sort(key=lambda r: r[1])
names = [r[0].split("/")[-1] + "\n" + r[0].split("/")[0][:3] for r in rows]
vals = [r[1] for r in rows]
cols = [S.C_V6 if v < 25 else S.C_TRIM for v in vals]
b = ax.barh(np.arange(len(rows)), vals, color=cols, alpha=0.9)
ax.bar_label(b, fmt="%.1f%%  (%ds)", labels=[f"{v:.1f}%  ({r[2]:.0f}s)" for v, r in zip(vals, rows)],
             fontsize=8.6, padding=2)
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(names, fontsize=8.6)
ax.set_xlim(0, max(vals) * 1.42)
ax.set_xlabel("5 s 电平 → 负载段末的涨幅 (%)")
ax.set_title("(d) 保压慢相：5 s 之后还要再涨 10~45%，段末仍未收敛")
ax.grid(axis="y", alpha=0)
S.note(ax, "这正是「1 s 稳住」要压住的量：\n显示停在 5 s 电平 = 无蠕变真值",
       loc=(0.985, 0.06), fs=9)

fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.065, wspace=0.22, hspace=0.30)
out = os.path.join(C.FIG, "F1_two_phase.png")
fig.savefig(out)
S.figcheck(fig, out)
plt.close(fig)
print("saved", out)

# 供正文引用的数字
print("\n[F1 数字]")
print(f"  形状中位 f(0.3/0.5/1.0/2.0/3.0) = "
      + " ".join(f"{med[int(np.argmin(np.abs(TGRID-g)))]:.3f}" for g in (0.3, 0.5, 1.0, 2.0, 3.0)))
print(f"  逐点极差(p90-p10) 0.2/1.0/3.0 s = "
      + " ".join(f"{hi[int(np.argmin(np.abs(TGRID-g)))]-lo[int(np.argmin(np.abs(TGRID-g)))]:.3f}"
                 for g in (0.2, 1.0, 3.0)))
print(f"  onset  n={len(on)}  t50/t80/t90/t95 = "
      + " ".join(f"{on[q].median():.2f}" for q in qs))
print(f"  restep n={len(rs)}  t50/t80/t90/t95 = "
      + " ".join(f"{rs[q].median():.2f}" for q in qs))
print("  慢相 5s→段末涨幅: " + "  ".join(f"{r[0]} {r[1]:+.1f}%({r[2]:.0f}s)" for r in rows))
