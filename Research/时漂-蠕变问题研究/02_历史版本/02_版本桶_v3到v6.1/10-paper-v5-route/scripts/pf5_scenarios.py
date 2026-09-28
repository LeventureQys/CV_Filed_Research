# -*- coding: utf-8 -*-
"""图 5：合成砝码 7 场景的稳态保真度（受载沿 + 逐帧偏差 + 偏差汇总）。

合成方式（与 scripts/bv_scenarios_v51.py 完全同一套）：蠕变律参数由实采保压段拟合
得到（0.00%·τ=42.6 s + 15.64%·τ=199 s），场景按砝码叠加构造，采样 100 Hz、共 600 s。
理想值 = 各档砝码之和（基准 10 N = 10000 ADC）；稳态取末端 40 s 均值。

产出：figures/F5_scenarios.png、results/f5_scenarios.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize

import pd as P
from figstyle import (C_ALG, C_FAST, C_GRAY, C_RAW, C_SLOW, plt, save_figure)

FLASH = P.FLASH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ad_lib as L                                                     # noqa: E402
from glm53_v51 import GLM53v51                                         # noqa: E402

# ── 蠕变律：从实采保压段拟合（与图 1(c) 同一窗口 133.84~182.0 s）──────
d0 = L.prep(os.path.join(P.TEMP, "变化负载", "切换负载-快相无责的测试",
                         "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"))
a, b = int(133.84 / d0["dtm"]), int(182.0 / d0["dtm"])
uu = d0["tu"][a:b] - d0["tu"][a]
yy = d0["tot"][a:b] / d0["tot"][a + int(5.0 / d0["dtm"])] - 1.0


def ft(x, a1, t1, a2, t2):
    return a1 * (1 - np.exp(-x / t1)) + a2 * (1 - np.exp(-x / t2))


Pc, _ = optimize.curve_fit(ft, uu, yy, p0=[0.03, 3.0, 0.05, 60.0],
                           bounds=([0, 0.2, 0, 5], [1, 60, 1, 3000]))
print(f"[F5] 蠕变律（实采保压段拟合）：{100*Pc[0]:.2f}%·τ={Pc[1]:.1f}s + {100*Pc[2]:.2f}%·τ={Pc[3]:.0f}s")


def cr(x):
    return np.where(x > 0, ft(np.maximum(x, 0), *Pc), 0.0)


FS, dt, DUR = 100.0, 0.01, 600.0
tt = np.arange(0.0, DUR, dt)


def build(steps, base_n=1.0):
    s = np.zeros_like(tt)
    m = tt > 20.0
    s[m] += base_n * 10000.0 * (1 + cr(tt[m] - 20.0))
    for t0, k in steps:
        mm = tt > t0
        s[mm] += k * 10000.0 * (1 + cr(tt[mm] - t0))
    return s


def run(sig):
    c = GLM53v51(1)
    Y = np.empty(len(tt))
    for i in range(len(tt)):
        Y[i] = c.process(tt[i], np.array([sig[i]]))[0]
    return Y, c


CASES = [("10 N → 280 s → +2.5 N（台阶 25%）", [(300.0, 0.25)], 1.0),
         ("10 N → 280 s → +5 N（台阶 50%）", [(300.0, 0.50)], 1.0),
         ("10 N → 280 s → +10 N（台阶 100%）", [(300.0, 1.00)], 1.0),
         ("20 N → 280 s → +5 N（台阶 12.5%）", [(300.0, 0.25)], 2.0),
         ("10 N → 200 s → +5 N → 80 s 后 +5 N", [(220.0, 0.50), (300.0, 0.50)], 1.0),
         ("10 N → 200 s → +5 N → 8 s 后 +5 N", [(220.0, 0.50), (228.0, 0.50)], 1.0),
         ("10 N → 200 s → +5 N → 3 s 后 +5 N", [(220.0, 0.50), (223.0, 0.50)], 1.0)]

tail = slice(int((DUR - 40) / dt), None)
rows, traces = [], {}
for title, steps, base_n in CASES:
    sig = build(steps, base_n)
    ideal = (base_n + sum(k for _, k in steps)) * 10000.0
    Y, c = run(sig)
    end = float(Y[tail].mean())
    rows.append(dict(case=title, ideal=ideal, raw=float(sig[tail].mean()), algo=end,
                     dev_pct=100 * (end - ideal) / ideal,
                     dev_trace_p99=float(np.percentile(np.abs(Y - ideal) / ideal * 100, 99))))
    traces[title] = (sig, Y, ideal)
    print(f"[F5] {title}: 理想 {ideal:.0f}、原始 {sig[tail].mean():.0f}、"
          f"本算法 {end:.0f}（{100*(end-ideal)/ideal:+.2f}%）")

df = pd.DataFrame(rows)
P.save_table(df, "f5_scenarios.csv")
print(f"[F5] 偏差范围 {df.dev_pct.min():+.2f}% ~ {df.dev_pct.max():+.2f}%，"
      f"|偏差| 中位 {df.dev_pct.abs().median():.2f}%")

# ── 绘图 ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.0, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.42,
                      left=0.055, right=0.985, top=0.885, bottom=0.135)

# (a) +2.5 N 场景的逐帧轨迹
title0 = CASES[0][0]
sig, Y, ideal = traces[title0]
ax = fig.add_subplot(gs[0, 0])
sel = (tt >= 260) & (tt <= 600)
ax.plot(tt[sel], sig[sel], color=C_RAW, lw=1.9, label="原始（含蠕变）", zorder=3)
ax.plot(tt[sel], Y[sel], color=C_ALG, lw=1.9, label="本算法", zorder=4)
ax.axhline(ideal, color=C_SLOW, lw=2.2, ls=(0, (4, 3)), alpha=0.95, label="理想值 12500",
           zorder=5)
ax.axvline(300.0, color=C_FAST, lw=1.6, ls=":", zorder=6)
ax.set_xlim(260, 600)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("显示值 (ADC)")
ax.set_title("(a) 保压 280 s 后加 2.5 N：显示立刻跟随台阶", fontsize=11)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=8.4, loc="lower right", framealpha=0.95)
ylo, yhi = ax.get_ylim()
ax.set_ylim(ylo - 0.06 * (yhi - ylo), yhi + 0.08 * (yhi - ylo))
ax.text(302, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        "台阶 @300 s", fontsize=8.5, color=C_FAST, va="bottom")
i_305 = int(305.5 / dt)
ax.annotate("台阶后 0.5 s 显示 %.0f" % Y[i_305], xy=(305.5, Y[i_305]),
            xytext=(352, Y[i_305] - 0.26 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
            fontsize=8.5, color=C_ALG,
            arrowprops=dict(arrowstyle="->", color=C_ALG, lw=1.0))

# (b) 七个场景的偏差
ax2 = fig.add_subplot(gs[0, 1])
y = np.arange(len(df))[::-1]
colors = [C_ALG if v < 0 else C_FAST for v in df.dev_pct]
ax2.barh(y, df.dev_pct, height=0.55, color=colors, alpha=0.88)
ax2.axvline(0.0, color=C_GRAY, lw=1.2)
ax2.set_yticks(y)
ax2.set_yticklabels([c.split("（")[0] for c in df.case], fontsize=8.0)
ax2.set_xlim(-0.60, 0.60)
ax2.set_ylim(-0.75, len(df) + 0.35)
ax2.set_xlabel("末端 40 s 均值相对理想值的偏差 (%)")
ax2.set_title("(b) 七个场景的稳态偏差：−0.22% ~ +0.33%", fontsize=11)
ax2.grid(alpha=0.25, axis="x", lw=0.6)
for yy, v in zip(y, df.dev_pct):
    ax2.text(v + (0.05 if v >= 0 else -0.05), yy, "%+.2f%%" % v, va="center",
             ha="left" if v >= 0 else "right", fontsize=8.3,
             color=C_FAST if v >= 0 else C_ALG)
ax2.text(0.02, 0.03, "原始同场景偏差 +%.0f%% ~ +%.0f%%\n（含蠕变）"
         % (((df.raw - df.ideal) / df.ideal * 100).min(),
            ((df.raw - df.ideal) / df.ideal * 100).max()),
         transform=ax2.transAxes, va="bottom", ha="left",
         fontsize=8.2, color=C_RAW,
         bbox=dict(fc="white", ec="#D5D8DC", lw=0.8, alpha=0.95, pad=2.5))

save_figure(fig, "F5_scenarios.png")
