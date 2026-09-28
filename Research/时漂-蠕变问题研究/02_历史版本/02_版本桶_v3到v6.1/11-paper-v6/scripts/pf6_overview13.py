# -*- coding: utf-8 -*-
"""论文图 F6：13 份录制的全量总览（raw / 无责1s / 无责3s / v6 / v6+trim）。

每格：灰=原始，蓝=无责 1 s，橙=无责 3 s（现役），绿=v6，红=v6+A 慢修正。
格标题给出该数据集的关键指标（恒载给慢相段时漂，实采给全程最大偏差）。
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

m = pd.read_csv(os.path.join(C.RES, "metrics_all.csv"))
st = pd.read_csv(os.path.join(C.RES, "metrics_settle.csv"))
order = [t for t, _ in C.ALL]
n, ncol = len(order), 4
nrow = int(np.ceil(n / ncol))

fig, axes = plt.subplots(nrow, ncol, figsize=(21.5, 3.0 * nrow))
fig.suptitle("图 F6  13 份录制总览：原始 / 无责 1 s / 无责 3 s（现役）/ v6 / v6+A 慢修正",
             fontsize=15.5)
for ax, tag in zip(axes.ravel(), order):
    try:
        z = C.load_npz(tag)
    except FileNotFoundError:
        ax.axis("off")
        ax.text(0.5, 0.5, f"缺少 {tag}", ha="center", transform=ax.transAxes)
        continue
    tu, tot = z["tu"], z["tot_s"]
    kind = C.KIND[tag]
    ax.plot(tu, tot, color=S.C_RAW, lw=0.9, zorder=1)
    for arm, col, ls, lw, lab in (("e1s", S.C_E1, "--", 1.0, "无责 1 s"),
                                  ("e3s", S.C_E3, "-.", 1.0, "无责 3 s"),
                                  ("v6", S.C_V6, "-", 1.25, "v6"),
                                  ("v6trim", S.C_TRIM, "-", 0.95, "v6+trim")):
        if "Y_" + arm in z:
            ax.plot(tu, z["Y_" + arm], color=col, ls=ls, lw=lw, label=lab, zorder=2)
    g = m[m.dataset == tag].set_index("algo")
    ss = st[st.dataset == tag].set_index("algo")
    if kind == "恒载":
        ttl = (f"{tag}｜慢相段时漂 3s {abs(g.loc['e3s','drift_slow']):.2f}% / "
               f"v6 {abs(g.loc['v6','drift_slow']):.2f}% / v6+trim "
               f"{abs(g.loc['v6trim','drift_slow']):.2f}%")
    else:
        ttl = (f"{tag}｜全程偏差 3s {g.loc['e3s','max_gap']:.0f} / v6 {g.loc['v6','max_gap']:.0f} / "
               f"v6+trim {g.loc['v6trim','max_gap']:.0f} ADC")
    if np.isfinite(ss.loc["v6", "t_stable"]):
        ttl += f"\nT_stable：3s {ss.loc['e3s','t_stable']:.2f} s → v6 {ss.loc['v6','t_stable']:.2f} s"
    ax.set_title(ttl, fontsize=8.7)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
for ax in axes.ravel()[n:]:
    ax.axis("off")
axes.ravel()[0].legend(fontsize=8.4, loc="lower left", ncol=2)
axes.ravel()[0].set_ylabel("阵列总量", fontsize=9)
fig.text(0.5, 0.012, "横轴：时间 (s)；纵轴：阵列总量（恒载为力值域 N，实采为 ADC 域，各格纵轴自适）",
         ha="center", fontsize=10, color="0.3")
fig.subplots_adjust(left=0.045, right=0.99, top=0.925, bottom=0.045, hspace=0.52, wspace=0.19)
outp = os.path.join(C.FIG, "F6_overview13.png")
fig.savefig(outp)
S.figcheck(fig, outp)
plt.close(fig)
print("saved", outp)
