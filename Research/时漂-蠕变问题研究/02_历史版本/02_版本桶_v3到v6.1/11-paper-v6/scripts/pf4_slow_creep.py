# -*- coding: utf-8 -*-
"""论文图 F4：慢相蠕变估计与扣除（本文的「纯蠕变」主体，逐行沿用 v5）。

(a) 逐通道归一化残差 rel_i=(Z_i−A_i)/A_i 的分布与中位共识 g_raw（保压段 3 个时刻）
(b) 中位共识 g 的全过程轨迹（恒载 9 组叠加）——扣除量真正的来源
(c) 逐通道增益 γ_i 的分布与限幅区间，附 A_i–γ_i 散点
(d) 逐通道扣除量 ded_i/A_i 的分布与限幅带（−0.5, +1.5），与输出封顶规则

数据：恒载 9 组录制（temp/右拇指指尖|左拇指指尖|四指指尖/数据1~3）
      + paper_v6/results/metrics_all.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                            # noqa: E402
import pv_style as S                                             # noqa: E402
import ad_lib as L                                               # noqa: E402
from glm53_v6 import GLM53v6                                     # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402


class Record(GLM53v6):
    """记录逐帧 g / γ / A / 扣除量（只在慢相段采样，内存可控）。"""

    def __init__(self, n):
        super().__init__(n)
        self.log = []

    def _slow_step(self, ts, v, dt):
        o = super()._slow_step(ts, v, dt)
        ded = self._deduction_vector(v)
        self.log.append((float(ts), float(self.g), float(self.A.sum()),
                         np.asarray(self.gamma, float).copy(),
                         np.asarray(self.A, float).copy(),
                         np.asarray(ded, float),
                         np.asarray(v, float).copy()))
        return o


rows = []
G_traces, gammas, ded_ratio, resids = {}, [], [], {}
for tag, path in C.HOLD:
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    s0, s1 = L.find_periods(tot, dtm)[0]
    c = Record(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    lg = c.log
    tlog = np.array([r[0] for r in lg])
    glog = np.array([r[1] for r in lg])
    m = (tlog >= tu[s0]) & (tlog <= tu[min(s1, len(tu) - 1)])
    if m.sum() > 10:
        tt = tlog[m] - tu[s0]
        gg = glog[m]
        grid = np.linspace(0, tt.max(), 400)
        G_traces[tag] = (grid, np.interp(grid, tt, gg))
    # 逐个快照取 γ / A / ded
    for (ts_, g_, asum, gam, A, ded, v) in lg[:: max(1, len(lg) // 400)]:
        for k in range(len(A)):
            if A[k] > 1e-9 and gam[k] != 1.0:
                gammas.append((tag, float(gam[k]), float(A[k]), float(ded[k] / A[k]), float(g_)))
    # 保压段中段的逐通道残差
    i_mid = (s0 + s1) // 2
    A_end = lg[-1][4]
    resids[tag] = ((Xu[i_mid] - A_end) / np.where(A_end > 1e-9, A_end, np.nan))
    rows.append(dict(tag=tag, g_end=glog[-1] if len(glog) else np.nan,
                     asum=float(lg[-1][3].sum()), slots=len(lg)))

fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.2))
fig.suptitle("图 F4  慢相蠕变估计与扣除：中位共识 g + 逐通道增益 γ + 限幅与输出封顶", fontsize=14)

# ── (a) 残差分布与中位共识 ───────────────────────────────────────
ax = axs[0, 0]
sel = ["右拇指指尖/数据1", "右拇指指尖/数据2", "左拇指指尖/数据1"]
offs = {s: 0.0 for s in sel}
for j, s in enumerate(sel):
    r = resids[s]
    r = r[np.isfinite(r)]
    r = r[r > -0.2]
    ax.scatter(np.full(len(r), j) + np.random.default_rng(0).normal(0, 0.045, len(r)), r,
               s=14, alpha=0.55, color=S.C_V6 if j == 0 else (S.C_E1 if j == 1 else S.C_E3),
               label=s)
    med = float(np.median(r))
    ax.plot([j - 0.30, j + 0.30], [med, med], color="#c62828", lw=2.2)
    ax.text(j + 0.34, med, f"$g_{{raw}}$={med:.3f}", fontsize=9, color="#c62828", va="center")
ax.set_xticks(range(len(sel)))
ax.set_xticklabels([s.split("/")[-1] + "\n" + s.split("/")[0][:3] for s in sel], fontsize=9)
ax.set_ylabel("归一化残差 $rel_i=(Z_i-A_i)/A_i$")
ax.set_title("(a) 逐通道残差（受载通道）与其中位共识 $g_{raw}$")
ax.legend(fontsize=8.4, loc="upper left")
S.note(ax, "取中位而非均值：器件蠕变是乘性差异、残差长尾，\n均值会被个别通道拉走", loc=(0.985, 0.03))

# ── (b) g 的全过程轨迹 ──────────────────────────────────────────
ax = axs[0, 1]
for tag, (grid, gg) in G_traces.items():
    ax.plot(grid, gg, lw=1.25, alpha=0.85,
            label=tag.split("/")[1] + " " + tag.split("/")[0][:3] if tag in
            ("右拇指指尖/数据1", "左拇指指尖/数据1", "四指指尖/数据1") else None)
med_grid = np.linspace(0, 140, 300)
A_ = np.vstack([np.interp(med_grid, g, v) for _, (g, v) in G_traces.items() if v.max() > 0.02])
ax.plot(med_grid, np.median(A_, axis=0), color="#c62828", lw=2.6, label="9 组中位")
ax.axhline(0.02, color="0.4", ls=":", lw=1.2)
ax.text(2, 0.024, "γ 更新门槛 $g>0.02$", fontsize=8.6, color="0.35")
ax.set_xlabel("负载段内时间 (s)")
ax.set_ylabel("蠕变场共识 $g$")
ax.set_title("(b) $g$ 的全过程：慢相内单调增长、段末仍未饱和（跟随而非扣完）")
ax.legend(fontsize=8.4, loc="upper left")

# ── (c) γ 分布与限幅 ────────────────────────────────────────────
ax = axs[1, 0]
G = pd.DataFrame(gammas, columns=["tag", "gamma", "A", "ded_ratio", "g"])
G = G[G.gamma > 0.05]
ax.hist(G.gamma, bins=np.arange(0.2, 2.1, 0.06), color=S.C_E1, alpha=0.85, edgecolor="white")
for xv, lab, col in ((0.3, "下限 0.3", "#c62828"), (2.0, "上限 2.0", "#c62828")):
    ax.axvline(xv, color=col, lw=1.8, ls="--")
ax.text(0.33, ax.get_ylim()[1] * 0.90, "限幅 [0.3, 2.0]", color="#c62828", fontsize=9.2)
ax.text(0.33, ax.get_ylim()[1] * 0.80, f"中位 γ = {G.gamma.median():.3f}\n"
        f"落在限幅内 {100*np.mean((G.gamma>0.3)&(G.gamma<2.0)):.1f}%",
        fontsize=9, color="0.2")
ax.set_xlabel("逐通道增益 $\\gamma_i$")
ax.set_ylabel("采样点计数")
ax.set_title("(c) $\\gamma_i$ = 过原点增量最小二乘（只在 $g>0.02$ 时更新，不随 epoch 重置）")

# ── (d) 扣除量与限幅、输出封顶 ───────────────────────────────────
ax = axs[1, 1]
dr = G.ded_ratio.to_numpy(float)
ax.hist(np.clip(dr, -0.6, 1.6), bins=np.arange(-0.6, 1.65, 0.05), color=S.C_V6,
        alpha=0.85, edgecolor="white")
for xv, lab in ((-0.5, "−0.5$A_i$"), (1.5, "+1.5$A_i$")):
    ax.axvline(xv, color="#c62828", lw=1.8, ls="--")
    ax.text(xv + (0.03 if xv > 0 else -0.03), ax.get_ylim()[1] * 0.90, lab, color="#c62828",
            fontsize=9.2, ha="left" if xv > 0 else "right")
ax.axvline(float(np.median(dr)), color="0.25", lw=1.6)
ax.text(float(np.median(dr)) + 0.03, ax.get_ylim()[1] * 0.72,
        f"中位 {np.median(dr):.3f}$A_i$", fontsize=9.2, color="0.2")
ax.set_xlabel("单通道扣除量 / 该通道幅度  $ded_i/A_i$")
ax.set_ylabel("采样点计数")
ax.set_title("(d) 扣除量限幅带 [−0.5$A_i$, +1.5$A_i$]，末级再做输出封顶")
S.note(ax, "输出封顶：$v_i=Z_i-\\min(ded_i,\\max(Z_i,0))$\n只改输出值，$g/\\gamma/A/$carry 的内部状态照常演进",
       loc=(0.985, 0.62), fs=8.8)

fig.subplots_adjust(left=0.06, right=0.985, top=0.925, bottom=0.07, wspace=0.22, hspace=0.30)
outp = os.path.join(C.FIG, "F4_slow_creep.png")
fig.savefig(outp)
S.figcheck(fig, outp)
plt.close(fig)
print("saved", outp)

m = pd.read_csv(os.path.join(C.RES, "metrics_all.csv"))
hold = m[m.kind == "恒载"]
print("\n[F4 数字]")
print(f"  γ 中位 {G.gamma.median():.3f}，p10~p90 {G.gamma.quantile(.1):.3f}~{G.gamma.quantile(.9):.3f}，"
      f"限幅内占比 {100*np.mean((G.gamma>0.3)&(G.gamma<2.0)):.1f}%")
print(f"  ded/A 中位 {np.median(dr):.3f}，p10~p90 {np.quantile(dr,.1):.3f}~{np.quantile(dr,.9):.3f}")
print(f"  g 末端（恒载 9 组均值）{hold[hold.algo=='v6'].g_end.mean():.3f}；"
      f"A 占幅度均值 {hold[hold.algo=='v6'].a_max.mean():.3f}")
print(f"  慢相段时漂残余：raw {hold[hold.algo=='raw'].drift_slow.abs().mean():.3f}% → "
      f"v6 {hold[hold.algo=='v6'].drift_slow.abs().mean():.3f}%")
