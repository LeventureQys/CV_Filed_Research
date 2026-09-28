# -*- coding: utf-8 -*-
"""b4：图（读 b1/b2/b3 产出的 CSV，不重算）。产出 figures/b_*.png。"""
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_common as B                                             # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
ARMC = {"raw": "#7f7f7f", "e3s": "#ff7f0e", "v6": "#2ca02c", "v6trim": "#d62728"}
LBL = B.ARM_LABEL


def save(fig, name):
    path = os.path.join(B.FIG, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("->", os.path.relpath(path, B.FLASH))


def fig1():
    tr = pd.read_csv(os.path.join(B.RES, "b_unload_traj.csv"))
    ev = pd.read_csv(os.path.join(B.RES, "b_unload_events.csv"))
    ul = pd.read_csv(os.path.join(B.RES, "b_unload_raw.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, knd in zip(axes[:2], ("恒载", "实采")):
        sub = tr[tr.kind == knd]
        for arm in B.ARMS:
            s = sub[sub.arm == arm].sort_values("t")
            ax.plot(s.t, s["median"], marker="o", ms=3, color=ARMC[arm], label=LBL[arm])
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(f"卸载沿对齐轨迹（{knd}）")
        ax.set_xlabel("卸载沿后时间 (s)")
        ax.set_ylabel("(显示 − 卸载后稳态) / 原负载幅度")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[2]
    w = 0.2
    x = np.arange(2)
    for i, arm in enumerate(B.ARMS):
        vals = []
        for knd in ("恒载", "实采"):
            g = ev[(ev.kind == knd) & (ev.arm == arm)]
            vals.append(100.0 * g.dip_vs_post.median() / abs(g.drop_ev.median()))
        ax.bar(x + (i - 1.5) * w, vals, w, color=ARMC[arm], label=LBL[arm])
    ax.set_xticks(x)
    ax.set_xticklabels(["恒载 (力域)", "实采 (ADC)"])
    ax.set_ylabel("卸载后下冲 / 原幅度 (%)")
    ax.set_title("卸载沿下冲：各臂中位（相对该臂自身稳态）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    save(fig, "b_fig1_unload_traj.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    for knd, mk in (("恒载", "o"), ("实采", "s")):
        g = ul[ul.kind == knd]
        ax.semilogy(np.abs(g["drop"]), g.dip_post_pct.abs(), mk, label=knd)
    ax.set_xlabel("负载幅度 |drop|（原始单位，力域=N / 实录=ADC）")
    ax.set_ylabel("|下冲| / 幅度 (%)")
    ax.set_title("下冲深度 vs 负载幅度")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    for knd, mk in (("恒载", "o"), ("实采", "s")):
        g = ul[(ul.kind == knd) & (ul.post_src == "long")]
        ax.plot(g.hold_s, g.t_to5, mk, label=knd)
    ax.set_xlabel("保压时长 (s)")
    ax.set_ylabel("回到稳态 ±5% 的时间 (s)")
    ax.set_title("恢复时间 vs 保压时长（仅长窗事件）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[2]
    g = ul[(ul.kind == "恒载") & (ul.ev_class == "full")]
    ax.scatter(g.creep_slow_pct, g.resid_pct_drop, c="C0")
    for _, r in g.iterrows():
        ax.annotate(r.rec.split("/")[0][:3] + r.rec[-1], (r.creep_slow_pct, r.resid_pct_drop),
                    fontsize=7)
    from scipy import stats
    rr = stats.spearmanr(g.creep_slow_pct, g.resid_pct_drop)
    ax.set_xlabel("慢相蠕变 / A_5s (%)")
    ax.set_ylabel("卸载后残余偏移 / 幅度 (%)")
    ax.set_title(f"残余偏移 vs 慢相蠕变（恒载 n={len(g)}，ρ={rr.statistic:.2f}, p={rr.pvalue:.3f}）")
    ax.grid(alpha=0.3)
    save(fig, "b_fig2_unload_regulation.png")


def fig2():
    rec = pd.read_csv(os.path.join(B.RES, "b_repeat_records.csv"))
    disp = pd.read_csv(os.path.join(B.RES, "b_repeat_disp.csv"))
    shp = pd.read_csv(os.path.join(B.RES, "b_repeat_shape.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    mets = ["err5s", "err_plat5_pct", "plat_lvl", "move3060_pct", "t_stable", "epoch_total"]
    names = ["5s 电平误差\n(%)", "平台误差\n(%·5s)", "平台绝对\n电平", "平台移动\n30→60s(%)",
             "T_stable\n(s)", "epoch 数"]
    x = np.arange(len(mets))
    w = 0.2
    for i, arm in enumerate(B.ARMS):
        vals = []
        for m in mets:
            g = disp[(disp.arm == arm) & (disp.metric == m)]
            vals.append(float(g.rng.median()) if len(g) else np.nan)
        ax.bar(x + (i - 1.5) * w, vals, w, color=ARMC[arm], label=LBL[arm])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("组内极差（3 个传感器组的中位）")
    ax.set_title("可重复性：组内极差（越小越可重复）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    ax = axes[1]
    for arm in B.ARMS:
        g = rec[rec.arm == arm]
        for i, (grp, gg) in enumerate(g.groupby("group")):
            ax.scatter(np.full(len(gg), i) + (list(B.ARMS).index(arm) - 1.5) * 0.06,
                       gg.err_plat5_pct, color=ARMC[arm], s=28,
                       label=LBL[arm] if i == 0 else None)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["右拇指", "左拇指", "四指"])
    ax.set_ylabel("平台误差 @40–60 s（%·5 s 电平）")
    ax.set_title("同一负载 3 次重复的平台误差")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[2]
    v6 = rec[rec.arm == "v6"].set_index("rec")
    xv = shp.m_shape.to_numpy(float) * 100 - 100
    yv = [v6.loc[r, "err_plat5_pct"] for r in shp.rec]
    ax.scatter(xv, yv, c="C2")
    for i, r in enumerate(shp.rec):
        ax.annotate(r.split("/")[0][:3] + r[-1], (xv[i], yv[i]), fontsize=7)
    lim = [min(xv.min(), min(yv)) - 1, max(xv.max(), max(yv)) + 1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("形状库匹配因子 m−1 (%)  →  Â 的预测误差")
    ax.set_ylabel("v6 实测平台误差 (%)")
    ax.set_title("pin 语义：显示稳态 ≡ pre+Â（虚线 y=x）")
    ax.grid(alpha=0.3)
    save(fig, "b_fig3_repeatability.png")


def fig3():
    rec = pd.read_csv(os.path.join(B.RES, "b_improve_records.csv"))
    ul = pd.read_csv(os.path.join(B.RES, "b_improve_unload.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    order = list(dict.fromkeys(rec.variant))
    ax = axes[0]
    for i, v in enumerate(order):
        g = rec[rec.variant == v]
        ax.scatter(np.full(len(g), i), g.err_plat5_pct, s=30, color="C%d" % (i % 9))
        ax.hlines(g.err_plat5_pct.mean(), i - 0.2, i + 0.2, color="k", lw=2)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8, rotation=15)
    ax.set_ylabel("平台误差 @40–60 s（%）")
    ax.set_title("变体：平台精度（点=9 组，横线=均值）")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    for i, v in enumerate(order):
        g = ul[ul.variant == v]
        if not len(g):
            continue
        ax.scatter(np.full(len(g), i) + np.random.RandomState(0).randn(len(g)) * 0.03,
                   g.dip_pct, s=30, color="C%d" % (i % 9))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8, rotation=15)
    ax.set_ylabel("卸载沿下冲 / 幅度 (%)")
    ax.set_title("变体：卸载沿下冲（恒载 9 组卸载事件）")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[2]
    for i, v in enumerate(order):
        g = rec[rec.variant == v]
        ax.scatter(g.move3060_pct.abs(), g.err_plat5_pct.abs(), s=30,
                   color="C%d" % (i % 9), label=v)
    ax.set_xlabel("平台 30→60 s 移动 |%|（绝对平的代价）")
    ax.set_ylabel("平台误差 |%|（绝对准）")
    ax.set_title("「绝对平」与「绝对准」的权衡")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    save(fig, "b_fig4_improve.png")


def main():
    B.log_reconfigure()
    fig1()
    fig2()
    if os.path.exists(os.path.join(B.RES, "b_improve_records.csv")):
        fig3()
    else:
        print("!! 缺少 b_improve_records.csv，跳过 fig4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
