# -*- coding: utf-8 -*-
"""T3-A 03：图件（全部英文标签，避免中文字体缺失出现方框；每图有标题/轴标签(含单位)/图例）。

  T3A_01_shape_spread.png  形状变异带（onset/restep 分面；中位 + p10~p90 + 逐事件细线）
  T3A_02_loo_error.png     乐观（自身标定）vs 悲观（留一/跨录制/跨族）误差分布 + >5% 占比
  T3A_03_strata.png        分层稳定性（输入形态 / 相对载荷量级 / 族 / 批次）
  T3A_04_time_stability.png 时不变性（录制内早/中/晚 + 组内 trend）

运行：python scripts/t3a_03_figures.py
"""
import os

import numpy as np
import pandas as pd

import t3a_common as C
import matplotlib.pyplot as plt

CL = dict(onset="#1f77b4", restep="#d62728", ok="#2ca02c", bad="#d62728", grey="#888888")

# 图内一律英文（DejaVu Sans 无 CJK 字形，出方框即违反 00 号文档 §4.4）
LBL = {
    "onset 快速阶跃": "onset, hard step", "onset 慢压": "onset, slow press",
    "restep": "restep", "onset(T_ramp缺)": "onset (T_ramp n/a)",
    "右拇指": "right thumb", "左拇指": "left thumb", "四指": "four fingers",
    "实录": "switching-load recordings",
    "数据1": "rep 1", "数据2": "rep 2", "数据3": "rep 3", "数据4": "rep 4 (SW)",
    "早期(前1/3)": "early (first third)", "中期(中1/3)": "middle (second third)",
    "晚期(后1/3)": "late (last third)",
    "早期": "early", "中期": "middle", "晚期": "late",
    "≤5s": "<=5 s", "5~30s": "5~30 s", ">30s": ">30 s",
}


def en(s):
    return LBL.get(str(s), str(s))


def fig1():
    grid = pd.read_csv(os.path.join(C.RES, "t3a_shape_grid.csv"))
    aud = pd.read_csv(os.path.join(C.RES, "t3a_inlier_audit.csv"))
    gridf = grid[grid["grid"] == C.TAU_FINE_LABEL]
    gridm = grid[grid["grid"] == C.TAU_MAIN_LABEL]
    us = set(aud[aud["usable"]]["uid"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for j, kind in enumerate(["onset", "restep"]):
        for i, (sname, uids, ttl) in enumerate([
                ("all", set(aud[aud["is_load"] & (aud["kind"] == kind)]["uid"]), "all load events"),
                ("usable", set(aud[aud["usable"] & (aud["kind"] == kind)]["uid"]), "usable subset")]):
            ax = axes[j][i] if j < 2 else None
            ax = axes[j, i]
            g = gridf[gridf["uid"].isin(uids)]
            if g.empty:
                ax.set_title("%s / %s: no data" % (kind, sname))
                continue
            piv = g.pivot_table(index="tau", columns="uid", values="f")
            for c in piv.columns:
                ax.plot(piv.index, piv[c], color=CL[kind], lw=0.6, alpha=0.35, zorder=2)
            ax.fill_between(piv.index, piv.quantile(0.10, axis=1), piv.quantile(0.90, axis=1),
                            color=CL[kind], alpha=0.18, zorder=3, label="p10~p90 band")
            ax.plot(piv.index, piv.median(axis=1), color="k", lw=2.0, zorder=4, label="median")
            gm = gridm[gridm["uid"].isin(uids)]
            for t in C.TAU_MAIN:
                v = gm[np.isclose(gm["tau"], t)]["f"]
                ax.plot([t], [v.median()], "o", ms=4, color="#ff7f0e", zorder=5,
                        label=("T4-A z_at_* grid" if t == C.TAU_MAIN[0] else None))
            ax.axvspan(0.0, 0.2, color="grey", alpha=0.18, zorder=1)
            ax.set_xscale("log")
            ax.set_xlim(0.01, 5.2)
            ax.set_ylim(-0.6, 1.45)
            ax.set_xlabel("tau after t_on  (s, log scale)")
            ax.set_ylabel("normalized shape  f(tau) = (Z(t_on+tau)-pre)/J")
            ax.set_title("%s / %s  (n=%d)   grey band: tau<0.2 s distortion zone"
                         % (kind, sname, piv.shape[1]), fontsize=9)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("T3A_01  Normalized fast-phase shape spread "
                 "(base = T4-A pre-window median, Z = sum over channels)", fontsize=11)
    C.savefig(fig, "T3A_01_shape_spread.png")


def fig2():
    g = pd.read_csv(os.path.join(C.RES, "t3a_generalization_raw.csv"))
    s = pd.read_csv(os.path.join(C.RES, "t3a_generalization.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    scen = ["own", "loo_event", "loo_rec", "cross_fam"]
    lab = ["own\n(optimistic)", "LOO event\n(same family)", "LOO recording\n(same family)",
           "cross family"]
    for ax, kind in zip(axes[:2], ["onset", "restep"]):
        data, share = [], []
        for sc in scen:
            q = g[(g["usable"]) & (g["scenario"] == sc) & (g["kind"] == kind)
                  & (g["tau_d"] == 0.20) & (g["inv"] == "single@tau") & (g["ref"] == "post")]
            v = np.abs(q["err_pct"].to_numpy(float))
            v = v[np.isfinite(v)]
            data.append(v if v.size else np.array([np.nan]))
            share.append(float(np.mean(v > 5.0) * 100.0) if v.size else np.nan)
        bp = ax.boxplot(data, tick_labels=lab, showfliers=True, widths=0.55)
        for b in bp["boxes"]:
            b.set_color(CL[kind])
        for xi, v in enumerate(data):
            ax.text(xi + 1, 92, ">5%%: %.0f%%\nn=%d" % (share[xi], v.size), ha="center", fontsize=7.5)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 400)
        ax.axhline(5.0, color="k", ls="--", lw=1.0)
        ax.text(0.62, 5.6, "5% acceptance line", fontsize=7.5)
        ax.set_ylabel("|amplitude inversion error|  (%)")
        ax.set_title("%s   single-point inversion at tau_d=0.20 s" % kind, fontsize=9.5)
        ax.grid(alpha=0.25, axis="y")
    # 第三格：τ_d 曲线
    ax = axes[2]
    for kind, ls in [("onset", "-"), ("restep", "--")]:
        for sc, mk in [("own", "o"), ("loo_event", "s"), ("cross_fam", "^")]:
            q = s[(s["sample"] == "usable") & (s["scenario"] == sc) & (s["kind"] == kind)
                  & (s["inv"] == "single@tau") & (s["ref"] == "post")]
            if q.empty:
                continue
            q = q.sort_values("tau_d")
            ax.plot(q["tau_d"], q["med_abs_pct"], ls, marker=mk, ms=4,
                    color=CL[kind], alpha=0.85,
                    label="%s / %s" % (kind, sc))
    ax.axhline(5.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("decision time tau_d  (s)")
    ax.set_ylabel("median |error|  (%)")
    ax.set_title("error vs decision time (usable sample)", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, ncol=1)
    fig.suptitle("T3A_02  Shape-library generalization: optimistic (self-calibrated) vs "
                 "pessimistic (leave-one-out / cross-family)", fontsize=11)
    C.savefig(fig, "T3A_02_loo_error.png")


def fig3():
    st = pd.read_csv(os.path.join(C.RES, "t3a_strata.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    dims = [("otype", "input morphology (T4-A T_ramp class)"),
            ("magbin", "relative load magnitude |J|/record peak (no calibration data)"),
            ("fam", "sensor family / recording group"),
            ("batch", "batch (per-family repetition index)")]
    for ax, (dim, ttl) in zip(axes.ravel(), dims):
        sub = st[(st["dim"] == dim) & (st["sample"] == "usable") & (st["col"] == "z_at_03")]
        if sub.empty:
            continue
        sub = sub.sort_values("med")
        y = np.arange(len(sub))
        ax.barh(y, sub["med"], color=CL["onset"], alpha=0.75, height=0.55)
        ax.errorbar(sub["med"], y, xerr=[sub["med"] - sub["p10"], sub["p90"] - sub["med"]],
                    fmt="none", ecolor="k", capsize=3, lw=1.0)
        for yi, (_, r) in zip(y, sub.iterrows()):
            ax.text(max(r["p90"], r["med"]) + 0.012, yi, "n=%d" % int(r["n_events_layer"]),
                    va="center", fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels([en(v) for v in sub["layer"]], fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.axvline(float(sub["med"].median()), color=CL["restep"], ls="--", lw=1.0)
        ax.set_xlabel("f(0.30 s)  [-],  bar = median, whiskers = p10~p90")
        ax.set_title("strata: %s" % ttl, fontsize=9)
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("T3A_03  Shape stability by stratum (usable sample; n<20 -> median + p10~p90, "
                 "n<=3 qualitative only)", fontsize=11)
    C.savefig(fig, "T3A_03_strata.png")


def fig4():
    ts = pd.read_csv(os.path.join(C.RES, "t3a_time_stability.csv"))
    tro = pd.read_csv(os.path.join(C.RES, "t3a_time_trend.csv"))
    tra = pd.read_csv(os.path.join(C.RES, "t3a_shape_trajectory.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
    # (a) 录制内早/中/晚（onset）
    ax = axes[0]
    sub = ts[(ts["strat"] == "pos_bin(onset)") & (ts["col"] == "z_at_10")]
    if not sub.empty:
        x = np.arange(len(sub))
        ax.errorbar(x, sub["med"], yerr=[sub["med"] - sub["p10"], sub["p90"] - sub["med"]],
                    fmt="o-", color=CL["onset"], capsize=4)
        for xi, (_, r) in zip(x, sub.iterrows()):
            ax.text(xi, r["p90"] + 0.012, "n=%d" % int(r["n_events"]), ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([en(v) for v in sub["layer"]], fontsize=8)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("f(1.00 s)  [-]")
    ax.set_title("(a) onset shape vs within-recording epoch\n(early/middle/late third)", fontsize=9)
    ax.grid(alpha=0.25)
    # (b) 录制内 trend
    ax = axes[1]
    s = tro[tro["metric"] == "f(1.00s)"].copy()
    s["lbl"] = s["key"] + " " + s["kind"]
    y = np.arange(len(s))
    cols = [CL["restep"] if p < 0.05 else CL["grey"] for p in s["p"]]
    ax.barh(y, s["rho"], color=cols, alpha=0.8, height=0.5)
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.text(0.02 if r["rho"] >= 0 else -0.02, yi, "n=%d" % int(r["n"]),
                va="center", ha="left" if r["rho"] >= 0 else "right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(s["lbl"], fontsize=8)
    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("Spearman rho( f(1.00 s), event time within recording )")
    ax.set_title("(b) within-recording shape trend\n(red = p<0.05)", fontsize=9)
    ax.grid(alpha=0.25, axis="x")
    # (c) 载重分层轨迹（形状是否随幅度变）
    ax = axes[2]
    for kind, ls in [("onset", "-"), ("restep", "--")]:
        for lab, lo, hi, col in [("|J|/peak <15%", 0.0, 0.15, CL["ok"]),
                                 ("|J|/peak >=15%", 0.15, 9.9, CL["onset"])]:
            q = tra[(tra["kind"] == kind) & tra["usable"]
                    & (tra["J_over_peak"].abs() >= lo) & (tra["J_over_peak"].abs() < hi)]
            if q.empty:
                continue
            q = q[q["tau"] > 0]
            g = q.groupby("tau")["f"]
            m = g.median()
            ax.plot(m.index, m.to_numpy(), ls, marker="o", ms=3.5, color=col,
                    label="%s, %s (n=%d ev.)" % (kind, lab, q["uid"].nunique()))
    ax.set_xscale("log")
    ax.set_xlabel("tau  (s, log)")
    ax.set_ylabel("median f(tau)  [-]")
    ax.set_title("(c) shape vs relative load magnitude", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5)
    fig.suptitle("T3A_04  Time invariance and load-magnitude dependence of the fast-phase shape",
                 fontsize=11)
    C.savefig(fig, "T3A_04_time_stability.png")


if __name__ == "__main__":
    C.start_log("t3a_03_figures")
    fig1()
    fig2()
    fig3()
    fig4()
    print("四张图已写入 %s" % C.FIG)
