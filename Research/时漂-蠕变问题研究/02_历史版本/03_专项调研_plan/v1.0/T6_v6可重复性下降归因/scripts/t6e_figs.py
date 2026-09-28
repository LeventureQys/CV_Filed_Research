# -*- coding: utf-8 -*-
"""T6-E：出图（只读 results/*.csv，不重跑实验）。

图：
  T6_01_three_layers.png   L1 输入层 / L2 输出层 / L3 路径层 三层分解
  T6_02_jitter_path.png    微扰下的输出轨线族 + 抖动-输出差异曲线
  T6_03_attribution.png    消融瀑布图（Δ 相对 v6 基线）
  T6_04_improve_pareto.png 改进方向的可重复性 vs 稳定时间 Pareto
标签一律用英文（避免中文字体缺失出方框）。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
COL = {"raw": "0.45", "v5.1": "#ff7f0e", "v6": "#2ca02c", "v6.1": "#1f77b4",
       "v6trim": "#d62728"}
# 图中一律用 ASCII 标签（DejaVu Sans 无 CJK 字形，中文会出方框）
GEN = {"右拇指": "right thumb", "左拇指": "left thumb", "四指": "four fingers"}


def rd(name):
    f = os.path.join(RES, name)
    return pd.read_csv(f) if os.path.exists(f) else None


def fig01():
    disp = rd("t6_repeat_dispersion.csv")
    sa, sb = rd("t6_jitter_summary_a.csv"), rd("t6_jitter_summary_b.csv")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    if disp is not None:
        l1 = disp[(disp.layer == "L1_input") & disp.metric.str.startswith("CV_shape@")]
        for g, gg in l1.groupby("group"):
            tau = [float(m.split("@")[1]) for m in gg.metric]
            ax[0].plot(tau, gg.value, "o-", label=f"input (raw), {GEN.get(g, g)}")
        ax[0].set_xscale("log")
        ax[0].set_xlabel(r"$\tau$ after load edge  (s)")
        ax[0].set_ylabel(r"$CV_{shape}(\tau)$ of raw  $f(\tau)$   (%)")
        ax[0].set_title("L1 input layer: reproducibility of the RAW signal\n"
                        "(3 repeated records per sensor, within-group CV)")
        ax[0].legend(fontsize=7)
        # L2 bars
        l2 = disp[disp.layer == "L2_output"]
        arms = ["raw", "v5.1", "v6", "v6.1"]
        x = np.arange(len(arms))
        for i, (key, lab) in enumerate((("plat_err_std_R5_pp", "std vs 5 s raw level (R5)"),
                                        ("plat_err_std_Rstep_pp", "std vs mechanical step"))):
            v = [float(l2[(l2.impl == a) & (l2.metric == key)].value.median()) for a in arms]
            ax[1].bar(x + (i - 0.5) * 0.38, v, 0.38, label=lab)
            for xi, vi in zip(x + (i - 0.5) * 0.38, v):
                ax[1].text(xi, vi + 0.05, f"{vi:.2f}", ha="center", fontsize=7)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(arms)
        ax[1].set_ylabel("within-group std of plateau error (pp)")
        ax[1].set_title("L2 output layer: dispersion of the DISPLAYED plateau\n"
                        "(3 repeats, median over 3 sensor groups)")
        ax[1].legend(fontsize=7)
        # L3 bars
        x2 = np.arange(3)
        w = 0.38
        for i, (s, nm) in enumerate(((sa, "scene a: variable-load (ADC)"),
                                     (sb, "scene b: constant-load (force domain)"))):
            if s is None:
                continue
            ss = s[(s.cond == "timing_J100P")]
            v = [float(ss[ss.impl == a].rms_pct_lvl_med.iloc[0]) for a in ("v5.1", "v6", "v6.1")]
            ax[2].bar(x2 + (i - 0.5) * w, v, w, label=nm)
            for xi, vi in zip(x2 + (i - 0.5) * w, v):
                ax[2].text(xi, max(vi, 0.004) * 1.15, f"{vi:.3f}" if vi > 0 else "0.00",
                           ha="center", fontsize=7)
        ax[2].set_yscale("log")
        ax[2].set_xticks(x2)
        ax[2].set_xticklabels(["v5.1", "v6", "v6.1"])
        ax[2].set_ylabel("output-trajectory RMS difference\n(% of record median level, log)")
        ax[2].set_title("L3 path layer: same data + packet timing jitter\n"
                        r"($\pm$1 packet, 30 seeds, median)")
        ax[2].legend(fontsize=7)
    fig.suptitle("T6-Q1  Three-layer decomposition of \"reproducibility\" "
                 "(n=9 constant-load repeats / n=30 jitter seeds)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "T6_01_three_layers.png"), dpi=150)
    plt.close(fig)


def fig02():
    tr = rd("t6_jitter_traj_a.csv")
    sa = rd("t6_jitter_summary_a.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
    if tr is not None:
        t = tr["t"].to_numpy()
        for k, a in enumerate(("v5.1", "v6")):
            ax[k].plot(t, tr[f"base_{a}"], "k", lw=1.6, label="baseline (uniform 100 Hz)")
            for s in range(8):
                c = f"jit_{a}_s{s}"
                if c in tr.columns:
                    ax[k].plot(t, tr[c], lw=0.7, alpha=0.75,
                               label=("8 jittered runs (±1 packet)" if s == 0 else None))
            ax[k].set_xlabel("time (s)")
            ax[k].set_ylabel("displayed total (ADC)")
            ax[k].set_title(f"scene a: {a} — output trajectory family")
            ax[k].legend(fontsize=7)
            ax[k].set_xlim(0, float(t[-1]))
    if sa is not None:
        for a in ("v5.1", "v6", "v6.1"):
            ss = sa[sa.impl == a].copy()
            tt = ss[ss.cond.str.startswith("timing_")].sort_values("J_rel")
            tt = tt[tt.J_rel > 0]
            if len(tt):
                ax[2].plot(100 * tt.J_rel, tt.rms_pct_lvl_med, "o-", color=COL[a],
                           label=f"{a} (timing jitter)")
            oth = ss[~ss.cond.str.startswith("timing_")]
            for _, row in oth.iterrows():
                mk = {"noise_1ADC": "s", "noise_5ADC": "^", "dropout_1frame": "v"}[row.cond]
                ax[2].scatter([np.nan], [np.nan], marker=mk, color=COL[a], alpha=0.6)
        for cd, mk, lab in (("noise_1ADC", "s", "±1 ADC RMS noise"),
                            ("noise_5ADC", "^", "±5 ADC RMS noise"),
                            ("dropout_1frame", "v", "1-frame dropout")):
            v6 = sa[(sa.impl == "v6") & (sa.cond == cd)]
            v51 = sa[(sa.impl == "v5.1") & (sa.cond == cd)]
            if len(v6) and len(v51):
                ax[2].scatter([30], [float(v6.rms_pct_lvl_med.iloc[0])], marker=mk,
                              color=COL["v6"], zorder=5)
                ax[2].scatter([35], [float(v51.rms_pct_lvl_med.iloc[0])], marker=mk,
                              color=COL["v5.1"], zorder=5)
                ax[2].annotate(lab, (30, float(v6.rms_pct_lvl_med.iloc[0])),
                               textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax[2].set_yscale("log")
        ax[2].set_xlabel("timing jitter amplitude  (% of one packet period)")
        ax[2].set_ylabel("output-trajectory RMS difference\n(% of record median level, log)")
        ax[2].set_title("T6-Q2  jitter → output difference curve\n(scene a, 30 seeds/point; "
                        "right-side markers = noise / dropout)")
        ax[2].legend(fontsize=7)
    fig.suptitle("T6-Q2  Path jitter: identical input, only packet arrival times perturbed",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "T6_02_jitter_path.png"), dpi=150)
    plt.close(fig)


def fig03():
    at = rd("t6_attribution.csv")
    if at is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8))
    w = at[at.arm != "v6_baseline"].copy()
    base_r5 = float(at[at.arm == "v6_baseline"].std_R5_pp.iloc[0])
    base_rs = float(at[at.arm == "v6_baseline"].std_Rstep_pp.iloc[0])
    x = np.arange(len(w))
    for k, (col, base, lab) in enumerate((("d_std_R5_pp", base_r5, "vs 5 s raw level (R5)"),
                                          ("d_std_Rstep_pp", base_rs, "vs mechanical step"))):
        ax[k].bar(x, w[col], 0.6, color=["#2ca02c" if v < 0 else "#d62728" for v in w[col]])
        ax[k].axhline(0, color="k", lw=1)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(w.arm, rotation=35, ha="right", fontsize=8)
        ax[k].set_ylabel(f"Δ within-group std of plateau error (pp)\n"
                         f"(positive = reverting this change makes it WORSE)")
        ax[k].set_title(f"Δ {lab}\nv6 baseline std = {base:.2f} pp")
    fig.suptitle("T6-Q4  Attribution: revert ONE v6 change at a time "
                 "(9 constant-load records, 80 s, x3 implementations in report)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "T6_03_attribution.png"), dpi=150)
    plt.close(fig)


def fig04():
    tr = rd("t6_tradeoff.csv")
    if tr is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.0))
    ax[0].scatter(tr.std_R5_pp, tr.T_stable_med_s, s=55, c="#2ca02c")
    for _, r in tr.iterrows():
        ax[0].annotate(r.arm, (r.std_R5_pp, r.T_stable_med_s), textcoords="offset points",
                       xytext=(5, 4), fontsize=7)
    ax[0].set_xlabel("reproducibility: within-group std of plateau error, R5 (pp, lower better)")
    ax[0].set_ylabel("T_stable median (s, lower better)")
    ax[0].set_title("T6-Q6  Pareto: reproducibility vs settling time\n"
                    "(constant-load repeats, legacy T_stable caliber)")
    ax[1].scatter(tr.std_R5_pp, tr.L3_rms_pct_lvl_med, s=55, c="#d62728")
    for _, r in tr.iterrows():
        ax[1].annotate(r.arm, (r.std_R5_pp, r.L3_rms_pct_lvl_med), textcoords="offset points",
                       xytext=(5, 4), fontsize=7)
    ax[1].set_xlabel("within-group std of plateau error, R5 (pp, lower better)")
    ax[1].set_ylabel("L3 path jitter RMS (% of level, ±1 packet, lower better)")
    ax[1].set_title("T6-Q5  does the improvement also fix PATH jitter?\n"
                    "(2 scenes × 5 seeds)")
    fig.suptitle("T6-Q5/Q6  Improvement directions: numerical feasibility and cost", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG, "T6_04_improve_pareto.png"), dpi=150)
    plt.close(fig)


def main():
    fig01()
    fig02()
    fig03()
    fig04()
    for f in ("T6_01_three_layers.png", "T6_02_jitter_path.png", "T6_03_attribution.png",
              "T6_04_improve_pareto.png"):
        fp = os.path.join(FIG, f)
        print(f"{'OK ' if os.path.exists(fp) else 'MISSING'} {fp} "
              f"({os.path.getsize(fp) if os.path.exists(fp) else 0} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
