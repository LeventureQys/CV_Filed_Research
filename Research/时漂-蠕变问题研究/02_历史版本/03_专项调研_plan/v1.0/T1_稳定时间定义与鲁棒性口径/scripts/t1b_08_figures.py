# -*- coding: utf-8 -*-
"""T1-B / 08：图件（figures/T1B_01~04）。全部英文标签（避免中文字体缺字变方框）。

T1B_01_noise_pareto.png     噪声-指标 Pareto：显示域（相对噪声）与 ADC 域（绝对噪声）+ 拐点
T1B_02_jitter_sensitivity.png 时序抖动 / 丢包 / 事件起点不确定性的敏感性
T1B_03_tap_recovery.png      拍击-恢复：真实扰动 vs 合成拍击轨迹 + 触发率/峰值/恢复时间
T1B_04_cost_benefit.png      代价-收益矩阵（ΔT_stable 代价 vs 误触发/偏差收益）
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
sys.path.insert(0, HERE)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
C = {"white": "#1f77b4", "band": "#d62728", "common": "#2ca02c",
     "raw": "#7f7f7f", "v6": "#1f77b4", "v51": "#ff7f0e", "v61": "#9467bd"}
LBL = {"white": "white noise", "band": "band-limited 0.3-40 Hz", "common": "common-mode (in-phase)"}


def band(ax, x, y, ylo, yhi=None, color="C0", label=None, marker="o"):
    ax.plot(x, y, marker=marker, color=color, label=label, lw=1.4, ms=4)
    if yhi is None:
        ax.fill_between(x, ylo, y, color=color, alpha=0.15)
    else:
        ax.fill_between(x, ylo, yhi, color=color, alpha=0.15)


def fig01():
    h = pd.read_csv(os.path.join(RES, "t1b_summary_noise_hold.csv"))
    a = pd.read_csv(os.path.join(RES, "t1b_summary_noise_adc.csv"))
    knee = pd.read_csv(os.path.join(RES, "t1b_knee.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))

    ax = axes[0, 0]
    ax.axhspan(1.0, 2.0, color="green", alpha=0.12)
    ax.text(0.0007, 1.15, "1-2 s target", color="green", fontsize=8)
    for kind in ("white", "band", "common"):
        g = h[(h.kind == kind) & (h.arm == "v6")].sort_values("amp_rel")
        if not len(g):
            continue
        band(ax, g.amp_rel, g.t_stable_med, g.t_stable_p10, g.t_stable_p90,
             C[kind], LBL[kind])
        ax.errorbar(g.amp_rel, g.t_stable_p90, yerr=None, fmt="none")
    ax.set_xscale("log")
    ax.set_xlabel("total noise RMS / hold level  r  (display domain)")
    ax.set_ylabel("T_stable (s), median [p10-p90]")
    ax.set_title("(a) Display domain (9 hold recs, 7 clean onsets): T_stable vs noise")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for kind in ("white", "band", "common"):
        g = h[(h.kind == kind) & (h.arm == "v6")].sort_values("amp_rel")
        if not len(g):
            continue
        ax.plot(g.amp_rel, 100 * g.frac_meet_2s, marker="o", color=C[kind],
                label=LBL[kind], lw=1.4, ms=4)
    for _, r in knee.iterrows():
        if r.dom == "显示域" and r.criterion == "T_stable_p90>2s" and np.isfinite(r.r_first_over):
            ax.axvline(r.r_first_over, color=C[r.kind], ls="--", lw=1.2, alpha=0.8)
            ax.annotate(f"knee {r.kind}\nr={r.r_first_over:.3f}", (r.r_first_over, 55),
                        fontsize=7, color=C[r.kind], rotation=0)
    ax.set_xscale("log")
    ax.set_ylim(-5, 105)
    ax.set_xlabel("r = total noise RMS / hold level")
    ax.set_ylabel("pass rate  T_stable <= 2 s  (%)")
    ax.set_title("(b) Acceptance-rate drop and knee (dashed)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for kind in ("white", "band", "common"):
        g = a[(a.kind == kind) & (a.arm == "v6")].sort_values("amp")
        if not len(g):
            continue
        band(ax, g.amp, g.maxdev_med, (g.maxdev_med * 0.5), None, C[kind], LBL[kind])
    ax.set_xscale("log")
    ax.set_xlabel("total noise RMS  A  (ADC)")
    ax.set_ylabel("max |display deviation| (ADC), median")
    ax.set_title("(c) ADC domain (4 change-load recs): deviation vs noise")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for kind in ("white", "band", "common"):
        g = a[(a.kind == kind) & (a.arm == "v6")].sort_values("amp")
        if not len(g):
            continue
        ax.plot(g.amp, g.n_miss_mean, marker="o", color=C[kind], label="miss (band/white/common)",
                lw=1.4, ms=4)
    g = a[(a.kind == "white") & (a.arm == "v6")].sort_values("amp")
    ax.plot(g.amp, g.n_extra_mean, marker="s", ls=":", color="#8c564b", label="false trip (white)")
    ax.set_xscale("log")
    ax.set_xlabel("total noise RMS  A  (ADC)")
    ax.set_ylabel("events per recording")
    ax.set_title("(d) Missed vs false events (truth = T4-A event table)")
    ax.legend(fontsize=7)
    fig.suptitle("T1B_01  Noise vs stability metrics (v6, injected on real trajectories)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "T1B_01_noise_pareto.png"))
    plt.close(fig)


def fig02():
    j = pd.read_csv(os.path.join(RES, "t1b_summary_jitter.csv"))
    bh = pd.read_csv(os.path.join(RES, "t1b_baseline_arms.csv"))
    base = bh[(bh.arm == "v6") & (bh.t_stable_ok == 1)].t_stable_v1.median()
    jv = j[(j.arm == "v6") & (j.scope == "jitter_hold")].copy()
    order = ["origin_pm2", "origin_pm1", "jitter1", "jitter2", "delay1",
             "drop_f0.005", "drop_f0.01", "drop_f0.02", "drop_f0.05",
             "droppkt0.01", "droppkt0.05", "agg3", "agg6"]
    jv = jv.set_index("pert").reindex([p for p in order if p in set(jv.pert)]).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3))
    x = np.arange(len(jv))
    ax = axes[0]
    ax.bar(x, jv.t_stable_med - base, color="#1f77b4")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.04, color="r", ls=":", lw=1)
    ax.text(0.1, 0.05, "+40 ms", color="r", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(jv.pert, rotation=70, fontsize=7)
    ax.set_ylabel("dT_stable vs clean (s), median")
    ax.set_title(f"(a) Timing perturbation cost (clean median {base:.2f} s)")

    ax = axes[1]
    ax.bar(x, 100 * jv.frac_meet_2s, color="#2ca02c")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(jv.pert, rotation=70, fontsize=7)
    ax.set_ylabel("pass rate  T_stable <= 2 s (%)")
    ax.set_title("(b) Acceptance rate under timing perturbations")

    ax = axes[2]
    ja = j[(j.arm == "v6") & (j.scope == "jitter_adc")].set_index("pert") \
        .reindex([p for p in order if p in set(j[j.scope == "jitter_adc"].pert)]).reset_index()
    xx = np.arange(len(ja))
    ax.bar(xx - 0.2, ja.n_miss_mean, width=0.4, color="#d62728", label="missed events")
    ax.bar(xx + 0.2, ja.n_extra_mean, width=0.4, color="#8c564b", label="false events")
    ax.set_xticks(xx)
    ax.set_xticklabels(ja.pert, rotation=70, fontsize=7)
    ax.set_ylabel("events per recording")
    ax.set_title("(c) ADC domain: event accounting")
    ax.legend(fontsize=7)
    fig.suptitle("T1B_02  Timing jitter / dropout / event-origin sensitivity "
                 "(+/-1 packet = 15.4 ms fingertip, 39.9 ms change-load)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "T1B_02_jitter_sensitivity.png"))
    plt.close(fig)


def fig03():
    t = pd.read_csv(os.path.join(RES, "t1b_summary_tap.csv"))
    fig = plt.figure(figsize=(12.2, 7.4))
    gs = fig.add_gridspec(2, 3)

    # (a)(b) 轨迹：真实扰动 vs 同幅度合成拍击
    z = np.load(os.path.join(RES, "t1b_real_tap_traj.npz"))
    names = sorted({k.split("|")[0] for k in z.files})
    for i, nm in enumerate(names[:2]):
        ax = fig.add_subplot(gs[0, i])
        tu = z[f"{nm}|tu"]
        ax.plot(tu, z[f"{nm}|real"], color="#1f77b4", lw=1.4, label="real trajectory (v6)")
        ax.plot(tu, z[f"{nm}|syn"], color="#d62728", lw=1.2, ls="--",
                label="same segment + synthetic tap")
        tp = float(z[f"{nm}|t_peak"][0])
        ax.axvline(tp, color="k", ls=":", lw=1)
        ax.annotate(f"real disturbance\n{float(z[f'{nm}|pre'][0]):.1f} level", (tp, ax.get_ylim()[0]),
                    fontsize=7)
        ax.set_xlabel("time in segment (s)")
        ax.set_ylabel("display total")
        ax.set_title(f"({'ab'[i]}) {nm}")
        ax.legend(fontsize=7)

    tv = t[(t.arm == "v6") & (t.noise_r == 0.0)]
    ax = fig.add_subplot(gs[0, 2])
    w = 0.35
    xs = np.arange(len(tv))
    ax.bar(xs - w / 2, tv.n_ep_tap_med, width=w, color="#1f77b4", label="new epochs in tap window")
    ax.bar(xs + w / 2, tv.n_revoke_mean, width=w, color="#ff7f0e", label="revokes")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r.scope.replace('tap_','')}\n{100*r.frac:.0f}%L" for _, r in tv.iterrows()],
                       fontsize=7)
    ax.set_ylabel("count per tap")
    ax.set_title("(c) Does a tap create an epoch? (and is it revoked?)")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    for arm, g0 in t[t.noise_r == 0.0].groupby("arm"):
        g = g0.groupby("frac").peak_dev_pct_med.median()
        ax.plot(100 * g.index, g.values, marker="o", color=C.get(arm, "k"), label=arm, lw=1.4)
    ax.plot([10, 50], [10, 50], "k:", lw=1, label="1:1 transmission")
    ax.set_xlabel("tap peak (% of hold level)")
    ax.set_ylabel("peak display deviation (% of hold level)")
    ax.set_title("(d) Display excursion vs tap size")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    for arm, g0 in t[t.noise_r == 0.0].groupby("arm"):
        g = g0.groupby("frac").level_shift_med.median()
        ax.plot(100 * g.index, g.values, marker="s", color=C.get(arm, "k"), label=arm, lw=1.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("tap peak (% of hold level)")
    ax.set_ylabel("permanent level shift after tap")
    ax.set_title("(e) Residual baseline shift (0 = baseline intact)")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    for arm, g0 in t[t.noise_r == 0.0].groupby("arm"):
        g = g0.groupby("frac").t_recover_med.median()
        ax.plot(100 * g.index, g.values, marker="^", color=C.get(arm, "k"), label=arm, lw=1.4)
    ax.set_xlabel("tap peak (% of hold level)")
    ax.set_ylabel("recovery time (s), median")
    ax.set_title("(f) Time back to +/-2% of pre-tap level")
    ax.legend(fontsize=7)
    fig.suptitle("T1B_03  Tap (50/100/50 ms) on an established hold: real vs injected", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "T1B_03_tap_recovery.png"))
    plt.close(fig)


def fig04():
    c = pd.read_csv(os.path.join(RES, "t1b_summary_costbenefit.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ax = axes[0]
    for _, r in c.iterrows():
        col = "#d62728" if r.knob.startswith("capf") else (
            "#2ca02c" if r.knob.startswith("dwell") else (
                "#9467bd" if r.knob.startswith("rescue") else "#7f7f7f"))
        ax.scatter(r.dT_stable, r.tap30_ep_mean, s=45, color=col)
        ax.annotate(r.knob, (r.dT_stable, r.tap30_ep_mean), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("cost: dT_stable on clean onset (s), vs prototype")
    ax.set_ylabel("benefit: epochs created by a 30%-level tap")
    ax.set_title("(a) Cost (onset slowdown) vs benefit (tap rejection)")
    ax.axvline(0, color="k", lw=0.8)

    ax = axes[1]
    ax.scatter(c.dT_stable, c.noise_maxdev_med, s=45, color="#1f77b4")
    for _, r in c.iterrows():
        ax.annotate(r.knob, (r.dT_stable, r.noise_maxdev_med), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("cost: dT_stable on clean onset (s), vs prototype")
    ax.set_ylabel("display max-deviation under white noise r=2%")
    ax.set_title("(b) Cost vs noise robustness")
    ax.axvline(0, color="k", lw=0.8)
    fig.suptitle("T1B_04  Cost-benefit of robustness knobs (v6 prototype variants)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "T1B_04_cost_benefit.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig01()
    fig02()
    fig03()
    fig04()
    print("figures written to", FIG)
