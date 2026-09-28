# -*- coding: utf-8 -*-
"""T5-A 图件（`figures/T5A_*.png`，中文标签；缺字自动回退英文）。

  T5A_01_overcharge_trace.png  过充/下冲的实测波形（选出最典型的 4 个事件：raw / v6 / v6.1）
  T5A_02_filter_pareto.png     滤波 Pareto（过充   稳定时间   台阶保真 G），两种落点
  T5A_03_kappa_sweep.png       κ 扫描四联指标 + 留一泛化散布（C-4）
  T5A_04_attribution.png       归因四量随事件类别的分布（H1~H4 证据）
  T5A_05_filter_cost.png       滤波代价（G 台阶保真 / err_1s / MD 随滤波强度）
  T5A_06_v6_vs_v61.png         v6 vs v6.1 逐事件超调散点 + 恒载/实录分组

每图必须有标题 / 轴标签（含单位）/ 图例；底层 csv 见 results/。
用法：`python scripts/t5a_figs.py`
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402

FIG = os.path.join(C.TASK, "figures")
RES = os.path.join(C.TASK, "results")
os.makedirs(FIG, exist_ok=True)

# ── 中文字体自检：有就用中文，没有就整图英文（不出方框）──
CJK = None
for fam in ["Microsoft YaHei", "SimHei", "Noto Sans SC", "SimSun", "DejaVu Sans"]:
    try:
        if any(fam.lower() in f.name.lower()
               for f in font_manager.fontManager.ttflist):
            CJK = fam
            break
    except Exception:
        pass
ZH = CJK is not None and CJK != "DejaVu Sans"
plt.rcParams["font.family"] = CJK or "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def L(zh, en):
    return zh if ZH else en


def load_csv(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"  [skip] 缺 {name}")
        return None
    return pd.read_csv(p)


def fig01():
    ev = load_csv("t5a_overcharge_events.csv")
    if ev is None:
        return
    ev = ev[ev.J_frac >= 0.05]
    recs = C.recordings()
    picks = ev.reindex(ev.OS_pct.abs().sort_values(ascending=False).index).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (_, e) in zip(axes.ravel(), picks.iterrows()):
        k = e["key"]
        d = recs[k]
        t0 = float(e["t_on"])
        o6 = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12)
        o61 = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12, cls=C.KV61)
        tu = d["tu"]
        i0, i1 = C.clip_win(tu, t0 - 1.0, t0 + 8.0)
        tt = tu[i0:i1] - t0
        ax.plot(tt, d["Z"][i0:i1], "k-", lw=1.6, label=L("原始", "raw"))
        ax.plot(tt, o6["Y"].sum(axis=1)[i0:i1], "C3-", lw=1.3, label="v6")
        ax.plot(tt, o61["Y"].sum(axis=1)[i0:i1], "C0--", lw=1.3, label="v6.1")
        Zf = float(e["Z_final"])
        ax.axhline(Zf, color="gray", ls=":", lw=1,
                   label=L("Z_final (t+4.6~5.4 s)", "Z_final (t+4.6~5.4 s)"))
        ax.axvline(0, color="green", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(f"{k} @{t0:.2f} s  ({e['kind']})  OS%={e['OS_pct']:.1f}%",
                     fontsize=9)
        ax.set_xlabel(L("相对 t_on 的时间 (s)", "time since t_on (s)"))
        ax.set_ylabel(L("总量读数 (ADC/显示单位)", "total reading (ADC/display units)"))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(L("T5A-01 过充/下冲的实测波形：原始 vs v6 vs v6.1（OS% 最大的 4 个加载事件）",
                   "T5A-01 overcharge traces: raw vs v6 vs v6.1 (top-4 loading events by OS%)"),
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "T5A_01_overcharge_trace.png")
    fig.savefig(p)
    plt.close(fig)
    print("  ->", p)


def fig02():
    S = load_csv("t5a_filter_pareto.csv")
    if S is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for layer, mk, col in [("display", "o", "C0"), ("internal_input", "s", "C1")]:
        s = S[S.layer == layer]
        if not len(s):
            continue
        c = [ "C3" if not b else "C2" for b in s.causal]
        axes[0].scatter(s.T_med, s.OS_abs_med, s=34, marker=mk, c=c,
                        edgecolors="k", linewidths=0.4,
                        label=(L("显示层滤波", "display layer") if layer == "display"
                               else L("内部状态层（输入前滤波）", "internal (input filter)")))
        axes[1].scatter(s.T_med, s.G_med, s=34, marker=mk, c=c,
                        edgecolors="k", linewidths=0.4, label=layer)
        axes[2].scatter(s.OS_abs_med, s.G_med, s=34, marker=mk, c=c,
                        edgecolors="k", linewidths=0.4, label=layer)
        for _, r in s[s.pareto].iterrows():
            axes[0].annotate(r.fid, (r.T_med, r.OS_abs_med), fontsize=6,
                             xytext=(3, 3), textcoords="offset points")
            axes[1].annotate(r.fid, (r.T_med, r.G_med), fontsize=6,
                             xytext=(3, 3), textcoords="offset points")
            axes[2].annotate(r.fid, (r.OS_abs_med, r.G_med), fontsize=6,
                             xytext=(3, 3), textcoords="offset points")
    axes[0].axvspan(1.0, 2.0, color="green", alpha=0.10,
                    label=L("1~2 s 目标带", "1-2 s target band"))
    axes[0].set_xlabel(L("T_stable 中位 (s)", "median T_stable (s)"))
    axes[0].set_ylabel(L("|过充| 中位 (%of J)", "median |OS| (% of J)"))
    axes[0].set_title(L("过充   稳定时间", "overshoot vs settling time"))
    axes[0].set_xlim(left=0)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(L("T_stable 中位 (s)", "median T_stable (s)"))
    axes[1].set_ylabel(L("G 台阶捕获比中位", "median step gain G"))
    axes[1].set_title(L("台阶保真   稳定时间", "step fidelity vs settling time"))
    axes[1].set_xlim(left=0)
    axes[1].legend(fontsize=7)
    axes[2].set_xlabel(L("|过充| 中位 (%of J)", "median |OS| (% of J)"))
    axes[2].set_ylabel(L("G 台阶捕获比中位", "median step gain G"))
    axes[2].set_title(L("过充   台阶保真", "overshoot vs step fidelity"))
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(L("T5A-02 滤波 Pareto（红点 = 非因果/需延迟；绿点 = 因果可上线）",
                   "T5A-02 filter Pareto (red = non-causal/needs delay; green = causal)"),
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "T5A_02_filter_pareto.png")
    fig.savefig(p)
    plt.close(fig)
    print(" ->", p)


def fig03():
    S = load_csv("t5a_kappa_four_metrics.csv")
    LOO = load_csv("t5a_kappa_loo.csv")
    if S is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    order = {"uniform": ("C0", "o", L("统一 κ", "uniform κ")),
             "onset_only": ("C1", "^", L("仅 onset κ（restep 固定 1.12）",
                                         "onset κ only (restep=1.12)")),
             "restep_only": ("C2", "v", L("仅 restep κ（onset 固定 1.30）",
                                          "restep κ only (onset=1.30)"))}
    for fam, (col, mk, lab) in order.items():
        s = S[S.family == fam].sort_values("kappa_onset")
        axes[0][0].plot(s.kappa_onset, s.OS5_med, mk + "-", color=col, label=lab)
        axes[0][1].plot(s.kappa_onset, s.T10_med, mk + "-", color=col, label=lab)
        axes[1][0].plot(s.kappa_onset, s.err1_med_onset, mk + "-", color=col, label=lab)
        axes[1][1].plot(s.kappa_onset, s.MD_med, mk + "-", color=col, label=lab)
    axes[0][0].axhline(0, color="k", lw=0.8)
    axes[0][0].axvspan(1.05, 1.15, color="orange", alpha=0.15,
                       label=L("推荐区 κ=1.05~1.15", "recommended 1.05-1.15"))
    axes[0][0].set_ylabel(L("过充中位 OS5% (% of J)", "median OS5% (% of J)"))
    axes[0][0].set_title(L("① 阶跃瞬态过充（5 s 窗）", "1) transient overshoot (5 s window)"))
    axes[0][1].axhspan(1.0, 2.0, color="green", alpha=0.12,
                       label=L("1~2 s 目标带", "1-2 s target band"))
    axes[0][1].set_ylabel(L("T_stable(10 s 窗) 中位 (s)", "median T_stable (10 s win) (s)"))
    axes[0][1].set_title(L("② 稳定时间", "2) settling time"))
    axes[1][0].axhline(0, color="k", lw=0.8)
    axes[1][0].set_ylabel(L("1 s 时刻误差中位 (%)", "median error at 1 s (%)"))
    axes[1][0].set_title(L("③ 1 s 误差（onset 子样本）", "3) error at 1 s (onset subsample)"))
    axes[1][1].set_ylabel(L("最大偏差中位 (ADC)", "median max deviation (ADC)"))
    axes[1][1].set_title(L("④ 最大偏差", "4) max deviation"))
    for ax in axes.ravel():
        ax.set_xlabel("κ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(L("T5A-03 统一 κ 扫描（加载类 39 事件，同一事件集 / 同一口径）",
                   "T5A-03 unified κ sweep (39 loading events, same event set & gauge)"),
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "T5A_03_kappa_sweep.png")
    fig.savefig(p)
    plt.close(fig)
    print(" ->", p)

    # ── T5A_03c：κ 上限触发诊断（直接实测）↔ 四联指标 ──
    CAP = load_csv("t5a_kappa_cap_summary.csv")
    if CAP is None:
        return
    on = CAP[CAP.kind == "onset"]
    F = S[S.family == "onset_only"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(on.kappa, on.trigger_rate * 100, "C4o-",
                 label=L("触发率（出现过 r>κ 的事件占比）", "trigger rate (events with r>κ)"))
    axes[0].plot(on.kappa, on.trunc_frame_frac * 100, "C5s--",
                 label=L("截断时长占比（被 cap 截断的帧）", "truncated time fraction"))
    axes[0].axvline(1.10, color="orange", ls=":", lw=1.4,
                    label=L("κ=1.10（cap 开始大量咬人）", "κ=1.10 (cap starts biting)"))
    axes[0].axvline(1.05, color="green", ls=":", lw=1.4, label=L("κ=1.05（推荐）", "κ=1.05 (recommended)"))
    axes[0].set_xlabel("κ")
    axes[0].set_ylabel(L("百分比 (%)", "percent (%)"))
    axes[0].set_title(L("① κ 上限何时真正咬人（onset, n=22）",
                        "1) when the κ cap actually bites"), fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=6.5)
    axes[1].plot(F.kappa_onset, F.OS5_med, "C0o-", label=L("过充中位 OS5%", "median OS5%"))
    axes[1].plot(F.kappa_onset, F.OS5_max, "C3s-", label=L("过充最大 OS5%", "max OS5%"))
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].axvspan(1.05, 1.10, color="orange", alpha=0.15, label=L("推荐区", "recommended"))
    axes[1].set_xlabel("κ")
    axes[1].set_ylabel(L("过充 (% of J)", "overshoot (% of J)"))
    axes[1].set_title(L("② 过充 vs κ", "2) overshoot vs κ"), fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    axes[2].plot(F.kappa_onset, F.T5_med, "C0o-", label=L("T_stable 中位", "median T_stable"))
    axes[2].plot(F.kappa_onset, F.T5_p90, "C3s-", label=L("T_stable p90", "p90 T_stable"))
    axes[2].axhspan(1.0, 2.0, color="green", alpha=0.12, label=L("1~2 s 目标带", "1-2 s band"))
    axes[2].set_xlabel("κ")
    axes[2].set_ylabel(L("T_stable（5 s 窗, s）", "T_stable (5 s win, s)"))
    axes[2].set_title(L("③ 稳定时间 vs κ（κ=1.05 的 p90 才在带内）",
                        "3) settling time vs κ"), fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.suptitle(L("T5A-03c κ 是单侧上限（Â=min(A_raw, κ·inc)）：触发率解释了收益的来源",
                   "T5A-03c κ is a one-sided cap: the trigger rate explains where the gain comes from"),
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "T5A_03c_kappa_cap_trigger.png")
    fig.savefig(p)
    plt.close(fig)
    print(" ->", p)

    # ── Â/J−1 随 κ 的实测变化 ──
    B = load_csv("t5a_ahat_bias_by_kappa.csv")
    if B is not None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(B.kappa, B.bias_med, "C0o-", label=L("中位", "median"))
        ax.plot(B.kappa, B.bias_p90, "C3s-", label=L("p90", "p90"))
        ax.plot(B.kappa, B.bias_max, "C4^--", label=L("max", "max"))
        ax.axhline(0, color="k", lw=0.8)
        ax.axvspan(1.05, 1.10, color="orange", alpha=0.15, label=L("推荐区", "recommended"))
        ax.set_xlabel("κ")
        ax.set_ylabel(L("Â/|J| − 1 (%)", "Â/|J| − 1 (%)"))
        ax.set_title(L("T5A-03d κ 对反演高估的压制（onset, n=22, τ=1 s）",
                       "T5A-03d how κ suppresses the inversion bias"), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(FIG, "T5A_03d_ahat_bias.png")
        fig.savefig(p)
        plt.close(fig)
        print(" ->", p)

    if LOO is not None and len(LOO):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for fam, (col, mk, lab) in order.items():
            s = LOO[LOO.family == fam].sort_values("kappa_onset")
            if not len(s):
                continue
            axes[0].plot(s.kappa_onset, s.OS5_med_fold_max, mk + "-", color=col,
                         label=lab + L(" 最坏折", " worst fold"))
            axes[0].plot(s.kappa_onset, s.OS5_med_fold_min, mk + ":", color=col,
                         alpha=0.6, label=lab + L(" 最好折", " best fold"))
            axes[1].plot(s.kappa_onset, s.T10_med_fold_max, mk + "-", color=col, label=lab)
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_ylabel(L("留出录制上的过充中位 OS5% (%)", "held-out median OS5% (%)"))
        axes[0].set_ylim(-350, 60)
        axes[1].set_ylabel(L("留出录制上的 T_stable 最坏折 (s)",
                             "held-out worst-fold T_stable (s)"))
        for ax in axes:
            ax.set_xlabel("κ")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(L("T5A-03b κ 的留一泛化（13 折，留出一份录制；折间散布 >> κ 效应）",
                       "T5A-03b leave-one-recording-out generalisation (13 folds)"),
                     fontsize=11)
        fig.tight_layout()
        p = os.path.join(FIG, "T5A_03b_kappa_loo.png")
        fig.savefig(p)
        plt.close(fig)
        print(" ->", p)


def fig04():
    A = load_csv("t5a_attribution.csv")
    if A is None:
        return
    A = A[A.kind.isin(["onset", "restep"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, col, ttl, unit in [
            (axes[0], "H1_ahat_over_J_pct", L("H1 形状反演偏差 Â/|J|−1", "H1 shape-inversion bias"),
             "%"),
            (axes[1], "H2_rate_sat_frames", L("H2 滑行器速率饱和帧数", "H2 glider rate-saturated frames"),
             L("帧(10 ms)", "frames (10 ms)")),
            (axes[2], "H3_handoff_jump_ratio", L("H3 交接帧跳变比", "H3 handoff jump ratio"),
             L("倍数(相对邻域中位)", "x vs local median")),
            (axes[3], "H4_g_slope_per_s", L("H4 交接后 g 斜率", "H4 post-handoff g slope"),
             "1/s")]:
        for i, kind in enumerate(["onset", "restep"]):
            v = A[A.kind == kind][col].dropna().to_numpy()
            if len(v):
                ax.boxplot([v], positions=[i], widths=0.5, showfliers=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["onset", "restep"])
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel(unit)
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(L("T5A-04 过充归因四量（T4-A 事件集，v6 现状 κ=1.30/1.12）",
                   "T5A-04 attribution metrics (T4-A event set, v6 κ=1.30/1.12)"),
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "T5A_04_attribution.png")
    fig.savefig(p)
    plt.close(fig)
    print(" ->", p)


def fig05():
    S = load_csv("t5a_filter_cost.csv")
    if S is None:
        return
    S = S[S.layer == "display"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fams = {"iir_ema": ("C0", "o", "IIR-EMA"), "causal_median": ("C1", "s", "causal median"),
            "causal_mean": ("C2", "^", "causal mean"), "rate_limit": ("C3", "v", "rate limit"),
            "clip_raw": ("C4", "D", "hard clip"), "centered_median": ("C5", "P", "centered median*"),
            "savgol": ("C6", "X", "Savitzky-Golay*")}
    for fam, (col, mk, lab) in fams.items():
        s = S[S.family == fam]
        if not len(s):
            continue
        axes[0].scatter(s.OS_med, s.G_med, s=28, marker=mk, color=col,
                        edgecolors="k", linewidths=0.3, label=lab)
        axes[1].scatter(s.OS_med, s.err1_abs_med, s=28, marker=mk, color=col,
                        edgecolors="k", linewidths=0.3, label=lab)
        axes[2].scatter(s.OS_med, s.MD_med, s=28, marker=mk, color=col,
                        edgecolors="k", linewidths=0.3, label=lab)
    axes[0].set_ylabel(L("G 台阶捕获比（中位）", "step gain G (median)"))
    axes[0].set_title(L("台阶保真代价", "step-fidelity cost"))
    axes[1].set_ylabel(L("|1 s 误差| 中位 (%)", "median |error at 1 s| (%)"))
    axes[1].set_title(L("1 s 精度代价", "1-s accuracy cost"))
    axes[2].set_ylabel(L("最大偏差中位 (ADC)", "median max deviation (ADC)"))
    axes[2].set_title(L("最大偏差代价", "max-deviation cost"))
    for ax in axes:
        ax.set_xlabel(L("过充中位 (% of J)", "median OS (% of J)"))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(L("T5A-05 滤波代价（显示层；* = 非因果，上线需 r 帧延迟）",
                   "T5A-05 filter cost (display layer; * = non-causal, needs r-frame delay)"),
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "T5A_05_filter_cost.png")
    fig.savefig(p)
    plt.close(fig)
    print(" ->", p)


def fig06():
    A = load_csv("t5a_v6_vs_v61.csv")
    if A is None:
        return
    A = A[A.J_frac >= 0.05]
    piv = A.pivot_table(index=["key", "t_on", "kind", "dom"], columns="arm",
                        values="OS_pct", aggfunc="first").reset_index()
    if "v6" not in piv or "v61_s106" not in piv:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for dom, col, mk in [("显示域", "C0", "o"), ("ADC域", "C3", "s")]:
        s = piv[piv.dom == dom]
        axes[0].scatter(s["v6"], s["v61_s106"], s=40, marker=mk, color=col,
                        edgecolors="k", linewidths=0.4, label=dom)
    lim = [min(-20, piv["v6"].min()), max(25, piv["v61_s106"].max())]
    axes[0].plot(lim, lim, "k--", lw=0.9, label=L("y = x", "y = x"))
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[0].axvline(0, color="gray", lw=0.7)
    axes[0].set_xlabel(L("v6 过充 OS% (% of J)", "v6 OS% (% of J)"))
    axes[0].set_ylabel(L("v6.1 过充 OS% (% of J)", "v6.1 OS% (% of J)"))
    axes[0].set_title(L("逐事件：v6.1 把正过充全部压到 0 侧", "per-event: v6.1 pushes overshoot to <=0"))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    sc = axes[1].scatter(np.arange(len(piv)), piv["v6"], s=26, color="C3",
                         label="v6")
    axes[1].scatter(np.arange(len(piv)), piv["v61_s106"], s=26, color="C0",
                    label="v6.1", marker="x")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel(L("加载事件序号（按 v6 的 OS% 排序）", "loading-event index"))
    axes[1].set_ylabel(L("过充 OS% (% of J)", "OS% (% of J)"))
    axes[1].set_title(L("同一批事件的配对对比", "paired comparison, same events"))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(L("T5A-06 v6 vs v6.1（ROM_SCALE=1.06）的过充对照",
                   "T5A-06 v6 vs v6.1 (ROM_SCALE=1.06) overshoot comparison"), fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG, "T5A_06_v6_vs_v61.png")
    fig.savefig(p)
    plt.close(fig)
    print(" ->", p)


def main():
    print(f"中文字体：{CJK}（ZH={ZH}）")
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06]:
        try:
            fn()
        except Exception as e:
            print(f"  [warn] {fn.__name__} 失败：{type(e).__name__}: {e}")
    print("done")


if __name__ == "__main__":
    main()
