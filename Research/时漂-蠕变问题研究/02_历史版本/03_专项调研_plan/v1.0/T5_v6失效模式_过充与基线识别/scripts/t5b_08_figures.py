# -*- coding: utf-8 -*-
"""T5-B / 08：出图（figures/T5B_*.png）。

中文字体缺失时自动退化为英文标签（不输出方框）。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)


def setup_font():
    from matplotlib import font_manager
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name),
                                  fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


CJK = setup_font()
print("中文字体可用:", CJK)


def T(zh, en):
    return zh if CJK else en


# ── 图 1：状态机 / 机制链 ──
def fig1():
    p = os.path.join(RES, "t5b_internal_trace.csv")
    if not os.path.exists(p):
        print("缺 t5b_internal_trace.csv，跳过图 1")
        return
    df = pd.read_csv(p, encoding="utf-8-sig")
    cases = [("C2_tap_50pct_100ms", "C2 injected tap 50% level / 100 ms"),
             ("C4_white1000_reaIevent", "C4 white noise 1000 ADC RMS + real edge")]
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex="col")
    for j, (c, title) in enumerate(cases):
        d = df[df["case"] == c]
        if not len(d):
            continue
        t = d["ts"].to_numpy()
        d = d.reset_index(drop=True)
        ax = axes[0, j]
        ax.plot(t, d["total"], lw=0.8, color="0.55", label=T("输入总量", "input total"))
        ax.plot(t, d["out"], lw=1.0, color="C0", label=T("v6 显示总量", "v6 display"))
        ax.set_ylabel(T("总量 (ADC)", "total (ADC)"))
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax = axes[1, j]
        ax.plot(t, d["d"], lw=0.9, color="C3", label="d = level(T-0.21)-level(T-0.66)")
        ax.plot(t, d["gate"], lw=0.9, color="k", label=T("门限 gate", "gate"))
        ax.plot(t, -d["gate"], lw=0.9, color="k", ls=":")
        hit = d["raw_hit"].to_numpy(bool)
        ax.fill_between(t, 0, 1, where=hit, transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.15, label=T("raw_hit", "raw_hit"))
        ax.set_ylabel(T("电平差 (ADC)", "level diff (ADC)"))
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax = axes[2, j]
        ax.plot(t, d["sumA"], lw=1.0, color="C2", label=T("基线锚点 ΣA（扰动）", "ΣA perturbed"))
        ax.plot(t, d["sumA_pre"], lw=1.0, color="C1", ls="--",
                label=T("ΣA（扰动，帧前快照）", "ΣA (pre-frame)"))
        ax.plot(t, d["dA"], lw=1.0, color="C4", label=T("ΔΣA = 扰动 − 干净", "ΔΣA vs clean"))
        ax.set_ylabel(T("ΣA (ADC)", "ΣA (ADC)"))
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax = axes[3, j]
        ax.plot(t, d["state_code"], drawstyle="steps-post", lw=1.2, color="C5",
                label=T("状态 0=idle 1=event 2=slow", "state 0=idle 1=event 2=slow"))
        for i, r in d[d["n_epoch"] > d["n_epoch"].shift(1).fillna(0)].iterrows():
            ax.axvline(r["ts"], color="purple", lw=1.0)
            ax.annotate(T("建事件", "epoch"), (r["ts"], 1.5), fontsize=8, color="purple")
        if c.startswith("C4"):
            ax.axvline(28.63 - 26.0, color="green", lw=1.2, ls="--")
            ax.annotate(T("真实沿 28.63 s", "real edge 28.63 s"), (28.63 - 26.0, 1.0),
                        fontsize=8, color="green")
        ax.set_ylim(-0.3, 2.5)
        ax.set_ylabel(T("状态码", "state code"))
        ax.set_xlabel(T("时间 (s，窗内坐标)", "time (s, window coords)"))
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle(T("T5B-01 v6 检测器/基线状态机的逐帧内部量（拍击 vs 抖动+真实沿）",
                   "T5B-01 v6 detector/baseline state machine internals (tap vs noise+real edge)"))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIG, "T5B_01_state_machine.png"), dpi=130)
    plt.close(fig)
    print("图 1 完成")


# ── 图 2：失效边界 ──
def fig2():
    p = os.path.join(RES, "t5b_detector_roc.csv")
    if not os.path.exists(p):
        print("缺 t5b_detector_roc.csv，跳过图 2")
        return
    df = pd.read_csv(p, encoding="utf-8-sig")
    df = df[df.get("family", "boundary") == "boundary"] if "family" in df else df
    classes = [c for c in df["cls"].unique() if not c.startswith("tap")]
    taps = df[df["cls"] == "tap"]
    n_panel = len(classes) + (1 if len(taps) else 0)
    ncol = 2
    nrow = int(np.ceil(n_panel / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow), squeeze=False)
    k = 0
    for cls in classes:
        ax = axes[k // ncol][k % ncol]
        k += 1
        for w in sorted(df["win"].unique()):
            s = df[(df["cls"] == cls) & (df["win"] == w)].sort_values("amp")
            s = s[s["amp"].notna()]
            if not len(s):
                continue
            y = s["p_fail"].to_numpy(float)
            lo = np.clip(y - s["ci_lo"].to_numpy(float), 0, 1)
            hi = np.clip(s["ci_hi"].to_numpy(float) - y, 0, 1)
            ax.errorbar(s["amp"], y, yerr=[lo, hi], marker="o", ms=4, lw=1.0,
                        capsize=2, label=w)
            a50 = s["A50"].iloc[0]
            if a50 == a50:
                ax.axvline(a50, color="0.7", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel(T("扰动总量 RMS (ADC)", "perturbation total RMS (ADC)"))
        ax.set_ylabel(T("失效概率", "failure probability"))
        ax.set_title(f"{cls} / {T('失败率与 Wilson 95% CI', 'failure rate, Wilson 95% CI')}",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if len(taps):
        ax = axes[k // ncol][k % ncol]
        for dur in sorted(taps["dur_ms"].dropna().unique()):
            s = taps[taps["dur_ms"] == dur].sort_values("amp_pct")
            ax.errorbar(s["amp_pct"], s["p_fail"], marker="o", ms=4, lw=1.0, capsize=2,
                        label=f"dur={int(dur)} ms")
        ax.set_xlabel(T("拍击峰值（% 平台电平）", "tap peak (% of plateau level)"))
        ax.set_ylabel(T("失效概率", "failure probability"))
        ax.set_ylim(-0.05, 1.08)
        ax.set_title(T("拍击：幅值 × 时长", "tap: amplitude x duration"), fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for kk in range(k + 1, nrow * ncol):
        axes[kk // ncol][kk % ncol].axis("off")
    fig.suptitle(T("T5B-02 v6 基线识别失效边界（失败=任一 M1~M4；样本量见 results/t5b_baseline_failure.csv）",
                   "T5B-02 v6 baseline-identification failure boundary (fail = any of M1-M4)"))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "T5B_02_failure_boundary.png"), dpi=130)
    plt.close(fig)
    print("图 2 完成")


# ── 图 3：判据 ROC / PR ──
def fig3():
    p = os.path.join(RES, "t5b_criteria_roc.csv")
    if not os.path.exists(p):
        print("缺 t5b_criteria_roc.csv，跳过图 3")
        return
    df = pd.read_csv(p, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for name, g in df.groupby("criterion"):
        g = g.sort_values("fpr")
        lab = name + ("*" if str(g["causality"].iloc[0]).startswith("非因果") else "")
        axes[0].plot(g["fpr"], g["tpr"], marker=".", ms=4, lw=1.2,
                     label=f"{lab} (AUC={g['auc'].iloc[0]:.3f})")
        axes[1].plot(g["recall"], g["precision"].fillna(0), marker=".", ms=4, lw=1.2, label=lab)
    axes[0].plot([0, 1], [0, 1], "k:", lw=1)
    axes[0].set_xlabel(T("假正率 FPR（拍击/抖动被当加载）", "FPR (tap/noise accepted as load)"))
    axes[0].set_ylabel(T("真正率 TPR（真实沿被接受）", "TPR (real edge accepted)"))
    axes[0].set_title(T("ROC（* = 需未来样本，非因果）", "ROC (* = non-causal, needs lookahead)"))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(T("召回率", "recall"))
    axes[1].set_ylabel(T("精确率", "precision"))
    axes[1].set_title("PR")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(T("T5B-03 候选判据的 ROC/PR（正例=真实沿，负例=拍击/抖动触发）",
                   "T5B-03 candidate-criterion ROC/PR (pos=real edge, neg=tap/noise trigger)"))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "T5B_03_roc.png"), dpi=130)
    plt.close(fig)
    print("图 3 完成")


# ── 图 4：Pareto ──
def fig4():
    p = os.path.join(RES, "t5b_detector_pareto.csv")
    if not os.path.exists(p):
        print("缺 t5b_detector_pareto.csv，跳过图 4")
        return
    df = pd.read_csv(p, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    for capf, mk in ((None, "o"), (1.0, "s"), (0.5, "^")):
        s = df[(df["CAPF"].astype(str) == ("None" if capf is None else str(capf)))]
        ax.scatter(s["lat_t_det_med"], s["tap_surv_mean"], marker=mk, s=32,
                   label=f"CAPF={capf}")
    fr = df[df["is_pareto_lat_tap"]].sort_values("lat_t_det_med")
    ax.plot(fr["lat_t_det_med"], fr["tap_surv_mean"], "k--", lw=1.2, label=T("Pareto 前沿", "Pareto front"))
    for _, r in fr.iterrows():
        ax.annotate(f"{r['dwell_s']:.2f}s/K{r['DET_K']:g}", (r["lat_t_det_med"], r["tap_surv_mean"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(T("真实阶跃建立事件延迟 (s)", "epoch-build latency for real step (s)"))
    ax.set_ylabel(T("拍击后存活伪事件（均值/次）", "surviving false epochs per tap"))
    ax.set_title(T("代价：检测延迟", "cost: detection latency") + " vs " +
                 T("风险：拍击误捕获", "risk: tap false capture"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(df["lat_t_det_med"], df["noise_exc_p90"], s=32, color="C3")
    fr = df[df["is_pareto_lat_noise"]].sort_values("lat_t_det_med")
    ax.plot(fr["lat_t_det_med"], fr["noise_exc_p90"], "k--", lw=1.2, label=T("Pareto 前沿", "Pareto front"))
    ax.set_xlabel(T("真实阶跃建立事件延迟 (s)", "epoch-build latency for real step (s)"))
    ax.set_ylabel(T("抖动下显示超额偏差 p90 (×电平)", "noise excess display dev p90 (x level)"))
    ax.set_title(T("代价：检测延迟", "cost: detection latency") + " vs " +
                 T("风险：抖动下显示偏差", "risk: noise display deviation"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(T("T5B-04 误触发率与检测延迟的 Pareto（驻留 × 门限系数 × 封顶策略）",
                   "T5B-04 Pareto: false-trigger rate vs detection latency"))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "T5B_04_pareto.png"), dpi=130)
    plt.close(fig)
    print("图 4 完成")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
