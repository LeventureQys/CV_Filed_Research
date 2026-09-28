# -*- coding: utf-8 -*-
"""a5 出图：抖动下的检测器失守（σ_d 膨胀 / 门限 / 漏事件 / 显示偏差）与拍击 vs 真实阶跃。"""
import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import REC, load_uniform, make_perturb, add_tap, add_step, run_traced, ROOT  # noqa: E402
from glm53_v6 import GLM53v6                                                              # noqa: E402
from a1_detector import detector_trace                                                    # noqa: E402
from a_common import TRACED_V6                                                            # noqa: E402

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
FIG = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "figures")
os.makedirs(FIG, exist_ok=True)

REC1 = "零负载-切换负载-零负载-再切换负载"
REC2 = "中途切换-最终测试目标"


def perturb_on_mask(X, A, kind, rng, kw):
    E = make_perturb(X, A, kind, rng, **kw)
    t = E.sum(axis=1)
    s = t.std()
    return E * (A / s) if s > 1e-12 else E


def fig_detector():
    """图1：σ_d 膨胀 → 门限抬升 → 漏事件 → 显示偏差（抖动幅度扫描）。"""
    df = pd.read_csv(os.path.join(RES, "a2b_jitter_ext.csv"))
    d = df[(df.rec == REC2) & (df.kind.isin(["独立白噪", "带限0.3-5Hz", "同相白噪"]))]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    colors = {"独立白噪": "tab:blue", "带限0.3-5Hz": "tab:red", "同相白噪": "tab:green"}
    for lbl, g in d.groupby("kind"):
        g = g.sort_values("amp")
        c = colors.get(lbl, "k")
        ax[0, 0].plot(g.amp, g.sig_med, "o-", color=c, label=lbl)
        ax[0, 1].plot(g.amp, g.thr_med, "o-", color=c, label=lbl)
        ax[1, 0].plot(g.amp, g.n_miss, "o-", color=c, label=lbl)
        ax[1, 1].plot(g.amp, g.maxdev, "o-", color=c, label=lbl)
    ax[0, 0].axhline(np.median(d[d.amp == 200].sig_med), ls=":", c="gray")
    ax[0, 0].set_title("在线 σ_d 随抖动幅度膨胀（中介切换录制）")
    ax[0, 0].set_ylabel("σ_d (ADC)")
    ax[0, 1].axhline(1216, ls="--", c="gray", label="5%·平台电平 = 1216")
    ax[0, 1].set_title("检测门限 thr_d = max(5σ_d, 5%·lv_ref, 1%·max_tot)")
    ax[0, 1].set_ylabel("门限 (ADC)")
    ax[1, 0].set_title("真实变载被漏掉的事件数")
    ax[1, 0].set_ylabel("漏事件数")
    ax[1, 1].set_title("显示相对无扰动基线的最大偏差")
    ax[1, 1].set_ylabel("|Δ显示| (ADC)")
    for a in ax.ravel():
        a.set_xlabel("抖动幅度（总量 RMS, ADC）")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("v6 检测器在快速扰动下的失守机理：抖动抬高 σ_d ⇒ 门限抬高 ⇒ 漏事件 ⇒ 基线/显示错误")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a_jitter_detector.png"), dpi=130)
    plt.close(fig)


def fig_epochs():
    """图2：epoch 数与误触发数 vs 抖动幅度（两份录制 × 两种带宽）。"""
    df = pd.read_csv(os.path.join(RES, "a2b_jitter_ext.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lbl, g in df[df.kind.isin(["独立白噪", "带限0.3-5Hz"])].groupby("kind"):
        for rec, gg in g.groupby("rec"):
            gg = gg.sort_values("amp")
            style = "-" if lbl == "独立白噪" else "--"
            ax[0].plot(gg.amp, gg.n_epoch, "o" + style, label=f"{lbl}/{rec[:6]}")
            ax[1].plot(gg.amp, gg.n_miss, "o" + style, label=f"{lbl}/{rec[:6]}")
            ax[2].plot(gg.amp, gg.n_extra, "o" + style, label=f"{lbl}/{rec[:6]}")
    for a, t in zip(ax, ["epoch 总数（含漏/误）", "漏掉的真实事件数", "多出来的误触发事件数"]):
        a.set_title(t)
        a.set_xlabel("抖动幅度（总量 RMS, ADC）")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("抖动幅度 vs 事件识别：主要失效是\"漏\"而不是\"误\"")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a_jitter_epochs.png"), dpi=130)
    plt.close(fig)


def fig_tap_vs_step():
    """图3：拍击 vs 真实阶跃的时域对照（同一注入位置、同幅度）。"""
    d = load_uniform(REC[REC1])
    tu, X = d["tu"], d["Xu"]
    i0 = int(round(21.5 / 0.01))
    amp = 4000
    Xt, _ = add_tap(X, i0, amp, rise_ms=50, hold_ms=100)
    Xs = add_step(X, i0, amp, rise_ms=50)
    cases = [("拍击 50/100 ms +4000 ADC", Xt), ("真实 restep +4000 ADC", Xs)]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for j, (title, Xp) in enumerate(cases):
        r = run_traced(TRACED_V6, tu, Xp)
        w = slice(max(0, i0 - 300), i0 + 900)
        ax[j].plot(tu[w], Xp.sum(axis=1)[w], c="gray", lw=1.0, label="原始（真值）总量")
        ax[j].plot(tu[w], r["Y"].sum(axis=1)[w], c="tab:blue", lw=1.6, label="v6 显示总量")
        for e in r["epoch"]:
            if i0 - 0.3 <= i0 + (e[0] - tu[i0]) and abs(e[0] - tu[i0]) < 9:
                ax[j].axvline(e[0], c="tab:red", ls=":", lw=1.2)
        for rv in r["revoke"]:
            ax[j].axvline(rv[0], c="tab:green", ls="-.", lw=1.2)
        ax[j].set_title(f"{title}   epoch 新增 {len(r['epoch']) - 6}（红点线=epoch 时刻，绿点线=撤销）")
        ax[j].set_ylabel("ADC")
        ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=9)
    ax[1].set_xlabel("时间 (s)")
    fig.suptitle("拍击被当成一次加载（epoch 触发），但 0.4 s 撤销窗内回落 ⇒ 不重锚（ΣA 不变）；真实阶跃不回落 ⇒ 交接重锚")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a_tap_vs_step.png"), dpi=130)
    plt.close(fig)


def fig_mech():
    """图4：抖动注入下 d / 门限的逐帧对照（定位"门限被抬高 ⇒ 漏事件"）。"""
    d = load_uniform(REC[REC2], tmax=70.0)
    tu, X = d["tu"], d["Xu"]
    Xp = X + perturb_on_mask(X, 2000, "white", np.random.default_rng(7), dict(f_hi=40.0))
    t0 = detector_trace(tu, X)
    t1 = detector_trace(tu, Xp)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(tu, np.abs(t0["d"]), c="gray", lw=0.7, label="|d| 无扰动")
    ax[0].plot(tu, np.abs(t1["d"]), c="tab:red", lw=0.7, label="|d| +2000ADC 白噪")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("|d| (ADC, log)")
    ax[0].legend(fontsize=9)
    ax[0].set_title("检测统计量 |d| 与门限：抖动把门限从 ~190 抬到 ~2000 ADC，同时真实变载的 |d| 分布不变")
    ax[0].grid(alpha=0.3)
    ax[1].plot(tu, t0["thr"], c="gray", lw=1.2, label="thr 无扰动")
    ax[1].plot(tu, t1["thr"], c="tab:red", lw=1.2, label="thr +2000ADC 白噪")
    ax[1].axhline(0.05 * np.median(X.sum(axis=1)[X.sum(axis=1) > 0.5 * X.sum(axis=1).max()]),
                  ls="--", c="gray", label="5%·平台电平")
    ax[1].set_ylabel("thr (ADC)")
    ax[1].set_xlabel("时间 (s)")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "a_jitter_threshold.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_detector()
    fig_epochs()
    fig_tap_vs_step()
    fig_mech()
    print("figures written to", FIG)
