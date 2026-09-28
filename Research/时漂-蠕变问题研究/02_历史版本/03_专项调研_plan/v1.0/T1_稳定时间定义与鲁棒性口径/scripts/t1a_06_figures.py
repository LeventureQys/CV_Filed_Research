# -*- coding: utf-8 -*-
"""t1a_06_figures：T1-A 图件（英文标签，避免中文字体缺失出方框）。

产出：
  figures/T1A_01_settle_ecdf.png    三实现（+raw）的 T_stable 分布：ECDF + 箱线 + 删失计数
  figures/T1A_02_caliber_compare.png 口径对照：逐事件三口径、各臂中位、第一轮 5 值裁决、D1 vs D2
  figures/T1A_03_lowerbound.png      输入下界：T_ramp 分布（onset vs restep）+ 实测 vs 输入下界
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

import t1a_common as C

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
COL = {"raw": "#7f7f7f", "v5.1": "#1f77b4", "v6": "#2ca02c", "v6.1": "#d62728"}
LBL = {"raw": "raw (no compensation)", "v5.1": "v5.1 (3 s exemption)",
       "v6": "v6 (shape inversion)", "v6.1": "v6.1 (conservative ROM)"}


def ecdf(a):
    a = np.sort(np.asarray(a, float))
    return a, np.arange(1, len(a) + 1) / max(len(a), 1)


def fig1(m):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))
    # (a) 恒载 9 组 onset：ECDF（冻结口径 ch5）
    s = m[(m.kind == "onset") & (m.dom == "显示域")]
    for arm in C.ARMS:
        a, y = ecdf(s.loc[(s.arm == arm) & (s.cens_ch5 == False), "T_stable_ch5"])   # noqa: E712
        n_c = int(s[(s.arm == arm)].cens_ch5.sum())
        ax[0, 0].step(a, y, where="post", color=COL[arm], lw=1.8,
                      label="%s (n=%d, censored=%d)" % (LBL[arm], len(a), n_c))
    ax[0, 0].axvspan(1.0, 2.0, color="#c8e6c9", alpha=.6, label="target 1-2 s")
    ax[0, 0].set_xlim(0, 14)
    ax[0, 0].set_xlabel("T_stable (s) - main channel, +5 s reference  [frozen caliber D1]")
    ax[0, 0].set_ylabel("ECDF")
    ax[0, 0].set_title("(a) 13 static-load onsets (9 recordings)")
    ax[0, 0].legend(fontsize=7.5, loc="lower right")
    ax[0, 0].grid(alpha=.3)

    # (b) 箱线（同集合）
    data = [s.loc[(s.arm == arm) & (s.cens_ch5 == False), "T_stable_ch5"].to_numpy(float)
            for arm in C.ARMS]
    bp = ax[0, 1].boxplot(data, tick_labels=[a for a in C.ARMS], showmeans=False,
                          patch_artist=True, widths=.55)
    for patch, arm in zip(bp["boxes"], C.ARMS):
        patch.set_facecolor(COL[arm])
        patch.set_alpha(.55)
    for i, d in enumerate(data, start=1):
        ax[0, 1].scatter(np.full(len(d), i) + np.random.default_rng(i).normal(0, .05, len(d)),
                         d, s=14, color="k", alpha=.6, zorder=3)
    ax[0, 1].axhspan(1.0, 2.0, color="#c8e6c9", alpha=.6)
    ax[0, 1].axhline(2.0, color="r", ls="--", lw=1.2)
    ax[0, 1].set_ylabel("T_stable (s)")
    ax[0, 1].set_title("(b) same set: box + individual events (red = 2 s limit)")
    ax[0, 1].grid(alpha=.3, axis="y")

    # (c) 实录族：D1-ev 修订口径（窗在下一事件处截断）
    s2 = m[m.family == "实录"]
    for arm in C.ARMS:
        a, y = ecdf(s2.loc[(s2.arm == arm) & (s2.cens_ev_ch5 == False), "T_stable_ev_ch5"])
        n_c = int(s2[s2.arm == arm].cens_ev_ch5.sum())
        ax[1, 0].step(a, y, where="post", color=COL[arm], lw=1.8,
                      label="%s (n=%d, censored=%d)" % (arm, len(a), n_c))
    ax[1, 0].axvspan(1.0, 2.0, color="#c8e6c9", alpha=.6, label="target 1-2 s")
    ax[1, 0].set_xlim(0, 14)
    ax[1, 0].set_xlabel("T_stable_ev (s) - window cut at next real event")
    ax[1, 0].set_ylabel("ECDF")
    ax[1, 0].set_title("(c) field recordings (ADC domain, 39 events)")
    ax[1, 0].legend(fontsize=7.5, loc="lower right")
    ax[1, 0].grid(alpha=.3)

    # (d) 删失计数（各口径 × 各臂）
    cals = [("cens_ch5", "D1 main+5s"), ("cens_tot5", "D1 total+5s"),
            ("cens_chcreep", "D1 main+creep"), ("cens_ev_ch5", "D1-ev main+5s")]
    w = .2
    for i, arm in enumerate(C.ARMS):
        v = [int(m[(m.arm == arm)][c].sum()) for c, _ in cals]
        ax[1, 1].bar(np.arange(len(cals)) + (i - 1.5) * w, v, w, color=COL[arm], label=arm)
    ax[1, 1].set_xticks(np.arange(len(cals)))
    ax[1, 1].set_xticklabels([n for _, n in cals], fontsize=8)
    ax[1, 1].set_ylabel("censored events (out of 60)")
    ax[1, 1].set_title("(d) events that never settle in the window")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=.3, axis="y")
    fig.suptitle("T1-A: settling time of three implementations on 13 real recordings", fontsize=12)
    fig.tight_layout()
    p = os.path.join(C.FIG, "T1A_01_settle_ecdf.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig2(m, aud):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))
    # (a) 逐事件三口径（v6，恒载 9 组 onset）
    s = m[(m.arm == "v6") & (m.kind == "onset") & (m.dom == "显示域")].sort_values("ds")
    x = np.arange(len(s))
    ax[0, 0].bar(x - .25, s["T_stable_ch5"], .25, label="main ch, +5 s ref", color="#2ca02c")
    ax[0, 0].bar(x, s["T_stable_tot5"], .25, label="total Z, +5 s ref", color="#98df8a")
    ax[0, 0].bar(x + .25, s["T_stable_chcreep"], .25, label="main ch, +creep ref", color="#c5b0d5")
    ax[0, 0].axhline(2.0, color="r", ls="--", lw=1.2, label="2 s limit")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([k.replace("/数据", "") for k in s["key"]], rotation=45,
                             ha="right", fontsize=7)
    ax[0, 0].set_ylabel("T_stable (s)")
    ax[0, 0].set_title("(a) v6, per-event: the same run, three reference calibers")
    ax[0, 0].legend(fontsize=7.5)
    ax[0, 0].grid(alpha=.3, axis="y")

    # (b) 中位对比（各臂 × 三口径，恒载 9 组 onset）
    med = {}
    for arm in C.ARMS:
        sub = m[(m.arm == arm) & (m.kind == "onset") & (m.dom == "显示域")]
        med[arm] = [sub["T_stable_ch5"].median(), sub["T_stable_tot5"].median(),
                    sub["T_stable_chcreep"].median()]
    w = .2
    for i, arm in enumerate(C.ARMS):
        ax[0, 1].bar(np.arange(3) + (i - 1.5) * w, med[arm], w, color=COL[arm], label=arm)
        for j, v in enumerate(med[arm]):
            ax[0, 1].text(j + (i - 1.5) * w, v, "%.2f" % v, ha="center", va="bottom", fontsize=7)
    ax[0, 1].axhspan(1.0, 2.0, color="#c8e6c9", alpha=.6)
    ax[0, 1].axhline(2.0, color="r", ls="--", lw=1.2)
    ax[0, 1].set_xticks(np.arange(3))
    ax[0, 1].set_xticklabels(["main+5s\n(frozen D1)", "total+5s", "main+creep"], fontsize=8)
    ax[0, 1].set_ylabel("median T_stable (s)")
    ax[0, 1].set_title("(b) median over 9 static-load onsets")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=.3, axis="y")

    # (c) 第一轮公开值的裁决：公布值 / 第一轮口径复算 / 本任务冻结口径
    a = aud.copy()

    def ascii_label(n):
        for key, lab in (("第一轮口径", ""), ("主通道+5s（恒载", "v6 main ch +5s ref"),
                        ("总通道+5s（恒载", "v6 total Z +5s ref"),
                        ("含蠕变口径", "v6 main ch +creep ref"),
                        ("早期旧值", "v6 OLD value (index bug)"),
                        ("raw / v5.1 主通道", "raw main ch +5s ref"),
                        ("raw 总通道", "raw total Z +5s ref"),
                        ("回溯真沿", "v6 total Z + backtracked edge")):
            if key in n:
                return lab
        return n[:28]
    lab = [ascii_label(n) for n in a["value_name"]]
    y = np.arange(len(a))
    ax[1, 0].barh(y - .24, a["published_s"], .22, color="#ff9896", label="published (round 1)")
    ax[1, 0].barh(y, a["recomputed_round1caliber_s"], .22, color="#1f77b4",
                  label="T1-A recomputed, round-1 caliber")
    ax[1, 0].barh(y + .24, a["recomputed_frozen_s"], .22, color="#2ca02c",
                  label="T1-A frozen caliber (T4-A edge)")
    ax[1, 0].set_yticks(y)
    ax[1, 0].set_yticklabels(lab, fontsize=7)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("T_stable for v6 (s, log scale)")
    ax[1, 0].set_title("(c) adjudication of the published v6 values")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].grid(alpha=.3, which="both", axis="x")

    # (d) D1 vs D2（T_settle 2%/5%）逐事件，v6 主通道
    s = m[(m.arm == "v6") & (m.cens_ch5 == False)]                                 # noqa: E712
    ok = s["T_settle_ch5"].notna()
    ax[1, 1].scatter(s.loc[ok, "T_stable_ch5"], s.loc[ok, "T_settle_ch5"], s=18,
                     c=np.where(s.loc[ok, "kind"] == "onset", "#2ca02c", "#ff7f0e"),
                     label="eps=5%")
    ok2 = s["T_settle_ch2"].notna()
    ax[1, 1].scatter(s.loc[ok2, "T_stable_ch5"], s.loc[ok2, "T_settle_ch2"], s=18, marker="^",
                     c=np.where(s.loc[ok2, "kind"] == "onset", "#98df8a", "#ffbb78"),
                     label="eps=2%")
    lim = [0.05, 120]
    ax[1, 1].plot(lim, lim, "k--", lw=.8, label="identity")
    ax[1, 1].axvspan(1.0, 2.0, color="#c8e6c9", alpha=.6)
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlim(*lim)
    ax[1, 1].set_ylim(*lim)
    ax[1, 1].set_xlabel("T_stable (s)  [D1, main ch +5 s]")
    ax[1, 1].set_ylabel("T_settle(eps) (s)  [D2]")
    ax[1, 1].set_title("(d) D1 vs D2, v6, per event (green=onset, orange=other)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=.3, which="both")
    fig.suptitle("T1-A: caliber comparison and round-1 value adjudication", fontsize=12)
    fig.tight_layout()
    p = os.path.join(C.FIG, "T1A_02_caliber_compare.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig3(lb):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for kd, c in (("onset", "#2ca02c"), ("restep", "#ff7f0e")):
        s = lb[lb.kind == kd]
        val = s["T_ramp"].dropna()
        ax[0].hist(val, bins=np.arange(0, 1.35, .1), alpha=.6, color=c,
                   label="%s (n=%d, median %.2f s)" % (kd, len(val), val.median()))
    ax[0].set_xlabel("equivalent input ramp T_ramp (s)   [T4-A deconvolution]")
    ax[0].set_ylabel("events")
    ax[0].set_title("(a) how long the INPUT itself needs")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    s = lb[(lb.kind == "restep") & (lb.cens_in_ch == False)]                        # noqa: E712
    ax[1].scatter(s["T_in5_raw_ch"], s["T_stable_ev_ch5_v6"], s=22, color="#2ca02c",
                  label="v6")
    ax[1].scatter(s["T_in5_raw_ch"], s["T_stable_ev_ch5_v5.1"], s=22, marker="s",
                  color="#1f77b4", label="v5.1")
    m2 = s[["T_in5_raw_ch", "T_ramp"]].dropna()
    ax[1].scatter(m2["T_in5_raw_ch"], m2["T_ramp"], s=22, marker="x", color="k",
                  label="input ramp T_ramp")
    lim = [0.03, 40]
    ax[1].plot(lim, lim, "k--", lw=.8)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(*lim)
    ax[1].set_ylim(0.01, 60)
    ax[1].set_xlabel("input-side settle T_in5 (s), same caliber")
    ax[1].set_ylabel("measured T_stable_ev (s)")
    ax[1].set_title("(b) restep: measured vs input-side floor")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")

    for i, kd in enumerate(("onset", "restep")):
        s = lb[lb.kind == kd]
        v = pd.to_numeric(s["z_at_10"], errors="coerce").dropna()
        ax[2].hist(v, bins=np.arange(0, 1.05, .05), alpha=.6,
                   color=("#2ca02c" if kd == "onset" else "#ff7f0e"),
                   label="%s (median %.2f, n=%d)" % (kd, v.median(), len(v)))
    ax[2].axvline(0.95, color="r", ls="--", lw=1, label="95%")
    ax[2].set_xlabel("fraction of step reached at t_on + 1 s  (z_at_10)")
    ax[2].set_ylabel("events")
    ax[2].set_title("(c) is the 1 s promise reachable by INPUT?")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    fig.suptitle("T1-A Q5: theoretical lower bound of settling time (input-limited cases)", fontsize=12)
    fig.tight_layout()
    p = os.path.join(C.FIG, "T1A_03_lowerbound.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    C.start_log("06_figures")
    m = pd.read_csv(os.path.join(C.RES, "t1a_settle_metrics.csv"), encoding="utf-8-sig")
    aud = pd.read_csv(os.path.join(C.RES, "t1a_round1_audit.csv"), encoding="utf-8-sig")
    lb = pd.read_csv(os.path.join(C.RES, "t1a_lowerbound.csv"), encoding="utf-8-sig")
    os.makedirs(C.FIG, exist_ok=True)
    print("图 1: 冻结口径 ECDF / 箱线 / 实录修订口径 / 删失计数")
    print("-> %s" % fig1(m))
    print("图 2: 口径对照 / 第一轮裁决 / D1 vs D2")
    print("-> %s" % fig2(m, aud))
    print("图 3: 输入下界")
    print("-> %s" % fig3(lb))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
