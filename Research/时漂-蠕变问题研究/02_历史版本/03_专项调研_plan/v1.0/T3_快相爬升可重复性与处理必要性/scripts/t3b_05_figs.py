# -*- coding: utf-8 -*-
"""t3b_05_figs.py —— T3-B 的图件（3 张，全部有标题/轴标签含单位/图例，底层 csv 见 results/）。

  figures/T3B_01_route_roi.png    ROI 曲线与拐点（4 维：rom_scale / κ / 滑行器速率 / 只对 onset）
  figures/T3B_02_cost_benefit.png 代价-收益：T_stable ↔ OS% / MD / epoch（全部臂散点）
  figures/T3B_03_verdict.png      H1/H2 三态裁决一览 + 「≤2 s 达标录制比」条形

运行：python scripts/t3b_05_figs.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_common as C   # noqa: E402

RES = C.RES
FIG = C.FIG
PRIM = "T_stable_ev_tot5"

import matplotlib            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

C_T = "#1565c0"      # T_stable
C_O = "#c62828"      # 代价
C_K = "#2e7d32"      # 拐点
BAND = "#c8e6c9"


def roi_curve(ax, df, xcol, xlab, title, knee=None, logx=False):
    d = df.dropna(subset=[xcol, "Tstab_tot_ev_med"]).sort_values(xcol)
    ax.plot(d[xcol], d.Tstab_tot_ev_med, "o-", color=C_T, label="T_stable 中位（总通道 D1-ev）")
    ax.fill_between(d[xcol], d.Tstab_tot_ev_p10, d.Tstab_tot_ev_p90, color=C_T, alpha=.15,
                    label="p10~p90（跨事件）")
    ax.axhline(2.0, color="k", ls="--", lw=.9)
    ax.text(d[xcol].min(), 2.02, "2 s 上限", fontsize=8, va="bottom")
    ax.axhspan(1.0, 2.0, color=BAND, alpha=.5, zorder=0)
    ax.set_xlabel(xlab)
    ax.set_ylabel("T_stable（s）", color=C_T)
    ax.tick_params(axis="y", labelcolor=C_T)
    if logx:
        ax.set_xscale("log")
    for _, r in d.iterrows():
        ax.annotate(r.arm, (r[xcol], r.Tstab_tot_ev_med), fontsize=7,
                    xytext=(2, 3), textcoords="offset points")
    ax2 = ax.twinx()
    ax2.plot(d[xcol], d.OS_ch_med, "s--", color=C_O, label="超调 OS% 中位（主通道）")
    ax2.set_ylabel("OS%（%×J）", color=C_O)
    ax2.tick_params(axis="y", labelcolor=C_O)
    if knee is not None and np.isfinite(knee):
        ax.axvline(knee, color=C_K, ls=":", lw=1.2)
        ax.text(knee, ax.get_ylim()[1], " 拐点 %.3g" % knee, color=C_K, fontsize=8, va="top")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=10)
    return ax


def fig1(roi, knee):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    kn = {r["dim"]: r for _, r in knee.iterrows()} if knee is not None and len(knee) else {}

    def kv(key):
        for k, r in kn.items():
            if key in k:
                return r.knee_Tstable
        return None

    d = roi.dropna(subset=["rom_scale"])
    roi_curve(axs[0, 0], d, "rom_scale", "形状库缩放（g(0.2 s)/g_v6(0.2 s)；>1 = 更早到位 = 更保守）",
              "① 形状库快慢缩放（rom_scale）", kv("形状库"))
    d = roi.dropna(subset=["kappa_onset"]).copy()
    d["strength"] = 1.30 - d.kappa_onset
    roi_curve(axs[0, 1], d, "strength", "κ 封顶强度 = 1.30 − κ_onset（越大 = 单侧上限越紧）",
              "② κ 单侧上限 min(A, κ·inc)", kv("κ"))
    d = roi.dropna(subset=["glide_rate"]).copy()
    d["strength"] = -d.glide_rate
    roi_curve(axs[1, 0], d, "strength", "滑行器速率上限的负值 = −RATE_MAX（越大 = 越慢 = 摊得越平）",
              "③ 滑行器速率上限 RATE_MAX（0.8/s 为默认）", kv("滑行器"))
    d = roi[roi.arm.isin(["v6", "onset_only"])].copy()
    if len(d) == 2:
        x = np.arange(2)
        w = 0.35
        on = [d[d.arm == "v6"].Tstab_tot_ev_med_onset.iloc[0],
              d[d.arm == "onset_only"].Tstab_tot_ev_med_onset.iloc[0]]
        re_ = [d[d.arm == "v6"].Tstab_tot_ev_med_restep.iloc[0],
               d[d.arm == "onset_only"].Tstab_tot_ev_med_restep.iloc[0]]
        axs[1, 1].bar(x - w / 2, on, w, color=C_T, label="onset（n=%d）" % int(
            d[d.arm == "v6"].n_onset.iloc[0]))
        axs[1, 1].bar(x + w / 2, re_, w, color=C_O, label="restep（n=%d）" % int(
            d[d.arm == "v6"].n_restep.iloc[0]))
        axs[1, 1].axhline(2.0, color="k", ls="--", lw=.9)
        axs[1, 1].axhspan(1.0, 2.0, color=BAND, alpha=.5, zorder=0)
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(["全部事件走反演\n(v6)", "只对 onset 反演\n(onset_only)"])
        axs[1, 1].set_ylabel("T_stable 中位（s）")
        axs[1, 1].set_title("④ 只对 onset 生效（事件过滤模拟）", fontsize=10)
        for i, (a, b) in enumerate(zip(on, re_)):
            axs[1, 1].text(i - w / 2, a, "%.2f" % a, ha="center", va="bottom", fontsize=8)
            axs[1, 1].text(i + w / 2, b, "%.2f" % b, ha="center", va="bottom", fontsize=8)
        axs[1, 1].legend(fontsize=8)
    fig.suptitle("T3-B · 处理强度 ROI（13 份录制、40 事件；每臂相对 v6 只改一个参数）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(FIG, "T3B_01_route_roi.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> %s" % p)


def fig2(roi):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.4))
    d = roi.dropna(subset=["Tstab_tot_ev_med", "OS_ch_med"])
    ax = axs[0]
    ax.axvspan(1.0, 2.0, color=BAND, alpha=.6, label="1~2 s 目标区")
    ax.axvline(2.0, color="k", ls="--", lw=.9)
    sc = ax.scatter(d.Tstab_tot_ev_med, d.OS_ch_med, c=d.trigger_rate.fillna(0), cmap="viridis",
                    s=48, edgecolor="k", lw=.4)
    for _, r in d.iterrows():
        ax.annotate(r.arm, (r.Tstab_tot_ev_med, r.OS_ch_med), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="κ 上限触发率（截断 Â 的比例）")
    ax.set_xlabel("T_stable 中位（s，总通道 D1-ev；n=40 事件/13 录制）")
    ax.set_ylabel("超调 OS%（%×J，主通道中位）")
    ax.set_title("代价-收益：越快越容易过充吗？", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    ax = axs[1]
    ax.scatter(d.Tstab_tot_ev_med, d.MD_tot_med, s=48, color=C_O, edgecolor="k", lw=.4,
               label="最大偏差 MD 中位（ADC）")
    for _, r in d.iterrows():
        ax.annotate(r.arm, (r.Tstab_tot_ev_med, r.MD_tot_med), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvspan(1.0, 2.0, color=BAND, alpha=.6)
    ax.axvline(2.0, color="k", ls="--", lw=.9)
    ax.set_xlabel("T_stable 中位（s，总通道 D1-ev）")
    ax.set_ylabel("MD 中位（ADC，总量）", color=C_O)
    ax.tick_params(axis="y", labelcolor=C_O)
    ax2 = ax.twinx()
    ax2.scatter(d.Tstab_tot_ev_med, d.epoch_per100s, s=40, marker="^", color="#6a1b9a",
                edgecolor="k", lw=.4, label="epoch 数 /100 s")
    ax2.set_ylabel("epoch 数 /100 s", color="#6a1b9a")
    ax2.tick_params(axis="y", labelcolor="#6a1b9a")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    ax.set_title("代价-收益：偏差与状态机活动", fontsize=11)
    ax.grid(alpha=.25)
    fig.suptitle("T3-B · 路线代价-收益（全部 22 条臂；底层表 results/t3b_route_roi.csv）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(FIG, "T3B_02_cost_benefit.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> %s" % p)


def _safe(s):
    """matplotlib 中文字体缺字时会把字符画成方框；把已知缺字的符号换成 ASCII。"""
    return (str(s).replace("⇒", "->").replace("⇔", "<->").replace("→", "->")
            .replace("←", "<-"))


def fig3(m):
    vp = os.path.join(RES, "t3b_h2_verdict.csv")
    fig = plt.figure(figsize=(13.5, 7.6))
    ax = fig.add_axes([0.02, 0.34, 0.96, 0.62])
    ax.axis("off")
    if os.path.isfile(vp):
        v = pd.read_csv(vp, encoding="utf-8-sig")
        col = {"支持": "#2e7d32", "反驳": "#c62828", "有条件支持": "#ef6c00"}
        y = 0.98
        ax.text(0.005, y, "T3-Q10 · H1/H2 三态裁决（数值口径：T_stable 总通道 D1-ev 中位 / n=13 录制、40 事件）",
                fontsize=12, weight="bold", va="top")
        y -= 0.09
        for _, r in v.iterrows():
            ax.text(0.005, y, _safe(r["item"]), fontsize=10, weight="bold", va="top")
            ax.text(0.13, y, _safe(r["claim"]), fontsize=9.5, va="top")
            ax.text(0.62, y, _safe(r["three_state"]), fontsize=11, weight="bold", va="top",
                    color=col.get(r["three_state"], "#000000"))
            y -= 0.055
            ax.text(0.03, y, "理由：" + _safe(r["one_line_reason"]), fontsize=9, va="top")
            y -= 0.05
            ax.text(0.03, y, "数字：" + _safe(r["key_numbers"]), fontsize=9, va="top",
                    color="#333333")
            y -= 0.075
    else:
        ax.text(0.01, 0.9, "缺 results/t3b_h2_verdict.csv（先跑 t3b_04_verdict.py 生成）",
                fontsize=12)

    ax2 = fig.add_axes([0.06, 0.07, 0.88, 0.22])
    dz = pd.read_csv(os.path.join(RES, "t3b_route_q6q7.csv"), encoding="utf-8-sig")
    d = dz[dz.tag == "tot_ev"].dropna(subset=["frac_le_2s"])
    order = ["raw", "v51_f5", "v51_f3", "v51_f1", "v6", "v61_rs106", "onset_only",
             "rt_filter030", "rt_onesided"]
    d = d[d.arm.isin(order)]
    d["o"] = d.arm.map({a: i for i, a in enumerate(order)})
    d = d.sort_values("o")
    cols = ["#9e9e9e" if a in ("raw", "v51_f5", "v51_f3") else
            ("#ef6c00" if a == "v51_f1" else "#1565c0") for a in d.arm]
    ax2.bar(d.arm, 100 * d.frac_le_2s, color=cols)
    for i, (_, r) in enumerate(d.iterrows()):
        ax2.text(i, 100 * r.frac_le_2s + 2, "%.0f%%\n中位%.2fs" % (100 * r.frac_le_2s, r.med),
                 ha="center", fontsize=8)
    ax2.set_ylabel("T_stable ≤ 2 s 的录制比（%）")
    ax2.set_xlabel("臂（灰 = 不处理快相；橙 = 只压免责期；蓝 = 形状反演及其变体）")
    ax2.set_ylim(0, 125)
    ax2.set_title("「能不能进 2 s」——13 份录制上 D1-ev 总通道口径的达标录制比", fontsize=10)
    ax2.grid(alpha=.25, axis="y")
    fig.suptitle("T3-B · H1/H2 裁决一览", fontsize=13)
    p = os.path.join(FIG, "T3B_03_verdict.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> %s" % p)


def main():
    roi = pd.read_csv(os.path.join(RES, "t3b_route_roi.csv"), encoding="utf-8-sig")
    m = pd.read_csv(os.path.join(RES, "t3b_arm_metrics.csv"), encoding="utf-8-sig")
    kp = os.path.join(RES, "t3b_roi_knee.csv")
    knee = pd.read_csv(kp, encoding="utf-8-sig") if os.path.isfile(kp) else None
    fig1(roi, knee)
    fig2(roi)
    fig3(m)
    print("完成：3 张图写入 %s" % FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
