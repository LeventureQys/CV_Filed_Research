# -*- coding: utf-8 -*-
"""t4b_figs.py —— T4-B 图件（中文字体已确认存在：Microsoft YaHei / SimHei / Noto Sans SC）。

产出：
  figures/T4B_01_separability.png —— 四个候选判据的特征分离（条带 + AUC + 门限）
  figures/T4B_02_branch_cost.png  —— 误判代价：各臂的 T1 口径指标对比（T_stable / 过充 / 平台偏差）
  figures/T4B_03_chattering.png   —— 门限附近抖动与迟滞抑制
  figures/T4B_04_shape_by_arm.png —— 两形态归一化形状 + 变异带（支撑"形状重叠大"的结论）

运行：python scripts/t4b_figs.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C            # noqa: E402

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
RES, FIG = C.RES, C.FIG


def fig1_separability():
    sep = pd.read_csv(os.path.join(RES, "t4b_separability.csv"))
    ll = pd.read_csv(os.path.join(RES, "t4b_morphology_by_arm.csv"))
    ll = ll[ll.kind.isin(["onset", "restep"])]
    panels = [
        ("P1_pre_frac", "① pre 电平 ÷ 记录峰值\n（Δ=0，纯因果）", "pre_frac", None),
        ("P2_slope@0.2s", "② 上升沿斜率 J(0.2)/0.2\n÷ 峰值（Δ=0.20 s）", None, "Jpc_020"),
        ("P3_J_over_pre@0.2s", "③ 台阶幅度/电平比\nJ(0.2)/pre（Δ=0.20 s）", None, "ratio_pre_020"),
        ("P4_singleframe@020", "④ 单帧跃变占比\nJ(0)/J(0.2)（Δ=0.20 s）", None, "JF_020"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6))
    rng = np.random.default_rng(7)
    for ax, (cid, title, col_a, col_b) in zip(axes, panels):
        if col_a == "pre_frac":
            xo, xr = ll[ll.kind == "onset"].pre_frac, ll[ll.kind == "restep"].pre_frac
        elif col_b == "ratio_pre_020":
            xo = ll[ll.kind == "onset"].inc_020 / np.maximum(ll[ll.kind == "onset"].pre, 1e-9)
            xr = ll[ll.kind == "restep"].inc_020 / np.maximum(ll[ll.kind == "restep"].pre, 1e-9)
        else:
            xo = ll[ll.kind == "onset"][col_b]
            xr = ll[ll.kind == "restep"][col_b]
        lo = min(xo.min(), xr.min())
        hi = max(xo.max(), xr.max())
        pad = 0.06 * max(hi - lo, 1e-9)
        ax.set_xlim(lo - pad, hi + pad)
        for y, arr, c, lbl in ((1, xo, "tab:blue", "零基线起 (onset)"),
                               (0, xr, "tab:red", "带载起 (restep)")):
            ax.hlines(y, np.percentile(arr, 10), np.percentile(arr, 90), color=c, lw=9, alpha=.35)
            ax.plot(arr, y + 0.14 * (rng.random(len(arr)) - 0.5) * 2 * 0.6, "o", ms=4, color=c,
                    alpha=.85, label=lbl)
        row = sep[sep.crit == cid]
        if len(row):
            r0 = row.iloc[0]
            ax.axvline(r0["thr"], color="k", ls="--", lw=1.2)
            ax.set_title("%s\nAUC=%.3f  重叠=%.3f  平衡准确率=%.3f"
                         % (title, r0.auc_eff, r0.p10p90_overlap, r0.balanced_acc), fontsize=9.5)
        else:
            ax.set_title(title, fontsize=9.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["restep", "onset"])
        ax.set_ylim(-0.5, 1.5)
        ax.grid(alpha=.25, axis="x")
    axes[0].legend(loc="upper center", fontsize=8)
    fig.suptitle("T4B_01 四种因果判据的可分性（13 份录制，onset n=%d / restep n=%d；"
                 "条带 = p10~p90，虚线 = 最佳门限）"
                 % ((ll.kind == "onset").sum(), (ll.kind == "restep").sum()), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(FIG, "T4B_01_separability.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> " + p)


def fig2_branch_cost():
    conf = pd.read_csv(os.path.join(RES, "t4b_branch_confusion.csv"))
    cost = pd.read_csv(os.path.join(RES, "t4b_branch_cost.csv"))
    arms = ["M0_now", "M1_flat112", "M2_correctA", "M2_correct", "M3_causal_P1",
            "M3_causal_thr0.40", "M4_warped", "M2_inverse"]
    lab = {"M0_now": "现状\n(单库 κ1.30)", "M1_flat112": "单库 κ1.12",
           "M2_correctA": "单库 κ1.30/1.12", "M2_correct": "真值分支\n(上界)",
           "M3_causal_P1": "因果判据分支\n(θ=0.30)", "M3_causal_thr0.40": "因果判据分支\n(θ=0.40)",
           "M4_warped": "连续参数化\n(不分支)", "M2_inverse": "完全反着分\n(最坏)"}
    arms = [a for a in arms if a in set(conf.arm)]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    x = np.arange(len(arms))
    for ax, col, ttl, unit in (
            (axes[0], "T_stable_med", "事件后 T_stable 中位（越低越好）", "s"),
            (axes[1], "os_med", "超调中位 OS%（越低越好）", "%"),
            (axes[2], "plat_dev_abs_med", "平台偏差中位（÷事件绝对幅度，越接近 0 越好）", "%"),
            (axes[3], "md_med", "最大偏差中位 MD（越低越好）", "ADC/单位")):
        w = 0.38
        for j, kind in enumerate(("onset", "restep")):
            v = [float(conf[(conf.arm == a) & (conf.true_kind == kind)][col].iloc[0])
                 if len(conf[(conf.arm == a) & (conf.true_kind == kind)]) else np.nan
                 for a in arms]
            ax.bar(x + (j - 0.5) * w, v, w, label=("零基线起 (onset)" if kind == "onset"
                                                   else "带载起 (restep)"),
                   color=("tab:blue" if kind == "onset" else "tab:red"), alpha=.85)
        ax.set_xticks(x)
        ax.set_xticklabels([lab[a] for a in arms], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(alpha=.25, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("T4B_02 分支/不分支的 T1 口径代价（同一算法、同一批事件，只改形状库与 κ；"
                 "误差棒另见 results/t4b_branch_cost.csv）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "T4B_02_branch_cost.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> " + p)


def fig3_chattering():
    pe = pd.read_csv(os.path.join(RES, "t4b_chattering_per_event.csv"))
    ch = pd.read_csv(os.path.join(RES, "t4b_chattering_hyst.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    ax = axes[0]
    for kind, c in (("onset", "tab:blue"), ("restep", "tab:red")):
        s = pe[pe.kind == kind]
        ax.scatter(s.s_nom, s.s_400_med, s=30, color=c, alpha=.8,
                   label="%s（A=400 ADC）" % kind)
    ax.axhline(0.30, color="k", ls="--", lw=1.2, label="门限 θ=0.30")
    ax.axvline(0.30, color="k", ls=":", lw=1)
    ax.set_xlabel("无噪声时的判据统计量 s = pre/滚动峰值")
    ax.set_ylabel("注入 400 ADC 带限噪声后的 s（20 次中位）")
    ax.set_title("(a) 判据统计量的噪声漂移", fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    for A, c in ((0, "tab:green"), (100, "tab:orange"), (400, "tab:red")):
        s = ch[ch.A == A]
        g = s.groupby("theta").cross_per_ev.mean()
        ax.plot(g.index, g.values, "o-", color=c, label="A=%d ADC" % A)
    ax.set_xlabel("分支门限 θ（s ≤ θ ⇒ 零基线支）")
    ax.set_ylabel("事件后 0.2 s 内门限穿越次数/事件")
    ax.set_title("(b) 噪声下的门限穿越（无迟滞）", fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    ax = axes[2]
    s = ch[ch.A == 400]
    for th, c in zip(sorted(s.theta.unique()), plt.cm.viridis(np.linspace(0, .85, 5))):
        g = s[s.theta == th].set_index("hyst").cross_per_ev
        ax.plot(g.index, g.values, "o-", color=c, label="θ=%.2f" % th)
    ax.set_xlabel("迟滞宽度 h")
    ax.set_ylabel("事件后 0.2 s 内门限穿越次数/事件（1 次 = 正常切换）")
    ax.set_title("(c) 迟滞抑制（A=400 ADC）", fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.suptitle("T4B_03 分支门限附近的切换抖动与迟滞设计", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "T4B_03_chattering.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> " + p)


def fig4_shape():
    ll = pd.read_csv(os.path.join(RES, "t4b_morphology_by_arm.csv"))
    ll = ll[ll.kind.isin(["onset", "restep"])]
    cols = [c for c in ll.columns if c.startswith("sh_") and c.endswith("b")]
    taus = np.array([int(c[3:6]) / 100 for c in cols])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    for kind, c in (("onset", "tab:blue"), ("restep", "tab:red")):
        s = ll[ll.kind == kind]
        med = np.array([s[c].median() for c in cols])
        p10 = np.array([s[c].quantile(.1) for c in cols])
        p90 = np.array([s[c].quantile(.9) for c in cols])
        ax.plot(taus, med, "o-", color=c, label="%s 中位 (n=%d)" % (kind, len(s)))
        ax.fill_between(taus, p10, p90, color=c, alpha=.18, label="%s p10~p90" % kind)
    ax.set_xscale("log")
    ax.set_xlabel("τ（相对 t_on，s，对数轴）")
    ax.set_ylabel("归一化形状 J(τ)/J(5 s)")
    ax.set_title("(a) 两形态形状：中位差集中在前 0.3 s，但 p10~p90 大幅重叠", fontsize=9.5)
    ax.grid(alpha=.25, which="both")
    ax.legend(fontsize=8)
    ax = axes[1]
    ll2 = ll[ll.label_confident]
    ax.scatter(ll2[ll2.kind == "onset"].pre_frac, ll2[ll2.kind == "onset"].sh_020b,
               s=32, color="tab:blue", label="onset")
    ax.scatter(ll2[ll2.kind == "restep"].pre_frac, ll2[ll2.kind == "restep"].sh_020b,
               s=32, color="tab:red", label="restep")
    ax.axvline(0.30, color="k", ls="--", lw=1.2, label="P1 门限 θ=0.30")
    ax.set_xlabel("pre 电平 ÷ 记录峰值（因果、Δ=0）")
    ax.set_ylabel("J(0.2)/J(5 s)（形状）")
    ax.set_title("(b) 因果判据 vs 形状：幅度维度可分、形状维度重叠\n"
                 "（只画 20% 门限 ±5 pt 之外的确信事件）", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.suptitle("T4B_04 两种快相形态的形状与可分性", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "T4B_04_shape_by_arm.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> " + p)


def fig5_loadlevel():
    """T4-Q9 图：相对量级 vs 快相形状（补足"量级依赖性"的可视证据）。"""
    ll = pd.read_csv(os.path.join(RES, "t4b_loadlevels.csv"))
    sh = pd.read_csv(os.path.join(RES, "t4b_loadlevel_shape.csv"))
    cor = pd.read_csv(os.path.join(RES, "t4b_loadlevel_corr.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    ax = axes[0]
    for kind, c, mk in (("onset", "tab:blue", "o"), ("restep", "tab:red", "s")):
        s = ll[ll.kind == kind]
        ax.scatter(s.post_frac_rec, s.sh_050b, s=34, color=c, marker=mk, alpha=.85,
                   label="%s (n=%d)" % (kind, len(s)))
    ax.set_xlabel("加载后平台电平 ÷ 该录制峰值（相对量级）")
    ax.set_ylabel("快相形状 J(0.5 s)/J(5 s)")
    ax.set_title("(a) 形状 vs 相对量级（全部加载事件）", fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    sub = cor[cor.subset == "仅 onset"]
    ax.bar(sub.tau, sub.rho, width=0.035,
           color=["tab:orange" if p < 0.05 else "lightgray" for p in sub.p])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("τ（s）")
    ax.set_ylabel("Spearman ρ(形状, 相对量级)")
    ax.set_title("(b) 仅 onset 子样本：秩相关\n橙 = p<0.05（n=23，低档仅 4 个 ⇒ 不可作结论）",
                 fontsize=9.5)
    ax.grid(alpha=.25, axis="y")
    ax = axes[2]
    tiers = pd.read_csv(os.path.join(RES, "t4b_loadlevel_tiers.csv"))
    ax.hist(tiers.level_frac_peak, bins=np.arange(0, 1.05, 0.1),
            color="tab:green", alpha=.8, edgecolor="k")
    ax.set_xlabel("载荷档位 ÷ 该录制峰值（持续 >3 s 的平台）")
    ax.set_ylabel("平台段数")
    ax.set_title("(c) 4 份 ADC 实录里的载荷档位分布\n（共 %d 段；事件落点仍集中在高档）"
                 % len(tiers), fontsize=9.5)
    ax.grid(alpha=.25, axis="y")
    fig.suptitle("T4B_05 跨载荷量级：绝对量级缺数据（无 N 标定），相对量级看不出形态依赖",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "T4B_05_loadlevel.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> " + p)


def fig6_tramp_window():
    """T4-Q5 增量图：输入速率判据的**因果观测窗**代价（vs 电平门限）。"""
    W = pd.read_csv(os.path.join(RES, "t4b_tramp_window.csv"))
    R = pd.read_csv(os.path.join(RES, "t4b_tramp_ref_crit.csv"))
    V = pd.read_csv(os.path.join(RES, "t4b_tramp_vs_p1.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    ax = axes[0]
    ax.plot(W.lag_s, W.slope_auc, "o-", color="tab:blue", label="② 因果斜率 J(Δ)/Δ")
    ax.plot(W.lag_s, W.T50c_auc, "s-", color="tab:olive", label="② 因果到达时刻 T50c(Δ)")
    ax.plot(W.lag_s, W.FR_auc, "^-", color="tab:cyan", label="④ 早期到达比例 FR(Δ)")
    for _, r in R[(R.crit.str.startswith("P1 ")) | (R.crit.str.startswith("T4-A armed"))].iterrows():
        c = "tab:red" if r.crit.startswith("P1 ") else "tab:orange"
        ax.axhline(r.auc, color=c, ls="--", lw=1.2,
                   label="%s（Δ=0，AUC=%.4f）" % (r.crit.split(" ")[0], r.auc))
    for _, r in R[R.crit.str.contains("T_ramp")].iterrows():
        ax.axhline(r.auc, color="k", ls=":", lw=1.4,
                   label="T4-A T_ramp（**非因果**，AUC=%.4f）" % r.auc)
    ax.set_xlabel("判据观测窗 Δ（s，判据只用 t ≤ t_on+Δ 的帧）")
    ax.set_ylabel("AUC（onset 为正类，<0.5 表示 onset 更小）")
    ax.set_title("(a) 因果速率判据的窗口-可分性曲线\nAUC 越大越好；P1 在 Δ=0 就已 0.996", fontsize=9.5)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    d = V.dropna(subset=["spearman_vs_z_at_02"]).copy()
    d["lbl"] = d["var"].str.replace("（", "\n（", regex=False)
    ax.barh(range(len(d)), d["spearman_vs_z_at_02"].abs(),
            color=["tab:red" if "P1" in v else ("k" if "T_ramp" in v else "tab:gray")
                   for v in d["var"]])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["lbl"], fontsize=7.5)
    ax.set_xlabel("|Spearman ρ|（与 T4-A 的 z_at_02，即形态轴）")
    ax.set_title("(b) 谁携带'形态'信息？\nT_ramp ρ=−0.77（驱动量）/ P1 ρ=−0.39（工况量）", fontsize=9.5)
    ax.grid(alpha=.25, axis="x")
    ax = axes[2]
    grp = [("P1 电平门限\n(θ=0.4155)", 0.9960), ("T4-A armed\n(20% 门限)", 0.9783),
           ("T4-A T_ramp\n(非因果)", 0.6522), ("T4-A t50_ch_iqr\n(非因果)", 0.8913)]
    ax.bar([g[0] for g in grp], [g[1] for g in grp],
           color=["tab:red", "tab:orange", "k", "tab:gray"])
    for i, g in enumerate(grp):
        ax.text(i, g[1] + 0.01, "%.4f" % g[1], ha="center", fontsize=8)
    ax.set_ylim(0.5, 1.08)
    ax.set_ylabel("AUC（全样本 n=34，34 个配对事件）")
    ax.set_title("(c) 可用的判据里 P1 最强\n(T_ramp 强但非因果 ⇒ 不可用于在线判据)", fontsize=9.5)
    ax.grid(alpha=.25, axis="y")
    fig.suptitle("T4B_06 输入速率 vs 电平门限：驱动量很强但只能事后反演，因果判据仍是电平门限",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "T4B_06_tramp_window.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> " + p)


def fig7_cost_t1():
    """T4-Q6 增量图：clean 子集 vs 全样本的 T1 口径误判代价。"""
    p_arms = os.path.join(RES, "t4b_cost_t1_arms.csv")
    p_sum = os.path.join(RES, "t4b_cost_t1_summary.csv")
    if not (os.path.isfile(p_arms) and os.path.isfile(p_sum)):
        print("!! 缺 t4b_cost_t1_*.csv（先跑 t4b_cost_with_t1.py）→ 跳过 T4B_07")
        return
    A = pd.read_csv(p_arms)
    S = pd.read_csv(p_sum)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    arms = [a for a in ("M0_now", "M2_correct", "M3_causal_thr0.40", "M4_warped", "M2_inverse")
            if a in set(A.arm)]
    x = np.arange(len(arms))
    for ax, col, ttl, unit in ((axes[0], "T_stable_med", "T_stable 中位（越低越好）", "s"),
                               (axes[1], "OS_pct_med", "超调 OS% 中位（越低越好）", "%"),
                               (axes[2], "MD_med", "最大偏差 MD 中位（越低越好）", "ADC/单位")):
        w = 0.2
        for i, (tag, mk) in enumerate((("all", ""), ("clean", "//"))):
            for j, kind in enumerate(("onset", "restep")):
                v = []
                for a in arms:
                    s = A[(A.sample == tag) & (A.arm == a) & (A.true_kind == kind)]
                    v.append(float(s[col].iloc[0]) if len(s) else np.nan)
                ax.bar(x + (i * 2 + j - 1.5) * w, v, w, hatch=mk,
                       color=("tab:blue" if kind == "onset" else "tab:red"),
                       alpha=0.65 + 0.3 * i,
                       label="%s · %s" % ("全样本" if tag == "all" else "T4-A clean", kind))
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n") for a in arms], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(alpha=.25, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("T4B_07 误判代价的 T1 口径表达（实心 = 全样本 35 事件；斜纹 = T4-A clean 子集）",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "T4B_07_cost_t1_clean.png")
    fig.savefig(p)
    plt.close(fig)
    print("-> " + p)


if __name__ == "__main__":
    fig1_separability()
    fig4_shape()
    if os.path.isfile(os.path.join(RES, "t4b_branch_confusion.csv")):
        fig2_branch_cost()
    else:
        print("!! 缺 t4b_branch_confusion.csv（先跑 t4b_q6_branch.py）→ 跳过 T4B_02")
    if os.path.isfile(os.path.join(RES, "t4b_chattering.csv")):
        fig3_chattering()
    else:
        print("!! 缺 t4b_chattering.csv（先跑 t4b_q8_chattering.py）→ 跳过 T4B_03")
    if os.path.isfile(os.path.join(RES, "t4b_loadlevels.csv")):
        fig5_loadlevel()
    else:
        print("!! 缺 t4b_loadlevels.csv（先跑 t4b_q9_loadlevel.py）→ 跳过 T4B_05")
    if os.path.isfile(os.path.join(RES, "t4b_tramp_window.csv")):
        fig6_tramp_window()
    else:
        print("!! 缺 t4b_tramp_window.csv（先跑 t4b_q5_tramp.py）→ 跳过 T4B_06")
    fig7_cost_t1()
