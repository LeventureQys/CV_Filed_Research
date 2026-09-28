# -*- coding: utf-8 -*-
"""t4b_report_tables.py —— T4-B 报告用汇总表（从各 results/*.csv 直接生成，只做汇总不重算）。

产物：
  results/t4b_headline.csv       —— 报告"结论速览"里每一个数字的出处（结论号/指标/数值/文件/列）
  results/t4b_cost_table.csv     —— 误判代价紧凑表（每臂 × 真值类别）
  results/t4b_crit_decision.csv  —— 各候选判据的"是否可用"裁决表（可分性 + 所需 Δ + 是否可落地）

运行：python scripts/t4b_report_tables.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402

RES = C.RES


def main():
    rows = []

    def add(concl, metric, value, src, col, note=""):
        rows.append(dict(conclusion=concl, metric=metric, value=value,
                         source_file=src, column=col, note=note))

    # ── 判据可分性 ──
    sep = pd.read_csv(os.path.join(RES, "t4b_separability.csv"))
    for cid, concl in (("P1_pre_frac", "C1"), ("P1b_pre_frac_frozenV", "C1"),
                       ("P2_slope@0.1s", "C2"), ("P3_J_over_pre@0.2s", "C2"),
                       ("P4_singleframe@020", "C3"), ("P4_shape020", "C3"),
                       ("P2_dur@0.20s", "C3")):
        r = sep[sep.crit == cid]
        if not len(r):
            continue
        r = r.iloc[0]
        add(concl, cid + ".auc_eff", r.auc_eff, "results/t4b_separability.csv", "auc_eff")
        add(concl, cid + ".balanced_acc", r.balanced_acc,
            "results/t4b_separability.csv", "balanced_acc")
        add(concl, cid + ".lag_s", r.lag_s, "results/t4b_separability.csv", "lag_s")
        add(concl, cid + ".p10p90_overlap", r.p10p90_overlap,
            "results/t4b_separability.csv", "p10p90_overlap")
    for t in ("10%", "20%", "30%", "40%"):
        r = sep[sep.crit == "P1_pre_frac@" + t]
        if len(r):
            add("C1", "P1 fixed-threshold acc @" + t, float(r.iloc[0].acc),
                "results/t4b_separability.csv", "acc")

    # ── 两形态形状 ──
    sh = pd.read_csv(os.path.join(RES, "t4b_shape_by_arm.csv"))
    for tau in (0.05, 0.1, 0.2, 0.5, 1.0):
        r = sh[np.isclose(sh.tau, tau)]
        if len(r):
            r = r.iloc[0]
            add("C3", "shape@%.2fs med_onset" % tau, round(float(r.med_on), 4),
                "results/t4b_shape_by_arm.csv", "med_on")
            add("C3", "shape@%.2fs med_restep" % tau, round(float(r.med_re), 4),
                "results/t4b_shape_by_arm.csv", "med_re")
            add("C3", "shape@%.2fs d_pt" % tau, round(float(r.d_pt), 1),
                "results/t4b_shape_by_arm.csv", "d_pt")
            add("C3", "shape@%.2fs auc" % tau, round(float(r.auc), 4),
                "results/t4b_shape_by_arm.csv", "auc")
            add("C3", "shape@%.2fs p10p90_overlap" % tau, round(float(r.overlap), 4),
                "results/t4b_shape_by_arm.csv", "overlap")

    # ── 事件清单/形态表 ──
    ma = pd.read_csv(os.path.join(RES, "t4b_morphology_by_arm.csv"))
    add("C0", "n_events_total", len(ma), "results/t4b_morphology_by_arm.csv", "kind")
    add("C0", "n_onset", int((ma.kind == "onset").sum()),
        "results/t4b_morphology_by_arm.csv", "kind")
    add("C0", "n_restep", int((ma.kind == "restep").sum()),
        "results/t4b_morphology_by_arm.csv", "kind")
    add("C0", "n_decrement+partial_unload",
        int(ma.kind.isin(["decrement", "partial_unload"]).sum()),
        "results/t4b_morphology_by_arm.csv", "kind")

    # ── 建议判据（θ=0.40）的事件级混淆矩阵 + 逐事件反事实代价 ──
    ma2 = pd.read_csv(os.path.join(RES, "t4b_morphology_by_arm.csv"))
    rise = ma2[ma2.kind.isin(["onset", "restep"])].copy()
    THR = 0.40
    rise["pred_onset"] = rise.pre_frac <= THR
    cm = []
    for true_kind in ("onset", "restep"):
        s = rise[rise.kind == true_kind]
        cm.append(dict(theta=THR, true_kind=true_kind, n=len(s),
                       pred_onset=int(s.pred_onset.sum()),
                       pred_restep=int((~s.pred_onset).sum()),
                       n_misbranch=int((s.pred_onset != (true_kind == "onset")).sum())))
    cm = pd.DataFrame(cm)
    cm.to_csv(os.path.join(RES, "t4b_branch_confmat_theta040.csv"), index=False,
              encoding="utf-8-sig")
    print("\n== 建议判据 P1（θ=0.40）的事件级混淆矩阵 ==")
    print(cm.to_string(index=False))

    am = pd.read_csv(os.path.join(RES, "t4b_arm_metrics.csv"))
    am = am[am.arm.isin(["M2_correct", "M2_inverse"])]
    piv = am.pivot_table(index=["rec", "t_on", "kind"], columns="arm",
                         values=["plat_dev_adc", "plat_mae_abs_pct", "md_adc", "T_stable"])
    piv = piv.dropna()
    piv["d_plat_adc"] = piv[("plat_dev_adc", "M2_inverse")] - piv[("plat_dev_adc", "M2_correct")]
    piv["d_mae_pct"] = piv[("plat_mae_abs_pct", "M2_inverse")] - piv[("plat_mae_abs_pct", "M2_correct")]
    piv["d_md_adc"] = piv[("md_adc", "M2_inverse")] - piv[("md_adc", "M2_correct")]
    piv["d_T"] = piv[("T_stable", "M2_inverse")] - piv[("T_stable", "M2_correct")]
    piv = piv.reset_index()
    piv.to_csv(os.path.join(RES, "t4b_cost_per_event_reverse.csv"), index=False,
               encoding="utf-8-sig")
    print("\n== 逐事件上界（完全反着分 vs 真值分支）==")
    print(piv[["rec", "t_on", "kind", "d_plat_adc", "d_mae_pct", "d_md_adc", "d_T"]]
          .to_string(index=False))

    # ── 抖动/迟滞 ──
    ch = pd.read_csv(os.path.join(RES, "t4b_chattering_hyst.csv"))
    ex = []
    for th in sorted(ch.theta.unique()):
        for h in sorted(ch.hyst.unique()):
            c0 = ch[(ch.A == 0) & (ch.theta == th) & (ch.hyst == h)].cross_per_ev.iloc[0]
            c4 = ch[(ch.A == 400) & (ch.theta == th) & (ch.hyst == h)].cross_per_ev.iloc[0]
            e4 = ch[(ch.A == 400) & (ch.theta == th) & (ch.hyst == h)].final_err_rate.iloc[0]
            ex.append(dict(theta=th, hyst=h, cross_A0=c0, cross_A400=c4,
                           excess=round(c4 - c0, 4), final_err_A400=e4))
    ex = pd.DataFrame(ex)
    ex.to_csv(os.path.join(RES, "t4b_chattering_excess.csv"), index=False,
              encoding="utf-8-sig")
    for _, r in ex.iterrows():
        add("C4", "excess_cross θ=%.2f h=%.2f A=400" % (r.theta, r.hyst), r.excess,
            "results/t4b_chattering_excess.csv", "excess")
        add("C4", "final_err_A400 θ=%.2f h=%.2f" % (r.theta, r.hyst), r.final_err_A400,
            "results/t4b_chattering_excess.csv", "final_err_A400")
    pe = pd.read_csv(os.path.join(RES, "t4b_chattering_per_event.csv"))
    for kind in ("onset", "restep"):
        s = pe[pe.kind == kind]
        add("C4", "s_std@400 %s" % kind, round(float(s.s_400_std.median()), 4),
            "results/t4b_chattering_per_event.csv", "s_400_std")

    # ── 时序裕度 ──
    mg = pd.read_csv(os.path.join(RES, "t4b_dbg_margin.csv"))
    for off in ("+000", "+020", "+040", "+060", "+100"):
        c = "s_off_%s" % off
        if c not in mg.columns:
            continue
        for kind in ("onset", "restep"):
            add("C5", "s_off%s %s" % (off, kind),
                round(float(mg[mg.kind == kind][c].median()), 4),
                "results/t4b_dbg_margin.csv", c)

    # ── 分支代价（若已生成）──
    pc = os.path.join(RES, "t4b_branch_cost.csv")
    if os.path.isfile(pc):
        cost = pd.read_csv(pc)
        cost.to_csv(os.path.join(RES, "t4b_cost_table.csv"), index=False,
                    encoding="utf-8-sig")
        for _, r in cost.iterrows():
            for col in ("dT_med", "dT_absmax", "dOS_med", "dOS_absmax",
                        "dPlatAbs_med", "dPlatAbs_absmax", "dMAE_abs_med", "dMAE_abs_max",
                        "dADC_med", "dADC_absmax", "dMD_med", "dMD_absmax"):
                add("C6", "%s.%s" % (r.arm, col), r[col], "results/t4b_branch_cost.csv", col,
                    note="true_kind=%s n=%d" % (r.true_kind, r.n))

    # ── 判据裁决表 ──
    crit = [
        dict(crit="① pre 电平门限（P1）", lag_need_s=0.0, auc_eff=float(
            sep[sep.crit == "P1_pre_frac"].iloc[0].auc_eff),
             balanced_acc=float(sep[sep.crit == "P1_pre_frac"].iloc[0].balanced_acc),
             verdict="**推荐**：可分且 Δ=0，不需要等任何数据", note="需冻结门限值 V，见 C4/C5"),
        dict(crit="① pre 电平门限（冻结 V，P1b）",
             lag_need_s=0.0,
             auc_eff=float(sep[sep.crit == "P1b_pre_frac_frozenV"].iloc[0].auc_eff),
             balanced_acc=float(sep[sep.crit == "P1b_pre_frac_frozenV"].iloc[0].balanced_acc),
             verdict="**推荐（工程实现口径）**", note="与 P1 同 AUC，且可在线实现"),
        dict(crit="② 上升沿斜率（Δ=0.10 s）", lag_need_s=0.10,
             auc_eff=float(sep[sep.crit == "P2_slope@0.1s"].iloc[0].auc_eff),
             balanced_acc=float(sep[sep.crit == "P2_slope@0.1s"].iloc[0].balanced_acc),
             verdict="可用但比 ① 差；仅在拿不到 pre 电平的历史时用",
             note="与 ① 高度共线（大台阶多来自零基线）"),
        dict(crit="③ 台阶幅度/电平比（Δ=0.20 s）", lag_need_s=0.20,
             auc_eff=float(sep[sep.crit == "P3_J_over_pre@0.2s"].iloc[0].auc_eff),
             balanced_acc=float(sep[sep.crit == "P3_J_over_pre@0.2s"].iloc[0].balanced_acc),
             verdict="可用；但需 0.20 s 数据，比 ① 慢 0.20 s", note=""),
        dict(crit="④ 上升沿形状（单帧跃变占比，Δ=0.20 s）", lag_need_s=0.20,
             auc_eff=float(sep[sep.crit == "P4_singleframe@020"].iloc[0].auc_eff),
             balanced_acc=float(sep[sep.crit == "P4_singleframe@020"].iloc[0].balanced_acc),
             verdict="**不建议**：AUC 0.62、p10~p90 重叠 0.42，样本级不可分",
             note="两形态的形状差异是分布性的，不是单体可判的"),
        dict(crit="③ 上升沿时长（0.9·J(Δ)=Δt90，Δ=0.20 s）", lag_need_s=0.20,
             auc_eff=float(sep[sep.crit == "P2_dur@0.20s"].iloc[0].auc_eff),
             balanced_acc=float(sep[sep.crit == "P2_dur@0.20s"].iloc[0].balanced_acc),
             verdict="**不建议**：Δ 小时大量 NaN（Δ 内走不到 90%）",
             note="实测 Δ=0.20 s 时 p10~p90 重叠 0.67"),
    ]
    pd.DataFrame(crit).to_csv(os.path.join(RES, "t4b_crit_decision.csv"), index=False,
                              encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(os.path.join(RES, "t4b_headline.csv"), index=False,
                             encoding="utf-8-sig")
    print("-> results/t4b_headline.csv (%d 行) / t4b_cost_table.csv / "
          "t4b_crit_decision.csv / t4b_chattering_excess.csv" % len(rows))
    print(pd.DataFrame(crit)[["crit", "lag_need_s", "auc_eff", "balanced_acc", "verdict"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
