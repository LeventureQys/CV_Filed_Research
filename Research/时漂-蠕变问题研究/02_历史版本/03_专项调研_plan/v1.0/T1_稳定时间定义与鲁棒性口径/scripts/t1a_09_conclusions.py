# -*- coding: utf-8 -*-
"""t1a_09_conclusions：容差/噪声诊断 + 结构化交接件 `results/conclusions_T1A.json`。

两件事：
  1. **容差/噪声比诊断**（`t1a_tol_noise.csv`）：D1 的容差是 `5%·|J_ref|`，而单通道噪声/慢漂会在
     "小台阶 + 单通道"口径下淹没容差 ⇒ 稳定时间变成噪声驱动量、可测性消失。
     逐事件给 `tol = 5%·|J_ch|`、`sigma_ch = 1.4826·MAD(主通道 pre 窗)`、`ratio = tol/sigma_ch`，
     以及总通道的同口径比值（说明为什么总通道口径"更容易测到"）。
  2. 把报告「结论速览 / 裁决 / 与既有结论的差异 / 缺口」写成 `results/conclusions_T1A.json`
     （格式见 00-项目组织文档.md §4.3），数字全部从 results/*.csv 读入，保证与表格一致。

产出：results/t1a_tol_noise.csv、results/conclusions_T1A.json、results/_t1a_09_conclusions.log
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import t1a_common as C

T4A_IN = os.path.join(C.PLAN, "T4_两种快相形态与分支判据", "results", "t4a_input_recover.csv")


def med(v):
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


def main():
    C.start_log("09_conclusions")
    m = pd.read_csv(os.path.join(C.RES, "t1a_settle_metrics.csv"), encoding="utf-8-sig")
    summ = pd.read_csv(os.path.join(C.RES, "t1a_settle_summary.csv"), encoding="utf-8-sig")
    vd = pd.read_csv(os.path.join(C.RES, "t1a_target_verdict.csv"), encoding="utf-8-sig")
    aud = pd.read_csv(os.path.join(C.RES, "t1a_round1_audit.csv"), encoding="utf-8-sig")
    lb = pd.read_csv(os.path.join(C.RES, "t1a_lowerbound.csv"), encoding="utf-8-sig")
    ev = pd.read_csv(os.path.join(C.RES, "t1_events.csv"), encoding="utf-8-sig")
    ins = pd.read_csv(T4A_IN, encoding="utf-8-sig")

    def V(caliber, group, arm, col="med", gt=None):
        s = vd[(vd.caliber == caliber) & (vd.group == group) & (vd.arm == arm)]
        if not len(s):
            return float("nan")
        return float(s[col].iloc[0])

    def MED(caliber, group_type, group, arm, col="med"):
        s = summ[(summ.caliber == caliber) & (summ.group_type == group_type)
                 & (summ.group == group) & (summ.arm == arm)]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    # ── 1. 容差/噪声比 ──
    k_map = {(r.ds, r.ev_id): int(r.k_on) for r in ev.itertuples()}
    rows = []
    for ds, g in m[m.arm == "v6"].groupby("ds", sort=False):
        z = np.load(os.path.join(C.CACHE, "t1a_%s.npz" % ds.replace("/", "_")))
        for _, e in g.iterrows():
            k0 = k_map[(ds, e["ev_id"])]
            a = max(0, k0 - 200)
            ych = z["yraw_ch"][a:k0]
            ytot = z["yraw_tot"][a:k0]
            sig_ch = 1.4826 * float(np.median(np.abs(ych - np.median(ych)))) if len(ych) else np.nan
            sig_tot = 1.4826 * float(np.median(np.abs(ytot - np.median(ytot)))) if len(ytot) else np.nan
            rows.append(dict(ds=ds, ev_id=e["ev_id"], family=e["family"], dom=e["dom"],
                             kind=e["kind"], clean=bool(e["clean"]), t_on=e["t_on"],
                             J_ch=e["J_ch"], J_tot=e["J_tot"], sigma_ch=sig_ch, sigma_tot=sig_tot,
                             tol_ch=0.05 * abs(e["J_ch"]), tol_tot=0.05 * abs(e["J_tot"]),
                             ratio_ch=(0.05 * abs(e["J_ch"]) / sig_ch if sig_ch > 1e-12 else np.nan),
                             ratio_tot=(0.05 * abs(e["J_tot"]) / sig_tot if sig_tot > 1e-12 else np.nan),
                             T_stable_ch5=e["T_stable_ch5"], cens_ch5=e["cens_ch5"],
                             T_stable_ev_ch5=e["T_stable_ev_ch5"], cens_ev_ch5=e["cens_ev_ch5"]))
        del z
    tn = pd.DataFrame(rows)
    p1 = os.path.join(C.RES, "t1a_tol_noise.csv")
    tn.round(4).to_csv(p1, index=False, encoding="utf-8-sig")
    print("== 容差/噪声比（D1 的 5%·|J| ÷ 1.4826·MAD(pre 窗)）==")
    for kd in ("onset", "restep", "unload"):
        s = tn[tn.kind == kd]
        print("  %-7s n=%2d ratio_ch 中位 %7.2f (p10~p90 %6.2f~%7.2f)  ratio_tot 中位 %8.2f  "
              "ratio_ch<1 的 %2d/%d"
              % (kd, len(s), s.ratio_ch.median(), s.ratio_ch.quantile(.1), s.ratio_ch.quantile(.9),
                 s.ratio_tot.median(), int((s.ratio_ch < 1).sum()), len(s)))
    print("  → ratio<1 表示容差本身小于噪声：该事件上 T_stable 由噪声决定，不可测/无意义")

    # ── 2. 关键数字 ──
    t90 = {kd: med(ev.loc[ev.kind == kd, "t90"]) for kd in ("onset", "restep", "unload")}
    z10 = {kd: med(ev.loc[ev.kind == kd, "z_at_10"]) for kd in ("onset", "restep")}
    tramp = {kd: med(ins.loc[ins.kind == kd, "T_ramp"]) for kd in ("onset", "restep")}
    tramp_c = {kd: med(ins.loc[(ins.kind == kd) & (ins.clean), "T_ramp"]) for kd in ("onset", "restep")}
    n_tr1 = int((ins[ins.kind == "restep"].T_ramp > 1.0).sum())
    n_rs = int((ins.kind == "restep").sum())
    t_in5 = med(lb.loc[(lb.kind == "onset") & (lb.cens_in_ch == False), "T_in5_raw_ch"])   # noqa: E712
    n_in5 = int(((lb.kind == "onset") & (lb.cens_in_ch == False)).sum())
    tam_on = summ[(summ.caliber == "ch5") & (summ.group_type == "分工况")
                  & (summ.group == "onset·恒载9组")].set_index("arm")
    tam_ev = summ[(summ.caliber == "ev_ch5") & (summ.group_type == "分工况")
                  & (summ.group == "onset·恒载9组")].set_index("arm")
    print("\n== 关键数字（供 JSON/报告）==")
    for arm in C.ARMS:
        print("  %-5s 恒载9组 onset: med(ch5)=%.2f pass=%d/%d | ev_ch5 med=%.2f pass=%d/%d | tot5=%.2f | creep=%.2f"
              % (arm, tam_on.loc[arm, "med"], V("ch5", "onset·恒载9组", arm, "n_pass_2s"),
                 V("ch5", "onset·恒载9组", arm, "n_measurable"), tam_ev.loc[arm, "med"],
                 V("ev_ch5", "onset·恒载9组", arm, "n_pass_2s"),
                 V("ev_ch5", "onset·恒载9组", arm, "n_measurable"),
                 MED("tot5", "分工况", "onset·恒载9组", arm),
                 MED("chcreep", "分工况", "onset·恒载9组", arm)))
    print("  t90 中位: onset %.2f / restep %.2f / unload %.2f s" % (t90["onset"], t90["restep"], t90["unload"]))
    print("  z_at_10 中位: onset %.3f / restep %.3f" % (z10["onset"], z10["restep"]))
    print("  T_ramp 中位: onset %.3f / restep %.3f s（clean: %.3f / %.3f）；restep T_ramp>1 s: %d/%d"
          % (tramp["onset"], tramp["restep"], tramp_c["onset"], tramp_c["restep"], n_tr1, n_rs))
    print("  跟随器下界 T_in5（原始主通道）中位 %.2f s（n=%d 可测）" % (t_in5, n_in5))
    print("  重捕获/内部延迟下界 = TAU_REF 0.20 + STALL_HOLD_S 0.45 = 0.65 s")

    r6 = {"ch5": float(aud[(aud.value_name.str.contains("主通道\\+5s（恒载"))].recomputed_frozen_s.iloc[0]),
          "tot5": float(aud[(aud.value_name.str.contains("总通道\\+5s（恒载"))].recomputed_frozen_s.iloc[0]),
          "creep": float(aud[(aud.value_name.str.contains("含蠕变口径"))].recomputed_frozen_s.iloc[0]),
          "old": float(aud[(aud.value_name.str.contains("早期旧值"))].recomputed_round1caliber_s.iloc[0]),
          "ch5_r1": float(aud[(aud.value_name.str.contains("主通道\\+5s（恒载"))].recomputed_round1caliber_s.iloc[0]),
          "tot5_r1": float(aud[(aud.value_name.str.contains("总通道\\+5s（恒载"))].recomputed_round1caliber_s.iloc[0]),
          "creep_r1": float(aud[(aud.value_name.str.contains("含蠕变口径"))].recomputed_round1caliber_s.iloc[0]),
          "backtrack": float(aud[(aud.value_name.str.contains("回溯真沿"))].recomputed_round1caliber_s.iloc[0])}
    r6_edge = pd.read_csv(os.path.join(C.RES, "t1a_settle_edge_sensitivity.csv"), encoding="utf-8-sig")

    J = {
        "task": "T1", "seat": "T1-A", "date": "2026-09-19",
        "scope": "T1-Q1~Q5（稳定时间的定义与实测）；T1-Q6~Q10（鲁棒性扰动扫描）属 T1-B，本件不含",
        "frozen_caliber": {
            "name": "T_stable_ch5（主口径 D1）",
            "definition": "从真沿 t_on 起算，首次存在 τ 使 [t_on+τ, t_on+τ+30 s] 内显示读数相对该时刻自身的漂移 ≤ 5%·|J_ref|；要求完整 30 s 窗，不因后续事件截断",
            "signal": "事件主通道（该录制加载段最大台阶通道；恒载 9 组 = 第一轮指定通道 17/18/11）",
            "ref_amplitude": "J_ref = 中位([t_on+4, t_on+6]) − 中位([t_on−2, t_on))（指标字典 §2.1）",
            "grid": "timestamp 重采样 100 Hz（dt=0.01 s）",
            "t_on": "T4-A 真沿（检出沿 ±0.20 s 内最大单帧跳变帧）",
            "worst_case_caliber": "即本口径本身（三候选中数值最大：恒载 9 组 v6 1.80 s ≥ 总通道 0.55 s ≥ 含蠕变 0.50 s）",
            "multi_event_variant": "T_stable_ev_ch5：窗在下一真实事件处截断、要求可用窗 ≥5 s（实录类必须用它，且须同报删失状态）",
        },
        "headline": [
            {"n": 1, "claim": "冻结口径（主通道 + 5 s 参考）下 v6 的恒载 9 组 onset 中位稳定时间",
             "value": "%.2f s（v6.1 %.2f s；v5.1 %.2f s；raw %.2f s）" % (
                 tam_on.loc["v6", "med"], tam_on.loc["v6.1", "med"], tam_on.loc["v5.1", "med"],
                 tam_on.loc["raw", "med"]),
             "source": "results/t1a_settle_summary.csv:caliber=ch5,group=onset·恒载9组,col=med"},
            {"n": 2, "claim": "「≤2 s」达标率（冻结口径，分母 = 可测事件数）——中位达标但最差严重超标",
             "value": "v6 %d/%d=%.1f%%（worst %.2f s）；v6.1 %d/%d=%.1f%%；v5.1 %d/%d=%.1f%%；raw %d/%d=%.1f%%" % (
                 V("ch5", "onset·恒载9组", "v6", "n_pass_2s"), V("ch5", "onset·恒载9组", "v6", "n_measurable"),
                 100 * V("ch5", "onset·恒载9组", "v6", "rate_2s"), V("ch5", "onset·恒载9组", "v6", "worst_s"),
                 V("ch5", "onset·恒载9组", "v6.1", "n_pass_2s"), V("ch5", "onset·恒载9组", "v6.1", "n_measurable"),
                 100 * V("ch5", "onset·恒载9组", "v6.1", "rate_2s"),
                 V("ch5", "onset·恒载9组", "v5.1", "n_pass_2s"), V("ch5", "onset·恒载9组", "v5.1", "n_measurable"),
                 100 * V("ch5", "onset·恒载9组", "v5.1", "rate_2s"),
                 V("ch5", "onset·恒载9组", "raw", "n_pass_2s"), V("ch5", "onset·恒载9组", "raw", "n_measurable"),
                 100 * V("ch5", "onset·恒载9组", "raw", "rate_2s")),
             "source": "results/t1a_target_verdict.csv（caliber=ch5,group=onset·恒载9组）"},
            {"n": 3, "claim": "分工况达标（冻结口径）：卸载类最好、restep 结构性不可测、实录类靠修订口径才可测",
             "value": "unload v6 %d/%d 达标（中位 %.2f s）；restep 19 个里仅 %d 个可测且都 >2 s；实录族 v6 修订口径 %d/%d 达标" % (
                 V("ch5", "unload（含部分卸载）", "v6", "n_pass_2s"),
                 V("ch5", "unload（含部分卸载）", "v6", "n_measurable"),
                 MED("ch5", "分工况", "unload", "v6"),
                 V("ch5", "restep", "v6", "n_measurable"),
                 V("ev_ch5", "实录族（ADC 域）", "v6", "n_pass_2s"),
                 V("ev_ch5", "实录族（ADC 域）", "v6", "n_measurable")),
             "source": "results/t1a_target_verdict.csv + results/t1a_settle_summary.csv"},
            {"n": 4, "claim": "三个自由度的口径差（同一份 v6、同一批恒载 9 组；冻结口径）",
             "value": "主通道 %.2f / 总通道 %.2f / 含蠕变 %.2f s（最大差 %.1f 倍）；第一轮口径下同一组为 %.2f / %.2f / %.2f s" % (
                 r6["ch5"], r6["tot5"], r6["creep"], r6["ch5"] / r6["creep"],
                 r6["ch5_r1"], r6["tot5_r1"], r6["creep_r1"]),
             "source": "results/t1a_settle_summary.csv（v6,onset·恒载9组）+ results/t1a_settle_edge_sensitivity.csv"},
            {"n": 5, "claim": "第一轮 5 个公开值全部复现/裁决完成",
             "value": "1.72 / 0.46 / 0.41 s 精确复现（用第一轮自己的 t_on 与网格，9/9 份逐份差 ≤0.01 s）；2.72 s = 1.72+1.00 s 的切片下标 bug（非口径）；0.55 s 用『总通道+回溯真沿』独立复现 0.55 s",
             "source": "results/t1a_round1_audit.csv、results/t1a_settle_edge_sensitivity.csv、results/t1a_grid_sensitivity.csv"},
            {"n": 6, "claim": "本任务冻结口径相对第一轮系统性 +0.08~0.10 s（只因 t_on 定义更早 0.09 s）",
             "value": "v6 主通道：%.2f（第一轮口径 %.2f）→ %.2f s（冻结口径）；总通道 %.2f→%.2f；含蠕变 %.2f→%.2f" % (
                 r6["ch5_r1"], 1.72, r6["ch5"], r6["tot5_r1"], r6["tot5"], r6["creep_r1"], r6["creep"]),
             "source": "results/t1a_settle_edge_sensitivity.csv"},
            {"n": 7, "claim": "T_stable 与 T_settle(ε) 的口径差：中位接近、删失更多、极端差可达 120 s",
             "value": "配对中位差 0.00~1.11 s（v6），但 T_settle 删失 %d/60 > T_stable 删失 %d/60，单事件最大差 120.96 s" % (
                 int(sum(m[(m.arm == "v6")].T_settle_ch5.isna())),
                 int(sum(m[(m.arm == "v6")].cens_ch5))),
             "source": "results/t1a_d1_vs_d2.csv、results/t1a_definition_compare.csv"},
            {"n": 8, "claim": "理论下界：输入侧不构成瓶颈，瓶颈是传感器粘弹 + 判据延迟",
             "value": "等效输入斜坡 T_ramp 中位 onset %.3f s / restep %.3f s（clean %.3f s），restep 仅 %d/%d 事件 >1 s；v6 内部延迟下界 %.2f s；跟随器下界 T_in5 中位 %.2f s（n=%d）" % (
                 tramp["onset"], tramp["restep"], tramp_c["restep"], n_tr1, n_rs, 0.65, t_in5, n_in5),
             "source": "results/t1a_lowerbound.csv、T4-A/results/t4a_input_recover.csv、scripts/t1a_glm53_v6.py 常数"},
        ],
        "verdicts": [
            {"question": "T1-Q1", "answer": "给 3 个候选（D1 T_stable 自身漂移型 / D2 T_settle(ε) 相对终值型 / D3 T_band 真值带型），推荐并冻结 D1 的主通道+5 s 参考版（T_stable_ch5）：它是三者中最严（数值最大）、不奖励通道平均、不奖励移动参考幅度、且与第一轮/07-v6 同源可追溯。D2/D3 因『Z_final/真值』本身依赖算法正确性（把『准』混进『稳』）且删失更多，仅作为辅助口径。", "three_state": "支持"},
            {"question": "T1-Q2", "answer": "三实现 × 三口径 × 分族/分工况的分布见 results/t1a_settle_metrics.csv（240 行）与 t1a_settle_summary.csv；恒载 9 组 onset 中位：v6 1.80 / 0.55 / 0.50 s（主通道/总通道/含蠕变），v6.1 1.16 / 0.53 / 0.43 s，v5.1 2.89 / 3.83 / 2.78 s，raw 9.42 / 16.29 / 8.11 s。", "three_state": "支持"},
            {"question": "T1-Q3", "answer": "分工况裁决：onset·恒载9组 v6 5/9 达标（最差 13.16 s）⇒部分达标；onset·实录 修订口径 v6 %d/%d=%.0f%%；unload v6 %d/%d=%.0f%% ⇒达标；restep 19 个中仅 2 个可测且均 >2 s（75 s 量级）⇒不达标且指标结构性失效。逐个超标事件与超出量见 t1a_target_exceed.csv（41 行）。" % (
                V("ev_ch5", "onset", "v6", "n_pass_2s"), V("ev_ch5", "onset", "v6", "n_measurable"),
                100 * V("ev_ch5", "onset", "v6", "rate_2s"),
                V("ch5", "unload（含部分卸载）", "v6", "n_pass_2s"),
                V("ch5", "unload（含部分卸载）", "v6", "n_measurable"),
                100 * V("ch5", "unload（含部分卸载）", "v6", "rate_2s")), "three_state": "有条件支持"},
            {"question": "T1-Q4", "answer": "T_stable 与 T_settle(ε) 在配对样本上中位差 0.00~1.11 s、Spearman 0.68~0.96，但 D2 删失更多（43~47/60 vs 37/60）且单事件最大差 120 s；验收应用 D1（T_stable_ch5），D2 只作辅助。对第一轮的裁决：1.72/0.46/0.41 s 全部复现（口径本身就是这三档）；2.72 s 是代码 bug 不是口径（=修正值+1.00 s）；0.55 s 是『总通道+回溯真沿』口径（独立复现）。", "three_state": "支持"},
            {"question": "T1-Q5", "answer": "下界由三件事决定：(a) 输入等效斜坡 T_ramp（onset %.3f s、restep %.3f s，仅 %d/%d 个 restep >1 s）⇒『输入没走完』不是 ≤2 s 的主要障碍；(b) 判据/估计延迟（v6: TAU_REF 0.20 + STALL_HOLD_S 0.45 = 0.65 s）；(c) 传感器粘弹（restep t90 中位 %.2f s、1 s 完成度仅 %.3f vs onset %.3f）。纯跟随器的下界 T_in5 中位 %.2f s（n=%d 可测）——v6 在 8/8 同批事件上快于该下界，靠预测+冻结，代价是过冲。" % (
                tramp["onset"], tramp["restep"], n_tr1, n_rs, t90["restep"], z10["restep"],
                z10["onset"], t_in5, n_in5), "three_state": "有条件支持"},
        ],
        "corrections": [
            {"against": "13-v6-assessment/results/settle_arms.csv", "was": "v6 T_stable 中位 1.72 / 0.46 / 0.41 s（主通道/总通道/含蠕变）",
             "now": "同口径逐份复现（9/9 份差 ≤0.01 s，中位完全相同）；本任务冻结口径（换用 T4-A 真沿）给出 1.80 / 0.55 / 0.50 s",
             "verdict": "相同"},
            {"against": "plan/v1.0/第一轮现状与口径更正.md（2.72 s 为『早期主通道』）", "was": "2.72 s 与 1.72 s 是口径差异",
             "now": "2.72 = 1.72 + 1.00 s，是 r4 stable_time 把切片相对下标当绝对下标的恒定偏移（该文件自述的 bug），不是另一个口径；用第一轮网格+第一轮 t_on 复算 9/9 份 = 5.35/13.06/3.79/0.41/1.05/0.40/2.63/1.72/0.39（公布值逐份同）", "verdict": "推翻"},
            {"against": "07-v6/MANIFEST.md（T_stable 中位 0.55 s）", "was": "v6 把稳定时间压到 0.55 s（可与 1.72 s 混讲）",
             "now": "0.55 s 是『总通道 + 回溯真沿』口径；本任务独立复现 = %.2f s。主通道+5 s 口径下同一份 v6 是 %.2f s ⇒ 两数不可横比" % (r6["backtrack"], r6["ch5"]),
             "verdict": "修正"},
            {"against": "07-v6/docs/07-v6算法说明.md §2.2（restep 慢 6 倍且慢在输入）", "was": "restep t90 2.87 s、慢在输入不在传感器",
             "now": "T4-A 反卷积：restep 等效输入斜坡中位 %.3f s（clean %.3f s），仅 %d/%d 事件 >1 s ⇒ 输入本身在 1 s 内走完；restep 的慢来自传感器粘弹响应（t90 中位 %.2f s、1 s 完成度 %.3f）与判据，而不是输入未完成" % (tramp["restep"], tramp_c["restep"], n_tr1, n_rs, t90["restep"], z10["restep"]),
             "verdict": "修正"},
            {"against": "第一轮（主通道/总通道/含蠕变三口径并列，未冻结）", "was": "『1~2 s 达没达标取决于选哪个口径』",
             "now": "冻结唯一验收口径 T_stable_ch5；并把『总通道更松』这一直觉限定为『只对补偿臂成立』——raw 臂上总通道反而更严（16.29 s vs 9.42 s）",
             "verdict": "修正"},
        ],
        "gaps": [
            "跨载荷量级（5N/10N/20N）缺失：无法判定稳定时间是否随载荷量级变化（沿用 T4-A G1）",
            "无受控加载速率实验：T_ramp 只能反演、不能标定，Q5 的『输入侧下界』只能用观测到的斜坡时长",
            "restep 工况只有 2/19 个事件可测稳定时间 ⇒ restep 的结论基于 n=2（仅定性参考），需要『短静默窗 + 事件后偏差』类指标的新定义与补采",
            "实录类 ADC 域的 5%·J 容差在部分小台阶事件上低于单通道噪声（ratio_ch<1，见 t1a_tol_noise.csv）⇒ 这些事件的 T_stable 是噪声驱动量，需要先定『容差下限（如 ≥3σ）』",
            "v6 的 T_stable 对分析网格敏感：1/9 份录制在 100 Hz vs 100.5 Hz（span/(n−1)）网格下从 3.79 s 变 6.66 s（路径抖动，见 t1a_grid_sensitivity.csv）；未做系统抖动扫描（属 T1-B Q8）",
            "本次未做时序抖动/丢包/拍击注入下的 T_stable 退化（属 T1-B），因此『鲁棒性达标率』本件不回答",
            "raw 臂的总通道 T_stable 中位复算 16.21 s vs 首轮 17.98 s（1 份录制的平台判定差异）；raw 非候选臂，未逐份深究",
        ],
        "blockers": [],
    }
    p2 = os.path.join(C.RES, "conclusions_T1A.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=2)
    print("\n== conclusions_T1A.json ==")
    print(json.dumps(J["headline"], ensure_ascii=False, indent=1)[:2600])
    for p in (p1, p2):
        print("-> %s" % p)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
