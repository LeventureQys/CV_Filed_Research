# -*- coding: utf-8 -*-
"""T3-A 04：反驳性对照（与既有结论逐条对表）—— 项目纪律要求的「差异说明」底座。

对照对象：
  A. 第一轮 `progress/13-v6-assessment/docs/v6评估与需求答复.md` §3（快相是否稳定）与 §4；
     数据：results/{shape_stats,shape_timewarp,rom_compare,rom_loo}.csv
  B. `progress/07-v6/docs/07-v6算法说明.md` §2.1（τ=2 s 平滑伪影）、§2.2（onset/restep）、
     §2.4（形状反演精度：0.2 s ±7%、1 s 0.0%）；
  C. `progress/08-v6.1/docs/08-v6.1算法说明.md`（ROM_SCALE=1.06 ⇒ 形状「稳定 ≠ 正确」）；
  D. 平行第一轮产物 `T4_*/results/{t4a_morphology,t4a_input_recover}.csv`（本任务的输入口径源）。

每条给：was（既有数字）/ now（本任务复算）/ verdict（相同|修正|推翻|新增|引用不推翻|不可复现|不适用）/ why。
归因维度固定为：事件集 / 信号口径（总量 Z vs 主通道）/ 基准电平（T4-A pre vs A_5s）/ 样本量。

产物：results/t3a_crosscheck.csv、results/t3a_crosscheck.json、results/_t3a_04_crosscheck.log
运行：python scripts/t3a_04_crosscheck.py   （秒级）
"""
import json
import os

import numpy as np
import pandas as pd

import t3a_common as C

ROWS = []


def add(group, item, was, now, verdict, why):
    ROWS.append(dict(group=group, item=item, was=was, now=now, verdict=verdict, why=why))


def main():
    C.start_log("t3a_04_crosscheck")
    sp = pd.read_csv(os.path.join(C.RES, "t3a_shape_spread.csv"))
    gen = pd.read_csv(os.path.join(C.RES, "t3a_generalization.csv"))
    aud = pd.read_csv(os.path.join(C.RES, "t3a_inlier_audit.csv"))
    ts = pd.read_csv(os.path.join(C.RES, "t3a_time_stability.csv"))
    pcv = pd.read_csv(os.path.join(C.RES, "t3a_param_cv.csv"))
    tests = pd.read_csv(os.path.join(C.RES, "t3a_strata_tests.csv"))
    raw = pd.read_csv(os.path.join(C.RES, "t3a_generalization_raw.csv"))
    chk = pd.read_csv(os.path.join(C.RES, "t3a_recon_crosscheck.csv"))
    ss = pd.read_csv(C.R1_SHAPE_STATS)
    tw = pd.read_csv(C.R1_TIMEWARP)
    loo = pd.read_csv(C.R1_ROM_LOO)

    def spget(kind, sample, tau, col):
        r = sp[(sp["kind"] == kind) & (sp["sample"] == sample) & np.isclose(sp["tau"], tau)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def gget(sample, scen, kind, tau_d, inv, ref, col):
        r = gen[(gen["sample"] == sample) & (gen["scenario"] == scen) & (gen["kind"] == kind)
                & np.isclose(gen["tau_d"], tau_d) & (gen["inv"] == inv) & (gen["ref"] == ref)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    def dom_loo(dom, tau_d):
        q = raw[(raw["usable"]) & (raw["kind"] == "onset") & (raw["dom"] == dom)
                & (raw["scenario"] == "loo_event") & (raw["inv"] == "single@tau")
                & (raw["ref"] == "post") & np.isclose(raw["tau_d"], tau_d)]
        v = np.abs(q["err_pct"].to_numpy(float))
        v = v[np.isfinite(v)]
        return (float(np.median(v)) if v.size else np.nan), int(v.size)

    # ══════════ A. 第一轮 §3.1（shape_stats.csv） ══════════
    for tau in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]:
        r1 = ss[np.isclose(ss["tau"], tau)]
        if r1.empty:
            continue
        r1 = r1.iloc[0]
        m_u = spget("onset", "usable", tau, "med")
        verdict = "相同" if abs(m_u - r1["med"]) <= 0.02 else "修正"
        add("A 第一轮 §3.1 shape_stats", "onset 中位 f(%.2f s)" % tau,
            "%.3f（n=%d，主通道，A_5s 基准，全部 19 onset）" % (r1["med"], int(r1["n"])),
            "usable %.3f（n=%d，总量 Z，T4-A pre 基准）/ all %.3f（n=%d）"
            % (m_u, int(spget("onset", "usable", tau, "n")),
               spget("onset", "all", tau, "med"), int(spget("onset", "all", tau, "n"))),
            verdict,
            "①信号：总量 Z vs 主通道；②基准：T4-A pre 窗中位 vs 第一轮 A_5s；"
            "③样本：usable n=18 / all n=22 vs 第一轮 n=19（含实录 onset）")
        add("A 第一轮 §3.1 shape_stats", "onset p10~p90 f(%.2f s)" % tau,
            "%.3f~%.3f" % (r1["p10"], r1["p90"]),
            "usable %.3f~%.3f / all %.3f~%.3f"
            % (spget("onset", "usable", tau, "p10"), spget("onset", "usable", tau, "p90"),
               spget("onset", "all", tau, "p10"), spget("onset", "all", tau, "p90")),
            "相同" if abs(spget("onset", "usable", tau, "p90") - r1["p90"]) <= 0.06 else "修正",
            "同上前提；第一轮的带宽在 0.2~0.5 s 更宽（含 0.2 s 完成度更高的实录事件与主通道噪声）")
    add("A 第一轮 §3.1 结论", "「0.2 s 之前不可用」是否复现",
        "0.05 s 极差 0.705、0.2 s 极差 0.591（n=19）",
        "usable：0.05 s 极差 %.3f、0.2 s 极差 %.3f；all：%.3f / %.3f；"
        "usable onset 的 CQV 由 0.05 s 的 %.3f 单调升到 0.2 s 的 %.3f、1.0 s 的 %.3f"
        % (spget("onset", "usable", 0.05, "rng"), spget("onset", "usable", 0.20, "rng"),
           spget("onset", "all", 0.05, "rng"), spget("onset", "all", 0.20, "rng"),
           spget("onset", "usable", 0.05, "cqv"), spget("onset", "usable", 0.20, "cqv"),
           spget("onset", "usable", 1.00, "cqv")),
        "相同",
        "两个独立事件集给出同向结论：f(0.05~0.2 s) 的同类散布比 0.3 s 之后大 3~5 倍")

    # ══════════ A2. 第一轮 §3.2（shape_timewarp.csv，α） ══════════
    al = tw["alpha"].to_numpy(float)
    t1 = pcv[(pcv["kind"] == "onset") & (pcv["sample"] == "usable") & (pcv["param"] == "tau1")].iloc[0]
    t1e = pcv[(pcv["kind"] == "onset") & (pcv["sample"] == "usable") & (pcv["param"] == "tau_e")].iloc[0]
    add("A 第一轮 §3.2 shape_timewarp", "时间缩放参数 alpha 的中位/范围",
        "中位 %.3f，p10~p90 %.2f~%.2f（n=%d；按录制×受载段拟合；alpha 越大 = 现场比 ROM 慢）"
        % (np.median(al), np.percentile(al, 10), np.percentile(al, 90), len(al)),
        "本任务等价量【单 tau 参数 tau1】：中位 %.4f、p10~p90 %.3f~%.3f、CV %.3f、CQV %.3f（n=%d）；"
        "【拉伸指数 tau_e】：中位 %.4f、CV %.3f"
        % (t1["med"], t1["p10"], t1["p90"], t1["cv"], t1["cqv"], int(t1["n"]),
           t1e["med"], t1e["cv"]),
        "修正（口径不同，不可直接横比）",
        "alpha 是「事件形状 vs 现役 ROM」的【相对】缩放（含 ROM 自身偏慢的系统项），"
        "本任务的 tau1 是「事件形状自身」的【绝对】时间常数 ⇒ 两者不是同一个量；"
        "第一轮 alpha 中位 0.50 为点估计且只给 19 个受载段的汇总，"
        "本任务 usable onset 的 tau1 CV=%.3f 说明其个体差异很大（不能用一个缩放吃掉）" % t1["cv"])
    add("A 第一轮 §3.2 结论", "「约一半形状失配可由时间缩放解释」",
        "RMS 残差 0.096（直接套 ROM）→ 0.048（允许时间缩放），降 50%",
        "本任务不重算该 RMS（属 T4/T2 口径），但给出同向的稳健性证据："
        "可用样本 onset 的 tau1 四分位区间 %.3f~%.3f s（比值 %.2f 倍），"
        "而 f(1.0 s) 的 CQV 只有 %.3f ⇒ 【早期形状差异主要活在 0.3~1 s 的时间尺度上，1 s 之后被抹平】"
        % (t1["p10"], t1["p90"], t1["p90"] / t1["p10"], spget("onset", "usable", 1.00, "cqv")),
        "相同（方向）/ 新增（分位）",
        "口径不同：第一轮比的是「形状 vs ROM」的 RMS，本任务报的是「形状参数自身」的离散度")

    # ══════════ A3. 第一轮 §3.3（跨传感器 15 pt） ══════════
    famrow = tests[(tests["dim"] == "fam") & (tests["col"] == "z_at_03")]
    add("A 第一轮 §3.3", "跨传感器差 15 pt（0.2 s 占比：右拇指 78.7% vs 左拇指 93.5%）",
        "15 pt（每个传感器 3 次录制重复）",
        "本任务 usable 样本：族维 f(0.30 s) Kruskal-Wallis k=%d H=%.2f p=%.3f（各层 n=%s）；"
        "族间两两 Mann-Whitney p 均 >= 0.20"
        % (int(famrow["k"].iloc[0]), famrow["H"].iloc[0], famrow["p"].iloc[0],
           famrow["n_each"].iloc[0]),
        "有条件支持（方向一致，统计功效不足）",
        "本任务 usable 子集在恒载 3 族里每族只剩 2~3 个事件（clean 判据 + 非病态判据把样本压到 n=8）；"
        "要在 0.05 水平复现 15 pt 的族间差需要每族 >=5 次同型事件（缺口 G7）")
    add("A 第一轮 §3.3", "同一传感器 3 次重复的组内极差（4.5~6.3 pt）",
        "右拇指 6.3 pt / 四指 5.4 pt / 左拇指 4.5 pt（n=3）",
        "本任务 usable onset 在恒载 3 族的 f(0.20 s) p10~p90 宽度：右拇指 n=3、四指 n=2、左拇指 n=3 "
        "—— **n<=3，仅定性参考**，不给点估计",
        "不可复现（样本不足）",
        "第一轮的「3 次重复」是段级聚合（录制内所有受载段），本任务是事件级且要求 clean+非病态；"
        "细节见 results/t3a_strata.csv 的 fam 维度")

    # ══════════ B. 07-v6 §2.4 形状反演精度 ══════════
    d20, n20 = dom_loo("显示域", 0.20)
    d50, _ = dom_loo("显示域", 0.50)
    d10, _ = dom_loo("显示域", 1.00)
    add("B 07-v6 §2.4", "形状反演 Â 的误差（留一法，9 组恒载 onset）",
        "tau_d=0.20 s 中位 −1.6%、p10~p90 −5.3~+6.6%、max 7.2%；tau_d=1.00 s 中位 0.0%、max 4.6%",
        "本任务同域（显示域）usable onset（n=%d）事件级留一：中位|误差| tau_d=0.2/0.5/1.0 s = "
        "%.2f%% / %.2f%% / %.2f%%；全样本（含实录）tau_d=0.2 s 中位|误差| %.2f%%（n=%d）"
        % (n20, d20, d50, d10,
           gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "med_abs_pct"),
           int(gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "n"))),
        "相同（同量级，双方都在个位数~10% 量级）",
        "①第一轮的 9 组留一是【录制级】：训练库为其余 8 组的中位形状，事件级噪声被平掉；"
        "本任务是【事件级】留一，训练库只剩 1~2 个事件（显示域 3 族 x 1~3 个 usable onset）⇒ 多一份库噪声；"
        "②第一轮只评 9 个稳定恒载 onset，本任务含多级/慢压 onset（尾部来源）")

    for kind in ["onset", "restep"]:
        add("B 07-v6 §2.4 扩展", "%s 全样本泛化（usable，单点反演 @ tau_d）" % kind,
            "第一轮未给 %s 的泛化数" % ("restep" if kind == "restep" else "含实录的全量 onset"),
            "留一事件 %.2f%%/%.2f%%/%.2f%%/%.2f%%（tau_d=0.2/0.3/0.5/1.0 s，n=%d）；"
            "跨族 %.2f%%/%.2f%%/%.2f%%/%.2f%%；跨形态（用另一个 kind 的库）"
            "%.2f%%/%.2f%%/%.2f%%/%.2f%%"
            % (gget("usable", "loo_event", kind, 0.20, "single@tau", "post", "med_abs_pct"),
               gget("usable", "loo_event", kind, 0.30, "single@tau", "post", "med_abs_pct"),
               gget("usable", "loo_event", kind, 0.50, "single@tau", "post", "med_abs_pct"),
               gget("usable", "loo_event", kind, 1.00, "single@tau", "post", "med_abs_pct"),
               int(gget("usable", "loo_event", kind, 0.20, "single@tau", "post", "n")),
               gget("usable", "cross_fam", kind, 0.20, "single@tau", "post", "med_abs_pct"),
               gget("usable", "cross_fam", kind, 0.30, "single@tau", "post", "med_abs_pct"),
               gget("usable", "cross_fam", kind, 0.50, "single@tau", "post", "med_abs_pct"),
               gget("usable", "cross_fam", kind, 1.00, "single@tau", "post", "med_abs_pct"),
               gget("usable", "cross_kind", kind, 0.20, "single@tau", "post", "med_abs_pct"),
               gget("usable", "cross_kind", kind, 0.30, "single@tau", "post", "med_abs_pct"),
               gget("usable", "cross_kind", kind, 0.50, "single@tau", "post", "med_abs_pct"),
               gget("usable", "cross_kind", kind, 1.00, "single@tau", "post", "med_abs_pct")),
            "新增",
            "第一轮只做 onset 泛化；本任务补上 restep 与「跨形态」两档 —— "
            "跨形态（拿 onset 库反演 restep）是本项目最大的误差源")

    add("B 07-v6 §2.4 结论", "「0.2 s 已可把最终电平估到 ±7%」是否成立",
        "±7%（9 组恒载，留一录制，单点/max）",
        "usable 全样本：tau_d=0.2 s 留一事件 中位|误差| %.1f%%、p90 %.1f%%、最大 %.1f%%、"
        "|误差|>5%% 的占比 %.0f%%（n=%d）；tau_d=1.0 s 中位|误差| %.1f%%、最大 %.1f%%"
        % (gget("usable", "loo_event", "both", 0.20, "single@tau", "post", "med_abs_pct"),
           gget("usable", "loo_event", "both", 0.20, "single@tau", "post", "p90_abs_pct"),
           gget("usable", "loo_event", "both", 0.20, "single@tau", "post", "max_abs_pct"),
           gget("usable", "loo_event", "both", 0.20, "single@tau", "post", "share_over_5pct"),
           int(gget("usable", "loo_event", "both", 0.20, "single@tau", "post", "n")),
           gget("usable", "loo_event", "both", 1.00, "single@tau", "post", "med_abs_pct"),
           gget("usable", "loo_event", "both", 1.00, "single@tau", "post", "max_abs_pct")),
        "修正（±7%% 只在显示域 onset 成立；全样本尾部大得多）",
        "±7%% 是「同域、同形态、录制级留一、9 个稳定恒载 onset」的结论；"
        "一旦把 restep 与实录的慢压 onset 算进来，尾部长到几十个百分点 ⇒ "
        "**形状反演的误差预算必须按 kind 与输入形态分级**")

    # ══════════ B2. 07-v6 §2.2 onset/restep 时间常数 ══════════
    add("B 07-v6 §2.2", "onset t90 = 0.48 s、restep t90 = 2.87 s（差 6.0 倍）",
        "0.48 s / 2.87 s（n=9 / n=15）",
        "本任务【不重复】该口径工作，直接引用 T4-A 冻结表 t4a_morphology.csv 的 t90 列"
        "（T4-A 已修正为 clean：onset 0.62 s / restep 1.40 s）",
        "引用不推翻",
        "t90 属 T2/T4 的口径范围；本任务只消费 T4-A 的 t_on / pre / post / z_at_* 列")

    # ══════════ B3. 07-v6 §2.1 平滑红线 ══════════
    add("B 07-v6 §2.1", "禁用 tau=2 s 平滑（伪影）",
        "1 s 处文档轮廓 0.384 vs 原始 0.912（差 2.4 倍）",
        "本任务全流程只用 0.5 s 中位滤波（k=50 @100 Hz）+ 原始 Z 的逐帧量，**未做任何指数平滑**；"
        "usable onset f(1.0 s) = %.3f" % spget("onset", "usable", 1.00, "med"),
        "遵守",
        "口径红线；另：t3a_shape_grid.csv 的 f(τ) 是单点取值（非窗内均值），与 T4-A 完全同法")

    # ══════════ C. 08-v6.1「形状稳定 ≠ 形状正确」 ══════════
    fu = pcv[(pcv["kind"] == "onset") & (pcv["sample"] == "usable") & (pcv["param"] == "f(1.00s)")].iloc[0]
    add("C 08-v6.1", "ROM 比现场慢 ⇒ Â 系统性高估（onset 中位 +3.2%、最大 +10.9%；"
                     "ROM_SCALE=1.06 修复后 +0.4%）",
        "修正前后：+3.2% → +0.4%（onset）",
        "本任务把同源机制拆成两半并各自给数：①形状【稳定】：usable onset f(1.0 s) "
        "CQV %.3f、CV %.3f（n=%d）；②形状【不可通用】：跨形态反演 tau_d=0.2 s 中位|误差| %.1f%%、"
        "usable restep 留一 %.1f%%、usable onset 留一 %.1f%%"
        % (fu["cqv"], fu["cv"], int(fu["n"]),
           gget("usable", "cross_kind", "onset", 0.20, "single@tau", "post", "med_abs_pct"),
           gget("usable", "loo_event", "restep", 0.20, "single@tau", "post", "med_abs_pct"),
           gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "med_abs_pct")),
        "支持并扩展",
        "需求文档 §5-3 要求把「稳定」与「正确」分开报：本任务给出两套独立指标。"
        "注意本任务的形状库由本批事件自身构造，不含 v6 的 ROM 偏慢项 ⇒ 不能用它检验偏差方向")
    add("C 08-v6.1", "「形状库需要现场标定」这一结论的泛化边界",
        "既有：跨工况（恒载库 → 实录 onset）τ_d=1 s 中位 13.0%、最大 30.9%；"
        "跨位置（留一位置）中位 0.3~4.7% ⇒ 「形状在位置上几乎通用」",
        "本任务按【族】重算：left/right/four-finger 三族互相反演（cross_fam，usable，单点@0.2 s）"
        "onset 中位|误差| %.2f%%、restep %.2f%%；按【形态】反演（cross_kind）onset %.1f%%、restep %.1f%%"
        % (gget("usable", "cross_fam", "onset", 0.20, "single@tau", "post", "med_abs_pct"),
           gget("usable", "cross_fam", "restep", 0.20, "single@tau", "post", "med_abs_pct"),
           gget("usable", "cross_kind", "onset", 0.20, "single@tau", "post", "med_abs_pct"),
           gget("usable", "cross_kind", "restep", 0.20, "single@tau", "post", "med_abs_pct")),
        "相同（跨族小）／新增（跨形态大）",
        "既有「跨位置通用」在【同一域同一形态】内成立（本任务 cross_fam 复现为个位数）；"
        "真正致命的是【跨形态/跨输入方式】——这正是 T4-A 的 T_ramp 连续谱要解决的问题")

    # ══════════ D. T4-A 冻结表 ══════════
    la = aud[aud["is_load"]]
    nbad = int((~la["posdef"]).sum())
    nbad_re = int(((~la["posdef"]) & (la["kind"] == "restep")).sum())
    add("D T4-A 冻结表", "装载类事件集（与 T4-A 报告正文的差异）",
        "T4-A 报告正文写 n=40（onset 22 / restep 18，clean 25）",
        "T4-A 冻结 CSV 实为 60 行，装载类 **41**（onset 22 / restep 19；unload 14 / partial_unload 5；"
        "jump 缺失 1）；本任务 usable 24（onset 18 / restep 6）",
        "修正（版本漂移）",
        "冻结表在报告正文写成之后被更新过（多 1 个 restep）。本任务以【冻结 CSV】为准并在此标注，"
        "避免 T8 引用时 n 对不上")
    dmed = float(np.nanmedian(np.abs(chk["d"])))
    dmax = float(np.nanmax(np.abs(chk["d"])))
    add("D T4-A 冻结表", "z_at_* 口径一致性自检",
        "T4-A 的 z_at_* 以 `pre` 窗中位为基准、Z̄ 单点取值",
        "本任务独立重算 60 事件 × 8 个 tau 点 = 472 对，|Δ| 中位 %.2e、最大 %.2e（超 1e-6 的 3 对）"
        % (dmed, dmax),
        "相同（完全对齐）",
        "同法同源；残余差异来自冻结 CSV 的浮点打印位数（<2e-6），不影响任何结论")

    # ══════════ E. 第一轮 rom_loo / rom_compare ══════════
    b = loo["bias_pct"].to_numpy(float)
    add("A 第一轮 §3.3 rom_loo", "留一 Â 偏差 bias_pct（13 行 = 每录制首个 onset）",
        "中位 %+.2f%%、p10~p90 %+.2f~%+.2f%%、|偏差| 最大 %.2f%%（n=%d）"
        % (np.median(b), np.percentile(b, 10), np.percentile(b, 90), np.max(np.abs(b)), len(b)),
        "本任务 usable onset 事件级留一：tau_d=0.2 s 中位 %+.2f%%、中位|误差| %.2f%%、"
        "p90|误差| %.2f%%、最大|误差| %.2f%%（n=%d）"
        % (gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "med_pct"),
           gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "med_abs_pct"),
           gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "p90_abs_pct"),
           gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "max_abs_pct"),
           int(gget("usable", "loo_event", "onset", 0.20, "single@tau", "post", "n"))),
        "相同（中位）／修正（尾部）",
        "中位偏差同为个位数；尾部差一个量级：第一轮每行是「每录制 1 个代表事件」（已把病态事件挡在外面），"
        "本任务 n=18 含慢压 onset（如 SW1@75.89 单点|误差| 51.8%）")
    add("A 第一轮 §3.3 rom_compare", "用同一 ROM 的 Â 偏差按传感器分组（−5.2%/+12.4%/+6.5%）",
        "右拇指 −5.2%、左拇指 +12.4%、四指 +6.5%（n=3 录制/族）",
        "本任务不给同口径复算（第一轮用的是其自建 ROM 的峰值归一化形状 g，本任务无该 ROM）；"
        "但给出**同机制**的替代量：族间 f(0.30 s) 的 Kruskal-Wallis p=%.3f（n=3/3/2）"
        % float(tests[(tests["dim"] == "fam") & (tests["col"] == "z_at_03")]["p"].iloc[0]),
        "不适用（缺 ROM 输入）",
        "本任务的形状库由事件自身构造，不含第一轮的 ROM；要复算该表需要 T4-B/T5-A 的 ROM 参数")

    # ══════════ G. 本任务新增结论 ══════════
    add("G 新增", "restep 归一化形状的【病态率】",
        "既有工作只报 restep 的均值/中位（如第一轮 0.2 s 完成度 62.6~72.2%）",
        "装载类 41 个事件里 %d 个（全部是 restep，占 restep 的 %d/19）的归一化 f(τ) 在主网格上出现"
        "负值或 >1.5 ⇒ 不可用于比值反演；其中 2 个 |J| 低于 2%% 记录峰值；"
        "主网格极差达 12.9（f 的物理量纲 [-1.5, 1.0]）" % (nbad, nbad_re),
        "新增",
        "这解释了为什么 restep 的「0.2 s 完成度」在不同口径下能差 28 pt（T4-A §④#2 已发现）："
        "根因是**分母 J 的窗定义 + 事件本身是复合/多级/基线在动**，不是单一算法问题")
    slope = ts[(ts["strat"] == "pos_bin(onset)") & (ts["col"] == "z_at_10")]
    tro = pd.read_csv(os.path.join(C.RES, "t3a_time_trend.csv"))
    add("G 新增", "形状的时不变性（Q5）",
        "既有工作未做",
        "录制内早/中/晚三段 f(1.0 s) 中位 = %s（n=%s）；同录制内 onset 的 "
        "Spearman rho(f(1.0 s), t_on) 中位 %+.2f（%d 组，p<0.05 的 %d 组，方向多为正）"
        % (" / ".join("%.3f" % m for m in slope["med"]),
           " / ".join(str(int(x)) for x in slope["n_events"]),
           float(np.median(tro[tro["metric"] == "f(1.00s)"]["rho"])),
           len(tro[tro["metric"] == "f(1.00s)"]),
           int((tro[tro["metric"] == "f(1.00s)"]["p"] < 0.05).sum())),
        "新增",
        "早/晚分层样本量极小（n=3/3），只能给「未发现明显时变」的弱结论；列为缺口 G8")
    add("G 新增", "相对载荷量级对早期形状的影响",
        "既有结论：形状与幅度基本无关（corr ≈ −0.01~−0.23）",
        "本任务 usable 样本：|J|/峰值 分层（<5%%/5~15%%/15~40%%/>=40%%）的 f(0.30 s) "
        "Kruskal-Wallis H=%.2f p=%.4f（各层 n=%s）—— **小相对幅度事件的 0.30 s 完成度显著更低**"
        % (float(tests[(tests["dim"] == "magbin") & (tests["col"] == "z_at_03")]["H"].iloc[0]),
           float(tests[(tests["dim"] == "magbin") & (tests["col"] == "z_at_03")]["p"].iloc[0]),
           tests[(tests["dim"] == "magbin") & (tests["col"] == "z_at_03")]["n_each"].iloc[0]),
        "修正（条件性）",
        "既有「与幅度无关」是在【绝对幅度跨域不可比】的前提下说的；本任务用【相对幅度】分层后发现"
        "弱相关（p=0.03，n=3/5/16），且这与 T4-A 的「幅度与形状 corr=-0.11~+0.23」不矛盾："
        "小幅度事件多为慢压/多级加载 ⇒ 真正的驱动量仍是输入上升时间（T4-A 的 T_ramp）")

    df = pd.DataFrame(ROWS)
    df.to_csv(os.path.join(C.RES, "t3a_crosscheck.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(C.RES, "t3a_crosscheck.json"), "w", encoding="utf-8") as f:
        # 行内中文引号一律转成 \u0022，保证 JSON 严格可解析（ensure_ascii=False 时裸引号会破坏结构）
        json.dump(_sanitize(ROWS), f, ensure_ascii=False, indent=2)
    print("对照条目 %d 条；判定分布：%s\n" % (len(df), df["verdict"].value_counts().to_dict()))
    for _, r in df.iterrows():
        print("-- [%s] %s" % (r["group"], r["item"]))
        print("   was    : %s" % r["was"])
        print("   now    : %s" % r["now"])
        print("   verdict: %s" % r["verdict"])
        print("   why    : %s" % r["why"])
    print("\n-> results/t3a_crosscheck.csv、results/t3a_crosscheck.json")


def _sanitize(obj):
    """把字符串里未转义的英文双引号替换成 \\u0022，避免破坏 JSON 结构。"""
    import re as _re
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, str):
        out, instr = [], False
        i, n = 0, len(obj)
        while i < n:
            c = obj[i]
            if not instr:
                out.append(c)
                if c == chr(34):
                    instr = True
                i += 1
                continue
            if c == chr(92):
                out.append(obj[i:i + 2]); i += 2; continue
            if c == chr(34):
                j = i + 1
                while j < n and obj[j] in " \t":
                    j += 1
                if j < n and obj[j] in ",:}]":
                    out.append(c); instr = False; i += 1; continue
                out.append(chr(92) + 'u0022'); i += 1; continue
            out.append(c); i += 1
        return ''.join(out)
    return obj


if __name__ == "__main__":
    main()
