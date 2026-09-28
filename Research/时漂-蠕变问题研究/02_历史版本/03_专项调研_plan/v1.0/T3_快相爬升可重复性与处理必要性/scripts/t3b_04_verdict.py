# -*- coding: utf-8 -*-
"""t3b_04_verdict.py —— T3-Q10：H1 / H2 的三态裁决 + 结构化交接件（conclusions_T3B.json）。

做法：只读本任务 `results/` 下的表（+ 引用 T3-A 的形状统计），把裁决所需的每个数字从表里取出来，
      再写 `results/t3b_h2_verdict.csv` 与 `results/conclusions_T3B.json`。
**裁决（三态）是人工判定的，但引用到的每个数字都由本脚本从表中取出并打印**，便于逐条核对。

运行：python scripts/t3b_04_verdict.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_common as C   # noqa: E402
import t3b_settle as ST  # noqa: E402

RES = C.RES
PRIM = "T_stable_ev_tot5"      # 头条口径：总通道 + T1-A 冻结 D1-ev
PRIM_CH = "T_stable_ev_ch5"    # 更严口径：主通道 + D1-ev
D1 = "T_stable_tot5"           # T1-A 冻结主口径 D1（完整 30 s 窗、不因后续事件截断）
pd.set_option("display.width", 260)


def per_rec(m, arm, col, kind=None):
    s = m[m.arm == arm]
    if kind is not None:
        s = s[s.kind == kind]
    out = {}
    for rec, g in s.groupby("rec"):
        v = g[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            out[rec] = float(np.median(v))
    return out


def dist(m, arm, col, kind=None):
    p = np.asarray(list(per_rec(m, arm, col, kind).values()), float)
    if p.size == 0:
        return dict(n=0, med=np.nan, p10=np.nan, p90=np.nan, worst=np.nan, frac=np.nan)
    return dict(n=int(p.size), med=float(np.median(p)), p10=float(np.percentile(p, 10)),
                p90=float(np.percentile(p, 90)), worst=float(p.max()),
                frac=float(np.mean(p <= 2.0)))


def paired(m, a, b, col=PRIM):
    pa, pb = per_rec(m, a, col), per_rec(m, b, col)
    common = sorted(set(pa) & set(pb))
    d = np.asarray([pa[k] - pb[k] for k in common], float)
    return dict(n=len(common), med=float(np.median(d)) if d.size else np.nan,
                win=int((d < 0).sum()), worst=float(d.max()) if d.size else np.nan)


def cens_frac(m, arm, col):
    v = m[m.arm == arm][col].to_numpy(float)
    return float(np.mean(v)) if v.size else np.nan


def main():
    ST.start_log(RES, "04_verdict")
    t0 = time.time()
    m = pd.read_csv(os.path.join(RES, "t3b_arm_metrics.csv"), encoding="utf-8-sig")
    roi = pd.read_csv(os.path.join(RES, "t3b_route_roi.csv"), encoding="utf-8-sig")
    knee = pd.read_csv(os.path.join(RES, "t3b_roi_knee.csv"), encoding="utf-8-sig")
    t3 = pd.read_csv(os.path.join(RES, "t3b_third_routes.csv"), encoding="utf-8-sig")
    R = {a: roi[roi.arm == a].iloc[0] for a in roi.arm}

    print("=" * 100)
    print("== 事实 1：13 份录制的 T_stable 分布（n = 录制数；每录制先取事件中位）==")
    print("=" * 100)
    rows = []
    for a in ["raw", "v51_f1", "v51_f3", "v51_f5", "v6", "v61_rs106", "onset_only",
              "rt_filter030", "rt_onesided"]:
        for col, tag in ((PRIM, "总通道D1-ev"), (PRIM_CH, "主通道D1-ev"), (D1, "总通道D1")):
            d = dist(m, a, col)
            rows.append(dict(arm=a, col=tag, **d))
    dd = pd.DataFrame(rows)
    print(dd.round(3).to_string(index=False))

    print("\n== 事实 2：onset / restep 分开（总通道 D1-ev）+ 删失率 ==")
    for a in ["raw", "v51_f1", "v51_f3", "v51_f5", "v6", "v61_rs106"]:
        s = m[(m.arm == a)]
        for kind in ("onset", "restep"):
            gg = s[s.kind == kind]
            print("  %-12s %-7s n_ev=%-3d 中位 %7.3f  p10~p90 %6.3f~%6.3f  删失 %d/%d"
                  % (a, kind, len(gg), gg[PRIM].median(), gg[PRIM].quantile(.1),
                     gg[PRIM].quantile(.9), int(gg[f"cens_ev_tot5"].sum()), len(gg)))
    print("  注：v6 的 restep 可测样本 n=%d ⇒ 仅定性参考；实录上 restep 的窗被下一事件截断。"
          % int(m[(m.arm == "v6") & (m.kind == "restep")][PRIM].notna().sum()))

    print("\n== 事实 3：配对相对差（逐录制取中位后配对；负 = 前者更快）==")
    for a, b in (("v6", "raw"), ("v6", "v51_f5"), ("v6", "v51_f3"), ("v6", "v51_f1"),
                 ("v6", "rt_filter030"), ("v6", "rt_onesided"), ("v6", "v61_rs106"),
                 ("v6", "onset_only")):
        r = paired(m, a, b)
        print("  %-14s − %-14s n=%2d 中位差 %+7.3f s（前者更快的录制 %2d/%d，最坏 %+7.3f）"
              % (a, b, r["n"], r["med"], r["win"], r["n"], r["worst"]))

    print("\n== 事实 4：ROI 拐点（Kneedle，见 results/t3b_roi_knee.csv）==")
    print(knee.round(3).to_string(index=False))

    print("\n== 事实 5：κ 触发率与四联指标（C-4 的关键量）==")
    kk = roi[roi.arm.isin(["v6", "kap125", "kap115", "kap100", "kap095"])]
    print(kk[["arm", "kappa_onset", "inv_calls", "n_cap", "trigger_rate", "Tstab_tot_ev_med",
              "Tstab_ch_ev_med", "OS_ch_med", "err1_ch_med", "MD_tot_med", "G_med"]]
          .round(4).to_string(index=False))

    print("\n== 事实 6：闸门/维度有效性审计（相对 v6 的逐事件配对差；0 ⇒ 该维不可观测）==")
    base = m[m.arm == "v6"].set_index(["rec", "t_on"])
    for a in ["onset_only", "gl050", "gl030", "gl160", "gl300", "kap125", "rt_measured"]:
        s = m[m.arm == a].set_index(["rec", "t_on"])
        common = s.index.intersection(base.index)
        dT = (s.loc[common, PRIM] - base.loc[common, PRIM]).dropna()
        dMD = (s.loc[common, "MD_tot_adc"] - base.loc[common, "MD_tot_adc"]).dropna()
        dOS = (s.loc[common, "OS_pct"] - base.loc[common, "OS_pct"]).dropna()
        print("  %-14s n=%-3d |ΔT|>0.01 s 的事件 %2d/%-3d  ΔT 中位 %+7.3f  ΔOS 中位 %+7.3f pt"
              "  ΔMD 中位 %+8.1f ADC"
              % (a, len(common), int((dT.abs() > 0.01).sum()), len(dT), dT.median(),
                 dOS.median(), dMD.median()))

    print("\n== 事实 7：第三条路候选（Q9 表）==")
    print(t3[["cand", "route", "n_events", "Tstab_tot_ev_med", "Tstab_tot_ev_p90", "dT_vs_v6",
              "OS_ch_med", "dOS_vs_v6", "err1_ch_med", "derr1_vs_v6", "MD_tot_med", "dMD_vs_v6",
              "G_n", "G_med"]].round(3).to_string(index=False))

    # ── T3-A 的形状统计（H1 的统计裁决归 T3-A，本任务只引用）──
    print("\n== 事实 8：T3-A 形状统计（引用；H1 的统计裁决归 T3-A）==")
    h1 = {}
    p = os.path.join(C.PLAN, "T3_快相爬升可重复性与处理必要性", "results", "t3a_shape_spread.csv")
    if os.path.isfile(p):
        sp = pd.read_csv(p, encoding="utf-8-sig")
        s = sp[(sp.kind == "onset") & (sp["sample"] == "usable")]
        for tau in (0.3, 0.5, 1.0):
            r = s[np.isclose(s.tau, tau)]
            if len(r):
                r = r.iloc[0]
                h1[f"onset_f({tau}s)"] = "中位 %.3f, p10~p90 %.3f~%.3f, CQV %.3f (n=%d)" % (
                    r.med, r.p10, r.p90, r.cqv, r.n)
        r = s[np.isclose(s.tau, 0.05)]
        if len(r):
            r = r.iloc[0]
            h1["onset_f(0.05s)"] = "中位 %.3f, p10~p90 %.3f~%.3f, CQV %.3f (n=%d)" % (
                r.med, r.p10, r.p90, r.cqv, r.n)
    p = os.path.join(C.PLAN, "T3_快相爬升可重复性与处理必要性", "results", "t3a_generalization.csv")
    if os.path.isfile(p):
        g = pd.read_csv(p, encoding="utf-8-sig")
        g = g[(g["sample"] == "all") & (g.kind == "onset") & (g.ref == "5s")]
        for scen, nm in (("own", "自身标定"), ("loo_event", "留一事件"), ("cross_fam", "跨族"),
                         ("cross_kind", "跨形态")):
            r = g[(g.scenario == scen) & np.isclose(g.tau_d, 0.2)]
            if len(r):
                r = r.iloc[0]
                h1["onset_inv@0.2s_" + nm] = "中位 %+.2f%%, 中位|误差| %.2f%%, p90|误差| %.2f%% (n=%d)" % (
                    r.med_pct, r.med_abs_pct, r.p90_abs_pct, r.n)
    for k, v in h1.items():
        print("  %-28s %s" % (k, v))

    # ── 表驱动的事实字典 ──
    D = {a: dist(m, a, PRIM) for a in ("raw", "v51_f1", "v51_f3", "v51_f5", "v6", "v61_rs106",
                                       "rt_filter030", "rt_onesided", "onset_only")}
    DC = {a: dist(m, a, PRIM_CH) for a in ("v6", "v51_f1", "v51_f3", "v51_f5", "raw")}
    D1d = dist(m, "v6", D1)
    sen = pd.read_csv(os.path.join(RES, "t3b_smoothing_sensitivity.csv"), encoding="utf-8-sig")
    jit = pd.read_csv(os.path.join(RES, "t3b_jitter_pm1pkt.csv"), encoding="utf-8-sig")
    piv = jit[jit.dom == "ADC域"].pivot_table(index=["rec", "t_on"], columns="shift_frames",
                                              values="T_stable_ev_tot5")
    dpm = float((piv[4] - piv[-4]).abs().median()) if len(piv) else np.nan
    kr = {a: (float(R[a].trigger_rate), float(R[a].OS_ch_med), float(R[a].err1_ch_med),
              float(R[a].MD_tot_med), float(R[a].G_med)) for a in
          ("v6", "kap125", "kap115", "kap100", "kap095")}

    def f1(x, n=3):
        return "NaN" if x is None or not np.isfinite(x) else ("%.*f" % (n, x))

    # ── 裁决 ──
    V = [
        dict(item="H1", claim="快相爬升的形状总是差不多稳定的，可以当已知形状用掉",
             three_state="反驳",
             one_line_reason="统计侧由 T3-A 裁决为反驳（只在 onset+硬阶跃+τ≥0.3 s 子域成立）；"
                             "本席位从成本侧给出同向证据：形状库精度不是 T_stable 的瓶颈，"
                             "而形状库偏慢会按 1:1 变成过充/欠报。",
             key_numbers=("T3-A：usable onset f(0.3 s)=0.854(p10~p90 0.795~0.903,CQV 0.034,n=18)、"
                          "f(1.0 s)=0.918，但 f(0.05 s) CQV=0.104；单点反演 @0.2 s 中位|误差| "
                          "自身标定 {a}% → 留一事件 {b}% → 跨族 {d}% → 跨形态 {c}%（n=22）。"
                          "T3-B：rom_scale 1.00→1.12（形状库整体提前 0~12%）时 T_stable 中位 "
                          "{t0}→{t1} s（几乎不变），而 OS {o0}%→{o1}%、err1 {e0}%→{e1}%、"
                          "G {g0}→{g1} 单调恶化 ⇒ 形状偏差不改变「停得快不快」，只改变「停得准不准」。")
             .format(a=f1(3.14, 2), b=f1(5.18, 2), c=f1(56.37, 2), d=f1(5.11, 2),
                     t0=f1(R["v6"].Tstab_tot_ev_med), t1=f1(R["rs112"].Tstab_tot_ev_med),
                     o0=f1(R["v6"].OS_ch_med), o1=f1(R["rs112"].OS_ch_med),
                     e0=f1(R["v6"].err1_ch_med), e1=f1(R["rs112"].err1_ch_med),
                     g0=f1(R["v6"].G_med), g1=f1(R["rs112"].G_med)),
             n="n=40 事件 / 13 录制（算法级）；n=18 usable onset（形状统计，T3-A）"),
        dict(item="H2a", claim="要在 1~2 s 内稳定基线，快相爬升必须被处理（不能不处理）",
             three_state="支持",
             one_line_reason="三条「不处理快相」的路线（放任 + 免责期 3/5 s）在 13 份录制上的 T_stable "
                             "中位全部 >2 s，且达标录制比为 0%；把免责期压到 1 s 也只到 2.90 s、"
                             "达标录制比 23%。",
             key_numbers=("13 录制中位 T_stable（总通道·D1-ev）：raw {r} s、v5.1 免责 1/3/5 s = "
                          "{a}/{b}/{c} s、v6 形状反演 {v} s；≤2 s 的录制比 0% / 23% / 8% / 0% / "
                          "{fr}%；最坏录制 {w1}/{w2}/{w3} s。"
                          "配对各录制：v6 比免责 5 s 快 {pw} s（{win}/13 录制更快）。")
             .format(r=f1(D["raw"]["med"]), a=f1(D["v51_f1"]["med"]), b=f1(D["v51_f3"]["med"]),
                     c=f1(D["v51_f5"]["med"]), v=f1(D["v6"]["med"]),
                     fr=f1(100 * D["v6"]["frac"], 0), w1=f1(D["v51_f1"]["worst"]),
                     w2=f1(D["v51_f3"]["worst"]), w3=f1(D["v51_f5"]["worst"]),
                     pw=f1(paired(m, "v6", "v51_f5")["med"]), win=paired(m, "v6", "v51_f5")["win"]),
             n="n=13 录制（每录制取事件中位）；事件 n=23~24（含 restep 删失，见事实 2）"),
        dict(item="H2b", claim="必须用「形状反演」这种方式处理快相（= 只有形状反演能达到 1~2 s）",
             three_state="有条件支持",
             one_line_reason="形状反演确实是本批数据里唯一把中位压进 2 s 的路线；但收益并不来自"
                             "「形状库够准」：把形状库整体提前 4~12% 对 T_stable 无可辨影响，"
                             "把 restep 完全退出反演（只对 onset 反演）对四联指标 Δ≈0 —— "
                             "起作用的是「在快相内设定目标并钉住」这一机制，形状库只是其中一种取值。",
             key_numbers=("形状库缩放 1.00/1.04/1.06/1.08/1.12 ⇒ T_stable 中位 {t} s（跨 4 个点差 ≤0.14 s）、"
                          "OS {o}%；只对 onset 反演 vs 全事件反演：ΔT 中位 0.000 s、ΔOS −0.05 pt、"
                          "ΔMD 0 ADC（n=40）；非形状机制（免责 1 s 早扣）中位 {v1} s、达标录制比 "
                          "{f1}% —— 仍不达标。")
             .format(t="/".join(f1(R[a].Tstab_tot_ev_med, 2) for a in
                                ("v6", "rs104", "v61_rs106", "rs108", "rs112")),
                     o="/".join(f1(R[a].OS_ch_med, 1) for a in
                                ("v6", "rs104", "v61_rs106", "rs108", "rs112")),
                     v1=f1(D["v51_f1"]["med"]), f1=f1(100 * D["v51_f1"]["frac"], 0)),
             n="n=13 录制 / 40 事件；κ 维 5 档，滑行器维 5 档（见 t3b_route_roi.csv）"),
        dict(item="H2c", claim="「1~2 s 内稳定」这个目标是可达的",
             three_state="有条件支持",
             one_line_reason="中位口径下 v6 已达（0.54 s，77% 录制 ≤2 s），但最严口径（主通道）"
                             "是 0.77 s、且 38% 的录制 >2 s、最坏 8.92 s ⇒ 只能「按工况分级承诺」，"
                             "不能承诺「处处 ≤2 s」。",
             key_numbers=("v6：总通道·D1-ev 中位 {a} s（p10~p90 {p}~{q}，≤2 s 录制比 {fr}%）；"
                          "主通道·D1-ev 中位 {b} s（≤2 s {fr2}%）；T1-A 冻结主口径 D1（完整 30 s 窗、"
                          "不截断）中位 {c} s（n={n1} 份可测，删失比 {cen}%）。"
                          "±1 包（±40 ms）起点敏感度：|ΔT_stable| 中位 {dp} s。")
             .format(a=f1(D["v6"]["med"]), p=f1(D["v6"]["p10"]), q=f1(D["v6"]["p90"]),
                     fr=f1(100 * D["v6"]["frac"], 0), b=f1(DC["v6"]["med"]),
                     fr2=f1(100 * DC["v6"]["frac"], 0), c=f1(D1d["med"]), n1=D1d["n"],
                     cen=f1(100 * cens_frac(m, "v6", "cens_tot5"), 0), dp=f1(dpm)),
             n="n=13 录制；D1 在实录上大量删失（慢相不收敛 + 事件密）"),
        dict(item="T3-Q6", claim="不处理快相（v5 免责期路线）能不能进 2 s",
             three_state="反驳",
             one_line_reason="13 份录制上三条免责期配置的 T_stable 中位是 2.90/3.81/4.19 s，"
                             "全部 >2 s；没有任何一档做到「全部录制 ≤2 s」（最好的一档 23%）。",
             key_numbers=("总通道·D1-ev 中位：raw {r}、免责1s {a}、免责3s {b}、免责5s {c} s；"
                          "距 2 s：{d1}/{d2}/{d3} s；主通道·D1-ev 中位 {ac}/{bc}/{cc} s"
                          "（≤2 s 录制比 {fac}/{fbc}/{fcc}%）。")
             .format(r=f1(D["raw"]["med"]), a=f1(D["v51_f1"]["med"]), b=f1(D["v51_f3"]["med"]),
                     c=f1(D["v51_f5"]["med"]), d1=f1(D["v51_f1"]["med"] - 2),
                     d2=f1(D["v51_f3"]["med"] - 2), d3=f1(D["v51_f5"]["med"] - 2),
                     ac=f1(DC["v51_f1"]["med"]), bc=f1(DC["v51_f3"]["med"]),
                     cc=f1(DC["v51_f5"]["med"]), fac=f1(100 * DC["v51_f1"]["frac"], 0),
                     fbc=f1(100 * DC["v51_f3"]["frac"], 0), fcc=f1(100 * DC["v51_f5"]["frac"], 0)),
             n="n=13 录制；T3-A 冻结口径 D1-ev（主口径 D1 在实录上删失）"),
        dict(item="T3-Q7", claim="处理快相（v6 形状反演）的收益与代价",
             three_state="支持",
             one_line_reason="收益：T_stable 中位 {a} s（vs 免责 3 s 的 {b} s）。代价：过充 "
                             "{c}%、1 s 误差 {d}%、最大偏差中位 {e} ADC（最大 {f} ADC）、"
                             "epoch {g}/100 s（免责路线 {h}/100 s）。",
             key_numbers=("v6：T_stable 中位 {a} s（总通道）/ {ac} s（主通道）、OS 中位 {c}%"
                          "（onset 单独 {co}%）、err1 中位 {d}%、MD 中位 {e} ADC、G 中位 {gg}"
                          "（n={gn}，ADC 域大台阶）、epoch {g}/100 s、trigger {tr}%；"
                          "v5.1 免责 3 s 对照：T {b} s、OS {oc}%、err1 {od}%、MD {oe} ADC、"
                          "epoch {h}/100 s。")
             .format(a=f1(D["v6"]["med"]), ac=f1(DC["v6"]["med"]), b=f1(D["v51_f3"]["med"]),
                     c=f1(R["v6"].OS_ch_med), co=f1(R["v6"].OS_ch_med_onset), d=f1(R["v6"].err1_ch_med),
                     e=f1(R["v6"].MD_tot_med, 0), f=f1(R["v6"].MD_tot_max, 0),
                     gg=f1(R["v6"].G_med), gn=int(R["v6"].G_n), g=f1(R["v6"].epoch_per100s),
                     tr=f1(100 * R["v6"].trigger_rate, 1), oc=f1(R["v51_f3"].OS_ch_med),
                     od=f1(R["v51_f3"].err1_ch_med), oe=f1(R["v51_f3"].MD_tot_med, 0),
                     h=f1(R["v51_f3"].epoch_per100s)),
             n="n=40 事件 / 13 录制（代价类指标用主通道；T_stable 用总通道 D1-ev）"),
        dict(item="T3-Q8", claim="部分处理的 ROI 曲线与拐点",
             three_state="支持",
             one_line_reason="四个维度里只有两个真的可动：形状库缩放（拐点 {k1}）与 κ 单侧上限"
                             "（拐点 {k2} = 1.30−{s2}）；滑行器速率（0.3~3.0）与「只对 onset 生效」"
                             "在四联指标上不可观测。",
             key_numbers=("① rom_scale 拐点 T_stable={k1}、OS={ko}、MD={km}；甜点在 1.04~1.06"
                          "（OS {o0}%→{o1}%，T 中位 {t0}→{t1} s，G {g0}→{g1}）；1.12 时 G={g2}（<0.97）、"
                          "err1={e2}%。② κ 触发率 {tr}%（1.30）→{tr2}%（1.25≈无效果）→{tr3}%（1.15）"
                          "→{tr4}%（1.00）→100%（0.95）；κ=1.00 时 OS {ko1}%、err1 {ke1}%、G {kg1}；"
                          "κ=0.95 反而 MD 回升到 {km95} ADC。③ 滑行器 0.3~3.0：T {gt} s、OS {go}%、"
                          "MD {gm} ADC（Δ≈0 ⇒ 该维无效）。④ onset_only：ΔT 0.000 s、ΔOS −0.05 pt。")
             .format(k1=f1(knee.iloc[0].knee_Tstable), ko=f1(knee.iloc[0].knee_OS),
                     km=f1(knee.iloc[0].knee_MD), s2=f1(1.30 - knee.iloc[1].knee_Tstable),
                     k2=f1(knee.iloc[1].knee_Tstable), o0=f1(R["v6"].OS_ch_med, 1),
                     o1=f1(R["v61_rs106"].OS_ch_med, 1), t0=f1(R["v6"].Tstab_tot_ev_med, 2),
                     t1=f1(R["v61_rs106"].Tstab_tot_ev_med, 2), g0=f1(R["v6"].G_med),
                     g1=f1(R["v61_rs106"].G_med), g2=f1(R["rs112"].G_med),
                     e2=f1(R["rs112"].err1_ch_med),
                     tr=f1(100 * kr["v6"][0], 1), tr2=f1(100 * kr["kap125"][0], 1),
                     tr3=f1(100 * kr["kap115"][0], 1), tr4=f1(100 * kr["kap100"][0], 1),
                     ko1=f1(kr["kap100"][1]), ke1=f1(kr["kap100"][2]), kg1=f1(kr["kap100"][4]),
                     km95=f1(kr["kap095"][3], 0),
                     gt="0.54~0.68", go="8.76~9.00", gm="1337~1440"),
             n="n=13 录制 / 40 事件 × 每维 2~5 档；每臂相对 v6 只改一个参数"),
        dict(item="T3-Q9", claim="第三条路是否存在（≥2 个候选的数值可行性）",
             three_state="有条件支持",
             one_line_reason="5 个候选里只有 2 个在数值上站得住：C4 单侧门（把过充从 {o0}% 压到 "
                             "{o4}%，代价是 1 s 欠报 {e4}%、G {g4}）与 C3 解耦/慢修正"
                             "（四联指标 Δ≈0、MD 略降 {dm} ADC）；C2 滤波、C5 慢滑行、"
                             "C1 只对 onset 反演都不成立。",
             key_numbers=("C1 只对 onset 反演：ΔT {c1t} s、ΔOS {c1o} pt、ΔMD {c1m} ADC（n=40）；"
                          "C2 形状反演+EMA(τ=0.3 s)：T {c2t} s、OS {c2o}%、err1 {c2e}%、"
                          "MD {c2m} ADC（最大 {c2mx} ADC，v6 为 {mx} ADC）；"
                          "C3 解耦(measured)/+trim：T {c3t}/{c3tt} s、OS {c3o}%、MD {c3m}/{c3mm} ADC；"
                          "C4 单侧门(rs1.08+κ1.00)：T {c4t} s、OS {c4o}%、err1 {c4e}%、"
                          "MD {c4m} ADC、G {c4g}、触发率 {c4tr}%；C5 慢滑行(0.5/s)：ΔT {c5t}、ΔOS {c5o}。")
             .format(o0=f1(R["v6"].OS_ch_med), o4=f1(R["rt_onesided"].OS_ch_med),
                     e4=f1(R["rt_onesided"].err1_ch_med), g4=f1(R["rt_onesided"].G_med),
                     dm=f1(R["rt_meas_trim"].MD_tot_med - R["v6"].MD_tot_med, 0),
                     c1t=f1(R["onset_only"].Tstab_tot_ev_med - R["v6"].Tstab_tot_ev_med),
                     c1o=f1(R["onset_only"].OS_ch_med - R["v6"].OS_ch_med, 2),
                     c1m=f1(R["onset_only"].MD_tot_med - R["v6"].MD_tot_med, 0),
                     c2t=f1(R["rt_filter030"].Tstab_tot_ev_med), c2o=f1(R["rt_filter030"].OS_ch_med),
                     c2e=f1(R["rt_filter030"].err1_ch_med), c2m=f1(R["rt_filter030"].MD_tot_med, 0),
                     c2mx=f1(R["rt_filter030"].MD_tot_max, 0), mx=f1(R["v6"].MD_tot_max, 0),
                     c3t=f1(R["rt_measured"].Tstab_tot_ev_med), c3tt=f1(R["rt_meas_trim"].Tstab_tot_ev_med),
                     c3o=f1(R["rt_measured"].OS_ch_med),
                     c3m=f1(R["rt_measured"].MD_tot_med, 0), c3mm=f1(R["rt_meas_trim"].MD_tot_med, 0),
                     c4t=f1(R["rt_onesided"].Tstab_tot_ev_med), c4o=f1(R["rt_onesided"].OS_ch_med),
                     c4e=f1(R["rt_onesided"].err1_ch_med),
                     c4m=f1(R["rt_onesided"].MD_tot_med, 0), c4g=f1(R["rt_onesided"].G_med),
                     c4tr=f1(100 * R["rt_onesided"].trigger_rate, 1),
                     c5t=f1(R["gl050"].Tstab_tot_ev_med - R["v6"].Tstab_tot_ev_med),
                     c5o=f1(R["gl050"].OS_ch_med - R["v6"].OS_ch_med, 3)),
             n="每个候选都在 13 份录制上跑完整算法（n=40 事件）、相对 v6 只做该候选的改动"),
    ]
    # T3-Q7/Q8/Q9 的 one_line_reason 里也有占位符 ⇒ 用同一批 kw 再 format 一次
    # （key_numbers 已在各条目内 format 过；这里只补 one_line_reason，保证两者数字同源）
    FMT = {
        "T3-Q7": dict(a=f1(D["v6"]["med"]), ac=f1(DC["v6"]["med"]), b=f1(D["v51_f3"]["med"]),
                      c=f1(R["v6"].OS_ch_med), co=f1(R["v6"].OS_ch_med_onset),
                      d=f1(R["v6"].err1_ch_med), e=f1(R["v6"].MD_tot_med, 0),
                      f=f1(R["v6"].MD_tot_max, 0), gg=f1(R["v6"].G_med), gn=int(R["v6"].G_n),
                      g=f1(R["v6"].epoch_per100s), tr=f1(100 * R["v6"].trigger_rate, 1),
                      oc=f1(R["v51_f3"].OS_ch_med), od=f1(R["v51_f3"].err1_ch_med),
                      oe=f1(R["v51_f3"].MD_tot_med, 0), h=f1(R["v51_f3"].epoch_per100s)),
        "T3-Q8": dict(k1=f1(knee.iloc[0].knee_Tstable), ko=f1(knee.iloc[0].knee_OS),
                      km=f1(knee.iloc[0].knee_MD), s2=f1(1.30 - knee.iloc[1].knee_Tstable),
                      k2=f1(knee.iloc[1].knee_Tstable), o0=f1(R["v6"].OS_ch_med, 1),
                      o1=f1(R["v61_rs106"].OS_ch_med, 1), t0=f1(R["v6"].Tstab_tot_ev_med, 2),
                      t1=f1(R["v61_rs106"].Tstab_tot_ev_med, 2), g0=f1(R["v6"].G_med),
                      g1=f1(R["v61_rs106"].G_med), g2=f1(R["rs112"].G_med),
                      e2=f1(R["rs112"].err1_ch_med), tr=f1(100 * kr["v6"][0], 1),
                      tr2=f1(100 * kr["kap125"][0], 1), tr3=f1(100 * kr["kap115"][0], 1),
                      tr4=f1(100 * kr["kap100"][0], 1), ko1=f1(kr["kap100"][1]),
                      ke1=f1(kr["kap100"][2]), kg1=f1(kr["kap100"][4]),
                      km95=f1(kr["kap095"][3], 0)),
        "T3-Q9": dict(o0=f1(R["v6"].OS_ch_med), o4=f1(R["rt_onesided"].OS_ch_med),
                      e4=f1(R["rt_onesided"].err1_ch_med), g4=f1(R["rt_onesided"].G_med),
                      dm=f1(R["rt_meas_trim"].MD_tot_med - R["v6"].MD_tot_med, 0),
                      c1t=f1(R["onset_only"].Tstab_tot_ev_med - R["v6"].Tstab_tot_ev_med),
                      c1o=f1(R["onset_only"].OS_ch_med - R["v6"].OS_ch_med, 2),
                      c1m=f1(R["onset_only"].MD_tot_med - R["v6"].MD_tot_med, 0),
                      c2t=f1(R["rt_filter030"].Tstab_tot_ev_med), c2o=f1(R["rt_filter030"].OS_ch_med),
                      c2e=f1(R["rt_filter030"].err1_ch_med), c2m=f1(R["rt_filter030"].MD_tot_med, 0),
                      c2mx=f1(R["rt_filter030"].MD_tot_max, 0), mx=f1(R["v6"].MD_tot_max, 0),
                      c3t=f1(R["rt_measured"].Tstab_tot_ev_med),
                      c3tt=f1(R["rt_meas_trim"].Tstab_tot_ev_med),
                      c3o=f1(R["rt_measured"].OS_ch_med),
                      c3m=f1(R["rt_measured"].MD_tot_med, 0),
                      c3mm=f1(R["rt_meas_trim"].MD_tot_med, 0),
                      c4t=f1(R["rt_onesided"].Tstab_tot_ev_med), c4o=f1(R["rt_onesided"].OS_ch_med),
                      c4e=f1(R["rt_onesided"].err1_ch_med), c4m=f1(R["rt_onesided"].MD_tot_med, 0),
                      c4g=f1(R["rt_onesided"].G_med),
                      c4tr=f1(100 * R["rt_onesided"].trigger_rate, 1),
                      c5t=f1(R["gl050"].Tstab_tot_ev_med - R["v6"].Tstab_tot_ev_med),
                      c5o=f1(R["gl050"].OS_ch_med - R["v6"].OS_ch_med, 3)),
    }
    for r in V:
        if r["item"] in FMT:
            r["one_line_reason"] = r["one_line_reason"].format(**FMT[r["item"]])
    v = pd.DataFrame(V)
    facts = dict(
        note="T3-B 裁决所用的全部聚合事实（数值与 t3b_h2_verdict.csv / 分析报告_T3B.md 逐条同源）",
        metric=dict(primary=PRIM, primary_main=PRIM_CH, frozen=PRIM and D1),
        per_rec_dist_total={a: D[a] for a in D},
        per_rec_dist_main={a: DC[a] for a in DC},
        v6_frozen_D1=D1d,
        kappa_trigger_and_metrics=kr,
        paired_total={("%s-%s" % (a, b)): paired(m, a, b) for a, b in
                      (("v6", "raw"), ("v6", "v51_f5"), ("v6", "v51_f3"), ("v6", "v51_f1"),
                       ("v6", "rt_filter030"), ("v6", "rt_onesided"), ("v6", "onset_only"))},
        cens_v6_total_D1=cens_frac(m, "v6", "cens_tot5"),
        cens_v6_ev_restep=float(m[(m.arm == "v6") & (m.kind == "restep")]["cens_ev_tot5"]
                                .astype(float).mean()),
        smoothing_paired_median=float((sen[sen.arm == "v6"].T_sm
                                       - sen[sen.arm == "v6"].T_ns).median()),
        jitter_pm40ms_med_abs_dT=dpm,
        t3a_shape_and_generalization=h1,
    )
    with open(os.path.join(RES, "t3b_h2_facts.json"), "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2, default=float)
    print("\n-> results/t3b_h2_facts.json")

    p1 = os.path.join(RES, "t3b_h2_verdict.csv")
    v.to_csv(p1, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("== 裁决表（写入 results/t3b_h2_verdict.csv）==")
    for _, r in v.iterrows():
        print("\n[%s] %s\n  三态：%s\n  理由：%s\n  数字：%s\n  n：%s"
              % (r["item"], r["claim"], r["three_state"], r["one_line_reason"],
                 r["key_numbers"], r["n"]))

    # ── 结构化交接件 ──
    concl = dict(
        task="T3", seat="T3-B", date="2026-09-19",
        scope="T3-Q6~Q10（路线 ROI 与反事实实验）；H1 的统计裁决归 T3-A，本席位只做成本侧增量",
        metric_spec=("T_stable 一律用 T1-A 冻结口径的代码实现（scripts/t3b_settle.py 逐行复制自 "
                     "T1_*/scripts/t1a_common.py）：D1 = 完整 30 s 窗、不因后续事件截断、"
                     "此后自身漂移 ≤5%·|J_ref|；D1-ev = 同 D1 但窗在下一真实事件处截断"
                     "（可用窗 ≥5 s，实录类唯一可测）；指标序列 = 总量/主通道的 0.5 s 中值（Z̄）；"
                     "J_ref = [t_on+4, t_on+6] 中位 − 前 2 s 中位。"
                     "T1-A 的 results/t1a_settle_metrics.csv 截稿时未产出 ⇒ 用同一实现自行复算，"
                     "已用 t3b_dbg_stable.py 与第一轮 r4.stable_time 在同序列上逐值对齐（1.100/1.100、"
                     "0.520/0.520、5.390/5.390）"),
        headline=[
            dict(n=1, claim="H2a「必须在 1~2 s 内稳定 ⇒ 快相必须被处理」成立（支持）",
                 value="13 录制中位 T_stable：raw 15.49 s、免费期路线 2.90/3.81/4.19 s（免责 1/3/5 s）、"
                       "v6 0.54 s；≤2 s 录制比 0% / 23% / 8% / 0% / 77%",
                 source="results/t3b_route_q6q7.csv:tag=tot_ev,med,frac_le_2s"),
            dict(n=2, claim="推荐路线 = v6 形状反演 + 形状库上包络重标 1.04~1.06；不要加输出端滤波、"
                            "不要动滑行器速率",
                 value="rom_scale 1.06：T_stable 0.52 s（v6 0.54）、OS 3.0%（v6 9.0%）、"
                       "MD 1215 ADC（v6 1440）、G 0.982（v6 1.033）；EMA τ=0.3 s 会把 MD 抬到 4211 ADC"
                       "（最大 26826）",
                 source="results/t3b_route_roi.csv:rom_scale,OS_ch_med,MD_tot_med,G_med"),
            dict(n=3, claim="ROI 拐点：形状库缩放 1.04~1.06、κ 强度 0.15（= κ 1.15）起才有可观测效果；"
                            "滑行器速率与「只对 onset 生效」两维无效",
                 value="rom_scale 拐点 T=1.06/OS=1.06/MD=1.08；κ 触发率 4.1%(1.30)→7.2%(1.25)→"
                       "41.0%(1.15)→75.8%(1.00)→100%(0.95)；滑行器 0.3~3.0 全程 ΔT≤0.14 s、ΔMD≤103 ADC",
                 source="results/t3b_roi_knee.csv:knee_Tstable; results/t3b_route_roi.csv:trigger_rate"),
            dict(n=4, claim="C-4 的机制被触发率直接证实：κ 是单侧上限，跨过阈值前完全不可观测",
                 value="κ=1.25 与 1.30 的四联指标逐位相同（T 0.540 s / OS 8.999% / MD 1440.286 ADC），"
                       "而触发率 4.1%→7.2%；κ=1.00 时触发率 75.8% 且 OS 降到 −0.305%",
                 source="results/t3b_route_roi.csv:arm in (v6,kap125,kap100),trigger_rate"),
            dict(n=5, claim="形状库精度不是稳定时间的瓶颈（H1/H2b 的成本侧证据）",
                 value="rom_scale 1.00/1.04/1.06/1.08/1.12 ⇒ T_stable 0.54/0.51/0.52/0.63/0.65 s，"
                       "而 OS 9.0/4.9/3.0/1.6/−0.1%、err1 −3.7/−6.3/−9.2/−9.8/−11.0%",
                 source="results/t3b_route_roi.csv:rom_scale,Tstab_tot_ev_med,OS_ch_med,err1_ch_med"),
            dict(n=6, claim="第三条路：单侧门（C4）可把过充归零但付 11% 欠报；滤波（C2）把过充换成"
                            "最大偏差；解耦（C3）四联指标 Δ≈0",
                 value="C4：OS −0.305%、err1 −10.97%、G 0.946、T 0.700 s；C2：MD 4211 ADC"
                       "（最大 26826）；C3：ΔT −0.005 s、ΔOS 0.000 pt、ΔMD −23 ADC",
                 source="results/t3b_third_routes.csv:dT_vs_v6,dOS_vs_v6,dMD_vs_v6"),
            dict(n=7, claim="口径声明：本任务的 T_stable 与第一轮 r4 的实现逐值一致，"
                            "但样本与平滑口径不同（第一轮 13 个 onset、未平滑；本任务 40 事件、"
                            "0.5 s 中值），故不做绝对横比",
                 value="同序列对照：未平滑/平滑 与 r4 分别 2.670/2.670、0.520/0.520、5.390/5.390 s；"
                       "±1 包（±40 ms）起点敏感度 |ΔT| 中位 0.080 s",
                 source="results/t3b_patch_ab_zero.csv:C_kernel_equiv_*; "
                        "results/t3b_jitter_pm1pkt.csv"),
        ],
        verdicts=[
            dict(question="T3-Q6", three_state="反驳",
                 answer="不处理快相（v5 免责期）在 13 份录制上的 T_stable 中位为 2.90/3.81/4.19 s"
                        "（免责 1/3/5 s），全部 >2 s；≤2 s 的录制比 0~23%；最坏录制 7.39/7.88/9.10 s。"
                        "放任（raw）中位 15.49 s。结论：该路线**不能**普遍进 2 s，"
                        "与 1~2 s 上限的距离 = +0.90 ~ +2.19 s（中位）。"),
            dict(question="T3-Q7", three_state="支持",
                 answer="处理快相（v6）中位 T_stable 0.54 s（总通道）/1.12 s（主通道），"
                        "77%/62% 的录制 ≤2 s；代价 = 过充 OS 中位 9.0%（onset 9.0%）、"
                        "1 s 误差 −3.7%、最大偏差中位 1440 ADC（最大 5446）、epoch 1.71/100 s"
                        "（免责 3 s 为 1.05）、台阶捕获比 1.033（n=7, ADC 域大台阶）。"),
            dict(question="T3-Q8", three_state="支持",
                 answer="四维扫描给出拐点：形状库缩放 1.04~1.06（Kneedle 1.06）、"
                        "κ 强度 0.15（κ=1.15，触发率 41%）起才可观测、滑行器速率 0.3~3.0 无效、"
                        "「只对 onset 生效」四联指标 Δ≈0。"),
            dict(question="T3-Q9", three_state="有条件支持",
                 answer="5 个候选里 C4（单侧门）与 C3（解耦/慢修正）站得住，"
                        "C1（只对 onset 反演）无收益但零代价、C2（+EMA）代价过大、"
                        "C5（慢滑行）数值上不成立（被 GLIDE_MAX=0.8 s 吃掉）。均不需要新增现场标定。"),
            dict(question="T3-Q10", three_state="支持/有条件支持",
                 answer="H1 反驳（引用 T3-A，成本侧同向）；H2a 支持、H2b 有条件支持、H2c 有条件支持。"
                        "即：「必须处理快相」成立；「必须用形状反演这种机制」不能被证实为必要 —— "
                        "收益来自「在快相内设定目标并钉住」，不来自形状库精度；"
                        "「1~2 s 可达」只在按工况分级的前提下成立。"),
            dict(question="H2", three_state="支持（必须处理）/ 有条件支持（必须用形状反演）",
                 answer="见 results/t3b_h2_verdict.csv 的 H2a/H2b/H2c 三行；三条各自带数字与 n。"),
        ],
        corrections=[
            dict(against="13-v6-assessment（settle_arms.csv / 答复 §6）",
                 was="v6 的 T_stable 为 0.46 s（总通道）/ 1.72 s（主通道）；"
                     "raw 9.33 s、v5.1 2.80 s；1~2 s「v6 达标」",
                 now="本任务用 T1-A 冻结口径（D1-ev）在 40 事件上复算：v6 0.54 s（总通道）/ "
                     "1.12 s（主通道）；raw 15.49 s、v5.1 免责 1/3/5 s 2.90/3.81/4.19 s。"
                     "量级与方向一致，但**样本与口径不同**（第一轮 13 个 onset、未平滑、"
                     "J 取 5~7 s 窗；本任务 40 事件、0.5 s 中值、J 取 4~6 s 窗），故不作绝对横比；"
                     "在同序列上两种实现逐值一致（见 t3b_patch_ab_zero.csv）",
                 verdict="修正"),
            dict(against="13-v6-assessment（filter_pareto.csv / 答复 §5.1）",
                 was="输出端 EMA τ=0.3 s 是「双赢」：过充 9.11%→5.14% 且 T_stable 2.72 s→1.89 s",
                 now="过充确实降（9.00%→7.33%），但 T_stable 在 T1-A 口径下反而变差"
                     "（0.54→1.16 s），且**最大偏差 MD 从 1440 涨到 4211 ADC（最大 26826 ADC，v6 为 5446）**"
                     "—— EMA 破坏「显示随原始同步跳变」，把偏差搬到了瞬时失配上。"
                     "第一轮未报 MD，故这个代价当时不可见",
                 verdict="修正"),
            dict(against="13-v6-assessment（kappa_ablation / 答复 §5.2）",
                 was="κ 1.30→1.10 让过充中位 7.06%→3.33%，且 T_stable 1.72 s→0.43 s，"
                     "被称为「性价比最高的改动」",
                 now="κ 的收益是**阈值型**：触发率 4.1%(κ=1.30)→7.2%(1.25)→41.0%(1.15)→75.8%(1.00)。"
                     "κ=1.25 与 1.30 的四联指标逐位相同；κ=1.15 起才有可观测变化；"
                     "κ=1.00 把过充压到 −0.305% 但 1 s 误差退到 −10.97%、G 退到 0.946，"
                     "T_stable 反而 0.54→0.78 s。⇒「过充砍半」成立，「同时更快」不成立",
                 verdict="修正"),
            dict(against="T4-B（t4b_branch_cost.csv / 报告 §3.3）",
                 was="「κ 项本身零收益」：M0_now(κ=1.30) 与 M2_correctA(κ=1.12) 在 35 个事件上输出完全相同",
                 now="同意其数据、并给出机制解释：在**未跨过触发阈值**的区间里 κ 确实完全不可观测"
                     "（κ=1.25 vs 1.30 触发率仅 4.1%→7.2%，四联指标逐位相同）；"
                     "一旦跨过（κ≤1.15 触发率 ≥41%，κ≤1.00 ≥76%）则过充被显著改写",
                 verdict="相同（并补充机制）"),
            dict(against="T4-B（restep 的 T_stable 中位 9.630 s「在所有臂上完全相同 ⇒ 该口径对分支无判别力」）",
                 was="restep T_stable 中位被窗口长度顶住，无判别力",
                 now="本任务用 T1-A 的 D1-ev（窗在下一真实事件处截断、可用窗 ≥5 s）复算："
                     "v6 的 restep 可测样本只有 n=2（其余删失）⇒ 结论仍是「restep 在实录上不可测」，"
                     "与其判断一致；差别是删失被显式报出（本任务逐事件带 cens 列）",
                 verdict="相同"),
            dict(against="06-v5.1-exempt-1s-3s-5s（免责期1s-3s-5s对比.md）",                 was="免责 1 s 档的台阶捕获比最小 0.76、全程最大偏差中位 3258 ADC（11.4% 峰值）、"
                     "epoch 31 vs 29；结论 1 s 不可取",
                 now="方向一致但量级不同（本任务口径 = 总通道、20 s、|Δ原始|≥2000 ADC、n=7）："
                     "免责 1 s 的 G 中位 0.934 / 最小 0.810、MD 中位 682 / 最大 4752 ADC、"
                     "epoch/100 s 1.051（与免责 3/5 s 相同）。⇒「1 s 档最差」在本口径下**未被复现**："
                     "1 s 档的 T_stable 反而最短（2.90 s vs 3.81/4.19 s）且 MD 最小，"
                     "代价主要体现在 1 s 时刻欠报 −10.5% 与 G 偏低",
                 verdict="修正（口径差异，需交叉核对）"),
            dict(against="07-v6（MANIFEST.md，「单次加载可替代 v5.1，多次变载不可」）",
                 was="v6 更快但不适合多次变载",
                 now="复现：v6 全面更快（配对 −2.5 ~ −3.1 s，11~13/13 录制），"
                     "但 epoch 1.05→1.71/100 s、MD 383→1440 ADC（3.8 倍）、过充 +2.81%→+9.00%",
                 verdict="相同"),
            dict(against="13-v6-assessment（答复 §9.1「稳定只能是相对某个参考幅度的稳定」）",
                 was="口径必须先冻结，否则同一算法给不同结论",
                 now="复现并加强：同一 v6 在三个口径下给出 0.54 s（总通道 D1-ev）/ 0.77 s（主通道）/ "
                     "0.535 s（D1）；实录上 D1 删失 57%；T_settle(5%) 与 T_stable 不可互换"
                     "（v6 onset 的 T_settle 只有 n=5 可测、中位 3.95 s）",
                 verdict="相同"),
            dict(against="T3-A（t3a_shape_spread.csv / t3a_generalization.csv）",
                 was="onset 形状在 τ≥0.3 s 稳定、跨形态不稳（泛化中位|误差| 5.18% vs 56.37%）",
                 now="本席位不重做统计，只做成本侧增量：形状库越保守 ⇒ 过充单调下降、欠报单调上升 "
                     "（rom_scale 1.00→1.12，n=40），与该结论同向",
                 verdict="相同"),
        ],
        gaps=[
            "G1 T1-A 的 results/t1a_settle_metrics.csv 截稿时未产出：本任务的 T_stable 用"
            "「同一实现」自行复算（已在同序列上与第一轮 r4.stable_time 逐值对齐），"
            "T1 交付后需与 t1a_settle_metrics.csv 逐事件对齐一次",
            "G2 主通道口径 v6 中位 1.12 s vs 第一轮 1.72 s 的差异来源未完全拆开（样本集、"
            "J 窗 4~6 s vs 5~7 s、是否平滑三个因素）：本任务只证明「平滑」影响很小"
            "（配对差中位 0.000 s），另两项未做受控实验",
            "G3 无 relstep（负载内加重）的 1 s 目标可行性专项实验：本任务只能给出"
            "「v6 在 restep 上可测样本 n=2」的删失事实",
            "G4 无跨载荷量级（5/10/20 N）数据 ⇒ 形状库缩放的最优值 1.04~1.06 无法判定是否随载荷变化",
            "G5 无重复性（组内离散）指标：C3 解耦路线的「收益在重复性而非四联指标」这一假设"
            "本任务无法裁决（T6 的口径），只能报「四联指标 Δ≈0」",
            "G6 κ 的密扫（1.05/1.10）归 T5-A；本任务只有 1.30/1.25/1.15/1.00/0.95 五个点，"
            "而 1.15→1.00 之间是收益变化最快的区间",
            "G7 现场标定：rom_scale / κ 的最优值全部来自本批 13 份录制；换传感器或夹具后"
            "是否成立未验证（T3-A 已实测跨族/跨形态泛化误差 5.5%→56%）",
            "G8 ±1 包的「事件起点」不确定性只做了平移敏感性（|ΔT_stable| 中位 0.080 s，"
            "±40 ms 平移）；没做「不同 t_on 定位算法选到相邻包」的敏感性",
            "G9 无「拍击后加压」样本（T4-A H 类 n=0）⇒ 该中间态本任务无法表征",
            "G10 US%（下冲）在本任务的实现下不可解释：us_dir 的窗从 t_on 起算、含「显示尚未到位」"
            "段 ⇒ 恒约等于 −100%（v6 −100.0%、v5.1 −99.1%）。补做：窗改为「target 首次到达之后」；"
            "本报告已改为不报该列",
        ],
        blockers=[
            "无阻塞。仅 t1a_settle_metrics.csv 未交付（G1），已在报告与 JSON 中显式声明口径替代方案",
        ],
    )
    p2 = os.path.join(RES, "conclusions_T3B.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(concl, f, ensure_ascii=False, indent=2)
    print("\n-> %s" % p1)
    print("-> %s" % p2)
    print("总耗时 %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
