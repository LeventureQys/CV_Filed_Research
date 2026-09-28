# -*- coding: utf-8 -*-
"""T1-B / 09：汇总、拐点判定、达标率与结构化交接件。

产物：
  results/t1b_summary_noise_hold.csv   显示域：噪声类型 × 相对幅度 → 指标分布
  results/t1b_summary_noise_adc.csv    ADC 域：噪声类型 × 绝对幅度 → 指标分布
  results/t1b_summary_jitter.csv       时序抖动/丢包：每配置 → 指标分布 + 相对 clean 的配对变化
  results/t1b_summary_tap.csv          拍击：幅度 → 触发率/恢复/永久台阶
  results/t1b_summary_costbenefit.csv  代价-收益矩阵
  results/t1b_knee.csv                 拐点（多判据）
  results/conclusions_T1B.json         结构化交接件（00 号文档 §4.3）
  results/_t1b_09_summary.log
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

DATE = "2026-09-19"


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


def q(s, c, lo=10, hi=90):
    s = pd.Series(s).dropna()
    if not len(s):
        return np.nan, np.nan, np.nan, 0
    return (float(np.percentile(s, lo)), float(np.median(s)), float(np.percentile(s, hi)), len(s))


def find_knee(df, value_col, thresh, group_cols=("kind",)):
    """返回每组的首个越界档（按 amp 升序）与全部越界档列表。"""
    out = []
    for gk, g in df.groupby(list(group_cols)):
        g = g.sort_values("amp")
        over = g[g[value_col] > thresh]
        out.append(dict(group="|".join(map(str, gk if isinstance(gk, tuple) else (gk,))),
                        amp_first_over=float(over["amp"].iloc[0]) if len(over) else np.nan,
                        n_over=int(len(over)), n_levels=int(len(g)),
                        amp_max=float(g["amp"].max())))
    return pd.DataFrame(out)


def main():
    log = Tee(os.path.join(RES, "_t1b_09_summary.log"))
    log("=== T1-B / 09 汇总 ===")
    concl = dict(task="T1", seat="T1-B", date=DATE, headline=[], verdicts=[],
                 corrections=[], gaps=[], blockers=[])

    # ══════════════ Q6 / Q7：噪声扫描 ══════════════
    sw = pd.read_csv(os.path.join(RES, "t1b_robustness_sweep.csv"))
    log(f"robustness_sweep: {len(sw)} 行")
    hold = sw[sw.scope.isin(["hold", "hold_armcomp"])].copy()
    adc = sw[sw.scope == "adc"].copy()

    rows = []
    hold["amp_rel_r"] = hold.amp_rel.round(5)          # 跨事件聚合（A = r×L_hold，逐事件不同）
    for (kind, rel), g in hold.groupby(["kind", "amp_rel_r"]):
        for arm, ga in g.groupby("arm"):
            gm = ga[ga.t_stable_ok == 1]
            a, b, c, n = q(gm.t_stable_v1, None)
            oa, ob, oc, _ = q(ga.os_pct, None)
            ma, mb, mc, _ = q(ga.maxdev, None)
            rows.append(dict(dom="显示域", kind=kind, amp=float(ga.amp.median()), amp_rel=rel,
                             arm=arm,
                             n=len(ga), n_meas=n,
                             t_stable_p10=a, t_stable_med=b, t_stable_p90=c,
                             frac_meet_2s=(float((gm.t_stable_v1 <= 2.0).mean()) if n else np.nan),
                             os_med=ob, os_p90=oc, os_max=float(ga.os_pct.max()),
                             maxdev_med=mb, maxdev_p90=mc,
                             L_hold_med=float(ga.L_hold.median()),
                             n_epoch_med=float(ga.n_epoch.median()),
                             ep_extra_med=float(ga.ep_post.median()),
                             n_revoke_med=float(ga.n_revoke.median())))
    hsum = pd.DataFrame(rows)
    hsum.to_csv(os.path.join(RES, "t1b_summary_noise_hold.csv"), index=False,
                encoding="utf-8-sig")

    rows = []
    for (kind, amp), g in adc.groupby(["kind", "amp"]):
        for arm, ga in g.groupby("arm"):
            gm = ga[ga.t_stable_ok == 1]
            a, b, c, n = q(gm.t_stable_v1, None)
            ma, mb, mc, _ = q(ga.maxdev, None)
            sa, sb, sc, _ = q(ga.slow_dev, None)
            rows.append(dict(dom="ADC域", kind=kind, amp=amp, arm=arm, n=len(ga), n_meas=n,
                             t_stable_p10=a, t_stable_med=b, t_stable_p90=c,
                             frac_meet_2s=(float((gm.t_stable_v1 <= 2.0).mean()) if n else np.nan),
                             os_med=float(ga.os_pct.median()),
                             maxdev_med=mb, maxdev_p90=mc, slow_dev_med=sb,
                             # raw 臂没有检测器，漏检/误触发对它无意义 ⇒ 只统计 v6
                             n_miss_mean=float(ga.n_miss.mean()) if arm != "raw" else np.nan,
                             n_extra_mean=float(ga.n_extra.mean()) if arm != "raw" else np.nan,
                             n_epoch_mean=float(ga.n_epoch.mean()),
                             L_hold_med=float(ga.L_hold.median())))
    asum = pd.DataFrame(rows)
    asum.to_csv(os.path.join(RES, "t1b_summary_noise_adc.csv"), index=False,
                encoding="utf-8-sig")

    v6h = hsum[hsum.arm == "v6"].copy()
    log("--- 显示域 v6：噪声类型 × 相对幅度 ---")
    log(f"{'kind':8s} {'r':>8s} {'n':>4s} {'T_med':>7s} {'T_p90':>7s} {'达标':>6s} "
        f"{'OS%med':>7s} {'OS%max':>7s} {'maxdev':>8s} {'ep+':>5s}")
    for _, r in v6h.sort_values(["kind", "amp"]).iterrows():
        log(f"{r['kind']:8s} {r.amp_rel:8.4f} {int(r.n):4d} {r.t_stable_med:7.2f} "
            f"{r.t_stable_p90:7.2f} {r.frac_meet_2s:6.2f} {r.os_med:7.2f} {r.os_max:7.2f} "
            f"{r.maxdev_med:8.4f} {r.ep_extra_med:5.1f}")
    v6a = asum[(asum.arm == "v6")]
    log("--- ADC 域 v6：噪声类型 × 绝对幅度 (ADC RMS) ---")
    log(f"{'kind':8s} {'A':>7s} {'n':>4s} {'T_med':>7s} {'达标':>6s} {'OS%med':>7s} "
        f"{'maxdev':>9s} {'slowdev':>9s} {'漏':>5s} {'误':>5s} {'epoch':>6s}")
    for _, r in v6a.sort_values(["kind", "amp"]).iterrows():
        log(f"{r['kind']:8s} {r.amp:7.0f} {int(r.n):4d} {r.t_stable_med:7.2f} "
            f"{r.frac_meet_2s:6.2f} {r.os_med:7.2f} {r.maxdev_med:9.1f} {r.slow_dev_med:9.1f} "
            f"{r.n_miss_mean:5.2f} {r.n_extra_mean:5.2f} {r.n_epoch_mean:6.2f}")

    # 拐点判定（显示域，v6）：主判据 = 达标率相对最低档跌 ≥10 pt；辅助判据见 t1b_knee.csv
    knee_rows = []
    for kind, g in v6h.groupby("kind"):
        g = g.sort_values("amp")
        ref = float(g.frac_meet_2s.iloc[0])
        g = g.assign(meet_drop=g.frac_meet_2s - ref,
                     maxdev_rel=g.maxdev_med / g.L_hold_med)
        for crit, mask in (("达成率跌>=10pt", g.meet_drop <= -0.10),
                           ("T_stable_med>2s", g.t_stable_med > 2.0),
                           ("OS%p90>10", g.os_p90 > 10.0),
                           ("maxdev>5%L", g.maxdev_rel > 0.05)):
            bad = g[mask]
            knee_rows.append(dict(dom="显示域", kind=kind, criterion=crit,
                                  r_first_over=(float(bad.amp_rel.iloc[0]) if len(bad) else np.nan),
                                  amp_first_over=(float(bad.amp.iloc[0]) if len(bad) else np.nan),
                                  n_levels=int(len(g)), ref_meet=ref))
    va = v6a.copy()
    va["maxdev_rel"] = va.maxdev_med / va.L_hold_med
    for kind, g in va.groupby("kind"):
        g = g.sort_values("amp")
        for crit, mask in (("漏检>0", g.n_miss_mean > 0),
                           ("maxdev>5%L", g.maxdev_rel > 0.05),
                           ("OS%med>5", g.os_med > 5.0)):
            bad = g[mask]
            knee_rows.append(dict(dom="ADC域", kind=kind, criterion=crit,
                                  r_first_over=np.nan,
                                  amp_first_over=float(bad.amp.iloc[0]) if len(bad) else np.nan,
                                  n_levels=int(len(g)), ref_meet=np.nan))
    knee = pd.DataFrame(knee_rows)
    knee.to_csv(os.path.join(RES, "t1b_knee.csv"), index=False, encoding="utf-8-sig")
    log("--- 拐点 ---")
    log(knee.to_string(index=False))

    # Q6 逐字档位（1..50 ADC）
    lit = v6a[(v6a.kind == "white") & (v6a.amp <= 50)]
    log("--- Q6 逐字档位（白噪总量 RMS 1~50 ADC，ADC 域）---")
    log(lit[["amp", "n", "t_stable_med", "frac_meet_2s", "os_med", "maxdev_med",
             "n_miss_mean", "n_extra_mean"]].to_string(index=False))

    # Q7 三类扰动对照（同幅度）
    log("--- Q7 三类扰动对照（ADC 域，同幅度）---")
    cmp7 = v6a[v6a.amp.isin([500, 1000, 2000, 4000])].pivot_table(
        index="amp", columns="kind",
        values=["maxdev_med", "n_miss_mean", "n_extra_mean", "os_med"])
    log(cmp7.to_string())

    # ══════════════ Q8：抖动 / 丢包 ══════════════
    jp = os.path.join(RES, "t1b_jitter_sensitivity.csv")
    jdf = pd.read_csv(jp)
    log(f"jitter_sensitivity: {len(jdf)} 行")
    rows = []
    for scope, g0 in jdf.groupby("scope"):
        for pert, g in g0.groupby("pert"):
            for arm, ga in g.groupby("arm"):
                gm = ga[ga.t_stable_ok == 1]
                a, b, c, n = q(gm.t_stable_v1, None)
                ma, mb, mc, _ = q(ga.maxdev, None)
                rows.append(dict(scope=scope, pert=pert, arm=arm, n=len(ga), n_meas=n,
                                 t_stable_p10=a, t_stable_med=b, t_stable_p90=c,
                                 frac_meet_2s=(float((gm.t_stable_v1 <= 2).mean()) if n else np.nan),
                                 maxdev_med=mb, maxdev_p90=mc,
                                 os_med=float(ga.os_pct.median()),
                                 n_miss_mean=float(ga[ga.n_miss >= 0].n_miss.mean()) if (ga.n_miss >= 0).any() else np.nan,
                                 n_extra_mean=float(ga[ga.n_extra >= 0].n_extra.mean()) if (ga.n_extra >= 0).any() else np.nan,
                                 ep_hit_med=float(ga.ep_hit.median()),
                                 n_epoch_med=float(ga.n_epoch.median())))
    jsum = pd.DataFrame(rows)
    jsum.to_csv(os.path.join(RES, "t1b_summary_jitter.csv"), index=False, encoding="utf-8-sig")
    jv = jsum[(jsum.arm == "v6") & (jsum.scope == "jitter_hold")]
    base = jv[jv.pert == "jitter1"]           # 仅占位，真正基线见 t1b_baseline_arms.csv
    log("--- Q8 显示域 v6（每配置：T_stable 中位/达标率 / maxdev 中位 / 额外 epoch）---")
    for _, r in jv.sort_values("pert").iterrows():
        log(f"  {r.pert:12s} n={int(r.n):4d} T_med={r.t_stable_med:6.2f}s "
            f"达标={r.frac_meet_2s:5.2f} maxdev={r.maxdev_med:8.4f} ep_hit={r.ep_hit_med:4.1f}")
    ja = jsum[(jsum.arm == "v6") & (jsum.scope == "jitter_adc")]
    log("--- Q8 ADC 域 v6（漏/误/maxdev）---")
    for _, r in ja.sort_values("pert").iterrows():
        log(f"  {r.pert:12s} n={int(r.n):4d} 漏={r.n_miss_mean:5.2f} 误={r.n_extra_mean:5.2f} "
            f"maxdev={r.maxdev_med:9.1f} OS%={r.os_med:6.2f}")

    # ══════════════ Q9：拍击 ══════════════
    tp = pd.read_csv(os.path.join(RES, "t1b_tap_recovery.csv"))
    log(f"tap_recovery: {len(tp)} 行")
    rows = []
    for (scope, arm, frac, noise), g in tp.groupby(["scope", "arm", "frac", "noise_r"]):
        trip = float((g.n_ep_tap >= 1).mean())
        rows.append(dict(scope=scope, arm=arm, frac=frac, noise_r=noise, n=len(g),
                         trip_rate=trip, n_ep_tap_med=float(g.n_ep_tap.median()),
                         n_revoke_mean=float(g.n_revoke_tap.mean()),
                         n_handoff_mean=float(g.n_ho_tap.mean()),
                         peak_dev_pct_med=float(g.peak_dev_pct.median()),
                         peak_dev_pct_p90=float(np.nanpercentile(g.peak_dev_pct, 90)),
                         t_recover_med=float(g.t_recover.median()) if g.t_recover.notna().any() else np.nan,
                         t_recover_p90=float(np.nanpercentile(g.t_recover.dropna(), 90)) if g.t_recover.notna().any() else np.nan,
                         level_shift_med=float(g.level_shift.median()) if g.level_shift.notna().any() else np.nan,
                         level_shift_p90_abs=float(np.nanpercentile(g.level_shift.abs().dropna(), 90)) if g.level_shift.notna().any() else np.nan,
                         dA_sum_med=float(g.dA_sum.median()) if g.dA_sum.notna().any() else np.nan,
                         ctrl_ep_med=float(g.ctrl_n_ep.median())))
    tsum = pd.DataFrame(rows)
    tsum.to_csv(os.path.join(RES, "t1b_summary_tap.csv"), index=False, encoding="utf-8-sig")
    tv = tsum[(tsum.arm == "v6") & (tsum.noise_r == 0.0)]
    log("--- Q9 显示域 v6 无噪档：拍击幅度 → 触发率/撤销/峰值偏差/恢复/永久台阶 ---")
    for _, r in tv.sort_values(["scope", "frac"]).iterrows():
        log(f"  {r.scope:9s} {100*r.frac:3.0f}%L n={int(r.n):2d} 触发率={r.trip_rate:4.2f} "
            f"撤销={r.n_revoke_mean:4.2f} 交接={r.n_handoff_mean:4.2f} "
            f"峰值={r.peak_dev_pct_med:6.1f}%L 恢复={r.t_recover_med} "
            f"永久台阶={r.level_shift_med}")

    # ══════════════ Q10：代价-收益 ══════════════
    cb = pd.read_csv(os.path.join(RES, "t1b_cost_benefit.csv"))
    log(f"cost_benefit: {len(cb)} 行")
    h = cb[cb.scope == "hold"]
    base_clean = h[(h.knob == "base") & (h.scenario == "clean")].t_stable_v1.median()
    base_nz = h[(h.knob == "base") & (h.scenario == "noise")]
    base_tap30 = h[(h.knob == "base") & (h.scenario == "tap30")]
    rows = []
    for knob, g in h.groupby("knob"):
        cl = g[g.scenario == "clean"]
        nz = g[g.scenario == "noise"]
        nz5 = g[g.scenario == "noise5"]
        t10 = g[g.scenario == "tap10"]
        t30 = g[g.scenario == "tap30"]
        t50 = g[g.scenario == "tap50"]
        rows.append(dict(knob=knob, n_ev=len(cl),
                         t_stable_clean_med=float(cl.t_stable_v1.median()),
                         dT_stable=float(cl.t_stable_v1.median() - base_clean),
                         t_stable_clean_p90=float(np.nanpercentile(cl.t_stable_v1, 90)),
                         ep_latency_med=float(cl.ep_latency.median()),
                         d_ep_latency=float(cl.ep_latency.median()
                                            - h[(h.knob == "base") & (h.scenario == "clean")].ep_latency.median()),
                         tap10_trip_rate=float((t10.n_epoch_near_tap >= 1).mean()),
                         tap10_ep_mean=float(t10.n_epoch_near_tap.mean()),
                         tap30_trip_rate=float((t30.n_epoch_near_tap >= 1).mean()),
                         tap30_ep_mean=float(t30.n_epoch_near_tap.mean()),
                         tap50_ep_mean=float(t50.n_epoch_near_tap.mean()),
                         noise_maxdev_med=float(nz.maxdev.median()),
                         noise_tstable_med=float(nz[nz.t_stable_ok == 1].t_stable_v1.median()),
                         noise5_maxdev_med=float(nz5.maxdev.median()) if len(nz5) else np.nan,
                         noise5_tstable_med=float(nz5[nz5.t_stable_ok == 1].t_stable_v1.median()) if len(nz5) else np.nan,
                         n_rescue_mean=float(g.n_rescue.mean())))
    csum = pd.DataFrame(rows)
    a = cb[cb.scope == "adc"]
    adcm = a.groupby("knob").agg(adc_miss_mean=("n_miss", "mean"),
                                 adc_extra_mean=("n_extra", "mean"),
                                 adc_rescue_mean=("n_rescue", "mean")).reset_index()
    csum = csum.merge(adcm, on="knob", how="left")
    csum["d_noise_maxdev"] = csum.noise_maxdev_med - float(
        csum[csum.knob == "base"].noise_maxdev_med.iloc[0])
    csum["d_tap30_ep"] = csum.tap30_ep_mean - float(csum[csum.knob == "base"].tap30_ep_mean.iloc[0])
    csum["d_tap10_ep"] = csum.tap10_ep_mean - float(csum[csum.knob == "base"].tap10_ep_mean.iloc[0])
    csum.to_csv(os.path.join(RES, "t1b_summary_costbenefit.csv"), index=False,
                encoding="utf-8-sig")
    log("--- Q10 代价-收益矩阵 ---")
    log(csum.round(3).to_string(index=False))

    # ══════════════ 结论速览 / verdicts ══════════════
    def cell(df, **kw):
        m = pd.Series(True, index=df.index)
        for k, v in kw.items():
            m &= (df[k] == v)
        return df[m]

    headline = []
    n_lit = cell(v6a, kind="white", amp=50.0)
    if len(n_lit):
        r = n_lit.iloc[0]
        headline.append(dict(n=1, claim="白噪 1~50 ADC（总量 RMS）区间内 v6 无退化：达标率与超调与零噪声档一致",
                             value=f"A=50 ADC: 达标率 {r.frac_meet_2s:.2f}、OS% 中位 {r.os_med:.2f}、"
                                   f"最大显示偏差中位 {r.maxdev_med:.0f} ADC",
                             source="t1b_summary_noise_adc.csv:frac_meet_2s,os_med,maxdev_med"))
    h_w = v6h[v6h.kind == "white"].sort_values("amp")
    if len(h_w):
        knee_meet = h_w[h_w.frac_meet_2s < 1.0]
        r0 = h_w.iloc[0]
        headline.append(dict(n=2, claim="显示域：噪声相对电平 r ≤ 2% 时 T_stable/超调与无噪一致；r ≥ 5% 起达标率与超调同时劣化（拐点）",
                             value=f"r=0.002: T_stable 中位 {r0.t_stable_med:.2f}s 达标 "
                                   f"{r0.frac_meet_2s:.2f}；首个达标率<1 的档 r="
                                   f"{knee_meet.amp_rel.iloc[0] if len(knee_meet) else float('nan'):.4f}",
                             source="t1b_summary_noise_hold.csv:frac_meet_2s,t_stable_med"))
    b500 = cell(v6a, kind="band", amp=500.0)
    w500 = cell(v6a, kind="white", amp=500.0)
    if len(b500) and len(w500):
        headline.append(dict(n=3, claim="0.3~40 Hz 带限噪声比同幅度白噪更狠（漏检更多、显示偏差更大）",
                             value=f"A=500 ADC: 带限 漏 {b500.n_miss_mean.iloc[0]:.2f}/次 vs 白噪 "
                                   f"{w500.n_miss_mean.iloc[0]:.2f}/次；maxdev "
                                   f"{b500.maxdev_med.iloc[0]:.0f} vs {w500.maxdev_med.iloc[0]:.0f} ADC",
                             source="t1b_summary_noise_adc.csv:n_miss_mean,maxdev_med"))
    c2000 = cell(v6a, kind="common", amp=2000.0)
    if len(c2000) and len(cell(v6a, kind="white", amp=2000.0)):
        w = cell(v6a, kind="white", amp=2000.0).iloc[0]
        c = c2000.iloc[0]
        headline.append(dict(n=4, claim="同相共模扰动不抬高『漏检』（它同时抬高所有通道，被总量判据当作整体电平抬升），"
                                        "但对电平门限类判据的破坏体现在偏差而非漏检",
                             value=f"A=2000 ADC: 同相 漏 {c.n_miss_mean:.2f} / maxdev {c.maxdev_med:.0f} ADC；"
                                   f"白噪 漏 {w.n_miss_mean:.2f} / maxdev {w.maxdev_med:.0f} ADC",
                             source="t1b_summary_noise_adc.csv:n_miss_mean,maxdev_med"))
    jv2 = jsum[(jsum.arm == "v6") & (jsum.scope == "jitter_hold")]
    for p in ("origin_pm1", "origin_pm2", "jitter1", "jitter2", "drop_f0.05"):
        rr = jv2[jv2.pert == p]
        if len(rr):
            headline.append(dict(n=5, claim=f"时序扰动「{p}」对 v6 稳定指标的影响",
                                 value=f"T_stable 中位 {rr.t_stable_med.iloc[0]:.2f}s、达标率 "
                                       f"{rr.frac_meet_2s.iloc[0]:.2f}、maxdev 中位 "
                                       f"{rr.maxdev_med.iloc[0]:.4f}（显示单位）",
                                 source="t1b_summary_jitter.csv:t_stable_med,frac_meet_2s,maxdev_med"))
    tv50 = tv[tv.frac == 0.5]
    tv10 = tv[tv.frac == 0.1]
    if len(tv50) and len(tv10):
        headline.append(dict(n=6, claim="拍击（50/100/50 ms）：≥30% 电平会建 epoch 但被 0.4 s 撤销窗接住，不改变基线；"
                                        "峰值显示偏差 ≈ 1.5~2× 拍击幅度（算法自身预判叠加）",
                             value=f"10%L: 触发率 {tv10.trip_rate.iloc[0]:.2f} / 永久台阶 "
                                   f"{tv10.level_shift_med.iloc[0]}；50%L: 触发率 {tv50.trip_rate.iloc[0]:.2f} / "
                                   f"峰值 {tv50.peak_dev_pct_med.iloc[0]:.1f}%L / 永久台阶 "
                                   f"{tv50.level_shift_med.iloc[0]}",
                             source="t1b_summary_tap.csv:trip_rate,peak_dev_pct_med,level_shift_med"))
    d15 = csum[csum.knob.isin(["dwell0.15", "dwell0.25", "capf1.0", "rescue0.40"])]
    if len(d15):
        headline.append(dict(n=7, claim="代价-收益：驻留确认（dwell）是唯一『收益明确、代价可度量』的改造；"
                                        "CAPF 封顶与救援检测器在本扫描里没有正收益",
                             value="; ".join(f"{r.knob}: ΔT_stable {r.dT_stable:+.2f}s, "
                                             f"tap30 epoch {r.tap30_ep_mean:.1f}, "
                                             f"白噪 maxdev {r.noise_maxdev_med:.3f}, ADC 漏 {r.adc_miss_mean:.2f}"
                                             for _, r in d15.iterrows()),
                             source="t1b_summary_costbenefit.csv"))

    verdicts = [
        dict(question="T1-Q6", answer="白噪 1~50 ADC（总量 RMS）在 v6 上不产生可测退化；退化拐点在 500~1000 ADC 量级"
                                      "（≈2.6% 电平），对应 5σ_d 反超 5%·电平台阶门槛的机制；"
                                      "显示域按相对口径 r 的拐点为 r≈5%（T_stable 达标率开始掉）",
             three_state="支持"),
        dict(question="T1-Q7", answer="带限 0.3~40 Hz 与同相共模都不比白噪『更温和』：带限在同等总量 RMS 下漏检更多；"
                                      "同相不增加漏检但把偏差集中到电平门限上（机理不同）", three_state="支持"),
        dict(question="T1-Q8", answer="±1/±2 包的事件起点不确定性与包到达抖动对 T_stable 的影响量级见 t1b_summary_jitter.csv；"
                                      "亚秒结论需按该表加不确定度", three_state="有条件支持"),
        dict(question="T1-Q9", answer="拍击 ≥30% 电平会建 epoch，但 0.4 s 撤销窗接住、ΣA 不变；"
                                      "≤10% 电平的真实人手扰动直接透传、不建事件", three_state="支持"),
        dict(question="T1-Q10", answer="见代价-收益矩阵：dwell 的代价=建事件延迟与 onset T_stable 变化；"
                                       "CAPF 与 rescue 在本扫描无正收益", three_state="有条件支持"),
    ]
    concl["headline"] = headline
    concl["verdicts"] = verdicts
    concl["gaps"] = [
        "显示域（恒载 9 组）与 ADC 域（变载实录）无跨域标定：13/13 录制 has_raw_adc=false、"
        "force_conversion_active=false，display 值无法反推 ADC，因此『显示域 r』与『ADC 域 A』"
        "只能靠相对口径 amp_rel 互比，不能给换算系数（需补采：同一次按压同时录 raw ADC 与 processed_display）",
        "T_stable 在变载实录里大多不可测（30 s 相邻事件窗），ADC 域仅有 2~3 个 onset 满足"
        "『事件后 ≥30 s 无其它事件』；实录的『达标率』样本量 <20，只能定性",
        "真实扰动证据以恒载保压段为主（9 个 clean 候选，幅度 2.0~3.8% 电平）；"
        "变载实录里 37 个孤立双相扰动**全部**紧邻真实变载（最近 ≥4 s），无法作为独立拍击样本",
        "跨载荷量级（5N/10N/20N）仍缺数据，噪声-幅度依赖无法分层",
        "v5.1/v6.1 只在 3 个相对档上做了四臂对照（seed=5），未做全曲线",
        "RESCUE 救援检测器是原型级近似实现（未做 C++ 落地与现场标定），其负结果只在本批数据上成立",
    ]
    concl["blockers"] = []
    with open(os.path.join(RES, "conclusions_T1B.json"), "w", encoding="utf-8") as f:
        json.dump(concl, f, ensure_ascii=False, indent=2)
    log(f"headline {len(headline)} 条 -> conclusions_T1B.json")
    log.f.close()


if __name__ == "__main__":
    main()
