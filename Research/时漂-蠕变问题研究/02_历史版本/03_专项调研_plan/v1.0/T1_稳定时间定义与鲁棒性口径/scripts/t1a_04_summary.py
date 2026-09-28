# -*- coding: utf-8 -*-
"""t1a_04_summary：分族/分工况汇总 + 「1~2 s 上限」达标裁决 + 口径对照 + 第一轮数字对账。

输入：results/t1a_settle_metrics.csv、results/t1_events.csv、results/cache/*.npz、
      第一轮 progress/13-v6-assessment/results/settle_arms.csv（只读）
产出：
  results/t1a_settle_summary.csv     分族/分工况汇总（中位 + p10~p90 + 最差 + 删失数）
  results/t1a_target_verdict.csv     分工况达标率（分子/分母写清）+ 最差超标量
  results/t1a_target_exceed.csv      逐事件超标清单（冻结口径）
  results/t1a_definition_compare.csv 口径对照（逐事件 × 臂：D1/D2/D3 + 差值）
  results/t1a_round1_audit.csv       第一轮 5 个公开值的复算对账与裁决
规则：n<20 报中位 + p10~p90（不报 mean±std）；n≤3 标"仅定性参考"；时间到 0.01 s、百分比到 0.1%。
"""
import os
import sys

import numpy as np
import pandas as pd

import t1a_common as C

R1_SETTLE = os.path.join(C.PROG, "13-v6-assessment", "results", "settle_arms.csv")

CAL = [("ch5", "T_stable_ch5", "主通道+5s参考【冻结主口径 D1】"),
       ("tot5", "T_stable_tot5", "总通道+5s参考"),
       ("chcreep", "T_stable_chcreep", "主通道+含蠕变参考"),
       ("chalt5", "T_stable_chalt5", "最大台阶通道+5s参考（主通道定义敏感性）"),
       ("ev_ch5", "T_stable_ev_ch5", "主通道+5s参考【D1-ev：窗在下一事件处截断】"),
       ("ev_tot5", "T_stable_ev_tot5", "总通道+5s参考【D1-ev】"),
       ("ev_chcreep", "T_stable_ev_chcreep", "主通道+含蠕变参考【D1-ev】")]
LIMIT = 2.0          # 用户口径：不超过 2 s
FLOOR = 1.0          # 1~2 s 的下沿


def summarize(m, key, arm, group, name, val, cens, note_extra=""):
    v = pd.to_numeric(val, errors="coerce").to_numpy(float)
    cn = pd.to_numeric(cens, errors="coerce").fillna(1).to_numpy(bool) if cens is not None \
        else np.zeros(len(v), bool)
    ok = np.isfinite(v) & (~cn)
    n_all, n_meas = int(len(v)), int(ok.sum())
    s = C.fill_rate_summary(v[ok])
    r = dict(arm=arm, caliber=key, caliber_desc=dict((a, b) for a, b, _ in CAL)[key],
             group_type=group, group=name, n_events=n_all, n_meas=n_meas,
             n_censored=int((~ok).sum()), med=s["med"], p10=s["p10"], p90=s["p90"],
             worst=s["worst"], n_le_1s=int((v[ok] <= FLOOR).sum()), n_le_2s=int((v[ok] <= LIMIT).sum()),
             note=s["note"] + (" " + note_extra if note_extra else ""))
    return r


def main():
    C.start_log("04_summary")
    m = pd.read_csv(os.path.join(C.RES, "t1a_settle_metrics.csv"), encoding="utf-8-sig")
    ev = pd.read_csv(os.path.join(C.RES, "t1_events.csv"), encoding="utf-8-sig")
    m["kind3"] = m["kind"].replace({"partial_unload": "unload"})
    print("指标表 %d 行；臂 %s" % (len(m), ", ".join(C.ARMS)))

    # ── 1. 分族 / 分工况汇总 ──
    rows = []
    for arm in C.ARMS:
        sub = m[m.arm == arm]
        for key, col, _ in CAL:
            ccol = "cens_" + col.replace("T_stable_", "")
            have = ccol in sub.columns
            rows.append(summarize(m, key, arm, "全部事件", "n=60", sub[col],
                                  sub[ccol] if have else None))
            for fam in ("右拇指", "左拇指", "四指", "实录"):
                s2 = sub[sub.family == fam]
                rows.append(summarize(m, key, arm, "分族", fam, s2[col],
                                      s2[ccol] if have else None))
            for kd in ("onset", "restep", "unload"):
                s2 = sub[sub.kind3 == kd]
                rows.append(summarize(m, key, arm, "分工况", kd, s2[col],
                                      s2[ccol] if have else None))
            s2 = sub[(sub.kind3 == "onset") & (sub.dom == "显示域")]
            rows.append(summarize(m, key, arm, "分工况", "onset·恒载9组", s2[col],
                                  s2[ccol] if have else None))
            s2 = sub[sub.clean & sub.kind3.isin(["onset", "restep"])]
            rows.append(summarize(m, key, arm, "分工况", "clean 加载事件", s2[col],
                                  s2[ccol] if have else None))
    summ = pd.DataFrame(rows)
    p1 = os.path.join(C.RES, "t1a_settle_summary.csv")
    summ.round(4).to_csv(p1, index=False, encoding="utf-8-sig")
    print("\n== 冻结主口径（ch5）分族中位（可测事件）==")
    t = summ[(summ.caliber == "ch5") & (summ.group_type.isin(["分族", "分工况"]))]
    print(t.pivot_table(index=["group_type", "group"], columns="arm", values="med").round(2).to_string())
    print("\n== 冻结主口径（ch5）删失数（n=60 中「没停下」的）==")
    print(summ[(summ.caliber == "ch5") & (summ.group_type == "全部事件")][
        ["arm", "n_meas", "n_censored", "med", "p90", "worst"]].round(2).to_string(index=False))

    # ── 2. 达标裁决（分工况，分子/分母写清） ──
    verdicts = []
    for arm in C.ARMS:
        sub0 = m[m.arm == arm]
        for key, col, desc in CAL:
            ccol = "cens_" + col.replace("T_stable_", "")
            for gname, gsel in (("全部事件", sub0),
                                ("onset", sub0[sub0.kind3 == "onset"]),
                                ("onset·恒载9组", sub0[(sub0.kind3 == "onset") & (sub0.dom == "显示域")]),
                                ("restep", sub0[sub0.kind3 == "restep"]),
                                ("unload（含部分卸载）", sub0[sub0.kind3 == "unload"]),
                                ("实录族（ADC 域）", sub0[sub0.family == "实录"]),
                                ("恒载族（显示域）", sub0[sub0.family != "实录"])):
                if not len(gsel):
                    continue
                v = pd.to_numeric(gsel[col], errors="coerce").to_numpy(float)
                cn = pd.to_numeric(gsel[ccol], errors="coerce").fillna(1).to_numpy(bool)
                ok = np.isfinite(v) & (~cn)
                nm = int(ok.sum())
                pass2 = int((v[ok] <= LIMIT).sum())
                pass1 = int((v[ok] <= FLOOR).sum())
                fail = nm - pass2
                wf = float(np.max(v[ok])) if nm else np.nan
                verdicts.append(dict(
                    arm=arm, caliber=key, caliber_desc=desc, group=gname,
                    n_events=int(len(gsel)), n_measurable=nm, n_censored=int((~ok).sum()),
                    n_pass_1s=pass1, n_pass_2s=pass2, n_fail_2s=fail,
                    rate_2s=(pass2 / nm if nm else np.nan),
                    rate_2s_lo=(pass2 / len(gsel)), rate_1s=(pass1 / nm if nm else np.nan),
                    worst_s=wf,
                    verdict=("达标" if nm and fail == 0 else
                             ("不达标" if nm == 0 else "部分达标")),
                    note=("无可测样本（30 s 窗内全部删失）" if nm == 0 else
                          ("仅定性参考(n<=3)" if nm <= 3 else ""))))
    vd = pd.DataFrame(verdicts)
    p2 = os.path.join(C.RES, "t1a_target_verdict.csv")
    vd.round(4).to_csv(p2, index=False, encoding="utf-8-sig")
    print("\n== 「≤2 s」达标裁决（冻结主口径 ch5；分母=可测事件数）==")
    show = vd[(vd.caliber == "ch5")]
    print(show[["arm", "group", "n_events", "n_measurable", "n_censored", "n_pass_2s",
                "n_fail_2s", "rate_2s", "worst_s"]].round(3).to_string(index=False))

    # ── 3. 超标清单（冻结口径，可测且 >2 s） ──
    ex = m[(m.cens_ch5 == False) & (m.T_stable_ch5 > LIMIT)].copy()   # noqa: E712
    ex["excess_s"] = ex["T_stable_ch5"] - LIMIT
    ex = ex.sort_values(["T_stable_ch5"], ascending=False)
    p3 = os.path.join(C.RES, "t1a_target_exceed.csv")
    ex[["ds", "ev_id", "family", "dom", "kind", "clean", "t_on", "arm", "J_ch", "J_tot",
        "T_stable_ch5", "T_stable_tot5", "T_stable_chcreep", "T_stable_ev_ch5", "win_avail_s",
        "excess_s", "OS_pct", "n_epoch_in_win"]].round(4).to_csv(p3, index=False,
                                                                 encoding="utf-8-sig")
    print("\n== 冻结口径超标清单（T_stable_ch5 > %.1f s 且可测）：%d 行 ==" % (LIMIT, len(ex)))
    print(ex.groupby(["arm", "family", "kind3"]).size().to_string() if len(ex) else "  （无）")
    if len(ex):
        print(ex[["ds", "ev_id", "kind", "arm", "t_on", "T_stable_ch5", "excess_s"]]
              .head(40).round(2).to_string(index=False))

    # ── 4. 口径对照（逐事件 × 臂） ──
    dc = m[["ds", "ev_id", "family", "dom", "kind3", "clean", "t_on", "arm", "J_ch", "J_tot",
            "J_creep", "T_stable_ch5", "T_stable_tot5", "T_stable_chcreep", "T_stable_chalt5",
            "T_stable_ev_ch5", "T_settle_ch2", "T_settle_ch5", "T_settle_tot2", "T_settle_tot5",
            "T_band_ch5", "cens_ch5", "cens_tot5", "cens_chcreep"]].copy()
    dc["d_tot5_minus_ch5"] = dc["T_stable_tot5"] - dc["T_stable_ch5"]
    dc["d_creep_minus_ch5"] = dc["T_stable_chcreep"] - dc["T_stable_ch5"]
    dc["d_settle5_minus_ch5"] = dc["T_settle_ch5"] - dc["T_stable_ch5"]
    dc["d_settle2_minus_ch5"] = dc["T_settle_ch2"] - dc["T_stable_ch5"]
    dc["d_band_minus_ch5"] = dc["T_band_ch5"] - dc["T_stable_ch5"]
    p4 = os.path.join(C.RES, "t1a_definition_compare.csv")
    dc.round(4).to_csv(p4, index=False, encoding="utf-8-sig")
    print("\n== 口径对照：可测样本上的中位/最差（各臂） ==")
    for arm in C.ARMS:
        s = dc[dc.arm == arm]
        a = s[["T_stable_ch5", "T_stable_tot5", "T_stable_chcreep", "T_settle_ch2",
               "T_settle_ch5", "T_band_ch5"]]
        print("  %-5s 可测(ch5) %2d | " % (arm, int(s.cens_ch5.eq(False).sum()))
              + " ".join("%s:%.2f" % (c, a[c].median()) for c in a.columns))

    # ── 5. 第一轮公开值对账（含"第一轮口径复算"与"本任务冻结口径"两列） ──
    up = m[(m.arm == "v6") & (m.kind3 == "onset") & (m.dom == "显示域")]
    med = lambda c: float(pd.to_numeric(up[c], errors="coerce").median())        # noqa: E731
    ep = os.path.join(C.RES, "t1a_settle_edge_sensitivity.csv")
    e = pd.read_csv(ep, encoding="utf-8-sig") if os.path.isfile(ep) else None

    def r1c(col):
        return float(e[col].median()) if e is not None and col in e.columns else np.nan

    def arm_med(arm, col):
        s = m[(m.arm == arm) & (m.kind3 == "onset") & (m.dom == "显示域")]
        return float(pd.to_numeric(s[col], errors="coerce").median())

    aud = []

    def add(name, src, was, round1, frozen, note):
        aud.append(dict(value_name=name, source=src, published_s=was,
                        recomputed_round1caliber_s=round1, recomputed_frozen_s=frozen,
                        delta_vs_round1=(np.nan if round1 != round1 else round1 - was),
                        delta_frozen_vs_pub=(np.nan if frozen != frozen else frozen - was),
                        note=note))

    add("T_stable v6 主通道+5s（恒载 9 组中位）",
        "13-v6-assessment/settle_arms.csv / 第一轮现状与口径更正.md（1.72）", 1.72,
        r1c("T_ch5_r4edge_v6"), med("T_stable_ch5"),
        "第一轮口径（r4 的 t_on + span/(n−1) 网格）逐份复现；冻结口径换用 T4-A 真沿 ⇒ +0.08 s")
    add("T_stable v6 总通道+5s（恒载 9 组中位）",
        "同上（0.46）", 0.46, r1c("T_tot5_r4edge_v6"), med("T_stable_tot5"),
        "同左")
    add("T_stable v6 含蠕变口径（恒载 9 组中位）",
        "同上（0.41）", 0.41, r1c("T_chcreep_r4edge_v6"), med("T_stable_chcreep"),
        "参考幅度 = 卸载沿前 5 s 电平 − pre")
    add("T_stable v6 主通道+5s（早期旧值，切片下标 bug）",
        "plan/v1.0/第一轮现状与口径更正.md §1（2.72）", 2.72,
        (None if r1c("T_ch5_r4edge_v6") != r1c("T_ch5_r4edge_v6")
         else r1c("T_ch5_r4edge_v6") + 1.00), np.nan,
        "旧值 = 修正值 + 1.00 s 的恒定偏移（切片相对下标当绝对下标），**不是另一个口径**")
    add("T_stable raw / v5.1 主通道+5s（恒载 9 组中位）",
        "同上（9.33 / 2.80）", 9.33, r1c("T_ch5_r4edge_raw"),
        arm_med("raw", "T_stable_ch5"), "v5.1 见 t1a_settle_summary.csv（第一轮口径 2.81）")
    add("T_stable raw 总通道+5s（恒载 9 组中位）",
        "同上（17.98）", 17.98, r1c("T_tot5_r4edge_raw"), arm_med("raw", "T_stable_tot5"),
        "raw 臂总通道 T_stable 处于删失边界，逐份平台判定不稳（非候选臂，未深究）")

    # 07-v6/11-paper 的 0.55 s：总通道 + 回溯真沿（first_onset）口径，独立复算
    vals = []
    for rec in [r["rec"] for r in C.RECS if r["dom"] == "显示域"]:
        z = np.load(os.path.join(C.CACHE, "t1a_%s.npz" % rec.replace("/", "_")))
        tu, ytot = z["tu"], z["yraw_tot"]
        k0, _ = C.first_onset_edge(ytot)
        if k0 is None:
            continue
        J, _, _ = C.amp_5s(ytot, k0)
        t, cens = C.t_stable(tu, z["v6_tot"], k0, J)
        if not cens:
            vals.append(t)
    add("T_stable v6 总通道+回溯真沿（恒载 9 组中位）",
        "07-v6/MANIFEST.md、11-paper-v6/results/metrics_settle.csv（0.55）", 0.55,
        (float(np.median(vals)) if vals else np.nan), med("T_stable_tot5"),
        "用 pv_common.first_onset 口径独立复算（n_meas=%d）；与『主通道+5s』不可横比" % len(vals))
    adf = pd.DataFrame(aud)
    p5 = os.path.join(C.RES, "t1a_round1_audit.csv")
    adf.round(4).to_csv(p5, index=False, encoding="utf-8-sig")
    print("\n== 第一轮公开值对账（裁决用） ==")
    print(adf[["value_name", "published_s", "recomputed_round1caliber_s", "delta_vs_round1",
               "recomputed_frozen_s", "delta_frozen_vs_pub"]].round(3).to_string(index=False))

    # ── 6. D1 vs D2/D3 配对比较（同一事件上两个口径差多少） ──
    print("\n== D1 vs D2(2%/5%) vs D3(带口径) 配对比较（两口径都可测的同一批事件） ==")
    pair = []
    for arm in C.ARMS:
        s = m[m.arm == arm]
        for name, col in (("T_settle(2%) ch", "T_settle_ch2"), ("T_settle(5%) ch", "T_settle_ch5"),
                          ("T_settle(5%) tot", "T_settle_tot5"), ("T_band(5%) ch", "T_band_ch5")):
            ok = s["T_stable_ch5"].notna() & s[col].notna()
            if ok.sum() == 0:
                continue
            a = s.loc[ok, "T_stable_ch5"].to_numpy(float)
            b = s.loc[ok, col].to_numpy(float)
            rho = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
            pair.append(dict(arm=arm, pair="T_stable_ch5 vs " + name, n_paired=int(ok.sum()),
                             med_D1=float(np.median(a)), med_other=float(np.median(b)),
                             med_delta=float(np.median(b - a)), max_delta=float(np.max(np.abs(b - a))),
                             spearman=rho,
                             n_D1_gt_other=int((a > b).sum()),
                             n_cens_D1=int(s["cens_ch5"].sum()),
                             n_cens_other=int(s[col].isna().sum())))
    pr = pd.DataFrame(pair)
    p6 = os.path.join(C.RES, "t1a_d1_vs_d2.csv")
    pr.round(4).to_csv(p6, index=False, encoding="utf-8-sig")
    print(pr.round(3).to_string(index=False))
    print("\n== 用不同口径判『≤2 s』的达标率差异（分母=该口径可测事件数，全部 60 事件） ==")
    for arm in C.ARMS:
        s = m[m.arm == arm]
        cells = []
        for name, col in (("D1 ch5", "T_stable_ch5"), ("D2 2%", "T_settle_ch2"),
                          ("D2 5%", "T_settle_ch5"), ("D3 band5", "T_band_ch5")):
            v = pd.to_numeric(s[col], errors="coerce").dropna()
            cells.append("%s %d/%d@%.0f%%" % (name, int((v <= LIMIT).sum()), len(v),
                                              100 * float((v <= LIMIT).mean()) if len(v) else 0))
        print("  %-6s %s" % (arm, "  ".join(cells)))

    print("\n可用对照：13-v6-assessment/settle_arms.csv 存在=%s" % os.path.isfile(R1_SETTLE))
    for p in (p1, p2, p3, p4, p5, p6):
        print("-> %s" % p)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
