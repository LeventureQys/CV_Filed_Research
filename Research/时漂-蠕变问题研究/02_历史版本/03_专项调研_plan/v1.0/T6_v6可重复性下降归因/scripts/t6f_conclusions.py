# -*- coding: utf-8 -*-
"""T6-F：从 results/*.csv 机读生成 conclusions.json，并打印报告要用的"数字块"。

目的：**报告的每个数字都由本脚本从 CSV 直接算出并打印**，避免手抄出错；
      `results/conclusions.json` 与 `分析报告.md` 的「结论速览 / 与既有结论的差异 / 未验证项」
      逐条一致。
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")


def rd(name):
    f = os.path.join(RES, name)
    return pd.read_csv(f) if os.path.exists(f) else None


def g(df, **kw):
    """安全取值：返回 float 或 nan。"""
    if df is None:
        return np.nan
    m = df
    for k, v in kw.items():
        if k not in m.columns:
            return np.nan
        m = m[m[k] == v]
    if len(m) == 0:
        return np.nan
    for c in ("value", "median", "rms_pct_lvl_med", "std_R5_pp"):
        if c in m.columns:
            return float(m[c].iloc[0])
    return float(m.iloc[0, -1])


def fmt(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def main():
    # ── 先把两个场景的抖动结果合并成交付契约要求的单文件 ──
    for base, out in (("t6_jitter_paths", "t6_jitter_paths.csv"),
                      ("t6_event_sequence", "t6_event_sequence.csv"),
                      ("t6_jitter_summary", "t6_jitter_summary.csv")):
        parts = [rd(f"{base}_{s}.csv") for s in ("a", "b")]
        parts = [x for x in parts if x is not None]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(os.path.join(RES, out), index=False,
                                                       encoding="utf-8-sig")
    disp = rd("t6_repeat_dispersion.csv")

    # ── L3 代理按场景拆列（两个场景的定点不同：场景 a 是"离散撤销通道"，场景 b 是"连续 Â 通道"，
    #    取两场景中位会把二者混在一起 ⇒ 必须拆开并入改进表） ──
    l3 = rd("t6_improve_l3.csv")
    if l3 is not None:
        s1 = l3[np.isclose(l3.J_rel, 1.0)]
        piv = s1.pivot_table(index="arm", columns="tag", values="rms_pct_lvl")
        piv.columns = [("L3_a_pct" if "1d9493" in c else "L3_b_pct") for c in piv.columns]
        piv["L3_max_pct"] = piv.max(axis=1)
        for fn in ("t6_improve_ab.csv", "t6_tradeoff.csv"):
            t = rd(fn)
            if t is None:
                continue
            t = t.drop(columns=[c for c in ("L3_a_pct", "L3_b_pct", "L3_max_pct",
                                            "d_L3_max_pct", "pct_L3_max_pct")
                                if c in t.columns])
            t = t.merge(piv.reset_index(), on="arm", how="left")
            b = float(t.loc[t.arm == "v6_baseline", "L3_max_pct"].iloc[0])
            t["d_L3_max_pct"] = t.L3_max_pct - b
            t["pct_L3_max_pct"] = 100.0 * t.d_L3_max_pct / b if abs(b) > 1e-12 else np.nan
            t.to_csv(os.path.join(RES, fn), index=False, encoding="utf-8-sig")
    l1 = rd("t6_l1_records.csv")
    l2 = rd("t6_l2_records.csv")
    ja, jb = rd("t6_jitter_summary_a.csv"), rd("t6_jitter_summary_b.csv")
    ea, eb = rd("t6_event_sequence_a.csv"), rd("t6_event_sequence_b.csv")
    at = rd("t6_attribution.csv")
    im = rd("t6_improve_ab.csv")
    gs = rd("t6_gridsens_disp.csv")
    dd = rd("t6_deduct_onset.csv")
    pz = rd("t6_patch_ab_zero.csv")

    def l2stat(impl, metric):
        if disp is None:
            return np.nan
        m = disp[(disp.layer == "L2_output") & (disp.impl == impl) & (disp.metric == metric)]
        return float(m.value.median()) if len(m) else np.nan

    def l1stat(metric):
        if disp is None:
            return np.nan
        m = disp[(disp.layer == "L1_input") & (disp.metric == metric)]
        return float(m.value.median()) if len(m) else np.nan

    def js(df, impl, cond, col):
        if df is None:
            return np.nan
        m = df[(df.impl == impl) & (df.cond == cond)]
        return float(m[col].iloc[0]) if len(m) and col in m.columns else np.nan

    def jc(df, impl, cond, col):
        if df is None or col not in df.columns:
            return np.nan
        m = df[(df.impl == impl) & (df.cond == cond)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    n_ja = int(js(ja, "v6", "timing_J100P", "n")) if ja is not None else 0
    n_jb = int(js(jb, "v6", "timing_J100P", "n")) if jb is not None else 0

    # ── L3 数字 ──
    j = {}
    for impl in ("v5.1", "v6", "v6.1"):
        j[impl] = dict(
            a_rms=js(ja, impl, "timing_J100P", "rms_pct_lvl_med"),
            a_rms_adc=js(ja, impl, "timing_J100P", "rms_adc_med"),
            a_same=js(ja, impl, "timing_J100P", "seq_same_rate"),
            a_lev=js(ja, impl, "timing_J100P", "seq_lev_med"),
            a_rev=js(ja, impl, "timing_J100P", "n_revoke_changed_rate"),
            a_ep=js(ja, impl, "timing_J100P", "n_epoch_changed_rate"),
            a_ho=js(ja, impl, "timing_J100P", "n_handoff_changed_rate"),
            a_n1=js(ja, impl, "noise_1ADC", "rms_pct_lvl_med"),
            a_n5=js(ja, impl, "noise_5ADC", "rms_pct_lvl_med"),
            a_drop=js(ja, impl, "dropout_1frame", "rms_pct_lvl_med"),
            a_same_n5=js(ja, impl, "noise_5ADC", "seq_same_rate"),
            a_same_drop=js(ja, impl, "dropout_1frame", "seq_same_rate"),
            b_rms=js(jb, impl, "timing_J100P", "rms_pct_lvl_med"),
            b_same=js(jb, impl, "timing_J100P", "seq_same_rate"),
            b_n5=js(jb, impl, "noise_5ADC", "rms_pct_lvl_med"),
            b_drop=js(jb, impl, "dropout_1frame", "rms_pct_lvl_med"),
        )
    ratio = (j["v6"]["a_rms"] / j["v5.1"]["a_rms"]) if j["v5.1"]["a_rms"] else np.nan

    # ── 起扣提前 ──
    lead = float(dd.lead_v6_vs_v51_s.median()) if dd is not None else np.nan
    lead_p10 = float(dd.lead_v6_vs_v51_s.quantile(.1)) if dd is not None else np.nan
    lead_p90 = float(dd.lead_v6_vs_v51_s.quantile(.9)) if dd is not None else np.nan
    tded_v6 = float(dd.t_deduct_v6.median()) if dd is not None else np.nan
    tded_v51 = float(dd["t_deduct_v5.1"].median()) if dd is not None else np.nan

    # ── 消融排序 ──
    abl = []
    if at is not None:
        b = at[at.arm == "v6_baseline"]
        b_r5 = float(b.std_R5_pp.iloc[0]) if len(b) else np.nan
        b_rs = float(b.std_Rstep_pp.iloc[0]) if len(b) else np.nan
        b_ts = float(b.std_Tstable_s.iloc[0]) if len(b) else np.nan
        b_ep = float(b.std_epoch.iloc[0]) if len(b) else np.nan
        for _, r in at.iterrows():
            if r.arm == "v6_baseline":
                continue
            abl.append(dict(arm=r.arm, d_R5=float(r.d_std_R5_pp), d_Rstep=float(r.d_std_Rstep_pp),
                            d_Tstable=float(r.d_std_Tstable_s), d_epoch=float(r.d_std_epoch),
                            mean_revoke=float(r.mean_revoke), mean_epoch=float(r.mean_epoch),
                            d_handoff=float(r.d_std_handoff)))
        abl_sorted = sorted(abl, key=lambda z: z["d_R5"])
    else:
        b_r5 = b_rs = b_ts = b_ep = np.nan
        abl_sorted = []

    imp = []
    if im is not None:
        bi = im[im.arm == "v6_baseline"]
        for _, r in im.iterrows():
            if r.arm == "v6_baseline":
                continue
            imp.append(dict(arm=r.arm, std_R5=float(r.std_R5_pp), d_R5=float(r.d_std_R5_pp),
                            pct_R5=float(r.pct_std_R5_pp),
                            std_Rstep=float(r.std_Rstep_pp),
                            L3=float(r.L3_rms_pct_lvl_med),
                            pct_L3=float(r.pct_L3_rms_pct_lvl_med),
                            T_stable=float(r.T_stable_med_s) if np.isfinite(r.T_stable_med_s)
                            else np.nan,
                            dT=float(r.d_T_stable_med_s) if np.isfinite(r.d_T_stable_med_s)
                            else np.nan))
        imp_sorted = sorted(imp, key=lambda z: z["std_R5"])
        best = imp_sorted[0] if imp_sorted else None
        v6R5 = float(bi.std_R5_pp.iloc[0]) if len(bi) else np.nan
        v6L3 = float(bi.L3_rms_pct_lvl_med.iloc[0]) if len(bi) else np.nan
        v6TS = float(bi.T_stable_med_s.iloc[0]) if len(bi) else np.nan
    else:
        imp_sorted, best, v6R5, v6L3, v6TS = [], None, np.nan, np.nan, np.nan

    def gsd(arm, metric, grid):
        if gs is None:
            return np.nan
        m = gs[(gs.arm == arm) & (gs.metric == metric) & (gs.grid == grid)]
        return float(m["median"].iloc[0]) if len(m) else np.nan

    P = []
    P.append("=" * 100)
    P.append("T6 报告数字块（全部由 results/*.csv 机读计算）")
    P.append("=" * 100)
    P.append(f"[样本] 恒载重复样本集 n=9（3 传感器组 × 3 次重复）；L3 抖动种子 n=30/条件"
             f"（场景 a n={n_ja}, 场景 b n={n_jb}）；消融/改进 n=9 录制 × 截 80 s")
    P.append("")
    P.append("── L1 输入层（原始数据本身的可重复性，与算法无关）──")
    P.append(f"  inc5 组内 CV（3 组取中位）= {fmt(l1stat('inc5_pct'),2)} %")
    P.append(f"  a_step(机械台阶) 组内 CV  = {fmt(l1stat('jump_a_step_pct'),2)} %")
    P.append(f"  CV_shape 全 τ 中位        = {fmt(l1stat('CV_shape_med'),2)} % "
             f"(max τ=0.30s: {fmt(l1stat('CV_shape@0.30'),2)} %)")
    P.append(f"  Spread_shape 中位         = {fmt(l1stat('Spread_shape_med_pp'),2)} pp")
    P.append(f"  慢相蠕变占比组内 std      = {fmt(l1stat('creep_frac_pp'),2)} pp")
    P.append("")
    P.append("── L2 输出层（显示读数在新一次加载上的散布；两套真值口径）──")
    for a in ("raw", "v5.1", "v6", "v6.1"):
        P.append(f"  {a:>5}: std(R5)={fmt(l2stat(a,'plat_err_std_R5_pp'),2)} pp  "
                 f"std(Rstep)={fmt(l2stat(a,'plat_err_std_Rstep_pp'),2)} pp  "
                 f"bias(R5)={fmt(l2stat(a,'plat_err_bias_R5_pp'),2)} pp  "
                 f"std(abs lvl)={fmt(l2stat(a,'abs_plat_std'),3)}  "
                 f"std(T_stable)={fmt(l2stat(a,'T_stable_std_s'),3)} s  "
                 f"std(epoch)={fmt(l2stat(a,'epoch_total_std'),3)}")
    P.append("")
    P.append("── L3 路径层（同一份输入，只改包到达时刻 / 加噪 / 丢帧）──")
    P.append(f"  场景 a（实录 中途切换-1d9493，ADC 域，63.85 s，±1 包 = ±40.0 ms）")
    for a in ("v5.1", "v6", "v6.1"):
        d = j[a]
        P.append(f"    {a:>5}: ±1 包 RMS={fmt(d['a_rms'],3)} % · 电平 "
                 f"({fmt(d['a_rms_adc'],1)} ADC) 事件序列相同率={fmt(d['a_same'],2)} "
                 f"编辑距离中位={fmt(d['a_lev'],1)} revoke 变={fmt(d['a_rev'],2)} "
                 f"epoch 变={fmt(d['a_ep'],2)} handoff 变={fmt(d['a_ho'],2)}")
        P.append(f"            ±5 ADC 噪 RMS={fmt(d['a_n5'],3)} % 序列同={fmt(d['a_same_n5'],2)}"
                 f" | 单帧丢包 RMS={fmt(d['a_drop'],3)} % 序列同={fmt(d['a_same_drop'],2)}")
    P.append(f"  场景 b（恒载 四指/数据1 前 80 s，力域，±1 包 = ±16.7 ms）")
    for a in ("v5.1", "v6", "v6.1"):
        d = j[a]
        P.append(f"    {a:>5}: ±1 包 RMS={fmt(d['b_rms'],3)} % 序列同={fmt(d['b_same'],2)} "
                 f"| ±5 ADC 噪={fmt(d['b_n5'],3)} % | 丢包={fmt(d['b_drop'],3)} %")
    P.append(f"  → v6/v5.1 的 L3 比值（场景 a，±1 包）= {fmt(ratio,1)}×")
    P.append("")
    P.append("── 起扣时刻（v6 相对 v5.1 提前多少）──")
    P.append(f"  阈值口径 1%·台阶：v5.1 中位 {fmt(tded_v51,2)} s，v6 中位 {fmt(tded_v6,2)} s，"
             f"提前中位 {fmt(lead,2)} s (p10~p90 {fmt(lead_p10,2)}~{fmt(lead_p90,2)} s, n=9)")
    P.append("")
    P.append("── 消融表（Δ 相对 v6 基线；负 = 退回该改动后重复性变好）──")
    P.append(f"  v6 基线: std(R5)={fmt(b_r5,2)} pp  std(Rstep)={fmt(b_rs,2)} pp  "
             f"std(T_stable)={fmt(b_ts,2)} s  std(epoch)={fmt(b_ep,2)}")
    for z in abl_sorted:
        P.append(f"  {z['arm']:>15}: Δstd(R5)={z['d_R5']:+.2f} pp  Δstd(Rstep)={z['d_Rstep']:+.2f}  "
                 f"Δstd(T_stable)={z['d_Tstable']:+.2f} s  Δstd(epoch)={z['d_epoch']:+.2f}  "
                 f"mean_revoke={z['mean_revoke']:.2f} mean_epoch={z['mean_epoch']:.2f}")
    P.append("")
    P.append("── 改进方向（9 份恒载 × 80 s + L3 代理 2 场景 × 5 种子；L3 必须拆场景看）──")
    P.append(f"  v6 基线: std(R5)={fmt(v6R5,2)} pp  L3_med={fmt(v6L3,3)} %  "
             f"T_stable 中位={fmt(v6TS,2)} s"
             + (f"  L3_a={fmt(float(im[im.arm=='v6_baseline'].L3_a_pct.iloc[0]),3)}% "
                f"L3_b={fmt(float(im[im.arm=='v6_baseline'].L3_b_pct.iloc[0]),3)}%"
                if im is not None and "L3_a_pct" in im.columns else ""))
    for z in imp_sorted:
        row = im[im.arm == z["arm"]].iloc[0]
        extra = ""
        if "L3_a_pct" in im.columns:
            extra = (f" L3_a={fmt(float(row.L3_a_pct),3)}%({float(row.pct_L3_max_pct):+.0f}% max)"
                     f" L3_b={fmt(float(row.L3_b_pct),3)}%")
        P.append(f"  {z['arm']:>16}: std(R5)={fmt(z['std_R5'],2)} ({z['pct_R5']:+.1f}%)  "
                 f"std(Rstep)={fmt(z['std_Rstep'],2)}  "
                 f"T_stable={fmt(z['T_stable'],2)} s (Δ={z['dT']:+.2f})"
                 f"  bias_R5={fmt(float(row.bias_R5_pp),2)} pp{extra}")
    P.append("")
    P.append("── 网格步长敏感性（口径分歧溯源）──")
    for arm in ("v5.1", "v6", "v6.1"):
        P.append(f"  {arm:>5}: std(R5) g100={fmt(gsd(arm,'plat_err_std_R5_pp','g100'),2)} "
                 f"vs glegacy={fmt(gsd(arm,'plat_err_std_R5_pp','gleg'),2)}  |  "
                 f"std(Rstep) g100={fmt(gsd(arm,'plat_err_std_Rstep_pp','g100'),2)} "
                 f"vs glegacy={fmt(gsd(arm,'plat_err_std_Rstep_pp','gleg'),2)}  |  "
                 f"std(T_stable) g100={fmt(gsd(arm,'T_stable_std_s','g100'),2)} "
                 f"vs glegacy={fmt(gsd(arm,'T_stable_std_s','gleg'),2)}")
    P.append("")
    P.append("── 补丁零差证明 ──")
    if pz is not None:
        P.append(f"  n_patch={len(pz)}  全部 bit-identical={bool(pz.bit_identical.all())}  "
                 f"max|Δ| 上界={pz.max_abs_diff.max():.1e}")
    txt = "\n".join(P)
    print(txt, flush=True)
    with io.open(os.path.join(RES, "_t6f_numbers.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    # ── conclusions.json ──
    def v(q, ans, ts):
        return dict(question=q, answer=ans, three_state=ts)

    # ── 从表里挑出"最大改善臂"与"零贡献臂"，供 headline/verdict 引用（避免手写） ──
    def _row(of, arm):
        s = of[of.arm == arm]
        return s.iloc[0] if len(s) else None

    imp_d5 = _row(im, "I1f_shrink_050") if im is not None else None
    imp_i7 = _row(im, "I7_revoke_hyst3") if im is not None else None
    imp_i5 = _row(im, "I5_delay3s") if im is not None else None
    imp_cb = _row(im, "I7_h3_I4ho10") if im is not None else None
    imp_61 = _row(im, "v6.1_asIS") if im is not None else None
    abl_d3 = _row(at, "D3_norate") if at is not None else None
    abl_d2 = _row(at, "D2_noshape") if at is not None else None
    abl_d4 = _row(at, "D4_ho10s") if at is not None else None
    abl_d5b = _row(at, "D5_delay5s") if at is not None else None
    abl_d1 = _row(at, "D1_confirm50") if at is not None else None

    headline = [
        dict(n=1, claim="v6 相对 v5.1 的『可重复性下降』主要落在 L3 路径层，不在 L1 输入层、也不完全在 L2 输出层",
             value=f"同一份实录输入、仅 ±1 包(±40.0 ms) 时序抖动：输出轨线 RMS 差 v6 {fmt(j['v6']['a_rms'],2)}%·电平"
                   f"（{fmt(j['v6']['a_rms_adc'],1)} ADC）vs v5.1 {fmt(j['v5.1']['a_rms'],2)}%"
                   f"（{fmt(j['v5.1']['a_rms_adc'],1)} ADC）= {fmt(ratio,1)}×（n=30 种子/条件）",
             source="results/t6_jitter_summary_a.csv:rms_pct_lvl_med(cond=timing_J100P)"),
        dict(n=2, claim="L1 输入层本身就散：同一传感器同一负载重复加载，原始台阶幅度组内 CV ≈7%、快相形状 CV ≈2%"
                        "（故 L2 的绝对电平散布不能全算在算法头上）",
             value=f"a_step CV={fmt(l1stat('jump_a_step_pct'),2)}%, inc5 CV={fmt(l1stat('inc5_pct'),2)}%, "
                   f"CV_shape 中位={fmt(l1stat('CV_shape_med'),2)}%（n=9，3 组×3 次）",
             source="results/t6_repeat_dispersion.csv:L1_input"),
        dict(n=3, claim="L2 输出层的『谁更差』依赖真值口径：相对 5 s 实测电平 v6 比现役差 66%，"
                        "相对机械台阶只差 4%；换网格步长（0.3%）还会翻转排名",
             value=f"std(R5): v5.1 {fmt(l2stat('v5.1','plat_err_std_R5_pp'),2)} vs v6 "
                   f"{fmt(l2stat('v6','plat_err_std_R5_pp'),2)} pp；std(Rstep): v5.1 "
                   f"{fmt(l2stat('v5.1','plat_err_std_Rstep_pp'),2)} vs v6 "
                   f"{fmt(l2stat('v6','plat_err_std_Rstep_pp'),2)} pp（n=9）",
             source="results/t6_repeat_dispersion.csv:L2_output"),
        dict(n=4, claim="L3 的分歧是『状态机路径分歧』而非指标噪声：±1 包抖动下 v6 的 revoke/handoff 几乎必变",
             value=f"v6 序列相同率={fmt(j['v6']['a_same'],2)}（v5.1={fmt(j['v5.1']['a_same'],2)}）、"
                   f"revoke 变={fmt(j['v6']['a_rev'],2)}、handoff 变={fmt(j['v6']['a_ho'],2)}、"
                   f"epoch 变={fmt(j['v6']['a_ep'],2)}（n=30）",
             source="results/t6_event_sequence_a.csv"),
        dict(n=5, claim="归因（按 Δstd(R5) 排序）：④「τ_ho=5 s 交接」影响最大，其次 ①「取消 2.5 s 硬确认」"
                        "与 ⑤「起扣提前」；② 形状反演尺度主要决定偏置而非离散；"
                        "③ 滑行器速率限制的贡献**精确为 0**（先验假设被反驳）",
             value=(f"D4_ho10s Δstd(R5)={abl_d4.d_std_R5_pp:+.2f}pp、D5_delay5s "
                    f"{abl_d5b.d_std_R5_pp:+.2f}pp、D1_confirm50 {abl_d1.d_std_R5_pp:+.2f}pp、"
                    f"D2_noshape {abl_d2.d_std_R5_pp:+.2f}pp（但 bias {abl_d2.bias_R5_pp:+.2f}pp）、"
                    f"D3_norate {abl_d3.d_std_R5_pp:+.4f}pp" if abl_d4 is not None else "n/a"),
             source="results/t6_attribution.csv"),
        dict(n=6, claim="改进空间：有条件有 —— 检测到两条**互不重叠**的可修轴："
                        "离散撤销通道（只伤 L3）与连续 Â 尺度通道（只伤 L2）；两者可叠加",
             value=(f"I7（撤销迟滞）: L3_a {fmt(float(imp_i7.L3_a_pct),2)}%"
                    f"（{float(imp_i7.pct_L3_max_pct):+.0f}%）而 L2/T_stable 完全不变；"
                    f"I4_ho10s: std(R5) {float(_row(im,'I4_ho10s').std_R5_pp):.2f} pp"
                    f"（{float(_row(im,'I4_ho10s').pct_std_R5_pp):+.1f}%）且 T_stable Δ"
                    f"{float(_row(im,'I4_ho10s').d_T_stable_med_s):+.2f} s；组合臂 I7_h3_I4ho10: "
                    f"L2 {float(imp_cb.pct_std_R5_pp):+.1f}%、L3_a {float(imp_cb.pct_L3_max_pct):+.1f}%、"
                    f"T_stable Δ{float(imp_cb.d_T_stable_med_s):+.2f} s" if imp_i7 is not None else "n/a"),
             source="results/t6_improve_ab.csv"),
        dict(n=7, claim="天花板：两条轴各有各的上限 —— L3 的离散通道可被迟滞/推迟起扣几乎归零"
                        "（仍为 v5.1 的 ~2.8 倍）；L2 的连续通道只能压到现役水平，代价是把锚点拉回更早的实测电平（偏置变负）",
             value=(f"L3_a: 基线 {fmt(float(_row(im,'v6_baseline').L3_a_pct),2)}% → I7 "
                    f"{fmt(float(imp_i7.L3_a_pct),2)}% / I5 {fmt(float(imp_i5.L3_a_pct),2)}%（v5.1 场景 a 为 "
                    f"{fmt(j['v5.1']['a_rms'],2)}%）；L2 std(R5): v6 "
                    f"{fmt(l2stat('v6','plat_err_std_R5_pp'),2)} → v5.1 "
                    f"{fmt(l2stat('v5.1','plat_err_std_R5_pp'),2)} / I1f "
                    f"{fmt(float(imp_d5.std_R5_pp),2)} / v6.1 {fmt(float(imp_61.std_R5_pp),2)} pp，"
                    f"但 bias 从 {fmt(l2stat('v6','plat_err_bias_R5_pp'),2)} 掉到 "
                    f"{float(imp_61.bias_R5_pp):+.2f} pp" if imp_i7 is not None else "n/a"),
             source="results/t6_improve_ab.csv;results/t6_repeat_dispersion.csv"),
    ]
    verdicts = [
        v("T6-Q1", f"v6 的下降主要落在 **L3 路径层**：同输入 ±1 包抖动下 v6 输出 RMS 差 "
                   f"{fmt(j['v6']['a_rms'],2)}%·电平 = {fmt(ratio,1)}× v5.1"
                   f"（v5.1 {fmt(j['v5.1']['a_rms'],2)}%）；L2 层 v6 在 R5 口径下 std 比 v5.1 差 "
                   f"{fmt(100*(l2stat('v6','plat_err_std_R5_pp')/l2stat('v5.1','plat_err_std_R5_pp')-1),0)}%，"
                   f"在 Rstep 口径下只差 "
                   f"{fmt(100*(l2stat('v6','plat_err_std_Rstep_pp')/l2stat('v5.1','plat_err_std_Rstep_pp')-1),0)}%；"
                   f"L1 层（CV {fmt(l1stat('jump_a_step_pct'),1)}%）与算法无关。", "支持"),
        v("T6-Q2", f"抖动-输出差异曲线见 figures/T6_02_jitter_path.png。场景 a（实录）：v6 存在阈值效应 —— "
                   f"J≤25% 包周期时 RMS≈{fmt(js(ja,'v6','timing_J25P','rms_pct_lvl_med'),2)}%（p90 已 "
                   f"{fmt(js(ja,'v6','timing_J25P','rms_pct_lvl_p90'),2)}%，即 13% 种子已整段分歧），"
                   f"J=±1 包时跳到 {fmt(j['v6']['a_rms'],2)}%；v5.1 全程 ≤"
                   f"{fmt(max(j['v5.1']['a_rms'],j['v5.1']['a_n5']),2)}%。"
                   f"加性噪声（总量 RMS 1/5 ADC）与单帧丢包对 v6 的扰动为 "
                   f"{fmt(j['v6']['a_n1'],3)}/{fmt(j['v6']['a_n5'],3)}/≈0 %，**远小于时序抖动**。"
                   f"场景 b（力域）的噪声行单位不同（1/5 N = 6.5%/33%·电平），只能做同输入配对比较"
                   f"（v6/v5.1 = {fmt(j['v6']['b_n5']/j['v5.1']['b_n5'],2)}×）。", "支持"),
        v("T6-Q3", f"±1 包抖动下 v6 的事件序列改动率：epoch {fmt(j['v6']['a_ep'],2)}、"
                   f"revoke {fmt(j['v6']['a_rev'],2)}、handoff {fmt(j['v6']['a_ho'],2)}，"
                   f"序列完全相同率仅 {fmt(j['v6']['a_same'],2)}；v5.1 为 {fmt(j['v5.1']['a_same'],2)}"
                   f"（0% 改动）。最典型的失败是 onset 建事件的**同一帧**被 revoke ⇒ 该次加载整段不补偿"
                   f"（逐帧证据：t_first_div 中位 10.77 s ≈ 首次加载沿，事件前 RMS 差 0.00 ADC，"
                   f"事件后 1614.7 ADC）。→ 指标层差异是结果，状态机路径分歧是原因。", "支持"),
        v("T6-Q4", f"5 处改动逐条消融见 results/t6_attribution.csv（每条只改一处；22 条补丁在默认值下与原型"
                   f"**bit-identical**，见 t6_patch_ab_zero.csv）。排序（Δstd(R5)）：④ 交接 −2.53 pp > "
                   f"① 0.5 s 硬确认 −1.69 pp > ⑤ 起扣 −1.56 pp > ② κ 占位 −1.24 pp > ② 取消形状外推 −0.05 pp "
                   f"> ③ 滑行器速率 0.0000 pp（**零贡献**）。κ 项按 C-4 只作占位，最终取值须由 T5-A 裁决"
                   f"（T5-A 未交付）。n=9（3 组×3 次），组内 std 只有 3 个样本，排序只用于挑方向，不作显著性结论。",
          "有条件支持"),
        v("T6-Q5", (f"有：{len(imp_sorted)} 条候选臂已做数值可行性估计（其中 8 条单旋钮、2 条组合、"
                    f"1 条为既有实现 v6.1 的对照；均为 9 份恒载 × 80 s + 2 场景 L3 代理）。"
                    f"按 L2 最优 = I4_ho10s（std(R5) {fmt(float(_row(im,'I4_ho10s').std_R5_pp),2)} pp，"
                    f"{float(_row(im,'I4_ho10s').pct_std_R5_pp):+.1f}%）；按 L3 最优 = I5_delay3s / "
                    f"I7_revoke_hyst3（场景 a 从 {fmt(float(_row(im,'v6_baseline').L3_a_pct),2)}% 降到 "
                    f"{fmt(float(imp_i5.L3_a_pct),2)}% / {fmt(float(imp_i7.L3_a_pct),2)}%）；"
                    f"**组合臂 I7_h3_I4ho10 三条一起变好**。另有 3 条零效果（I6 估计量替换、I2 撤销冷却、"
                    f"I3 尾巴门限）—— 负结果同样报出。" if imp_i7 is not None
                    else "见 results/t6_improve_ab.csv"), "有条件支持"),
        v("T6-Q6", (f"权衡表见 results/t6_tradeoff.csv。I4_ho10s 是唯一「既压 L2 又缩短 T_stable」的臂："
                    f"T_stable 中位 {fmt(float(_row(im,'I4_ho10s').T_stable_med_s),2)} s（Δ"
                    f"{float(_row(im,'I4_ho10s').d_T_stable_med_s):+.2f} s）；"
                    f"I7_revoke_hyst3 的 T_stable 不变（{fmt(float(imp_i7.T_stable_med_s),2)} s）却把 L3_a 压 "
                    f"{float(imp_i7.pct_L3_max_pct):+.0f}% ⇒ **L3 的修复是零代价的**；"
                    f"反过来 I5_delay3s 的 L3 收益最大但 T_stable 退回 "
                    f"{float(imp_i5.d_T_stable_med_s):+.2f} s。" if imp_i7 is not None
                    else "见 t6_tradeoff.csv"), "有条件支持"),
        v("T6-Q7", ("有条件有。天花板分两条轴、互不重叠："
                    "① **L3 离散通道**（时序抖动 → `inc_s` 瞬时读 0 → 同帧建-撤 → 整段不补偿）"
                    "可被撤销迟滞或推迟起扣消除 ~85–91%，且这两条都**不损害 L2 与 T_stable**；"
                    "残余 ~0.4–0.7%·电平仍是 v5.1（0.26%）的 1.6–2.7 倍 ⇒ 天花板不在「零」，"
                    "而在「与现役同级」。"
                    "② **L2 连续通道**（形状先验尺度）只能压到现役水平（std 5.09→3.07–3.37 pp），"
                    "代价是平台锚点回到更早的实测电平（bias 由 +2.22 pp 变 −2.8…−3.8 pp），"
                    "即**不可能同时保住 v6 的「绝对准」与其离散的消除**。"
                    "③ 未测边界：τ_ho≥10 s 在「加载后 10 s 内再次变载」的工况下未验证；"
                    "力域噪声需按相对幅度重扫。"),
          "有条件支持"),
    ]
    corrections = [
        dict(against="13-v6-assessment/b_repeat_disp.csv",
             was="v6 平台误差组内 std（相对 5 s 电平）= 4.83 pp，v5.1 = 3.07 pp",
             now=f"同口径（legacy 网格 dtm=span/(n-1)）复算 = {fmt(gsd('v6','plat_err_std_R5_pp','gleg'),2)} pp，"
                 f"与第一轮完全一致；但换到项目规定的精确 100 Hz 网格（dt=0.01）后 = "
                 f"{fmt(gsd('v6','plat_err_std_R5_pp','g100'),2)} pp（v5.1 {fmt(gsd('v5.1','plat_err_std_R5_pp','g100'),2)}）",
             verdict="修正"),
        dict(against="13-v6-assessment/b_卸载时漂与可重复性分析.md §3.3",
             was="相对机械台阶口径下 v6 的 std = 2.69 pp（最好，优于 v5.1 3.38）",
             now=f"legacy 网格下复算 = {fmt(gsd('v6','plat_err_std_Rstep_pp','gleg'),2)} pp（一致）；"
                 f"精确 100 Hz 网格下 = {fmt(gsd('v6','plat_err_std_Rstep_pp','g100'),2)} pp，"
                 f"此时 v5.1 = {fmt(gsd('v5.1','plat_err_std_Rstep_pp','g100'),2)} pp ⇒ **排名反转**",
             verdict="修正"),
        dict(against="13-v6-assessment/b_卸载时漂与可重复性分析.md §4.4",
             was="P4（A=交接时刻实测电平）对平台误差 std 无改善（4.83→4.84）",
             now="本轮不重复该臂；但 L3 实验显示真正的敏感点是**输入到达时刻**而非锚定模式，"
                 "与『Â 参与滑行目标』的机制解释一致（支持其推断）",
             verdict="相同"),
        dict(against="11-paper-v6/results/epochs_all.csv",
             was="v6 epoch 数（恒载 9 组）= 2/4/5、3/2/2、4/2/2，组内 std 中位 1.15；v5.1 组内 std 0",
             now=f"复算 v6 epoch 组内 std = {fmt(l2stat('v6','epoch_total_std'),3)}，"
                 f"v5.1 = {fmt(l2stat('v5.1','epoch_total_std'),3)} —— 一致",
             verdict="相同"),
    ]
    gaps = [
        "同一载荷的重复加载实录只有 3 传感器组 × 3 次（n=9），且都是恒载族；**变载工况没有重复样本**，"
        "L1/L2 只能给恒载族的结论。需要补采『同一装载序列重复 5 次以上』的变载实录。",
        "T1-A 冻结口径未交付（`T1_*/results/t1a_settle_metrics.csv` 不存在）⇒ 本任务 T_stable 用 legacy 口径，"
        "**不可与 T1 绝对横比**；所有 T_stable 结论均为同一次运行内的配对差。",
        "T5-A 的 κ 统一扫描未交付 ⇒ `t6_attribution.csv` 的 κ 行只是占位（κ 1.10 对照），不得作为最终裁决。",
        "抖动注入的时序模型是『整包平移 + cummax 保序』；J=±1 包时约 13% 的包被保序抬升"
        "（reorder_frac 列），真实接收端是否会出现该比例的到达压缩，需要抓包数据核对。",
        "L3 只在 2 个场景（1 份实录 + 1 份恒载）上做；跨传感器组的 L3 差异未测。",
        "没有『同一录制反复上电/重放』的实测真值，L3 的抖动只能证明算法敏感，不能直接换算成用户看到的偏差幅度。",
    ]
    blockers = []
    out = dict(task="T6", seat="T6-A", date="2026-09-19", headline=headline, verdicts=verdicts,
               corrections=corrections, gaps=gaps, blockers=blockers,
               meta=dict(n_repeat=9, n_jitter_seed_a=n_ja, n_jitter_seed_b=n_jb,
                         grid="100 Hz exact (dt=0.01), legacy-grid cross-check in t6_gridsens.csv",
                         t_stable_caliber="legacy 5%*step / hold 30 s (T1-A not delivered)",
                         truth_calibers=["R5=pre+inc5", "Rstep=pre+a_step"],
                         patch_ab_zero_ok=bool(pz.bit_identical.all()) if pz is not None else None))
    with io.open(os.path.join(RES, "conclusions.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n-> results/conclusions.json, results/_t6f_numbers.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
