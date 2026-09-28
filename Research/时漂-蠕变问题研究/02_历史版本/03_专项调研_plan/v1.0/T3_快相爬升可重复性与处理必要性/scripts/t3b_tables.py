# -*- coding: utf-8 -*-
"""t3b_tables.py —— T3-B 的汇总层：把「臂 × 事件」指标表聚合成报告用的表。

产出：
  results/t3b_route_roi.csv      ← 处理强度扫描（含 rom_scale/kappa/trigger_rate/glide_rate/
                                    onset_only 各维 + 四联指标 T_stable / OS% / 1 s 误差 / MD）
  results/t3b_route_q6q7.csv     ← 逐录制的稳定时间分布（Q6/Q7 的"13 份录制上的分布"）
  results/t3b_arm_kind.csv       ← 臂 × 事件类别（onset/restep/clean）的三态细分

报数规范（指标字典 §6）：n<20 报中位 + p10~p90；每个结论标 n 与口径。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_common as C   # noqa: E402
import t3b_settle as ST  # noqa: E402

RES = C.RES
M = "t3b_arm_metrics.csv"
RC = "t3b_route_recording.csv"

# 主口径：T1-A 冻结 D1（完整 30 s 窗、不因后续事件截断）
# 实录可测口径：D1-ev（窗在下一真实事件处截断，可用窗 ≥5 s）
PRIM = "T_stable_ev_tot5"     # 头条第 1 口径：总通道 + D1-ev
PRIM_CH = "T_stable_ev_ch5"   # 头条第 2 口径：主通道 + D1-ev（更严）


def _mc(df, arm, col, kind=None, clean=None, dom=None):
    s = df[df.arm == arm]
    if kind is not None:
        s = s[s.kind.isin(kind if isinstance(kind, (list, tuple)) else [kind])]
    if clean:
        s = s[s.clean_t4a.astype(bool)]
    if dom is not None:
        s = s[s.dom == dom]
    return s[col]


def per_rec_dist(df, recs, arm, col):
    """逐录制分布：先在该录制内取事件中位，再在 13 份录制上取中位/p10/p90（n=录制数）。"""
    s = df[df.arm == arm]
    vals = []
    for rec, g in s.groupby("rec"):
        v = g[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(float(np.median(v)))
    a = np.asarray(vals, float)
    if a.size == 0:
        return dict(n_rec=0, med=np.nan, p10=np.nan, p90=np.nan, frac_le_2s=np.nan)
    return dict(n_rec=int(a.size), med=float(np.median(a)),
                p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                frac_le_2s=float(np.mean(a <= 2.0)))


def arm_agg(df, recs, arm):
    """一条臂的全部汇总量（含 onset/restep 细分与触发率）。"""
    d = df[df.arm == arm]
    r = recs[recs.arm == arm]
    out = dict(arm=arm, n_events=int(len(d)),
               n_onset=int((d.kind == "onset").sum()),
               n_restep=int((d.kind == "restep").sum()),
               n_clean=int(d.clean_t4a.astype(bool).sum()),
               n_rec=int(r.rec.nunique()))
    # 四联指标（主口径 = 总通道 D1-ev）
    for col, tag in ((PRIM, "Tstab_tot_ev"), (PRIM_CH, "Tstab_ch_ev"),
                     ("T_stable_tot5", "Tstab_tot_D1"), ("T_stable_ch5", "Tstab_ch_D1"),
                     ("T_settle_tot5", "Tsettle5_tot"), ("T_settle_ch5", "Tsettle5_ch")):
        v = _mc(df, arm, col)
        out[tag + "_n"] = int(np.isfinite(v).sum())
        out[tag + "_med"] = ST.med(v)
        out[tag + "_p10"] = ST.q(v, 10)
        out[tag + "_p90"] = ST.q(v, 90)
    # 删失率（D1 在实录上大量删失，指标字典口径要求完整 30 s 窗）
    for col, tag in (("cens_tot5", "cens_tot_D1"), ("cens_ch5", "cens_ch_D1"),
                     ("cens_ev_tot5", "cens_tot_ev")):
        v = _mc(df, arm, col).to_numpy(float)
        out[tag + "_frac"] = float(np.mean(v)) if v.size else np.nan
    # 代价类
    for col, tag in (("OS_pct", "OS_ch"), ("OS_tot_pct", "OS_tot"), ("US_pct", "US_ch"),
                     ("err_1s_pct", "err1_ch"), ("err_2s_pct", "err2_ch"),
                     ("err_1s_tot_pct", "err1_tot"), ("MD_tot_adc", "MD_tot"),
                     ("MD_ch_adc", "MD_ch")):
        v = _mc(df, arm, col)
        out[tag + "_med"] = ST.med(v)
        out[tag + "_p90"] = ST.q(v, 90)
        out[tag + "_max"] = float(np.nanmax(np.abs(v.to_numpy(float)))) if v.notna().any() else np.nan
    # 台阶捕获比 G：只用 ADC 域且 |Δ原始(20 s)| ≥ 2000 ADC 的事件（与 06 号文档口径对齐）
    g = df[(df.arm == arm) & (df.dom == "ADC域") & (df.G_den_adc >= 2000) & df.G_20s.notna()]
    out["G_n"] = int(len(g))
    out["G_med"] = ST.med(g.G_20s)
    out["G_min"] = float(g.G_20s.min()) if len(g) else np.nan
    # epoch / 触发率
    out["epoch_per100s"] = ST.med(r.epoch_per100s) if len(r) else np.nan
    out["epoch_per100s_p90"] = ST.q(r.epoch_per100s, 90) if len(r) else np.nan
    ni, nc = float(r.n_inv.sum()), float(r.n_cap.sum())
    out["inv_calls"] = ni
    out["trigger_rate"] = (nc / ni) if ni > 0 else np.nan
    out["n_cap"] = nc
    # onset / restep 细分（主口径）
    for kind, tag in (("onset", "onset"), ("restep", "restep")):
        v = _mc(df, arm, PRIM, kind=kind)
        out["Tstab_tot_ev_med_" + tag] = ST.med(v)
        out["n_" + tag] = int(np.isfinite(v).sum())
        out["OS_ch_med_" + tag] = ST.med(_mc(df, arm, "OS_pct", kind=kind))
        out["err1_ch_med_" + tag] = ST.med(_mc(df, arm, "err_1s_pct", kind=kind))
        out["MD_tot_med_" + tag] = ST.med(_mc(df, arm, "MD_tot_adc", kind=kind))
    # ROI 维度（从臂定义里取，不从数据里猜）
    import t3b_arms as A
    a = A.ARM_BY_NAME.get(arm)
    if a is not None:
        out["group"] = a["group"]
        out["dim"] = a["dim"]
        out["dim_value"] = a["dim_value"]
        out["rom_scale"] = A._rom_dim(a)
        out["kappa_onset"] = a["params"].get("KAPPA_ONSET", np.nan)
        out["kappa_restep"] = a["params"].get("KAPPA_RESTEP", np.nan)
        out["glide_rate"] = a["params"].get("RATE_MAX", np.nan)
        out["onset_only"] = a["params"].get("N_GATE", 0)
        out["ema_tau"] = a["params"].get("EMA_TAU", 0.0)
        out["anchor_mode"] = a["params"].get("ANCHOR_MODE", "")
        out["trim_rate"] = a["params"].get("TRIM_RATE", 0.0)
    return out


def build_all(verbose=True):
    mp = os.path.join(RES, M)
    rp = os.path.join(RES, RC)
    df = pd.read_csv(mp, encoding="utf-8-sig")
    recs = pd.read_csv(rp, encoding="utf-8-sig")
    import t3b_arms as A
    order = [a["name"] for a in A.ARMS if a["name"] in set(df.arm.unique())]
    rows = [arm_agg(df, recs, a) for a in order]
    roi = pd.DataFrame(rows)
    roi.to_csv(os.path.join(RES, "t3b_route_roi.csv"), index=False, encoding="utf-8-sig")
    # 逐录制分布（Q6/Q7）
    dist = []
    for a in order:
        for col, tag in ((PRIM, "tot_ev"), (PRIM_CH, "ch_ev"), ("T_stable_tot5", "tot_D1")):
            dd = per_rec_dist(df, recs, a, col)
            dist.append(dict(arm=a, col=col, tag=tag, **dd))
    dd = pd.DataFrame(dist)
    dd.to_csv(os.path.join(RES, "t3b_route_q6q7.csv"), index=False, encoding="utf-8-sig")
    # 臂 × 类别
    g = df.groupby(["arm", "kind"]).agg(
        n=("t_on", "size"), T_ev_tot5_med=(PRIM, "median"),
        T_ev_tot5_p10=(PRIM, lambda s: s.quantile(.10)),
        T_ev_tot5_p90=(PRIM, lambda s: s.quantile(.90)),
        OS_ch_med=("OS_pct", "median"), err1_med=("err_1s_pct", "median"),
        MD_tot_med=("MD_tot_adc", "median"), MD_tot_max=("MD_tot_adc", "max")).reset_index()
    g.to_csv(os.path.join(RES, "t3b_arm_kind.csv"), index=False, encoding="utf-8-sig")
    if verbose:
        print("-> results/t3b_route_roi.csv（%d 臂）" % len(roi))
        print("-> results/t3b_route_q6q7.csv（%d 行）" % len(dd))
        print("-> results/t3b_arm_kind.csv（%d 行）" % len(g))
    return roi, dd, g


if __name__ == "__main__":
    build_all()
