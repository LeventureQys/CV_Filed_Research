# -*- coding: utf-8 -*-
"""T7-B 步骤 2（本席位核心）：加载 vs 卸载的时间常数对称性检验（T7-Q8）。

⚠ 关键方法要求：加载侧必须**按等效输入斜坡 `T_ramp` 分层**再比。
  否则会把"输入不同"误当成"响应不对称"（T4-A 已证 restep 的慢来自输入斜坡 0.55 s）。
  unload 侧 T4-A 未做反卷积 ⇒ 无 `T_ramp`；用**被动性上界**代替：
  单调/无源系统的输出不可能比输入更快 ⇒ 输出 t90 是输入 t90 的**上界**，
  故"卸载 T_ramp ≤ 输出 t90"是可用的保守边界（报告 §2 明写该假设）。

输入：results/t7b_events_metrics.csv（步骤 1，方向统一口径 + ±1 包敏感性）
输出：results/t7b_asymmetry.csv        分域/分层/分幅度档的对照表
      results/t7b_asymmetry_pairs.csv  同录制同 |J| 配对逐对明细
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t7b_ad_lib import RESULTS, ensure_dirs  # noqa: E402

ensure_dirs()
LOG = os.path.join(RESULTS, "_t7b_02_asymmetry.log")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0)
    return (float(np.median(v)), float(np.percentile(v, 10)),
            float(np.percentile(v, 90)), len(v))


def contrast(a, b, metric, cid, scope, stratum_var="", stratum="", note=""):
    """a = 加载侧值，b = 卸载侧值。ratio = 加载/卸载（>1 表示加载更慢）。"""
    am, ap10, ap90, na = q(a)
    bm, bp10, bp90, nb = q(b)
    ratio = am / bm if (np.isfinite(am) and np.isfinite(bm) and bm > 1e-9) else np.nan
    return dict(contrast_id=cid, scope=scope, stratum_var=stratum_var, stratum=stratum,
                metric=metric, n_load=na, load_med=am, load_p10=ap10, load_p90=ap90,
                n_unload=nb, unload_med=bm, unload_p10=bp10, unload_p90=bp90,
                ratio_load_over_unload=ratio, note=note)


def mw(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def main():
    print("CMD: python scripts/t7b_02_asymmetry.py")
    print("=" * 78)
    m = pd.read_csv(os.path.join(RESULTS, "t7b_events_metrics.csv"))
    # 减重侧改用**实测重新分档**（步骤 1 的 kind_eff）：T4-A 的 partial_unload 标签里
    # 有 2 例实测为全卸载（post 窗被后续事件污染，见报告 §④）
    amp = pd.read_csv(os.path.join(RESULTS, "t7b_partial_amp_response.csv"))
    m = m.merge(amp[["event_id", "kind_eff", "drop_frac_eff", "absJ_ad",
                     "t90_ad", "t50_ad", "f_005_ad", "f_01_ad", "f_02_ad", "f_05_ad",
                     "f_10_ad", "t90_ad_m1", "t90_ad_p1"]],
                on="event_id", how="left")
    m["kind_u"] = np.where(m["kind_eff"].notna(), m["kind_eff"], m["kind"])
    # 减重侧的时间/形状指标一律改用**自适应 post 窗**的值（污染窗会把 f 抬到 4~25）
    dec_mask = m["kind_eff"].isin(["unload", "partial_unload", "decrement_small"])
    for src_c, dst_c in (("t90_ad", "t90"), ("t50_ad", "t50"), ("f_005_ad", "f_005"),
                         ("f_01_ad", "f_01"), ("f_02_ad", "f_02"), ("f_05_ad", "f_05"),
                         ("f_10_ad", "f_10"), ("absJ_ad", "absJ"),
                         ("drop_frac_eff", "drop_frac_meas"),
                         ("t90_ad_m1", "t90_m1"), ("t90_ad_p1", "t90_p1")):
        m.loc[dec_mask, dst_c] = m.loc[dec_mask, src_c].to_numpy(float)
    rows = []

    # ── 0 组内基线 ──
    print("[0] 分组指标（方向统一 f(τ) 口径；中位）")
    g = m.groupby("kind_u").agg(
        n=("t90", "size"), absJ_med=("absJ", "median"),
        t50_med=("t50", "median"), t90_med=("t90", "median"),
        f_005=("f_005", "median"), f_01=("f_01", "median"),
        f_02=("f_02", "median"), f_05=("f_05", "median"), f_10=("f_10", "median"),
        step_frac=("step_frame_frac", "median"), top3_frac=("top3_frame_frac", "median"))
    print(g.to_string(float_format="%.3f"))

    load_all = m[m["kind_u"].isin(["onset", "restep"])]
    unl = m[m["kind_u"] == "unload"]
    par = m[m["kind_u"] == "partial_unload"]

    # ── 1 朴素对照（不按输入分层）──
    for met in ("t50", "t90", "f_005", "f_01", "f_02", "f_05", "f_10"):
        rows.append(contrast(load_all[met], unl[met], met, "A1_naive_all", "全部(13 份录制)",
                             "无", "未按输入分层（仅供对照，不作为裁决依据）",
                             "朴素对照"))
    # 分域
    for dom in ("显示域", "ADC域"):
        rows.append(contrast(load_all[load_all.domain == dom]["t90"],
                             unl[unl.domain == dom]["t90"], "t90",
                             "A1_naive_%s" % dom, dom, "无", "朴素对照（分域，避免跨域混淆）"))

    # ── 2 按 T_ramp 分层（加载侧）──
    bins = [(-0.001, 0.05, "T_ramp<=0.05s(近阶跃)"),
            (0.05, 0.30, "0.05<T_ramp<=0.30s"),
            (0.30, 99.0, "T_ramp>0.30s(人工慢压)")]
    print("\n[2] 加载侧按 T_ramp 分层")
    for lo, hi, lab in bins:
        sub = load_all[(load_all["T_ramp"] > lo) & (load_all["T_ramp"] <= hi)]
        print("  %-26s n=%d  kind=%s  t90 中位 %.3f  f_02 中位 %.3f"
              % (lab, len(sub), dict(sub["kind"].value_counts()),
                 sub["t90"].median(), sub["f_02"].median()))
        for met in ("t90", "f_02", "f_05"):
            rows.append(contrast(sub[met], unl[met], met, "A2_byTramp", "全部",
                                 "T_ramp_load", lab,
                                 "卸载侧 T_ramp 不可测；用被动性上界 T_ramp_unload<=t90_out"))
    # 分层 × 分域
    for lo, hi, lab in bins:
        for dom in ("显示域", "ADC域"):
            sub = load_all[(load_all["T_ramp"] > lo) & (load_all["T_ramp"] <= hi) &
                           (load_all.domain == dom)]
            if len(sub) >= 1:
                for met in ("t90", "f_02"):
                    rows.append(contrast(sub[met], unl[unl.domain == dom][met], met,
                                         "A2_byTramp_%s" % dom, dom, "T_ramp_load", lab,
                                         "n 小，参考"))

    # ── 3 同录制同 |J| 配对（|J| 差 ≤15%）──
    print("\n[3] 同录制、|J| 相差 ≤15% 的加载/卸载配对")
    pairs = []
    for key, gk in m.groupby("key"):
        lk = gk[gk["kind_u"].isin(["onset", "restep"])]
        uk = gk[gk["kind_u"] == "unload"]
        for _, a in lk.iterrows():
            for _, b in uk.iterrows():
                if abs(a["absJ"] - b["absJ"]) / max(b["absJ"], 1e-9) <= 0.15:
                    pairs.append(dict(
                        key=key, domain=a["domain"],
                        load_event=a["event_id"], load_kind=a["kind"],
                        unload_event=b["event_id"],
                        absJ_load=a["absJ"], absJ_unload=b["absJ"],
                        dJ_pct=100 * (a["absJ"] - b["absJ"]) / max(b["absJ"], 1e-9),
                        T_ramp_load=a["T_ramp"],
                        t50_load=a["t50"], t50_unload=b["t50"],
                        t90_load=a["t90"], t90_unload=b["t90"],
                        t90_ratio=(a["t90"] / b["t90"]) if b["t90"] > 1e-9 else np.nan,
                        t90_ratio_lo=((max(a["t90"] - a["pkt_dt"], 0.0)) /
                                      (b["t90"] + b["pkt_dt"])) if b["t90"] >= 0 else np.nan,
                        f02_load=a["f_02"], f02_unload=b["f_02"],
                        f05_load=a["f_05"], f05_unload=b["f_05"],
                        strict=bool(a["amp_ok"] and b["amp_ok"] and
                                    a["post_win_clean"] and b["post_win_clean"]),
                        pkt_dt=a["pkt_dt"]))
    pdf = pd.DataFrame(pairs)
    if len(pdf):
        print(pdf[["key", "load_event", "unload_event", "absJ_load", "absJ_unload",
                   "T_ramp_load", "t90_load", "t90_unload", "t90_ratio",
                   "t90_ratio_lo", "f02_load", "f02_unload", "strict"]]
              .to_string(index=False, float_format="%.3f"))
        print("  配对 t90 比值：中位 %.1f（p10~p90 %.1f~%.1f）；保守下界中位 %.2f"
              % (np.median(pdf["t90_ratio"]), np.percentile(pdf["t90_ratio"], 10),
                 np.percentile(pdf["t90_ratio"], 90), np.median(pdf["t90_ratio_lo"])))
        stp = pdf[pdf["strict"]]
        if len(stp):
            print("  严格配对 n=%d：比值中位 %.1f，保守下界中位 %.2f"
                  % (len(stp), np.median(stp["t90_ratio"]),
                     np.median(stp["t90_ratio_lo"])))
        print("  分域：%s" % pdf.groupby("domain")["t90_ratio"].median().to_dict())
        pdf.to_csv(os.path.join(RESULTS, "t7b_asymmetry_pairs.csv"), index=False,
                   encoding="utf-8-sig")
        rows.append(contrast(pdf["t90_load"], pdf["t90_unload"], "t90", "A3_matched",
                             "同录制同|J|", "|J| 差<=15%", "全部配对",
                             "n_pair=%d" % len(pdf)))
        rows.append(dict(contrast_id="A3_matched", scope="同录制同|J|",
                         stratum_var="ratio", stratum="逐对 t90_load/t90_unload",
                         metric="t90_ratio", n_load=len(pdf),
                         load_med=float(np.median(pdf["t90_ratio"])),
                         load_p10=float(np.percentile(pdf["t90_ratio"], 10)),
                         load_p90=float(np.percentile(pdf["t90_ratio"], 90)),
                         n_unload=len(pdf), unload_med=np.nan, unload_p10=np.nan,
                         unload_p90=np.nan,
                         ratio_load_over_unload=float(np.median(pdf["t90_ratio"])),
                         note="保守下界 t90_ratio_lo 中位 %.2f" %
                              np.median(pdf["t90_ratio_lo"])))
        for met in ("f02", "f05"):
            rows.append(contrast(pdf["%s_load" % met], pdf["%s_unload" % met], met,
                                 "A3_matched", "同录制同|J|", "|J| 差<=15%", "全部配对"))

    # ── 4 幅度档对照（drop_frac 分档 + 域内 |J| 分档）──
    print("\n[4] 幅度档 × 方向（drop_frac = |J|/pre）")
    dbands = [(0.0, 0.2, "<20%"), (0.2, 0.8, "20-80%"), (0.8, 1.5, ">=80%")]
    for lo, hi, lab in dbands:
        for kind in ("onset", "restep", "unload", "partial_unload", "decrement_small", "scan_decrement"):
            sub = m[m["kind_u"] == kind]
            sub = sub[(sub["drop_frac_meas"] > lo) & (sub["drop_frac_meas"] <= hi)]
            if len(sub) == 0:
                continue
            am, ap10, ap90, na = q(sub["t90"])
            fm, _, _, _ = q(sub["f_02"])
            print("  drop_frac %-9s %-15s n=%-3d t90 中位 %s  f_02 中位 %s"
                  % (lab, kind, na, ("%.3f" % am) if np.isfinite(am) else "n/a",
                     ("%.3f" % fm) if np.isfinite(fm) else "n/a"))
            rows.append(dict(contrast_id="A4_band", scope="幅度档", stratum_var="drop_frac",
                             stratum=lab, metric="t90|%s" % kind, n_load=na,
                             load_med=am, load_p10=ap10, load_p90=ap90,
                             n_unload=0, unload_med=np.nan, unload_p10=np.nan,
                             unload_p90=np.nan, ratio_load_over_unload=np.nan,
                             note="f_02 中位 %s" % (("%.3f" % fm) if np.isfinite(fm) else "n/a")))
    # 域内 |J| 分档
    for dom in ("显示域", "ADC域"):
        sub = m[(m.domain == dom) & (m["kind_u"].isin(["onset", "restep", "unload"]))]
        if len(sub) == 0:
            continue
        qs = np.percentile(sub["absJ"], [33, 67])
        for i, (lo, hi, lab) in enumerate([(-1, qs[0], "低幅"), (qs[0], qs[1], "中幅"),
                                           (qs[1], 1e18, "高幅")]):
            s2 = sub[(sub["absJ"] > lo) & (sub["absJ"] <= hi)]
            for kind in ("onset", "unload"):
                s3 = s2[s2["kind_u"] == kind]
                if len(s3) == 0:
                    continue
                am, ap10, ap90, na = q(s3["t90"])
                rows.append(dict(contrast_id="A4_absJ_%s" % dom, scope=dom,
                                 stratum_var="|J| 三分位", stratum="%s(%.0f~%.0f)" %
                                 (lab, max(lo, 0), hi if hi < 1e17 else s2["absJ"].max()),
                                 metric="t90|%s" % kind, n_load=na, load_med=am,
                                 load_p10=ap10, load_p90=ap90, n_unload=0,
                                 unload_med=np.nan, unload_p10=np.nan, unload_p90=np.nan,
                                 ratio_load_over_unload=np.nan, note=""))

    # ── 5 统计检验 + 保守边界 ──
    print("\n[5] 统计检验（Mann-Whitney U, two-sided）与保守边界")
    p_unl_ons = mw(m[m["kind_u"] == "unload"]["t90"], m[m["kind_u"] == "onset"]["t90"])
    p_unl_res = mw(m[m["kind_u"] == "unload"]["t90"], m[m["kind_u"] == "restep"]["t90"])
    p_f02 = mw(m[m["kind_u"] == "unload"]["f_02"],
               m[m["kind_u"].isin(["onset", "restep"])]["f_02"])
    print("  unload vs onset  t90: p=%.2e" % p_unl_ons)
    print("  unload vs restep t90: p=%.2e" % p_unl_res)
    print("  unload vs load   f_02: p=%.2e" % p_f02)
    rows.append(dict(contrast_id="A5_mw", scope="全部", stratum_var="", stratum="",
                     metric="t90|unload_vs_onset", n_load=len(m[m['kind'] == 'onset']),
                     load_med=m[m['kind'] == 'onset']['t90'].median(), load_p10=np.nan,
                     load_p90=np.nan, n_unload=len(unl), unload_med=unl['t90'].median(),
                     unload_p10=np.nan, unload_p90=np.nan,
                     ratio_load_over_unload=(m[m['kind'] == 'onset']['t90'].median() /
                                             max(unl['t90'].median(), 1e-9)),
                     note="Mann-Whitney p=%.2e" % p_unl_ons))
    # 保守边界：卸载 t90 加 1 包、加载 t90 减 1 包
    step_load = m[m["kind_u"].isin(["onset", "restep"])].copy()
    step_load["t90_lo"] = np.maximum(step_load["t90"] - step_load["pkt_dt"], 0.0)
    t90_hi_unl = unl["t90"] + unl["pkt_dt"]
    print("  保守边界（加载 t90−1包 vs 卸载 t90+1包）：加载中位 %.3f s、卸载上界中位 %.3f s、"
          "比值下界 %.2f" % (step_load["t90_lo"].median(), t90_hi_unl.median(),
                            step_load["t90_lo"].median() / max(t90_hi_unl.median(), 1e-9)))
    rows.append(contrast(step_load["t90_lo"], t90_hi_unl, "t90", "A5_conservative",
                         "全部", "±1 包", "加载 −1包 / 卸载 +1包",
                         "对对称性假设最有利的边界；仍 >1 即不对称"))
    print("  f(0.2 s)（≥5 包，不受 ±1 包影响）：加载中位 %.3f、卸载中位 %.3f"
          % (step_load["f_02"].median(), unl["f_02"].median()))
    rows.append(contrast(step_load["f_02"], unl["f_02"], "f_02", "A5_packet_robust",
                         "全部", "≥5 包", "τ=0.2 s 完成度", "包量化不敏感的指标"))

    # ── 6 严格子样本（幅度可分辨 & post 窗干净）──
    print("\n[6] 严格子样本（amp_ok & post_win_clean）")
    st = m[m["amp_ok"].astype(bool) & m["post_win_clean"].astype(bool)]
    print("  n=%d  kind=%s" % (len(st), dict(st["kind"].value_counts())))
    gst = st.groupby("kind_u")[["t90", "f_005", "f_01", "f_02", "f_05"]].median()
    print(gst.to_string(float_format="%.3f"))
    st_load = st[st["kind_u"].isin(["onset", "restep"])]
    st_unl = st[st["kind_u"] == "unload"]
    for met in ("t90", "f_02", "f_05"):
        rows.append(contrast(st_load[met], st_unl[met], met, "A6_strict", "严格子样本",
                             "amp_ok & post_win_clean", "全部",
                             "剔除了低信噪比与 post 窗污染事件"))
    rows.append(dict(contrast_id="A6_strict", scope="严格子样本", stratum_var="", stratum="",
                     metric="t90|partial_unload(n=2)", n_load=2,
                     load_med=float(st[st["kind_u"] == "partial_unload"]['t90'].median()),
                     load_p10=np.nan, load_p90=np.nan, n_unload=0, unload_med=np.nan,
                     unload_p10=np.nan, unload_p90=np.nan, ratio_load_over_unload=np.nan,
                     note="仅 2 例，且两例行为相反（见报告 §A.3）"))
    # 按录音分域统计严格子样本
    for dom in ("显示域", "ADC域"):
        for lab, sub in (("onset", st_load[st_load.domain == dom]),
                         ("restep", st_load[st_load.domain == dom])):
            if lab == "restep":
                continue
            for met in ("t90", "f_02"):
                rows.append(contrast(sub[met], st_unl[st_unl.domain == dom][met], met,
                                     "A6_strict_%s" % dom, dom, "严格子样本", "onset vs unload",
                                     "n_onset=%d n_unload=%d" % (len(sub),
                                                                 len(st_unl[st_unl.domain == dom]))))
    # 低信噪比/窗污染事件剔除前后
    print("  被剔除：低信噪比 %d 个、post 窗被后续事件污染 %d 个"
          % (int((~m["amp_ok"].astype(bool)).sum()), int((~m["post_win_clean"].astype(bool)).sum())))

    # ── 7 用 T4-A 自己的 t90（加载侧）复核 ──
    lc = m.dropna(subset=["t90_t4a"])
    lc_load = lc[lc["kind_u"].isin(["onset", "restep"])]["t90_t4a"]
    rows.append(contrast(lc_load, unl["t90"], "t90", "A7_t4a_t90", "全部",
                         "加载侧用 T4-A 的 t90 列", "cross-check",
                         "口径交叉核对：加载侧换 T4-A 原列后比值（卸载侧两口径一致）"))
    print("\n[7] 加载侧改用 T4-A 自报 t90 复核：加载中位 %.3f s、卸载中位 %.3f s、比值 %.1f"
          % (lc_load.median(), unl["t90"].median(),
             lc_load.median() / max(unl["t90"].median(), 1e-9)))
    dom_ck = lc.groupby(["domain", "kind"]).apply(
        lambda g: (g["t90"] - g["t90_t4a"]).abs().max(), include_groups=False)
    print("  t90 口径差（T7-B vs T4-A，最大 |Δ|）：\n%s" % dom_ck.to_string())

    # ── 8 双口径一致的最严子样本 ──
    print("\n[8] 最严子样本（amp_ok & post_win_clean & |T7-B t90 − T4-A t90| ≤ 0.5 s）")
    m["t90_agree"] = (m["t90"] - m["t90_t4a"]).abs() <= 0.5
    s8 = m[m["amp_ok"].astype(bool) & m["post_win_clean"].astype(bool) &
           m["t90_agree"].fillna(False)]
    l8 = s8[s8["kind_u"].isin(["onset", "restep"])]
    u8 = s8[s8["kind_u"] == "unload"]
    print("  n=%d（加载 %d / 卸载 %d / 部分卸载 %d）"
          % (len(s8), len(l8), len(u8), int((s8["kind_u"] == "partial_unload").sum())))
    print("  加载 t90 中位 %.3f s、卸载 t90 中位 %.3f s、f_02 %.3f vs %.3f"
          % (l8["t90"].median(), u8["t90"].median(), l8["f_02"].median(),
             u8["f_02"].median()))
    for met in ("t90", "f_02"):
        rows.append(contrast(l8[met], u8[met], met, "A8_tight", "最严子样本",
                             "双口径一致", "onset+restep vs unload",
                             "n_load=%d n_unload=%d" % (len(l8), len(u8))))
    print("  保守边界（加载−1包 vs 卸载+1包）比值 %.2f"
          % (np.maximum(l8["t90"] - l8["pkt_dt"], 0).median() /
             max((u8["t90"] + u8["pkt_dt"]).median(), 1e-9)))
    rows.append(contrast(np.maximum(l8["t90"] - l8["pkt_dt"], 0), u8["t90"] + u8["pkt_dt"],
                         "t90", "A8_tight_conservative", "最严子样本", "±1 包",
                         "加载 −1包 / 卸载 +1包", ""))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "t7b_asymmetry.csv"), index=False,
               encoding="utf-8-sig")
    print("\nWROTE results/t7b_asymmetry.csv (n=%d), results/t7b_asymmetry_pairs.csv (%d 对)"
          % (len(out), len(pdf)))


if __name__ == "__main__":
    tee = Tee(LOG)
    _o = sys.stdout
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = _o
        tee.flush()
        tee.f.close()
