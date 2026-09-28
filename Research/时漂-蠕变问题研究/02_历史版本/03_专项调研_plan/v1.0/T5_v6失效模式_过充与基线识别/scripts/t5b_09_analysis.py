# -*- coding: utf-8 -*-
"""T5-B / 09：后处理（不重跑算法）—— 把逐 trial / 逐事件明细汇总成报告可直接引用的表。

产物
    results/t5b_failure_by_event.csv  逐 (窗, 扰动类, 幅值, 真实沿) 的 M1/M2 率 + 台阶分层
    results/t5b_failure_strata.csv    按"大台阶/中台阶/小台阶"分层的失败率
    results/t5b_boundary_table.csv    分层 × 分量 × 扰动类 的 logistic A50（含 bootstrap CI）
    results/t5b_q1_headline.csv       Q1 结论数字（1w ADC 场景失败率，逐档次）
    results/_t5b_09.log
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)
from t5b_02_sweep import logit_fit, logit_boot, wilson     # noqa: E402


def stratum(jump_over_level):
    if jump_over_level >= 0.20:
        return "big(>=20%)"
    if jump_over_level >= 0.10:
        return "mid(10-20%)"
    if jump_over_level >= 0.05:
        return "small(5-10%)"
    return "tiny(<5%)"


def main():
    det = pd.read_csv(os.path.join(RES, "t5b_event_detail.csv"), encoding="utf-8-sig")
    tr = pd.read_csv(os.path.join(RES, "t5b_trials_raw.csv"), encoding="utf-8-sig")
    det["jol"] = (det["jump"].abs() / det["lvl_event"].abs()).round(4)
    det["stratum"] = det["jol"].map(stratum)
    det = det[det["should_detect"] == 1]
    # 只统计"干净运行也检出"的事件（增量口径）
    deti = det[det["n_match_clean"] > 0].copy()
    # ── 逐事件率 ──
    g = deti.groupby(["win", "cls", "amp", "amp_pct", "t_on", "kind_gt", "stratum"],
                     dropna=False)
    rows = []
    for k, gg in g:
        n = len(gg)
        m1, m2 = int(gg["M1_miss"].sum()), int(gg["M2"].sum())
        n2 = int(gg["ck_meas"].sum())
        rows.append(dict(zip(["win", "cls", "amp", "amp_pct", "t_on", "kind_gt", "stratum"], k),
                         n=n, M1=m1, p_M1=round(m1 / n, 3),
                         n_M2=n2, M2=m2, p_M2=round(m2 / n2, 3) if n2 else np.nan,
                         dA_rel_med=round(float(gg["dA_rel"].median()), 3) if n2 else np.nan,
                         dA_rel_p90=round(float(gg["dA_rel"].quantile(0.9)), 3) if n2 else np.nan))
    bye = pd.DataFrame(rows)
    bye.to_csv(os.path.join(RES, "t5b_failure_by_event.csv"), index=False, encoding="utf-8-sig")
    # ── 分层汇总（按 扰动类 × 幅值 × 分层）──
    rows = []
    for (cls, amp, st), gg in deti.groupby(["cls", "amp", "stratum"], dropna=False):
        n = len(gg)
        r = dict(cls=cls, amp=amp, stratum=st, n=n,
                 p_M1=round(float(gg["M1_miss"].mean()), 3),
                 p_M2=round(float(gg["M2"].mean()), 3),
                 dA_rel_med=round(float(gg["dA_rel"].median()), 4))
        rows.append(r)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(RES, "t5b_failure_strata.csv"), index=False, encoding="utf-8-sig")
    # ── 边界表：分层 × 分量 × 扰动类（对（幅值, 事件级 0/1）做 logistic）──
    fits = []
    for (cls, stn, comp) in [(c, s, p) for c in deti["cls"].unique()
                             for s in deti["stratum"].unique()
                             for p in ("M1_miss", "M2")]:
        gg = deti[(deti["cls"] == cls) & (deti["stratum"] == stn)]
        if cls in ("tap",):
            continue
        x = gg["amp"].to_numpy(float)
        y = gg[comp].to_numpy(float)
        if len(x) < 6 or len(np.unique(y)) < 2:
            continue
        a50, b, _ = logit_fit(x, y)
        lo, hi = logit_boot(x, y, n_boot=150, seed=3)
        fits.append(dict(class_=cls, stratum=stn, component=comp, n=len(x),
                         n_fail=int(y.sum()), p_min=round(float(y[np.argmin(x)]), 3),
                         A50=(round(a50, 1) if np.isfinite(a50) else np.nan),
                         A50_ci_lo=round(lo, 1), A50_ci_hi=round(hi, 1)))
    bt = pd.DataFrame(fits)
    bt.to_csv(os.path.join(RES, "t5b_boundary_table.csv"), index=False, encoding="utf-8-sig")

    # ── **实测**边界表：把主扫描 + 低幅补测按同口径合并（trial 级），逐 (win, cls, 分量) 拟合 ──
    low = os.path.join(RES, "t5b_lowamp_trials.csv")
    tr2 = tr[tr["kind"] == "noise"].copy()
    if os.path.exists(low):
        lw = pd.read_csv(low, encoding="utf-8-sig")
        keep = [c for c in tr2.columns if c in lw.columns]
        tr2 = pd.concat([tr2[keep], lw[keep]], ignore_index=True)
        print(f"[合并] 主表 noise trial {len(tr2)-len(lw)} + 低幅补测 {len(lw)}")
    rows = []
    for (w, c), g in tr2.groupby(["win", "cls"]):
        x = g["amp"].to_numpy(float)
        for comp, lab in (("fail", "复合（任一 M）"), ("M1_miss", "M1 漏触发"),
                          ("M2_wrong_anchor", "M2 锚点偏离>20%"),
                          ("M3_false_capture", "M3 错误重捕获"), ("M4_disp", "M4 显示越界")):
            y = (g[comp] > 0).to_numpy(float) if comp != "fail" else g[comp].to_numpy(float)
            n = len(x)
            k = int(y.sum())
            lo, hi = wilson(k, n)
            row = dict(win=w, cls=c, component=lab, n=n, n_fail=k, p_at_min_amp=round(k / n, 3),
                       p_ci_lo=round(lo, 3), p_ci_hi=round(hi, 3),
                       amp_min=float(x.min()), amp_max=float(x.max()))
            if len(np.unique(y)) >= 2:
                a50, b, _ = logit_fit(x, y)
                blo, bhi = logit_boot(x, y, n_boot=150, seed=5)
                row.update(A50=(round(a50, 1) if np.isfinite(a50) else np.nan),
                           A50_ci_lo=round(blo, 1), A50_ci_hi=round(bhi, 1),
                           extrapolated=int((not np.isfinite(a50)) or a50 < float(x.min())
                                            or a50 > float(x.max())))
            else:
                row.update(A50=np.nan, A50_ci_lo=np.nan, A50_ci_hi=np.nan, extrapolated=1)
            rows.append(row)
    bm = pd.DataFrame(rows)
    bm.to_csv(os.path.join(RES, "t5b_boundary_measured.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 实测 50% 失效边界（白噪/共模，trial 级 logistic；extrapolated=1 表示拟合值超出实测幅值区间）===")
    print(bm.to_string())
    # ── Q1 headline ──
    hl = []
    for cls in tr["cls"].unique():
        for amp in sorted(tr[tr["cls"] == cls]["amp"].dropna().unique()):
            gg = tr[(tr["cls"] == cls) & (tr["amp"] == amp)]
            for wname in sorted(gg["win"].unique()):
                h = gg[gg["win"] == wname]
                k, n = int(h["fail"].sum()), len(h)
                lo, hi = wilson(k, n)
                hl.append(dict(cls=cls, amp=amp, win=wname, n=n, n_fail=k,
                               p_fail=round(k / n, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                               p_fail_anchor=round(float(h["fail_anchor"].mean()), 3),
                               p_M1=round(float(h["M1_miss"].gt(0).mean()), 3),
                               p_M2=round(float(h["M2_wrong_anchor"].gt(0).mean()), 3),
                               p_M3=round(float(h["M3_false_capture"].gt(0).mean()), 3),
                               p_M4=round(float(h["M4_disp"].gt(0).mean()), 3),
                               excess_rel_med=round(float(h["excess_rel"].median()), 3)))
    q1 = pd.DataFrame(hl)
    q1.to_csv(os.path.join(RES, "t5b_q1_headline.csv"), index=False, encoding="utf-8-sig")
    print("=== Q1 headline（复合失败率 / 分量率）===")
    print(q1.to_string())
    print("\n=== 分层失败率（事件级，增量口径）===")
    print(st.to_string())
    print("\n=== 分层 50% 失效边界 ===")
    print(bt.to_string())
    print("\n=== tap 档（复合）===")
    tp = tr[tr["cls"] == "tap"]
    if len(tp):
        t2 = tp.groupby(["amp_pct", "dur_ms"]).agg(
            n=("fail", "size"), p_fail=("fail", "mean"),
            p_M1=("M1_miss", lambda s: float((s > 0).mean())),
            p_M2=("M2_wrong_anchor", lambda s: float((s > 0).mean())),
            p_M3=("M3_false_capture", lambda s: float((s > 0).mean())),
            tap_surv=("M3_false_capture", "mean")).round(3)
        t2.to_csv(os.path.join(RES, "t5b_q1_tap.csv"), encoding="utf-8-sig")
        print(t2.to_string())


if __name__ == "__main__":
    main()
