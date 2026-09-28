# -*- coding: utf-8 -*-
"""t8_probe.py -- T8 收口席的定向取证探针（只读）。

用途：为填空 `预置_跨任务数字对账表.md` 中 A/B/C/D/E 组里 T2/T5-A/T6/T7 的 ⬜ 单元格，
从各任务已交付的 results/*.csv 里抽取指定列的中位/p10/p90（不重跑任何扫描）。

用法：python scripts/t8_probe.py            # 全部探针
      python scripts/t8_probe.py t2 a1      # 只跑某一组
输出：results/_t8_probe.log（stdout 归档）+ prints
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
RESULTS = os.path.join(TASK, "results")

T1 = os.path.join(PLAN, "T1_稳定时间定义与鲁棒性口径", "results")
T2 = os.path.join(PLAN, "T2_三阶段时间特征实测", "results")
T3 = os.path.join(PLAN, "T3_快相爬升可重复性与处理必要性", "results")
T4 = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results")
T5 = os.path.join(PLAN, "T5_v6失效模式_过充与基线识别", "results")
T6 = os.path.join(PLAN, "T6_v6可重复性下降归因", "results")
T7 = os.path.join(PLAN, "T7_卸载与部分卸载时漂规律", "results")


def med(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return (np.nan, np.nan, np.nan, 0)
    return (float(np.median(s)), float(np.percentile(s, 10)), float(np.percentile(s, 90)), len(s))


def show(title, **kw):
    print("-" * 90)
    print(title)
    for k, v in kw.items():
        print("   ", k, "=", v)


def probe_t2():
    d = pd.read_csv(os.path.join(T2, "t2_shape_profile.csv"))
    print("=" * 90)
    print("[T2] t2_shape_profile.csv  (C-2 权威形状接口)")
    for kind in ["onset", "restep"]:
        for tau in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00, 5.00]:
            g = d[(d["kind"] == kind) & (np.isclose(d["tau"], tau))]
            if len(g):
                r = g.iloc[0]
                show(f"A1/A4/A5  kind={kind} tau={tau}s",
                     f_median=round(float(r["f_median"]), 4),
                     p10=round(float(r["f_p10"]), 4),
                     p90=round(float(r["f_p90"]), 4), n=int(r["n"]))
    d2 = pd.read_csv(os.path.join(T2, "t2_phase_boundary_conventions.csv"))
    print("=" * 90)
    print("[T2] t2_phase_boundary_conventions.csv (C-3 边界三口径)")
    print(d2.to_string())


def probe_t3():
    d = pd.read_csv(os.path.join(T3, "t3a_generalization.csv"))
    print("=" * 90)
    print("[T3-A] t3a_generalization.csv cols:", list(d.columns))
    print("scenarios:", sorted(d["scenario"].dropna().unique().tolist()))
    print("samples:", sorted(d["sample"].dropna().unique().tolist()))
    sub = d[(d["sample"] == "usable")]
    for tau_d in [0.2, 1.0]:
        for scen in ["own", "loo_event", "loo_rec", "cross_fam", "cross_kind"]:
            for kind in ["onset", "restep"]:
                g = sub[(sub["kind"] == kind) & (sub["scenario"] == scen) & (np.isclose(sub["tau_d"], tau_d))]
                if len(g):
                    m = med(g["med_abs_pct"])
                    show(f"T8-Q2  k={kind} scen={scen} tau_d={tau_d}",
                         med_abs_pct_med=round(m[0], 3) if m[3] else None,
                         p10=round(m[1], 3), p90=round(m[2], 3), n=m[3])


def probe_t5a():
    for fn in ["t5a_kappa_sweep.csv", "t5a_kappa_loo.csv", "t5a_kappa_summary.csv",
               "t5a_kappa_verdict.csv", "t5a_nonfilter_options.csv",
               "t5a_v6_vs_v61_summary.csv", "t5a_overcharge_summary.csv",
               "t5a_filter_pareto.csv"]:
        p = os.path.join(T5, fn)
        if not os.path.exists(p):
            print("MISSING", fn)
            continue
        d = pd.read_csv(p)
        print("=" * 90)
        print(f"[T5-A] {fn}  rows={len(d)}")
        print("cols:", list(d.columns))
        print(d.head(25).to_string())


def probe_t6():
    for fn in ["t6_repeat_dispersion.csv", "t6_jitter_summary_a.csv", "t6_improve_ab.csv",
               "t6_attribution.csv", "t6_tradeoff.csv", "t6_event_sequence_a.csv"]:
        p = os.path.join(T6, fn)
        if not os.path.exists(p):
            print("MISSING", fn)
            continue
        d = pd.read_csv(p)
        print("=" * 90)
        print(f"[T6] {fn} rows={len(d)}")
        print("cols:", list(d.columns))
        print(d.head(30).to_string())


def probe_t1():
    for fn in ["t1a_settle_summary.csv", "t1a_target_verdict.csv", "t1a_lowerbound.csv",
               "t1a_grid_sensitivity.csv", "t1a_settle_edge_sensitivity.csv"]:
        p = os.path.join(T1, fn)
        if not os.path.exists(p):
            print("MISSING", fn)
            continue
        d = pd.read_csv(p)
        print("=" * 90)
        print(f"[T1] {fn} rows={len(d)}")
        print("cols:", list(d.columns))
        print(d.head(40).to_string())


def probe_t7():
    for fn in ["t7_frozen_summary.csv", "t7b_partial_amp_response.csv",
               "t7b_asymmetry.csv", "t7_inject_amp.csv", "t7_zero_drift.csv"]:
        p = os.path.join(T7, fn)
        if not os.path.exists(p):
            print("MISSING", fn)
            continue
        d = pd.read_csv(p)
        print("=" * 90)
        print(f"[T7] {fn} rows={len(d)}")
        print("cols:", list(d.columns))
        print(d.head(30).to_string())


GROUPS = {"t1": probe_t1, "t2": probe_t2, "t3": probe_t3,
          "t5a": probe_t5a, "t6": probe_t6, "t7": probe_t7}


def main():
    want = sys.argv[1:] or list(GROUPS)
    os.makedirs(RESULTS, exist_ok=True)
    log = os.path.join(RESULTS, "_t8_probe.log")
    with open(log, "w", encoding="utf-8") as fh:
        old = sys.stdout

        class Tee:
            def write(self, s):
                old.write(s)
                fh.write(s)

            def flush(self):
                old.flush()
                fh.flush()

        sys.stdout = Tee()
        for w in want:
            if w in GROUPS:
                GROUPS[w]()
        sys.stdout = old
    print("log ->", log)


if __name__ == "__main__":
    main()
