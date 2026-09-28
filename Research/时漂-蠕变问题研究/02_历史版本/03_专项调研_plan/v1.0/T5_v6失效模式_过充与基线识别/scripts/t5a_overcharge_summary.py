# -*- coding: utf-8 -*-
"""T5A-Q1 汇总层：从逐事件指标表生成**分级统计**（薄壳脚本，不重跑算法）。

主口径样本分两档（都在同一 `t5a_event_metrics_by_arm.csv` 上算）：
  · 档 A「全部加载类」= `kind ∈ {onset, restep}` 的全部事件（n=41，含小台阶）
  · 档 B「主口径」    = 档 A 中 `J_frac ≥ 0.10`（台阶幅度 ≥ 10% 记录动态范围，n≈33）
  · 档 C「占比」      = 在档 B 上统计 OS% > 5% / > 10% 的事件数与占比

分 rung 报（同源数据、同一参考电平）：
  · `onset`（零基线起）与 `restep`（负载内加重）分开；
  · 同时给**绝对量**（ADC/显示单位的峰值偏差）与**相对量**（% of J）。

产出：`results/t5a_overcharge_summary.csv`、`results/t5a_overcharge_rates.csv`、
      `_t5a_overcharge_summary.log`。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402

LOG = []
ARMS = ["raw", "v5.1", "v6_now", "v6.1"]


def rec(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    allm = pd.read_csv(os.path.join(C.TASK, "results", "t5a_event_metrics_by_arm.csv"))
    ev = C.ev_load_frozen()
    rec(f"读入逐事件指标表：{len(allm)} 行 × {len(allm.columns)} 列（臂 {list(allm.arm.unique())}）")

    rows = []
    for arm in ARMS:
        a = allm[allm.arm == arm]
        if not len(a):
            continue
        for tier, sub in [("A_all_load", a[a.kind.isin(["onset", "restep"])]),
                          ("B_main_Jfrac>=0.10",
                           a[a.kind.isin(["onset", "restep"]) & (a.J_frac >= 0.10)]),
                          ("onset", a[(a.kind == "onset") & (a.J_frac >= 0.10)]),
                          ("restep", a[(a.kind == "restep") & (a.J_frac >= 0.10)]),
                          ("unload", a[a.kind == "unload"]),
                          ("partial_unload", a[a.kind == "partial_unload"])]:
            if not len(sub):
                continue
            rows.append(dict(
                arm=arm, tier=tier, n=len(sub),
                OS5_med=sub.OS_pct_5s.median(), OS5_p10=sub.OS_pct_5s.quantile(.10),
                OS5_p90=sub.OS_pct_5s.quantile(.90), OS5_max=sub.OS_pct_5s.max(),
                OS5_min=sub.OS_pct_5s.min(),
                OS30_med=sub.OS_pct_30s.median(), OS30_p90=sub.OS_pct_30s.quantile(.90),
                OS30_max=sub.OS_pct_30s.max(), OS30_min=sub.OS_pct_30s.min(),
                OS_med_adc=sub.dev_mid_max_adc.median(),
                OS_max_adc=sub.dev_mid_max_adc.max(),
                n_up_gt_2pct=int((sub.OS_pct_5s > 2).sum()),
                n_dn_gt_2pct=int((sub.OS_pct_5s < -2).sum()),
                US5_med=sub.US_pct_5s.median(), US5_max=sub.US_pct_5s.max(),
                US30_max=sub.US_pct_30s.max(),
                US_med_adc=(-sub.dev_mid_min_adc).median(),
                US_max_adc=(-sub.dev_mid_min_adc).max(),
                n_os5_gt5=int((sub.OS_pct_5s > 5).sum()),
                n_os5_gt10=int((sub.OS_pct_5s > 10).sum()),
                n_os5_gt0=int((sub.OS_pct_5s > 0).sum()),
                frac_os5_gt5=float((sub.OS_pct_5s > 5).mean()),
                frac_os5_gt10=float((sub.OS_pct_5s > 10).mean()),
                MD_med=sub.MD.median(), MD_max=sub.MD.max(),
                T_med=sub.T_stable.median(), T_n30=int(sub.T_stable30.notna().sum()),
                T_n10=int(sub.T_stable10.notna().sum()),
                err1_abs_med=sub.err_1s_pct.abs().median(),
                G_med=sub.G20.median(), G_min=sub.G20.min(),
            ))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(C.TASK, "results", "t5a_overcharge_summary.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_overcharge_summary.csv")

    # 速率表（v6 vs v6.1 vs raw）
    R = S[S.tier.isin(["B_main_Jfrac>=0.10", "onset", "restep", "unload"])].copy()
    R.to_csv(os.path.join(C.TASK, "results", "t5a_overcharge_rates.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_overcharge_rates.csv")

    lines = ["\n===== T5A-Q1 分级汇总 =====",
             "OS5 = 5 s 窗（与 08-v6.1 台账可比，量的是**阶跃瞬态过充**）；",
             "OS30 = 30 s 窗（《指标字典》§3 口径，量的是**瞬态 + 慢相平台偏差**）；",
             "Z_final = 原始 t_on+4.6~5.4 s 中位；J = 原始台阶；ADC 列 = 显示−原始 的峰值差（有符号）。"]
    for arm in ARMS:
        s = S[S.arm == arm]
        if not len(s):
            continue
        lines.append(f"\n--- {arm} ---")
        for _, r in s.iterrows():
            lines.append(
                f"  {r.tier:19s} n={r.n:2d} | OS5 中位 {r.OS5_med:8.2f} "
                f"p10~p90 {r.OS5_p10:8.2f}~{r.OS5_p90:8.2f} max {r.OS5_max:9.2f} "
                f"| OS30 中位 {r.OS30_med:9.2f} max {r.OS30_max:10.2f} "
                f"| ADC 上冲中位 {r.OS_med_adc:8.1f} max {r.OS_max_adc:9.1f} "
                f"| ADC 下冲中位 {r.US_med_adc:8.1f} max {r.US_max_adc:9.1f} "
                f"| 向上>2% {r.n_up_gt_2pct:2d} 向下>2% {r.n_dn_gt_2pct:2d} "
                f"| >5% {r.n_os5_gt5:2d}/{r.n:2d} >10% {r.n_os5_gt10:2d}/{r.n:2d} "
                f"| MD 中位 {r.MD_med:8.1f}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_overcharge_summary.log"),
                "python scripts/t5a_overcharge_summary.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
