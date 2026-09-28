# -*- coding: utf-8 -*-
"""C-4 裁决表：从 `t5a_kappa_perevent.csv` 重算「工作点 × 四联指标 + 留一」。

四联指标（全部在**同一批事件**上、同一实现、同一口径）：
  ① 过充 OS%（5 s 窗瞬态 / 30 s 窗两套；主结论用 5 s 窗，瞬态）
  ② `T_stable`（10 s 窗与 30 s 窗两套；30 s 窗在实录上几乎不可测，主结论用 10 s 窗）
  ③ 1 s 时刻误差（**带符号**中位 + |中位|，并用 onset-only 子样本，因为慢相段会污染 restep）
  ④ 最大偏差 MD（中位 / p90）

分层：`ALL`（加载类 39）/ `onset`（J_frac ≥ 0.05 的 onset）/ `restep`（同）
留一：13 折（每次留出一份录制），报留出集的过充中位与 T_stable 中位的最坏折。

产出：`results/t5a_kappa_verdict.csv`、`results/t5a_kappa_four_metrics.csv`、
      `_t5a_kappa_verdict.log`。
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


def rec(m):
    print(m, flush=True)
    LOG.append(m)


COLS4 = ["OS5_med", "OS5_max", "OS30_med", "T10_med", "T10_n", "T30_med", "T30_n",
         "err1_med_onset", "err1_abs_med_onset", "err1_med_all", "MD_med", "MD_p90",
         "G_med", "US5_max", "devmid_max_adc"]


def main():
    pe = pd.read_csv(os.path.join(C.TASK, "results", "t5a_kappa_perevent.csv"))
    pe["in_main"] = pe.kind.isin(["onset", "restep"]) & (pe.J_frac >= 0.05)
    PE = pe[pe.in_main].copy()
    rec(f"主口径：{len(PE)} 行（{PE.arm.nunique()} 臂 × {len(PE)//PE.arm.nunique()} 事件）")

    base = PE[(PE.kappa_onset == 1.30) & (PE.kappa_restep == 1.12)].set_index(
        ["key", "t_on", "kind"])

    rows = []
    for (arm, fam, ko, kr), g in PE.groupby(["arm", "family", "kappa_onset", "kappa_restep"],
                                            sort=False):
        on = g[g.kind == "onset"]
        re_ = g[g.kind == "restep"]
        gi = g.set_index(["key", "t_on", "kind"])
        common = gi.index.intersection(base.index)
        dOS5 = gi.loc[common, "OS_pct_5s"] - base.loc[common, "OS_pct_5s"]
        dOS30 = gi.loc[common, "OS_pct_30s"] - base.loc[common, "OS_pct_30s"]
        dT = gi.loc[common, "T_stable10"] - base.loc[common, "T_stable10"]
        dE = gi.loc[common, "err_1s_pct"] - base.loc[common, "err_1s_pct"]
        dM = gi.loc[common, "MD"] - base.loc[common, "MD"]
        rows.append(dict(
            arm=arm, family=fam, kappa_onset=ko, kappa_restep=kr,
            n=len(g), n_onset=len(on), n_restep=len(re_),
            OS5_med=g.OS_pct_5s.median(), OS5_p90=g.OS_pct_5s.quantile(.90),
            OS5_max=g.OS_pct_5s.max(),
            OS5_med_onset=on.OS_pct_5s.median(), OS5_max_onset=on.OS_pct_5s.max(),
            OS5_med_restep=re_.OS_pct_5s.median(),
            n_os5_gt5=int((g.OS_pct_5s > 5).sum()),
            n_os5_gt10=int((g.OS_pct_5s > 10).sum()),
            OS30_med=g.OS_pct_30s.median(), OS30_max=g.OS_pct_30s.max(),
            US5_max=g.US_pct_5s.max(), US5_med=g.US_pct_5s.median(),
            T10_med=g.T_stable10.median(), T10_p90=g.T_stable10.quantile(.90),
            T10_n=int(g.T_stable10.notna().sum()),
            T30_med=g.T_stable30.median(), T30_n=int(g.T_stable30.notna().sum()),
            T5_med=g.T_stable.median(), T5_p90=g.T_stable.quantile(.90),
            T5_n=int(g.T_stable.notna().sum()),
            err1_med_onset=on.err_1s_pct.median(),
            err1_abs_med_onset=on.err_1s_pct.abs().median(),
            err1_med_all=g.err_1s_pct.median(),
            err1_abs_med_all=g.err_1s_pct.abs().median(),
            MD_med=g.MD.median(), MD_p90=g.MD.quantile(.90),
            G_med=g.G20.median(), G_min=g.G20.min(),
            devmid_max_adc=g.dev_mid_max_adc.max(),
            devmid_med_adc=g.dev_mid_max_adc.median(),
            dOS5_med=dOS5.median(), dOS5_absmax=dOS5.abs().max(), dOS5_best=dOS5.min(),
            dOS30_med=dOS30.median(), dT10_med=dT.median(), dT10_absmax=dT.abs().max(),
            dE_med=dE.median(), dE_absmax=dE.abs().max(),
            dMD_med=dM.median(), dMD_absmax=dM.abs().max(),
            n_OS5_identical=int((dOS5.abs() < 1e-12).sum()), n_pair=len(common),
        ))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_verdict.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_kappa_verdict.csv")

    # 四联指标窄表（另给**显示域 / ADC 域**分层，供与第一轮「恒载 9 组」口径对表）
    F = S[["arm", "family", "kappa_onset", "kappa_restep", "n",
           "OS5_med", "OS5_max", "OS30_med",
           "T5_med", "T5_p90", "T5_n", "T10_med", "T10_n", "T30_med", "T30_n",
           "err1_med_onset", "err1_abs_med_onset", "MD_med", "MD_p90",
           "G_med", "US5_max", "devmid_max_adc"]].copy()
    F.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_four_metrics.csv"),
             index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_kappa_four_metrics.csv")

    # 分域（显示域 = 恒载 9 组，第一轮 κ 消融只在这批上做）
    drows = []
    for (arm, fam, ko, kr, dom), g in PE.groupby(
            ["arm", "family", "kappa_onset", "kappa_restep", "dom"], sort=False):
        drows.append(dict(arm=arm, family=fam, kappa_onset=ko, kappa_restep=kr, dom=dom,
                          n=len(g), n_onset=int((g.kind == "onset").sum()),
                          OS5_med=g.OS_pct_5s.median(), OS5_max=g.OS_pct_5s.max(),
                          OS5_med_onset=g[g.kind == "onset"].OS_pct_5s.median(),
                          OS5_max_onset=g[g.kind == "onset"].OS_pct_5s.max(),
                          OS30_med=g.OS_pct_30s.median(),
                          T10_med=g.T_stable10.median(), T30_med=g.T_stable30.median(),
                          err1_med=g.err_1s_pct.median(),
                          err1_abs_med=g.err_1s_pct.abs().median(),
                          err1_med_onset=g[g.kind == "onset"].err_1s_pct.median(),
                          MD_med=g.MD.median(), MD_p90=g.MD.quantile(.90),
                          G_med=g.G20.median()))
    DD = pd.DataFrame(drows)
    DD.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_by_domain.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_kappa_by_domain.csv")
    lines0 = ["\n===== 分域（显示域 = 恒载 9 组 ≈ 第一轮 κ 消融的样本）=====",
              f"{'arm':13s} {'域':6s} {'n':>3s} {'OS5中位':>8s} {'OS30中位':>9s} "
              f"{'T10中位':>8s} {'T30中位':>8s} {'err1中位':>9s} {'MD中位':>9s}"]
    for _, r in DD[DD.family.isin(["uniform", "onset_only"])].iterrows():
        lines0.append(f"{r['arm']:13s} {r.dom:6s} {r.n:3d} {r.OS5_med:8.2f} "
                      f"{r.OS30_med:9.2f} {r.T10_med:8.3f} {r.T30_med:8.3f} "
                      f"{r.err1_med:9.2f} {r.MD_med:9.1f}")

    # 留一
    loo = []
    for _, r in S.iterrows():
        g = PE[PE.arm == r["arm"]]
        for kk, gg in g.groupby("key"):
            loo.append(dict(arm=r["arm"], family=r["family"], kappa_onset=r["kappa_onset"],
                            kappa_restep=r["kappa_restep"], held_out=kk, n=len(gg),
                            OS5_med=gg.OS_pct_5s.median(), OS5_max=gg.OS_pct_5s.max(),
                            T10_med=gg.T_stable10.median(),
                            err1_med_onset=gg[gg.kind == "onset"].err_1s_pct.median(),
                            MD_med=gg.MD.median()))
    LO = pd.DataFrame(loo)
    LO.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_loo_raw.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    agg = []
    for (arm, family, ko, kr), g in LO.groupby(
            ["arm", "family", "kappa_onset", "kappa_restep"]):
        agg.append(dict(arm=arm, family=family, kappa_onset=ko, kappa_restep=kr,
                        n_folds=len(g),
                        OS5_med_fold_min=g.OS5_med.min(),
                        OS5_med_fold_max=g.OS5_med.max(),
                        OS5_med_fold_spread=g.OS5_med.max() - g.OS5_med.min(),
                        OS5_med_fold_std=g.OS5_med.std(),
                        OS5_max_worst_fold=g.OS5_max.max(),
                        T10_med_fold_max=g.T10_med.max(),
                        err1_abs_med_fold_max=g.err1_med_onset.abs().max(),
                        MD_med_fold_max=g.MD_med.max()))
    LA = pd.DataFrame(agg)
    LA.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_loo.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    rec("产出 results/t5a_kappa_loo.csv")

    lines = ["\n===== C-4 裁决表：κ 工作点 × 四联指标（加载类主口径 n=39；onset n=22 / restep n=17）=====",
             "口径：OS% = (显示 − Zf)/J；Zf = 原始 t_on+4.6~5.4 s 中位；OS5 = 5 s 窗（瞬态）；",
             "      OS30 = 30 s 窗（含慢相平台偏差）；T10 = 10 s 水平窗的 T_stable；",
             "      err1 = 显示在 t_on+1 s 相对 J 的带符号误差（onset 子样本，避开慢相污染）。",
             f"{'arm':13s} {'κon':>5s} {'κre':>5s} {'OS5中位':>8s} {'OS5max':>8s} "
             f"{'OS30中位':>9s} {'T5中位':>7s} {'T5p90':>7s} {'T5n':>4s} "
             f"{'T10中位':>8s} {'err1onset':>10s} "
             f"{'MD中位':>9s} {'G中位':>6s}"]
    for _, r in S.iterrows():
        lines.append(f"{r['arm']:13s} {r.kappa_onset:5.2f} {r.kappa_restep:5.2f} "
                     f"{r.OS5_med:8.2f} {r.OS5_max:8.2f} {r.OS30_med:9.2f} "
                     f"{r.T5_med:7.3f} {r.T5_p90:7.3f} {int(r.T5_n):4d} "
                     f"{r.T10_med:8.3f} {r.err1_med_onset:10.2f} "
                     f"{r.MD_med:9.1f} {r.G_med:6.3f}")
    txt = "\n".join(lines)
    rec(txt)
    rec("\n".join(lines0))

    lines2 = ["\n===== 留一泛化（13 折，留出一份录制）=====",
              f"{'arm':13s} {'OS5中位 最好折':>13s} {'最坏折':>9s} {'折间极差':>9s} "
              f"{'最坏折OS5max':>12s} {'最坏折T10':>10s} {'最坏折|err1|':>12s}"]
    for _, r in LA.iterrows():
        lines2.append(f"{r['arm']:13s} {r.OS5_med_fold_min:13.2f} "
                      f"{r.OS5_med_fold_max:9.2f} {r.OS5_med_fold_spread:9.2f} "
                      f"{r.OS5_max_worst_fold:12.2f} {r.T10_med_fold_max:10.3f} "
                      f"{r.err1_abs_med_fold_max:12.2f}")
    txt2 = "\n".join(lines2)
    rec(txt2)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_kappa_verdict.log"),
                "python scripts/t5a_kappa_verdict.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
