# -*- coding: utf-8 -*-
"""t4b_cost_with_t1.py —— T4-B × T4-A 接口增量：误判代价的 **T1 口径** 表 + 复用 T4-A 的 z_at_*。

要求（席位接口第 5 条）：误判代价用 T1 口径指标（`T_stable` / 超调 `OS%` / 最大偏差 `MD`）表达，
并**优先复用 T4-A 已算好的 `z_at_*` 列**，不另立归一化。

做法（全部只读既有产物，不重跑算法）：
  `results/t4b_arm_metrics{,_clean}.csv`（本任务 9 条臂 × 逐事件 T_stable/OS%/MD）
  ⨝ `results/t4b_t4a_crosscheck.csv`（本任务事件 ↔ T4-A `key`/`t_on` 的配对）
  ⨝ `results/t4a_morphology.csv`（`z_at_*`、`pre_over_peak`、`clean`）
  ⨝ `results/t4a_input_recover.csv`（`T_ramp`，按 `key` + `t_on`）
  ⨝ `results/t4a_channel_consistency.csv`（`t50_ch_iqr`，按 `key` + `kind`）

产物：results/t4b_cost_t1_clean.csv（clean 子集）、results/t4b_cost_t1_all.csv（全样本）、
      results/t4b_cost_t1_summary.csv（两套样本的关键差值并排）
运行：python scripts\t4b_cost_with_t1.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402

ARMS_SHOW = ["M0_now", "M2_correctA", "M2_correct", "M3_causal_P1", "M3_causal_thr0.40",
             "M4_warped", "M2_inverse"]


def build(arm_csv, mor, ir, ch):
    am = pd.read_csv(os.path.join(C.RES, arm_csv))
    x = pd.read_csv(os.path.join(C.RES, "t4b_t4a_crosscheck.csv"))
    # 配对键：本任务 (rec, t_on) ↔ T4-A (rec, t_on)
    am["t_on_r"] = am.t_on.round(3)
    x = x.rename(columns={"t_on_t4b": "t_on_r", "t_on_t4a": "t4a_t_on"})
    # 一个本任务事件可能落在 T4-A 两个事件的 ±0.6 s 窗内 ⇒ 只保留最近的那个，避免行数被放大
    x = x.assign(_ad=x.dt.abs()).sort_values("_ad").drop_duplicates(subset=["rec", "t_on_r"])
    key = x[["rec", "t_on_r", "t4a_t_on"]].drop_duplicates()
    out = am.merge(key, on=["rec", "t_on_r"], how="inner").drop_duplicates(
        subset=["rec", "t_on_r", "arm"])
    # T4-A 形态列（z_at_* 等）
    mm = mor.copy()
    mm["t4a_t_on"] = mm.t_on.round(3)
    keep = [c for c in ("z_at_005", "z_at_01", "z_at_02", "z_at_03", "z_at_05", "z_at_10",
                        "step_frame_frac", "frames_to_50", "n_frames_rise", "exp_tau",
                        "pre_over_peak", "clean", "armed", "key", "kind")
            if c in mm.columns]
    mm = mm[keep + ["rec", "t4a_t_on"]].rename(columns={"kind": "kind_t4a"})
    out = out.merge(mm, on=["rec", "t4a_t_on"], how="left")
    # T_ramp（key + t_on）
    if ir is not None and len(ir):
        t = ir[["key", "t_on", "T_ramp"]].copy()
        t["t4a_t_on"] = t.t_on.round(3)
        out = out.merge(t[["key", "t4a_t_on", "T_ramp"]], on=["key", "t4a_t_on"], how="left")
    else:
        out["T_ramp"] = np.nan
    # 通道一致性（key + kind）；T4-A 该表按 kind 分组、同 kind 多行 ⇒ 先按组取中位再并
    if ch is not None and len(ch):
        c = (ch.groupby(["key", "kind"], as_index=False)[["t50_ch_iqr", "n_active"]].median()
             .rename(columns={"kind": "kind_t4a"}))
        out = out.merge(c, on=["key", "kind_t4a"], how="left")
    else:
        out["t50_ch_iqr"] = np.nan
    return out


def main():
    mor = pd.read_csv(os.path.join(C.TASK, "results", "t4a_morphology.csv"))
    mor = mor[mor.kind.isin(["onset", "restep"])].copy()
    irp = os.path.join(C.TASK, "results", "t4a_input_recover.csv")
    chp = os.path.join(C.TASK, "results", "t4a_channel_consistency.csv")
    ir = pd.read_csv(irp) if os.path.isfile(irp) else None
    ch = pd.read_csv(chp) if os.path.isfile(chp) else None

    rows, summary = [], []
    for tag, fn, sub in (("all", "t4b_arm_metrics.csv", "全样本（本任务 35 个加载事件）"),
                         ("clean", "t4b_arm_metrics_clean.csv", "T4-A clean 主样本")):
        p = os.path.join(C.RES, fn)
        if not os.path.isfile(p):
            print("!! 缺 %s（先跑 t4b_q6_branch.py %s）" % (fn, "--clean-t4a" if tag == "clean" else ""))
            continue
        d = build(fn, mor, ir, ch)
        d["sample"] = tag
        d.to_csv(os.path.join(C.RES, "t4b_cost_t1_%s.csv" % tag), index=False,
                 encoding="utf-8-sig")
        print("\n== [%s] %s：臂 × 真值类别的 T1 口径指标（中位）==" % (tag, sub))
        print("   事件：本任务 %d 个 × %d 臂 = %d 行；其中 T4-A clean=True 的 %d 个"
              % (d.t_on_r.nunique(), d.arm.nunique(), len(d),
                 int(d.groupby("t_on_r").clean.first().sum()) if "clean" in d else -1))
        for arm in ARMS_SHOW:
            for kind in ("onset", "restep"):
                s = d[(d.arm == arm) & (d.kind == kind)]
                if not len(s):
                    continue
                rows.append(dict(sample=tag, arm=arm, true_kind=kind, n=len(s),
                                 T_stable_med=s.T_stable.median(),
                                 T_stable_p90=s.T_stable.quantile(.9),
                                 OS_pct_med=s.os_pct.median(),
                                 OS_pct_absmax=s.os_pct.abs().max(),
                                 US_pct_med=s.us_pct.median(),
                                 MD_med=s.md_adc.median(), MD_max=s.md_adc.max(),
                                 z_at_02_med=s.z_at_02.median() if "z_at_02" in s else np.nan,
                                 T_ramp_med=s.T_ramp.median() if "T_ramp" in s else np.nan))
        # 配对差值（相对真值分支）
        if "M2_correct" in set(d.arm):
            kk = ["rec", "t_on_r"]
            base = (d[d.arm == "M2_correct"][kk + ["kind", "T_stable", "os_pct",
                                                   "us_pct", "md_adc"]]
                    .drop_duplicates(subset=kk).set_index(kk))
            for arm in ARMS_SHOW:
                if arm == "M2_correct":
                    continue
                s = (d[d.arm == arm][kk + ["kind", "T_stable", "os_pct", "us_pct", "md_adc"]]
                     .drop_duplicates(subset=kk).set_index(kk))
                common = s.index.intersection(base.index)
                for kind in ("onset", "restep"):
                    m = [i for i in common if base.loc[[i], "kind"].iloc[0] == kind]
                    if not len(m):
                        continue
                    for col, out in (("T_stable", "dT"), ("os_pct", "dOS"),
                                     ("us_pct", "dUS"), ("md_adc", "dMD")):
                        dv = (s.loc[m, col] - base.loc[m, col]).dropna()
                        if not len(dv):
                            continue
                        summary.append(dict(sample=tag, arm=arm, true_kind=kind, n=len(m),
                                            metric=out, d_med=dv.median(),
                                            d_absmax=dv.abs().max()))
    R = pd.DataFrame(rows)
    S = pd.DataFrame(summary)
    R.to_csv(os.path.join(C.RES, "t4b_cost_t1_arms.csv"), index=False, encoding="utf-8-sig")
    S.to_csv(os.path.join(C.RES, "t4b_cost_t1_summary.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n== 臂 × 类别（中位 T_stable / OS% / MD；z_at_02 与 T_ramp 为该组中位）==")
    print(R.to_string(index=False))
    print("\n== 配对差值（相对真值分支 M2_correct）==")
    piv = S.pivot_table(index=["sample", "arm", "true_kind", "metric"],
                        values=["d_med", "d_absmax"])
    print(piv.to_string())
    print("\n-> results/t4b_cost_t1_{all,clean}.csv / t4b_cost_t1_arms.csv / t4b_cost_t1_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
