# -*- coding: utf-8 -*-
"""t4a_08_summary：报告用汇总数字（只读 results/*.csv，便于复现报告里的口径对照）。"""
import os
import sys

import numpy as np
import pandas as pd

import t4a_common as C


def main():
    C.start_log()
    d = pd.read_csv(os.path.join(C.RES, "t4a_morphology.csv"))
    ld = d[d.jump > 0]
    print("== 全部加载类（n=%d）按 kind 中位 ==" % len(ld))
    print(ld.groupby("kind")[["t50", "t80", "t90", "t95", "z_at_005", "z_at_02", "z_at_05",
                              "z_at_10"]].median().round(3).to_string())
    print("n:", ld.groupby("kind").size().to_dict())
    print("\n== clean 子集（n=%d）按 kind 中位 ==" % int(ld.clean.sum()))
    print(ld[ld.clean].groupby("kind")[["t50", "t80", "t90", "t95", "z_at_005", "z_at_02",
                                        "z_at_10"]].median().round(3).to_string())
    m = pd.read_csv(os.path.join(C.RES, "t4a_vs_firstround.csv"))
    mm = m[m.kind_r1.isin(["onset", "restep"])]
    print("\n== 与第一轮逐事件匹配上的加载类（n=%d）三口径 0.2 s 完成度 ==" % len(mm))
    for k in ("onset", "restep"):
        s = mm[mm.kind_r1 == k]
        if not len(s):
            continue
        print("  %-6s n=%2d | 第一轮(主通道) %5.1f%% [%.1f~%.1f] | 本任务Z %5.1f%% [%.1f~%.1f]"
              " | 本任务主通道 %5.1f%% [%.1f~%.1f]"
              % (k, len(s), s.r1_step_frac.median(), s.r1_step_frac.quantile(.1),
                 s.r1_step_frac.quantile(.9), 100 * s.t4a_z_at_02.median(),
                 100 * s.t4a_z_at_02.quantile(.1), 100 * s.t4a_z_at_02.quantile(.9),
                 100 * s.ch_at_02.median(), 100 * s.ch_at_02.quantile(.1),
                 100 * s.ch_at_02.quantile(.9)))
    print("\n== restep 的 t90 分布 ==")
    for tag, sub in (("全部 restep", ld[ld.kind == "restep"]),
                     ("clean restep", ld[(ld.kind == "restep") & ld.clean])):
        print("  %-14s n=%2d t90 中位 %.2f p10 %.2f p90 %.2f s;  t50 中位 %.2f s"
              % (tag, len(sub), sub.t90.median(), sub.t90.quantile(.1), sub.t90.quantile(.9),
                 sub.t50.median()))
    print("\n== onset 的 t90 分布 ==")
    for tag, sub in (("全部 onset", ld[ld.kind == "onset"]),
                     ("clean onset", ld[(ld.kind == "onset") & ld.clean])):
        print("  %-14s n=%2d t90 中位 %.2f p10 %.2f p90 %.2f s" %
              (tag, len(sub), sub.t90.median(), sub.t90.quantile(.1), sub.t90.quantile(.9)))
    se = pd.read_csv(os.path.join(C.RES, "t4a_separability.csv"))
    cl = se[se["sample"] == "clean"]
    print("\n== 干净样本效应量前 12（按 |AUC-0.5|）==")
    cl = cl.assign(sep=(cl.auc - 0.5).abs()).sort_values("sep", ascending=False)
    print(cl.head(12)[["feature", "med_onset", "med_restep", "med_diff", "auc",
                       "band_overlap", "n_onset_in_restep_band",
                       "n_restep_in_onset_band"]].round(4).to_string(index=False))

    # 幅度相关性（回答「形状/输入速率是否随载荷量级变」）
    ir = pd.read_csv(os.path.join(C.RES, "t4a_input_recover.csv"))
    ir["t_on"] = ir.t_on.round(3)
    d["t_on"] = d.t_on.round(3)
    mg = d.merge(ir[["key", "t_on", "T_ramp"]], on=["key", "t_on"], how="left")
    ld2 = mg[(mg.jump > 0) & mg.T_ramp.notna()]
    rows = []
    for tag, s in (("onset", ld2[ld2.kind == "onset"]), ("restep", ld2[ld2.kind == "restep"]),
                   ("all_load", ld2)):
        if len(s) < 5:
            continue
        x = np.log10(s.T_ramp.clip(lower=0.005))
        rows.append(dict(group=tag, n=len(s),
                         corr_logT_absJ=float(np.corrcoef(x, s.jump.abs())[0, 1]),
                         corr_logT_pre_over_peak=float(np.corrcoef(x, s.pre_over_peak)[0, 1]),
                         corr_z02_absJ=float(np.corrcoef(s.z_at_02, s.jump.abs())[0, 1]),
                         corr_z02_pre_over_peak=float(np.corrcoef(s.z_at_02, s.pre_over_peak)[0, 1])))
    amp = pd.DataFrame(rows)
    amp.round(4).to_csv(os.path.join(C.RES, "t4a_amplitude_shape.csv"), index=False,
                        encoding="utf-8-sig")
    print("\n== 幅度/预载 与 形状、输入速率 的相关性（Pearson）==")
    print(amp.round(4).to_string(index=False))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
