# -*- coding: utf-8 -*-
"""t4a_03_separability：T4-Q2 差异效应量（中位差 + p10~p90 重叠 + AUC），不是只给显著性。

样本：① `clean`=True 的加载类事件（指标字典 §2.1「干净事件」）为主样本；
      ② 全部加载类事件（含脏事件）为敏感性样本。
分组：`kind`（armed_20 口径，见 t4a_morphology.csv）。

产出：results/t4a_separability.csv
日志：results/_t4a_03_separability.log
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import t4a_common as C

FEATS = ["t10", "t50", "t80", "t90", "t95", "rise_frames_10_90", "n_frames_rise",
         "frames_to_50", "frames_to_90", "step_frame_frac", "top3_frame_frac",
         "z_at_005", "z_at_01", "z_at_02", "z_at_03", "z_at_05", "z_at_10", "z_at_20",
         "exp_tau", "exp_beta", "pow_p", "pow_tau", "T_slope",
         "pre_over_peak", "ratio", "pre_over_jump",
         "T_ramp", "rmse_ramp_pct", "rmse_step_pct", "gain_ramp", "A_ramp"]


def load():
    mor = pd.read_csv(os.path.join(C.RES, "t4a_morphology.csv"))
    ir = pd.read_csv(os.path.join(C.RES, "t4a_input_recover.csv"))
    keys = ["key", "t_on"]
    mor["t_on_r"] = mor["t_on"].round(3)
    ir["t_on_r"] = ir["t_on"].round(3)
    add = ir[["key", "t_on_r", "T_ramp", "rmse_ramp_pct", "rmse_step_pct", "gain_ramp", "A_ramp"]]
    return mor.merge(add, on=keys[:1] + ["t_on_r"], how="left")


def one_row(df, f, sample):
    o = df[df.kind == "onset"][f].to_numpy(float)
    r = df[df.kind == "restep"][f].to_numpy(float)
    o, r = o[np.isfinite(o)], r[np.isfinite(r)]
    if len(o) < 3 or len(r) < 3:
        return None
    mo, o10, o90, no = C.band(o)
    mr, r10, r90, nr = C.band(r)
    try:
        p = float(mannwhitneyu(o, r, alternative="two-sided").pvalue)
    except Exception:
        p = np.nan
    n_o_in_r = int(((o >= r10) & (o <= r90)).sum())
    n_r_in_o = int(((r >= o10) & (r <= o90)).sum())
    return dict(sample=sample, feature=f, n_onset=no, n_restep=nr,
                med_onset=mo, p10_onset=o10, p90_onset=o90,
                med_restep=mr, p10_restep=r10, p90_restep=r90,
                med_diff=mo - mr, ratio=(mo / mr if mr else np.nan),
                auc=C.auc(o, r), cliff_delta=(2 * C.auc(o, r) - 1 if np.isfinite(C.auc(o, r)) else np.nan),
                p_mannwhitney=p,
                band_overlap=C.overlap_ratio((o10, o90), (r10, r90)),
                n_onset_in_restep_band=n_o_in_r, n_restep_in_onset_band=n_r_in_o,
                overlap_event_frac=(n_o_in_r + n_r_in_o) / float(no + nr),
                direction=("onset 更大" if mo > mr else "restep 更大"))


def loo_centroid(df, fs):
    """留一法最近质心（用按类别标准化的欧氏距离），返回 score/AUC/准确率。
    ⚠ 特征由同一批标签选出 ⇒ 乐观上界，报告已声明。"""
    X = df[fs].to_numpy(float)
    y = (df.kind == "onset").to_numpy()
    m = np.isfinite(X).all(axis=1)
    X, y = X[m], y[m]
    if y.sum() < 3 or (~y).sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())
    sc = np.full(len(y), np.nan)
    acc = []
    for i in range(len(y)):
        t = np.ones(len(y), bool)
        t[i] = False
        mu, sd = X[t].mean(0), X[t].std(0) + 1e-12
        Z = (X - mu) / sd
        cp, cn = Z[t][y[t]].mean(0), Z[t][~y[t]].mean(0)
        d1, d0 = np.linalg.norm(Z[i] - cp), np.linalg.norm(Z[i] - cn)
        sc[i] = d0 - d1                       # >0 ⇒ 更像 onset
        acc.append(bool((sc[i] > 0) == y[i]))
    from scipy.stats import rankdata
    r = rankdata(sc)
    n1, n0 = int(y.sum()), int((~y).sum())
    a = float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    return a, float(np.mean(acc)), sc, len(y)


def main():
    C.start_log()
    df = load()
    ld = df[df.jump > 0].copy()
    rows = []
    for sample, sub in (("clean", ld[ld.clean]), ("all", ld)):
        for f in FEATS:
            if f not in sub.columns:
                continue
            r = one_row(sub, f, sample)
            if r:
                rows.append(r)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(os.path.join(C.RES, "t4a_separability.csv"), index=False,
                        encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("== 主样本（clean 干净事件）效应量，按 |AUC-0.5| 排序 ==")
    cl = out[out["sample"] == "clean"].copy()
    cl["sep"] = (cl.auc - 0.5).abs()
    print(cl.sort_values("sep", ascending=False)[
        ["feature", "n_onset", "n_restep", "med_onset", "med_restep", "med_diff", "ratio", "auc",
         "band_overlap", "n_onset_in_restep_band", "n_restep_in_onset_band", "p_mannwhitney"]
    ].round(4).to_string(index=False))
    print("\n== 敏感性：全部加载类事件（含脏事件）==")
    al = out[out["sample"] == "all"].copy()
    al["sep"] = (al.auc - 0.5).abs()
    print(al.sort_values("sep", ascending=False)[
        ["feature", "n_onset", "n_restep", "med_onset", "med_restep", "auc", "band_overlap"]
    ].head(14).round(4).to_string(index=False))
    # 两特征组合（乐观上界）
    print("\n== 组合可分性（留一最近质心；特征由同批标签选出 ⇒ 乐观上界）==")
    combos = [("z_at_02", "T_ramp"), ("z_at_005", "T_ramp"), ("z_at_01", "z_at_02"),
              ("T_ramp", "rmse_step_pct"), ("z_at_02", "frames_to_50")]
    for fs in combos:
        if not all(f in ld.columns for f in fs):
            continue
        for sample, sub in (("clean", ld[ld.clean]), ("all", ld)):
            a, ac, _, n = loo_centroid(sub, list(fs))
            print("  %-28s [%s] n=%2d  AUC=%.3f  LOO acc=%.3f" % ("+".join(fs), sample, n, a, ac))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
