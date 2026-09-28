# -*- coding: utf-8 -*-
"""把 **κ 上限触发率** 列并入 C-4 的四联表（回应对 κ 语义的纠正）。

κ 的语义 = 单侧上限 `Â = min(A_raw, κ·inc)`（不是增益）⇒ 没有"触发率"这一列，
κ 的收益就无法解释。本脚本把 `t5a_kappa_trigger.csv` 的触发率按 κ 对齐并入：
  · `results/t5a_kappa_four_metrics.csv`（四联指标 + 触发率）
  · `results/t5a_kappa_verdict.csv`
  · `results/t5a_kappa_sweep.csv`（逐类行）

列定义：
  `trig_onset_rate`  —— 该 κ 下 **onset 事件被上限触发（出现过 `A_raw > κ·inc`）的占比**
  `trunc_onset_frac` —— onset 被截断帧占其全部事件帧的比例（抽样 ×10 还原）
  `trig_restep_rate` / `trunc_restep_frac` —— restep 同义
  `cap_active`       —— 该工作点是否"上限真正起作用"（截断帧占比 ≥ 5%）

用法：`python scripts\t5a_kappa_trigger_join.py`
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


def main():
    trg = pd.read_csv(os.path.join(C.TASK, "results", "t5a_kappa_trigger.csv"))
    on = trg[trg.kind == "onset"].set_index("kappa")
    re_ = trg[trg.kind == "restep"].set_index("kappa")

    def lookup(ko, kr):
        """按 κ 取最近的档（触发率网格步长 0.05 / 0.04）。"""
        o = on.index[np.argmin(np.abs(on.index.to_numpy() - ko))]
        r = re_.index[np.argmin(np.abs(re_.index.to_numpy() - kr))]
        return o, r

    for fname in ["t5a_kappa_four_metrics.csv", "t5a_kappa_verdict.csv"]:
        p = os.path.join(C.TASK, "results", fname)
        df = pd.read_csv(p)
        trig_o, trig_r, trunc_o, trunc_r = [], [], [], []
        for _, row in df.iterrows():
            o, r = lookup(float(row["kappa_onset"]), float(row["kappa_restep"]))
            trig_o.append(float(on.loc[o, "trigger_rate"]))
            trunc_o.append(float(on.loc[o, "trunc_frame_frac"]))
            trig_r.append(float(re_.loc[r, "trigger_rate"]))
            trunc_r.append(float(re_.loc[r, "trunc_frame_frac"]))
        df["trig_onset_rate"] = trig_o
        df["trunc_onset_frac"] = trunc_o
        df["trig_restep_rate"] = trig_r
        df["trunc_restep_frac"] = trunc_r
        df["cap_active"] = (df["trunc_onset_frac"] >= 0.05) | (df["trunc_restep_frac"] >= 0.05)
        # 该臂实际用到的那个 κ 的截断率（onset_only 看 onset，其余看两者较大者）
        df["cap_active_relevant"] = np.where(
            df.get("family", pd.Series(["uniform"] * len(df))) == "onset_only",
            df["trunc_onset_frac"] >= 0.05,
            np.where(df.get("family", pd.Series(["uniform"] * len(df))) == "restep_only",
                     df["trunc_restep_frac"] >= 0.05,
                     (df["trunc_onset_frac"] >= 0.05) | (df["trunc_restep_frac"] >= 0.05)))
        df.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
        rec(f"已并入触发率列：{fname}")

    # sweep 也并（逐类行按 kind 取对应族）
    p = os.path.join(C.TASK, "results", "t5a_kappa_sweep.csv")
    sw = pd.read_csv(p)
    to, tro = [], []
    for _, row in sw.iterrows():
        if row["kind"] == "onset":
            key, k = on, float(row["kappa_onset"])
        elif row["kind"] == "restep":
            key, k = re_, float(row["kappa_restep"])
        else:
            key, k = on, float(row["kappa_onset"])
        j = key.index[np.argmin(np.abs(key.index.to_numpy() - k))]
        to.append(float(key.loc[j, "trigger_rate"]))
        tro.append(float(key.loc[j, "trunc_frame_frac"]))
    sw["trigger_rate"] = to
    sw["trunc_frame_frac"] = tro
    sw.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"已并入触发率列：t5a_kappa_sweep.csv")

    F = pd.read_csv(os.path.join(C.TASK, "results", "t5a_kappa_four_metrics.csv"))
    lines = ["\n===== C-4 四联指标 + **κ 上限触发率**（κ 是单侧上限，不是增益）=====",
             "触发率 = 该 κ 下「出现过 A_raw > κ·inc 的事件」占比；"
             "截断帧占比 = 被上限截断的帧 / 事件帧",
             f"{'arm':13s} {'κon':>5s} {'κre':>5s} {'OS5中位':>8s} {'OS5max':>8s} "
             f"{'T5中位':>7s} {'err1onset':>10s} {'MD中位':>9s} {'G中位':>6s} "
             f"{'触发率on':>9s} {'截断帧on':>9s} {'上限生效':>8s}"]
    for _, r in F.iterrows():
        lines.append(f"{r['arm']:13s} {r.kappa_onset:5.2f} {r.kappa_restep:5.2f} "
                     f"{r.OS5_med:8.3f} {r.OS5_max:8.2f} {r.T5_med:7.2f} "
                     f"{r.err1_med_onset:10.2f} {r.MD_med:9.1f} {r.G_med:6.3f} "
                     f"{r.trig_onset_rate*100:8.1f}% {r.trunc_onset_frac*100:8.1f}% "
                     f"{str(bool(r.cap_active_relevant)):>8s}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_kappa_trigger_join.log"),
                "python scripts/t5a_kappa_trigger_join.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
