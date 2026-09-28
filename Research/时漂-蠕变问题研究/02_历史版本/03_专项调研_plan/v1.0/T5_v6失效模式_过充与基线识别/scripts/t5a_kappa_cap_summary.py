# -*- coding: utf-8 -*-
"""κ 上限诊断的汇总件：把**直接实测**的触发率/截断时长与**实测**四联指标并列成一张表。

保留这一件、并只保留**直接可测**的量，理由：
  · 「反事实」需要把 τ=1 s 的 `Â/J` 快照与逐帧 `r(τ)=A_raw/inc` 对齐 —— 两者时间基准不同
    （`Â` 是 0.1 s 抽样的插值、`r` 是逐 0.1 s 抽样、`ev_kind` 可能在窗内由 onset 翻成 restep），
    实测对齐后会出现 `κ·r > Â/J` 的荒谬值 ⇒ **该反事实不可靠，本任务不采用**。
  · 直接可测的三样已经足够解释 κ 的收益来源：
      ① 触发率 = 「事件轨迹里出现过 `A_raw > κ·inc`」的事件占比；
      ② 截断时长占比 = 被上限截断的帧占事件帧的比例；
      ③ 实测四联指标（过充/`T_stable`/1 s 误差/MD）。

产出：`results/t5a_kappa_cap_summary.csv`（一行一 κ，onset / restep 两族）。
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
    TR = pd.read_csv(os.path.join(C.TASK, "results", "t5a_kappa_trigger.csv"))
    F = pd.read_csv(os.path.join(C.TASK, "results", "t5a_kappa_four_metrics.csv"))
    F = F[F.family == "onset_only"][["kappa_onset", "OS5_med", "OS5_max", "T5_med",
                                     "T5_p90", "err1_med_onset", "MD_med", "G_med"]]

    out = []
    for kind in ["onset", "restep"]:
        t = TR[TR.kind == kind]
        for _, r in t.iterrows():
            out.append(dict(kind=kind, kappa=r["kappa"], n_events=int(r.n_events),
                            n_trigger=int(r.n_trigger),
                            trigger_rate=r.trigger_rate,
                            trunc_frame_frac=r.trunc_frame_frac,
                            r_max_med=r.r_max_med, r_max_p10=r.r_max_p10))
    S = pd.DataFrame(out)
    # onset 侧并入实测四联
    J = S[S.kind == "onset"].merge(F, left_on="kappa", right_on="kappa_onset", how="left")
    J = J.drop(columns=["kappa_onset"])
    S2 = pd.concat([J, S[S.kind == "restep"]], ignore_index=True, sort=False)
    p = os.path.join(C.TASK, "results", "t5a_kappa_cap_summary.csv")
    S2.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"产出 {p}")

    lines = ["\n===== κ 上限：直接实测的触发诊断 ↔ 实测四联指标 =====",
             "κ 语义：`Â = min(A_raw, κ·inc)`（单侧上限）。`r = A_raw/inc`；`r > κ` ⇒ 该帧被截断。",
             f"\n{'类':7s} {'κ':>5s} {'n':>4s} {'触发事件数':>8s} {'触发率':>8s} "
             f"{'截断时长占比':>11s} {'r_max中位':>10s} {'r_max p10':>10s}"]
    for _, r in TR.sort_values(["kind", "kappa"], ascending=[True, False]).iterrows():
        lines.append(f"{r['kind']:7s} {r.kappa:5.2f} {int(r.n_events):4d} "
                     f"{int(r.n_trigger):8d} {r.trigger_rate*100:7.1f}% "
                     f"{r.trunc_frame_frac*100:10.1f}% {r.r_max_med:10.4f} "
                     f"{r.r_max_p10:10.4f}")
    lines.append("\n--- onset 侧：触发诊断 ↔ 四联指标 ---")
    lines.append(f"{'κ':>5s} {'触发率':>8s} {'截断时长':>9s} | {'OS5中位':>8s} {'OS5max':>8s} "
                 f"{'T5中位':>7s} {'T5p90':>7s} {'err1中位':>9s} {'MD中位':>9s} {'G中位':>6s}")
    for _, r in J.sort_values("kappa", ascending=False).iterrows():
        lines.append(f"{r.kappa:5.2f} {r.trigger_rate*100:7.1f}% "
                     f"{r.trunc_frame_frac*100:8.1f}% | {r.OS5_med:8.3f} {r.OS5_max:8.2f} "
                     f"{r.T5_med:7.2f} {r.T5_p90:7.2f} {r.err1_med_onset:9.2f} "
                     f"{r.MD_med:9.1f} {r.G_med:6.3f}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_kappa_cap_summary.log"),
                "python scripts/t5a_kappa_cap_summary.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
