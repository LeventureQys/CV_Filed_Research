# -*- coding: utf-8 -*-
"""T5A 口径交叉核对：把本任务事件集里与 `08-v6.1` 台账同名的事件逐个对表。

`08-v6.1/results/v61_overshoot.csv` 记了 13 份 × 38 个真阶跃的 `over_peak`（超调）与
`transient`（峰后回落）。本脚本用**本任务的公式**在这些事件上复算，看是否落在同一量级。

⚠ 两处口径不同，必须写明：
  · 08-v6.1 的 `over_peak` = (max 显示 τ∈[0,5 s] − P)/step，P = 事件后 4.6~5.4 s 原始中位；
    **减的是原始参考电平 P**，`step` 也是原始台阶。本任务的 `OS_pct` 定义字面上一样
    （`(max 显示 − Zf)/J`，Zf 同 P、J 同 step），因此两者应当**可比**。
  · 差别在于：① 事件集（本任务用 T4-A 冻结表，含小台阶；08-v6.1 用自建检测 × step ≥ 5% 峰值）；
    ② 本任务的窗口是 30 s，08-v6.1 是 5 s（30 s 会纳入慢相段的补偿误差，
    所以本任务的 OS% 系统性偏大——这一条是本任务对"用户看到的过充"的关键分解，见报告 §③.3）。

产出：`results/t5a_crosscheck_v61.csv`、`_t5a_crosscheck.log`。
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
    p61 = os.path.join(C.PROG, "08-v6.1", "results", "v61_overshoot.csv")
    v61 = pd.read_csv(p61)
    allm = pd.read_csv(os.path.join(C.TASK, "results", "t5a_event_metrics_by_arm.csv"))
    ev = C.ev_load_frozen()
    mine = allm[(allm.arm == "v6_now") & (allm.kind.isin(["onset", "restep"]))]
    rows = []
    for _, r in v61.iterrows():
        ds = r["dataset"]
        key = None
        for k, v in C.T4A_KEY.items():
            if k == ds or v == ds:
                key = v
        if key is None:
            # v61 的数据集名与 T4-A 的 rec 名不完全一致，做一次模糊匹配
            for k, v in C.T4A_KEY.items():
                if ds.startswith(k[:3]) or k.startswith(ds[:3]):
                    key = v
                    break
        if key is None:
            continue
        cand = mine[(mine.key == key) & (np.abs(mine.t_on - r["t0"]) < 0.35)]
        if not len(cand):
            rows.append(dict(v61_dataset=ds, v61_t0=r["t0"], v61_kind=r["ev"],
                             v61_over_peak=r["over_peak"], v61_transient=r["transient"],
                             key=key, matched=False))
            continue
        c = cand.iloc[0]
        rows.append(dict(v61_dataset=ds, v61_t0=r["t0"], v61_kind=r["ev"],
                         v61_over_peak=r["over_peak"], v61_transient=r["transient"],
                         key=key, t_on=c.t_on, kind=c.kind, J=c.J, J_frac=c.J_frac,
                         OS_pct_mine=c.OS_pct, US_pct_mine=c.US_pct,
                         MD_mine=c.MD, G20_mine=c.G20,
                         matched=True, d_t0=c.t_on - r["t0"]))
    D = pd.DataFrame(rows)
    p = os.path.join(C.TASK, "results", "t5a_crosscheck_v61.csv")
    D.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    m = D[D.matched] if len(D) else D
    rec(f"与 08-v6.1 台账对表：可匹配 {len(m)} / {len(D)} 条")
    if len(m):
        rec(f"  |Δt_on| 中位 {m.d_t0.abs().median():.3f} s，最大 {m.d_t0.abs().max():.3f} s")
        rec(f"  08-v6.1 over_peak 中位 {m.v61_over_peak.median():.3f}%  "
            f"max {m.v61_over_peak.max():.3f}%")
        rec(f"  本任务 OS_pct 中位 {m.OS_pct_mine.median():.3f}%  "
            f"max {m.OS_pct_mine.max():.3f}%")
        rec(f"  两者差（本任务 − 08-v6.1）中位 "
            f"{(m.OS_pct_mine - m.v61_over_peak).median():.3f} pt")
        big = m.reindex((m.OS_pct_mine - m.v61_over_peak).abs()
                        .sort_values(ascending=False).index).head(6)
        rec("  差异最大的 6 条（差异主要来自 30 s 窗纳入慢相段）：")
        for _, r in big.iterrows():
            rec(f"    {r.v61_dataset[:14]:14s} @{r.v61_t0:7.2f}s {r.v61_kind:6s} "
                f"08v61={r.v61_over_peak:8.3f}%  本任务={r.OS_pct_mine:9.3f}%  "
                f"J={r.J:9.1f}  J_frac={r.J_frac:5.3f}")
    rec(f"产出 {p}")
    C.write_log(os.path.join(C.TASK, "results", "_t5a_crosscheck.log"),
                "python scripts/t5a_crosscheck_v61.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
