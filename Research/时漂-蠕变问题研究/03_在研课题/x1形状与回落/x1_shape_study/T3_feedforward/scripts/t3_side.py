# -*- coding: utf-8 -*-
"""T3 步骤10：侧指标汇总（读 results/t3_arm_session.csv）：保压 std / 空载均值 / 一致性 / 触发。"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

ARMS = ["base", "A_now_a1.0", "A_lag_a1.0", "A_now_a0.8", "A_now_a0.6",
        "B_tb1.0_1.5s", "B_tb2.0_2.0s", "C_a0.8_tb2.0_1.5s",
        "C_a1.0_tb2.0_1.5s", "C_a0.8_lag_tb2.0_1.5s"]


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT, "t3_arm_session.csv"), encoding="utf-8")))
    for r in rows:
        for k in r:
            if k not in ("session", "arm"):
                r[k] = float(r[k])
    lines = ["T3 侧指标汇总（13 会话；保压 std = 沿后 1 s 至段末的显示 std 中位；",
             "空载均值 = 输入总量 < 15%·量程 帧的「显示−输入」均值；一致性 = 同类负载重复出现时 ±10%·台阶 判据）",
             "",
             "臂                   保压std中位 | 空载均值 中位(最差) | 一致性 合格/总 (合格率) | 触发总数 | 事件窗内占比"]
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        if not rr:
            continue
        ok = sum(r["cons_ok"] for r in rr)
        tot = sum(r["cons_tot"] for r in rr)
        idle = [r["idle_bias"] for r in rr if not np.isnan(r["idle_bias"])]
        lines.append("%-20s %9.0f | %+8.0f (%+7.0f) | %4d/%-4d (%5.1f%%) | %8.0f | %5.1f%%"
                     % (arm,
                        float(np.median([r["hold_std"] for r in rr if r["hold_std"] >= 0])),
                        float(np.median(idle)) if idle else float("nan"),
                        float(np.min(idle)) if idle else float("nan"),
                        int(ok), int(tot), 100.0 * ok / max(tot, 1),
                        sum(r["n_trig"] for r in rr),
                        100.0 * sum(r["n_trig_on_edge"] for r in rr) /
                        max(sum(r["n_trig"] for r in rr), 1)))
    with open(os.path.join(OUT, "t3_side.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
