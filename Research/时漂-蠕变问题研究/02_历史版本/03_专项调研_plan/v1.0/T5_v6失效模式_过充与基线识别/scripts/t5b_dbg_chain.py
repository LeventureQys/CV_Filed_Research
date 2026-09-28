# -*- coding: utf-8 -*-
"""T5-B 临时检视：打印机制链关键窗口的逐帧内部量（用于写报告，属留痕脚本）。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

COLS = ["ts", "total", "out", "lv_now", "lv_ref", "d", "gate", "ratio", "sig_d", "hit_run",
        "armed_pre", "raw_hit", "state_pre", "state_post", "ev_now", "sumA", "sumA_pre",
        "A_hat", "c_applied", "inc_s", "g", "n_epoch", "dev", "dA"]


def show(case, t0, t1, every=2):
    df = pd.read_csv(os.path.join(RES, "t5b_internal_trace.csv"), encoding="utf-8-sig")
    d = df[(df["case"] == case) & (df["ts"] >= t0) & (df["ts"] <= t1)]
    d = d.iloc[::every]
    print(f"\n===== {case}  t∈[{t0},{t1}] =====")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(d[COLS].to_string(index=False))


if __name__ == "__main__":
    show("C2_tap_50pct_100ms", 5.90, 6.80, 1)
    show("C3_tap_100pct_500ms", 5.90, 7.40, 2)
    show("C4_white1000_reaIevent", 2.40, 4.20, 2)
    show("C1_real_transient_SW4_99p56", 4.40, 6.60, 2)
