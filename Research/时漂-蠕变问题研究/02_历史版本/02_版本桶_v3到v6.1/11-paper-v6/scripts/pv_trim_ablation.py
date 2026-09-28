# -*- coding: utf-8 -*-
"""补充复算：v6 的 A 慢修正（trim）三档消融 —— 这是「绝对平 vs 绝对准」取舍的量化依据。

档位：
  T0  纯 pin          TRIM_RATE=0        （= 默认 v6，显示稳态 ≡ Â）
  T1  trim 2.5% 死区  TRIM_RATE=0.002    （本轮采用）
  T2  trim 无死区     TRIM_RATE=0.002, TRIM_DEAD_FRAC=0

产出：results/trim_ablation.csv、逐数据集指标 + 汇总打印。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                            # noqa: E402
import ad_lib as L                                               # noqa: E402
from glm53_v6 import GLM53v6                                     # noqa: E402
from pv_run import V6Trace                                       # noqa: E402

ARMS = [("T0_pin", 0.0, 0.025), ("T1_trim_dead", 0.002, 0.025), ("T2_trim_nodead", 0.002, 0.0)]

rows = []
for tag, path in C.ALL:
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_raw = Xu.sum(axis=1)
    tot_s = L.med_smooth(tot_raw, 0.5 / dtm)
    peak = float(tot_raw.max())
    kind = C.KIND[tag]
    fo = C.first_onset(tu, tot_raw, dtm)
    i0 = fo[0]
    tgt = float(np.median(tot_raw[i0 + int(4.6 / dtm):i0 + int(5.4 / dtm)]))
    step = tgt - fo[1]
    for name, rate, dead in ARMS:
        c = V6Trace(Xu.shape[1])
        c.TRIM_RATE, c.TRIM_DEAD_FRAC = rate, dead
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        c.A = np.full(Xu.shape[1], c.A_peak)
        mm = (C.hold_metrics(Y, c, tu, Xu, dtm, tot_s, peak) if kind == "恒载"
              else C.vary_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d))
        mm.update(dataset=tag, kind=kind, arm=name,
                  t_stable=C.stable_time(tu, Y.sum(axis=1), i0, step, dtm),
                  t_band=C.settle_time(tu, Y.sum(axis=1), i0, tgt, step, dtm),
                  platform_bias_pct=100 * (float(np.median(Y.sum(axis=1)[i0 + int(30 / dtm):
                                                                     i0 + int(50 / dtm)])) / tgt - 1))
        rows.append(mm)
        print(f"{tag:>18} {name:>14}  " + (
            f"全段 {mm['drift_main']:+6.2f}%  慢相 {mm['drift_slow']:+6.2f}%  平坦 {mm['flat']:5.2f}%  "
            f"平台偏置 {mm['platform_bias_pct']:+5.2f}%  T_stable {mm['t_stable']:6.2f}s"
            if kind == "恒载" else
            f"全程偏差 {mm['max_gap']:6.0f}  捕获 {mm['cap_med']:5.2f}  "
            f"平台偏置 {mm['platform_bias_pct']:+5.2f}%  T_stable {mm['t_stable']:6.2f}s"))

df = pd.DataFrame(rows)
C.save_table(df, "trim_ablation.csv")
h, v = df[df.kind == "恒载"], df[df.kind == "实采"]
print("\n===== 恒载 9 组（|·| 均值）=====")
print(h.groupby("arm").agg(全段时漂=("drift_main", lambda s: s.abs().mean()),
                           慢相段时漂=("drift_slow", lambda s: s.abs().mean()),
                           受载中位=("drift_loaded", lambda s: s.abs().mean()),
                           平坦度=("flat", "mean"),
                           平台偏置_abs=("platform_bias_pct", lambda s: s.abs().mean()),
                           T_stable中位=("t_stable", "median")).round(3).to_string())
print("\n===== 实采 4 份 =====")
print(v.groupby("arm").agg(全程偏差中位=("max_gap", "median"),
                           变载窗偏差中位=("gap_med", "median"),
                           捕获比中位=("cap_med", "mean"),
                           平台偏置_abs=("platform_bias_pct", lambda s: s.abs().mean()),
                           T_stable中位=("t_stable", "median")).round(3).to_string())
print("\n逐份全程最大偏差（ADC）")
print(v.pivot_table(index="dataset", columns="arm", values="max_gap").round(0).to_string())
print("\n逐份 T_stable（s）")
print(df.pivot_table(index="dataset", columns="arm", values="t_stable").round(2).to_string())
