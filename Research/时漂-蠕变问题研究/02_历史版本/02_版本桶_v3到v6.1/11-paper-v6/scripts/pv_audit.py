# -*- coding: utf-8 -*-
"""论文数字审计：把正文里引用的关键数字与复算产物逐条对照（只读，不改论文）。

用法：python pv_audit.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                            # noqa: E402

m = pd.read_csv(os.path.join(C.RES, "metrics_all.csv"))
st = pd.read_csv(os.path.join(C.RES, "metrics_settle.csv"))
ta = pd.read_csv(os.path.join(C.RES, "trim_ablation.csv"))
h, v = m[m.kind == "恒载"], m[m.kind == "实采"]
th, tv = ta[ta.kind == "恒载"], ta[ta.kind == "实采"]
ok = 0
bad = []


def check(name, got, want, tol=0.02):
    global ok
    good = abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want)))
    print(f"  [{'OK ' if good else 'BAD'}] {name:<46} 复算 {float(got):>10.4f}  正文 {want}")
    if good:
        ok += 1
    else:
        bad.append(name)


print("== §7.5 / §8.1 恒载 9 组（|·| 均值）==")
check("raw 全段时漂", h[h.algo == "raw"].drift_main.abs().mean(), 15.48, 0.01)
check("raw 慢相段时漂", h[h.algo == "raw"].drift_slow.abs().mean(), 11.76, 0.01)
check("raw 受载中位", h[h.algo == "raw"].drift_loaded.abs().mean(), 10.19, 0.01)
check("raw 平坦度", h[h.algo == "raw"].flat.mean(), 2.56, 0.01)
check("e1s 全段", h[h.algo == "e1s"].drift_main.abs().mean(), 1.50, 0.01)
check("e1s 慢相段", h[h.algo == "e1s"].drift_slow.abs().mean(), 1.75, 0.01)
check("e3s 全段", h[h.algo == "e3s"].drift_main.abs().mean(), 1.85, 0.01)
check("e3s 慢相段", h[h.algo == "e3s"].drift_slow.abs().mean(), 1.57, 0.01)
check("e3s 受载中位", h[h.algo == "e3s"].drift_loaded.abs().mean(), 0.91, 0.01)
check("e3s 平坦度", h[h.algo == "e3s"].flat.mean(), 1.85, 0.01)
check("e3s 噪声比", h[h.algo == "e3s"].noise_ratio.mean(), 0.71, 0.02)
check("e3s 首扣时延", h[h.algo == "e3s"].ded_delay.mean(), 6.42, 0.02)
check("v6 全段", h[h.algo == "v6"].drift_main.abs().mean(), 1.37, 0.02)
check("v6 慢相段", h[h.algo == "v6"].drift_slow.abs().mean(), 1.20, 0.02)
check("v6 受载中位", h[h.algo == "v6"].drift_loaded.abs().mean(), 0.38, 0.03)
check("v6 噪声比", h[h.algo == "v6"].noise_ratio.mean(), 0.68, 0.02)
check("v6 平坦度", h[h.algo == "v6"].flat.mean(), 1.67, 0.01)
check("v6 阶跃保真", h[h.algo == "v6"].step_ratio.mean(), 1.082, 0.01)
check("v6 首扣时延", h[h.algo == "v6"].ded_delay.mean(), 9.82, 0.02)
check("v6 epoch 和", h[h.algo == "v6"].epoch.sum(), 26, 0)
check("e3s epoch 和", h[h.algo == "e3s"].epoch.sum(), 16, 0)
check("v6trim 慢相段", h[h.algo == "v6trim"].drift_slow.abs().mean(), 1.42, 0.02)

print("== §7.5 / §8.2 实采 4 份 ==")
check("e1s 全程偏差中位", v[v.algo == "e1s"].max_gap.median(), 3258, 0.01)
check("e3s 全程偏差中位", v[v.algo == "e3s"].max_gap.median(), 1889, 0.01)
check("v6 全程偏差中位", v[v.algo == "v6"].max_gap.median(), 4583, 0.01)
check("v6trim 全程偏差中位", v[v.algo == "v6trim"].max_gap.median(), 3879, 0.01)
check("e3s 变载窗中位", v[v.algo == "e3s"].gap_med.median(), 1560, 0.01)
check("v6 变载窗中位", v[v.algo == "v6"].gap_med.median(), 2126, 0.01)
check("e3s 捕获比中位", v[v.algo == "e3s"].cap_med.mean(), 0.958, 0.01)
check("v6 捕获比中位", v[v.algo == "v6"].cap_med.mean(), 0.914, 0.02)
check("e3s 捕获比最小", v[v.algo == "e3s"].cap_min.min(), 0.798, 0.01)
check("v6 捕获比最小", v[v.algo == "v6"].cap_min.min(), 0.190, 0.05)
check("e3s epoch 和", v[v.algo == "e3s"].epoch.sum(), 29, 0)
check("e1s epoch 和", v[v.algo == "e1s"].epoch.sum(), 31, 0)
check("v6 epoch 和", v[v.algo == "v6"].epoch.sum(), 44, 0)
check("1d9493 v6 全程偏差", v[(v.algo == "v6") & (v.dataset == "中途切换-1d9493")].max_gap.iloc[0], 5407, 0.01)
check("1d9493 e3s 全程偏差", v[(v.algo == "e3s") & (v.dataset == "中途切换-1d9493")].max_gap.iloc[0], 1827, 0.01)
check("13ffca v6 捕获比", v[(v.algo == "v6") & (v.dataset == "中途切换-13ffca")].cap_med.iloc[0], 1.04, 0.02)
check("切换负载 v6 全程偏差", v[(v.algo == "v6") & (v.dataset == "切换负载-快相无责")].max_gap.iloc[0], 5445, 0.01)

print("== §8.3 T_stable ==")
for a, want_h, want_v in (("e1s", 2.98, 3.24), ("e3s", 3.82, 4.25), ("v6", 0.55, 1.00)):
    check(f"{a} T_stable 恒载中位", st[(st.kind == "恒载") & (st.algo == a)].t_stable.median(), want_h, 0.02)
    check(f"{a} T_stable 实采（切换负载）",
          st[(st.kind == "实采") & (st.algo == a)].t_stable.median(), want_v, 0.02)
check("v6 落在 ≤1 s 的组数", int((st[(st.algo == "v6")].t_stable <= 1.0).sum()), 6, 0)
for a, want in (("e1s", 0.54), ("e3s", 0.06), ("v6", 2.28)):
    check(f"{a} |err5s| 恒载中位", st[(st.kind == "恒载") & (st.algo == a)].err5s.abs().median(), want, 0.03)
check("v6 |err5s| 恒载 max", st[(st.kind == "恒载") & (st.algo == "v6")].err5s.abs().max(), 4.45, 0.02)
check("v6 |err5s| 实采中位", st[(st.kind == "实采") & (st.algo == "v6")].err5s.abs().median(), 9.09, 0.02)

print("== §7.3 trim 三档 ==")
for arm, slow, flat, bias, gap, tstab in (("T0_pin", 1.198, 1.665, 3.889, 4582.7, 0.995),
                                          ("T1_trim_dead", 1.415, 1.665, 3.574, 3879.3, 58.424),
                                          ("T2_trim_nodead", 2.403, 1.711, 1.961, 3807.8, 58.424)):
    check(f"{arm} 慢相段", th[th.arm == arm].drift_slow.abs().mean(), slow, 0.01)
    check(f"{arm} 平坦度", th[th.arm == arm].flat.mean(), flat, 0.01)
    check(f"{arm} 平台偏置（恒载 9 组）", th[th.arm == arm].platform_bias_pct.abs().mean(), bias, 0.01)
    check(f"{arm} 全程偏差", tv[tv.arm == arm].max_gap.median(), gap, 0.01)
    check(f"{arm} 实采 T_stable", tv[tv.arm == arm].t_stable.median(), tstab, 0.02)
check("T2 全段时漂", th[th.arm == "T2_trim_nodead"].drift_main.abs().mean(), 2.61, 0.02)
check("T0 全段时漂", th[th.arm == "T0_pin"].drift_main.abs().mean(), 1.37, 0.02)
check("T1 全段时漂", th[th.arm == "T1_trim_dead"].drift_main.abs().mean(), 1.65, 0.02)
check("切换负载 1d9493 T1 全程偏差",
      tv[(tv.arm == "T1_trim_dead") & (tv.dataset == "中途切换-1d9493")].max_gap.iloc[0], 5035, 0.01)
check("切换负载 T0 全程偏差",
      tv[(tv.arm == "T0_pin") & (tv.dataset == "切换负载-快相无责")].max_gap.iloc[0], 5445, 0.01)
check("切换负载 T2 全程偏差",
      tv[(tv.arm == "T2_trim_nodead") & (tv.dataset == "切换负载-快相无责")].max_gap.iloc[0], 3857, 0.01)
check("左拇指/数据1 T0 T_stable",
      th[(th.arm == "T0_pin") & (th.dataset == "左拇指指尖/数据1")].t_stable.iloc[0], 10.17, 0.02)
check("左拇指/数据1 T2 T_stable",
      th[(th.arm == "T2_trim_nodead") & (th.dataset == "左拇指指尖/数据1")].t_stable.iloc[0], 8.30, 0.02)

print(f"\n审计结果：OK {ok}，BAD {len(bad)}")
for b in bad:
    print("   不一致：", b)
