# -*- coding: utf-8 -*-
"""测量「短滞后差分」判据在真实数据上的台阶/蠕变分离度，用于定门限。

统计 |mean(近0.3s) − mean(0.3~0.8s前)| / 电平：
  · 排除已知事件附近（±6s）
  · 分「稳定保压窗」与「事件沿」两类，看两者差多少倍
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
RECS = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "最终测试目标", "device_001_seg000.csv"))]
for loc in ("右拇指指尖", "左拇指指尖", "四指指尖"):
    for i in (1, 2, 3):
        RECS.append((f"恒载-{loc[:2]}{i}",
                     os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(OUT))), loc, f"数据{i}", "device_001_seg000.csv")))

FAST, LAG = 0.3, 0.5
print(f"{'录制':>18}{'受载时长':>9}{'稳定窗 p99':>12}{'稳定窗 max':>12}{'事件沿 max':>12}{'倍数':>8}")
for tag, p in RECS:
    if not os.path.exists(p):
        print(f"[skip] {tag}")
        continue
    d = L.prep(p)
    tu, tot, dtm = d["tu"], d["tot"], d["dtm"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    periods = L.find_periods(tot, dtm)
    events = [e for e, _ in L.detect_events(tot, dtm)]
    lvl = tot_s.copy()
    # 短滞后差分
    d_lev = np.zeros(len(tu))
    for i in range(len(tu)):
        t = tu[i]
        w1 = (tu > t - FAST) & (tu <= t)
        w2 = (tu > t - FAST - LAG) & (tu <= t - FAST)
        if w1.any() and w2.any():
            d_lev[i] = abs(tot_s[w1].mean() - tot_s[w2].mean())
    rel = d_lev / np.maximum(lvl, 1e-9)
    inload = np.zeros(len(tu), bool)
    for a, b in periods:
        inload[a:b] = True
    near = np.zeros(len(tu), bool)
    for e in events:
        near[max(0, e - int(6 / dtm)):e + int(6 / dtm)] = True
    stable = inload & (~near)
    edge = inload & near
    if stable.sum() < 10:
        continue
    p99 = np.percentile(rel[stable], 99)
    mx = rel[stable].max()
    ex = rel[edge].max() if edge.any() else np.nan
    print(f"{tag:>18}{inload.sum()*dtm:9.0f}{100*p99:11.2f}%{100*mx:11.2f}%{100*ex:11.1f}%"
          f"{(ex/max(mx,1e-9)):8.1f}")
print("\n说明：'稳定窗' = 受载且不在任何检测事件 ±6s 内；'事件沿' = 事件 ±6s 内。")
print("门限应取在 稳定窗 max 之上、事件沿之下；倍数 = 事件沿 max / 稳定窗 max。")
