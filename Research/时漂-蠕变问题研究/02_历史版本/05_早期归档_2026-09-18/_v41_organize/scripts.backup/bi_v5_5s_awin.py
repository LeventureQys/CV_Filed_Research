# -*- coding: utf-8 -*-
"""菜单 5s 档的幅度窗口径核对：AWIN = 5/3(菜单实际) vs 1.5(此前 A/B 表用的值)。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
RECS = [("切换负载", os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "最终测试目标", "device_001_seg000.csv"))]

for tag, path in RECS:
    d = L.prep(path)
    d["periods"] = L.find_periods(d["tot"], d["dtm"])
    d["events"] = [e for e, _ in L.detect_events(d["tot"], d["dtm"])]
    tot_s = L.med_smooth(d["tot"], 0.5 / d["dtm"])
    line = f"{tag:>8}: "
    for awin in (1.5, 5.0 / 3.0):
        c = GLM53v5(d["Xu"].shape[1])
        c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = 5.0, awin, 5.0
        Y = np.empty_like(d["Xu"])
        for i in range(len(d["tu"])):
            Y[i] = c.process(d["tu"][i], d["Xu"][i])
        y = L.med_smooth(Y.sum(axis=1), 0.5 / d["dtm"])
        gap = np.abs(y - tot_s).max()
        d["Ys"] = {"v5": Y}
        ev = L.event_table(d, d["events"], d["Ys"], [], algos=["v5"])
        ml = L.mid_load_events(ev)
        cap = (ml["gain_v5"] / ml["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna()
        line += f"AWIN={awin:.3f}: 最大偏差 {gap:6.0f}（{100*gap/d['tot'].max():4.1f}%）" \
                f" 捕获比中位 {cap.median():.2f} 最小 {cap.min():.2f}   "
    print(line)
