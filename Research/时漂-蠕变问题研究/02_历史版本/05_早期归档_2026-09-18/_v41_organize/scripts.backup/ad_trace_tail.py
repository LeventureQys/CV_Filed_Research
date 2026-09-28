# -*- coding: utf-8 -*-
"""逐帧追踪 v3/v4 在 230~250s 多档快速切换段的内部量。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3          # noqa: E402
from ad_v4 import GLM53v4             # noqa: E402

REC = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
       r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
       r"\device_001_seg000.csv")
df = pd.read_csv(REC, skiprows=24)
ch = [c for c in df.columns if c.startswith("ch")]
t = df["timestamp"].to_numpy(float)
t = t - t[0]
X = df[ch].to_numpy(float)
span = t[-1] - t[0]
dtm = span / (len(t) - 1)
tu = np.arange(0.0, span, dtm)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T

for tag, cls in (("v3", GLM53v3), ("v4", GLM53v4)):
    c = cls(Xu.shape[1])
    if tag == "v4":
        c.FAST_S, c.A_W0_V4, c.A_W1_V4 = 5.0, 3.5, 5.0
    print(f"\n===== {tag} =====")
    print(f"{'t':>7}{'原始总':>9}{'显示总':>9}{'扣除':>8}{'u':>7}{'A_max':>9}{'g':>8}"
          f"{'γ中位':>7}{'hold':>6}{'pend':>6}{'inload':>7}")
    for i in range(len(tu)):
        Y = c.process(tu[i], Xu[i])
        if 230.0 <= tu[i] <= 250.0 and i % 25 == 0:
            raw = Xu[i].sum()
            print(f"{tu[i]:7.2f}{raw:9.0f}{Y.sum():9.0f}{raw-Y.sum():8.0f}"
                  f"{(tu[i]-c.onset_ts if c.in_load else -1):7.2f}{c.A.max():9.0f}{c.g:+8.4f}"
                  f"{np.median(c.gamma):7.3f}{int(c.hold):6d}{int(c.pending):6d}{int(c.in_load):7d}")
