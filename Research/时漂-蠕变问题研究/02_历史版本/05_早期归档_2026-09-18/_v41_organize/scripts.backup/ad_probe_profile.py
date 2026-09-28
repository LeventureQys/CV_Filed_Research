# -*- coding: utf-8 -*-
"""粗看新数据的整阵总量/主通道时序轮廓（不产图）。"""
import numpy as np
import pandas as pd

REC = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
       r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
       r"\device_001_seg000.csv")

df = pd.read_csv(REC, skiprows=24)
ch = [c for c in df.columns if c.startswith("ch")]
t = df["timestamp"].to_numpy(float)
t = t - t[0]
X = df[ch].to_numpy(float)
tot = X.sum(axis=1)

step = 25          # 0.25 s
tt = t[::step]
tt2 = tot[::step]
print(" t(s)   total   ch3    ch4    ch6    ch7   ch10   ch11")
for i in range(len(tt)):
    print(f"{tt[i]:7.2f} {tt2[i]:7.0f} " + " ".join(f"{X[i*step, j]:6.0f}" for j in (3, 4, 6, 7, 10, 11)))
