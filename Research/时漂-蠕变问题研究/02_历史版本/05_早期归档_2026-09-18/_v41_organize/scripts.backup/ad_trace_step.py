# -*- coding: utf-8 -*-
"""复算 v3 的阶跃检测量（三级 EMA / div / thr），定位 240.9s 台阶为何未被识别。"""
import os
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
span = t[-1] - t[0]
dtm = span / (len(t) - 1)
tu = np.arange(0.0, span, dtm)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
tot = Xu.sum(axis=1)

TAU_TOTAL, TAU_FAST, TAU_SLOW, TAU_LEVEL = 0.3, 0.7, 6.0, 10.0
STEP_REL, STEP_ABS = 0.18, 0.01
ts_s = fast = slow = level = tot[0]
min_ts = max_ts = tot[0]
last = tu[0]
print(f"{'t':>7}{'原始总':>9}{'fast':>9}{'slow':>9}{'div':>8}{'thr':>8}{'div/thr':>9}  判定")
prev = None
for i in range(1, len(tu)):
    dt = tu[i] - last
    last = tu[i]
    dt = min(dt, 0.1) if dt > 0 else 0.0
    if dt > 0:
        ts_s += (dt / TAU_TOTAL) * (tot[i] - ts_s)
        fast += (dt / TAU_FAST) * (tot[i] - fast)
        slow += (dt / TAU_SLOW) * (tot[i] - slow)
        level += (dt / TAU_LEVEL) * (ts_s - level)
    min_ts = min(min_ts, ts_s)
    max_ts = max(max_ts, ts_s)
    eps = 1e-6 * (1 + abs(max_ts))
    thr = max(STEP_REL * max(slow, eps), STEP_ABS * max_ts)
    div = abs(fast - slow)
    if 238.0 <= tu[i] <= 245.0 and (prev is None or tu[i] - prev >= 0.2):
        print(f"{tu[i]:7.2f}{tot[i]:9.0f}{fast:9.0f}{slow:9.0f}{div:8.0f}{thr:8.0f}"
              f"{div/thr:9.2f}  {'阶跃!' if div > thr else '未达阈'}")
        prev = tu[i]
