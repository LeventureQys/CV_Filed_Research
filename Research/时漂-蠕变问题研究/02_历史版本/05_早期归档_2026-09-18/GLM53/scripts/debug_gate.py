# -*- coding: utf-8 -*-
"""调试卸载门控：数据2 卸载前后 ts/min_ts/unloaded/b0/Z 轨迹"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "右拇指指尖")  # temp/右拇指指尖
name = "数据2"
MAIN = 17

df = pd.read_csv(os.path.join(BASE, name, "device_001_seg000.csv"), skiprows=24)
tc = [c for c in df.columns if c.startswith("ch")]
X = df[tc].to_numpy()
t = df["elapsed"].to_numpy()
n, m = X.shape
dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.1)
dtm = np.median(dt)
total = X.sum(axis=1)

a_s = np.clip(dtm / 0.3, 0, 1)
ts = np.empty(n)
ts[0] = total[0]
for i in range(1, n):
    ts[i] = ts[i - 1] + a_s * (total[i] - ts[i - 1])
min_ts = np.minimum.accumulate(ts)
unloaded = ts < 1.5 * min_ts

# 分段
thr = 0.15 * total.max()
loaded_mask = total > thr
d = np.diff(loaded_mask.astype(int))
s = np.where(d == 1)[0] + 1
e = np.where(d == -1)[0] + 1
s0, s1 = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]

print(f"s1={s1} t[s1]={t[s1]:.1f}s  total[s1-5s]={total[s1-int(5/dtm)]:.2f}  "
      f"total[s1+5s]={total[s1+int(5/dtm)]:.2f}  total[end]={total[-1]:.2f}")
print(f"min_ts@end={min_ts[-1]:.3f}  1.5*min_ts={1.5*min_ts[-1]:.3f}")
print(f"卸载后 unloaded 帧占比: {unloaded[s1:].mean()*100:.1f}%")
print(f"卸载后前10s unloaded 占比: {unloaded[s1:s1+int(10/dtm)].mean()*100:.1f}%")

# 基线跟踪
b0 = np.zeros(m)
a_b = np.clip(dt / 2.0, 0, 1)
baseline = np.empty((n, m))
for i in range(n):
    if unloaded[i]:
        b0 = b0 + a_b[i] * (X[i] - b0)
    baseline[i] = b0
Z = X - baseline
print(f"\nch17: X_pre={X[:s0,MAIN].mean():.4f} b0@s0={baseline[s0,MAIN]:.4f} "
      f"b0@s1={baseline[s1,MAIN]:.4f} b0@end={baseline[-1,MAIN]:.4f}")
post = slice(s1 + int(5 / dtm), s1 + int(30 / dtm))
print(f"ch17: X_post[5:30s]={X[post,MAIN].mean():.4f}  Z_post[5:30s]={Z[post,MAIN].mean():.4f}")
# 每5s一档
for k in range(6):
    sl = slice(s1 + int(5*k/dtm), s1 + int(5*(k+1)/dtm))
    print(f"  卸载后{5*k}-{5*(k+1)}s: X={X[sl,MAIN].mean():+.4f} b0={baseline[sl[-1] if sl.stop<n else -1,MAIN]:.4f} Z={Z[sl,MAIN].mean():+.4f} unloaded={unloaded[sl].mean()*100:.0f}%")
# total 与 1.5*min_ts 在卸载后前 30s 的轨迹
print("\n卸载后时间  ts    1.5*min_ts  unloaded")
for k in [0, 1, 2, 3, 5, 10, 20, 30]:
    i = min(n - 1, s1 + int(k / dtm))
    print(f"  {k:3d}s  {ts[i]:7.2f}  {1.5*min_ts[i]:7.2f}  {unloaded[i]}")
