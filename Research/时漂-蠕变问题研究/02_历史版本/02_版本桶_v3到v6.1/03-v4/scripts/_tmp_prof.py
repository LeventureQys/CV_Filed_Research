import numpy as np, pandas as pd
REC = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
       r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
       r"\device_001_seg000.csv")
df = pd.read_csv(REC, skiprows=24)
ch = [c for c in df.columns if c.startswith("ch")]
t = df["timestamp"].to_numpy(float); t = t - t[0]
X = df[ch].to_numpy(float); tot = X.sum(axis=1)
step = 100
print(" t(s)   total   d/dt")
prev = None
for i in range(0, len(t), step):
    d = "" if prev is None else f"{(tot[i]-prev)/1.0:+8.0f}"
    print(f"{t[i]:7.2f} {tot[i]:7.0f} {d}")
    prev = tot[i]
