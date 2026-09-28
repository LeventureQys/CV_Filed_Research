import os, subprocess, sys, numpy as np
OUT = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\v4.1flash"
sys.path.insert(0, os.path.join(OUT, "scripts"))
from glm53_v5 import GLM53v5
EXE = os.path.join(OUT, "cpp_v5_check", "build", "Debug", "v5_cpp_check.exe")
TAU, AMP, FS, DT, DUR, T_LOAD, T_ADD = 199.0, 0.1564, 100.0, 0.01, 400.0, 20.0, 300.0
out = subprocess.run([EXE, "0.5"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
ts = np.array([float(l.split(",")[0]) for l in out]); dc = np.array([float(l.split(",")[2]) for l in out])
c = GLM53v5(1); tt = np.arange(0.0, DUR, DT); Y = np.empty(len(tt))
for i in range(len(tt)):
    t = tt[i]; v = 0.0
    if t > T_LOAD: v += 10000.0*(1+AMP*(1-np.exp(-(t-T_LOAD)/TAU)))
    if t > T_ADD: v += 0.5*10000.0*(1+AMP*(1-np.exp(-(t-T_ADD)/TAU)))
    Y[i] = c.process(t, np.array([v]))[0]
dp = Y[np.arange(0, len(tt), 5)]
d = np.abs(dc - dp)
k = int(np.argmax(d))
print(f"最大差 {d.max():.2f} @ t={ts[k]:.2f}s  (C++ {dc[k]:.1f} / Py {dp[k]:.1f})")
print("差值分位: p50=%.2f p90=%.2f p99=%.2f max=%.2f" % tuple(np.percentile(d,[50,90,99,100])))
for w in (0, 19.9, 20.2, 25, 100, 299.9, 300.2, 305, 310, 350, 399):
    i = int(np.argmin(np.abs(ts-w)))
    print(f"  t={ts[i]:7.2f}  C++ {dc[i]:9.1f}  Py {dp[i]:9.1f}  差 {dc[i]-dp[i]:+8.2f}")
