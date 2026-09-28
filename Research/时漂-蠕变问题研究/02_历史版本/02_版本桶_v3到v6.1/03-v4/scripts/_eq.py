import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath("scripts/_eq.py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ad_v4 import GLM53v4 as A4
import importlib
R4 = importlib.import_module("r_fastphase").GLM53v4
REC = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
       r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv")
df = pd.read_csv(REC, skiprows=24)
ch = [c for c in df.columns if c.startswith("ch")]
t = df["timestamp"].to_numpy(float); t = t - t[0]
X = df[ch].to_numpy(float)
span = t[-1]-t[0]; dt = span/(len(t)-1)
tu = np.arange(0.0, span, dt)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
def run(cls):
    c = cls(Xu.shape[1]); c.FAST_S=5.0; c.A_W0_V4=3.5; c.A_W1_V4=5.0
    Y = np.empty_like(Xu)
    for i in range(len(tu)): Y[i] = c.process(tu[i], Xu[i])
    return Y
Ya, Yr = run(A4), run(R4)
print("ad_v4 与 r_fastphase.GLM53v4 最大逐帧差 =", np.abs(Ya-Yr).max())
