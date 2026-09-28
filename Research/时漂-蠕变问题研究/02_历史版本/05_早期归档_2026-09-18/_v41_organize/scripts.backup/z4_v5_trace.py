# -*- coding: utf-8 -*-
"""逐帧核对：v5 在「中途变载」事件前后到底做了什么。
判据：变载必须看得见（显示跟随原始），且变载后蠕变必须继续被补（显示相对原始不再漂）。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3  # noqa: E402
import importlib
GLM53v5 = importlib.import_module("z2_v5").GLM53v5

name = "零负载-中途切换负载-零负载-切换负载"
df = pd.read_csv(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"), skiprows=24)
ch = [c for c in df.columns if c.startswith("ch")]
tr = df["timestamp"].to_numpy(float)
t = tr - tr[0]
X = df[ch].to_numpy(float)
span = t[-1] - t[0]
dt = span / (len(t) - 1)
tu = np.arange(0.0, span, dt)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
tot = Xu.sum(axis=1)


def run(cls):
    c = cls(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


Y3, Y5 = run(GLM53v3), run(GLM53v5)
print("=" * 100)
print("数据B（中途 5N→10N 型变载）逐帧核对：总量")
print("=" * 100)
print(f"{'t(s)':>7}{'原始':>10}{'v3显示':>10}{'v5显示':>10}{'v3偏差':>9}{'v5偏差':>9}")
for tt in [16, 18, 19, 20, 20.5, 21, 21.5, 22, 22.5, 23, 24, 25, 26, 28, 30]:
    k = int(np.searchsorted(tu, tt))
    if k < len(tu):
        print(f"{tt:7.1f}{tot[k]:10.0f}{Y3[k].sum():10.0f}{Y5[k].sum():10.0f}"
              f"{Y3[k].sum()-tot[k]:9.0f}{Y5[k].sum()-tot[k]:9.0f}")

print("\n变载后（21s 起）显示相对原始是否重新开始补蠕变：")
for tt in [22, 24, 26, 28, 30]:
    k = int(np.searchsorted(tu, tt))
    print(f"  t={tt:5.1f}s  v3偏差={Y3[k].sum()-tot[k]:+8.0f}   v5偏差={Y5[k].sum()-tot[k]:+8.0f}")

print("\n首次 onset（10.98s 附近）v5 的免责期行为：")
for tt in [10.9, 11.0, 11.5, 12, 13, 14, 15, 16, 18, 20]:
    k = int(np.searchsorted(tu, tt))
    print(f"  t={tt:5.1f}s  原始={tot[k]:8.0f}  v5显示={Y5[k].sum():8.0f}  偏差={Y5[k].sum()-tot[k]:+8.0f}")

print("\n数据A 卸载事件（31.29s）逐帧：")
nameA = "零负载-切换负载-零负载-再切换负载"
dfA = pd.read_csv(os.path.join(TEMP, "变化负载", nameA, "device_001_seg000.csv"), skiprows=24)
chA = [c for c in dfA.columns if c.startswith("ch")]
trA = dfA["timestamp"].to_numpy(float)
tA = trA - trA[0]
XA = dfA[chA].to_numpy(float)
spanA = tA[-1] - tA[0]
dtA = spanA / (len(tA) - 1)
tuA = np.arange(0.0, spanA, dtA)
XuA = np.vstack([np.interp(tuA, tA, XA[:, c]) for c in range(XA.shape[1])]).T
totA = XuA.sum(axis=1)


def runA(cls):
    c = cls(XuA.shape[1])
    Y = np.empty_like(XuA)
    for i in range(len(tuA)):
        Y[i] = c.process(tuA[i], XuA[i])
    return Y


Y3A, Y5A = runA(GLM53v3), runA(GLM53v5)
print(f"{'t(s)':>7}{'原始':>10}{'v3显示':>10}{'v5显示':>10}{'v3偏差':>9}{'v5偏差':>9}")
for tt in [29, 30, 30.5, 31, 31.3, 31.5, 32, 33, 34, 35]:
    k = int(np.searchsorted(tuA, tt))
    print(f"{tt:7.1f}{totA[k]:10.0f}{Y3A[k].sum():10.0f}{Y5A[k].sum():10.0f}"
          f"{Y3A[k].sum()-totA[k]:9.0f}{Y5A[k].sum()-totA[k]:9.0f}")
