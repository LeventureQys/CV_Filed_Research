# -*- coding: utf-8 -*-
"""定点核对：v7 的免责期分支是否真的执行过；v3/v7 首次分歧在哪。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3
from glm53_v7 import GLM53v7


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


p = os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")
t, X = load_rec(p)
span = t[-1] - t[0]
dt = span / (len(t) - 1)
tu = np.arange(0.0, span, dt)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T


class V7T(GLMT := GLM53v7):
    n_exempt = 0

    def process(self, ts, v):
        before = getattr(self, "fast_on_", False) and not getattr(self, "fast_done_", True)
        out = super().process(ts, v)
        if self.in_load and getattr(self, "fast_on_", False) and not getattr(self, "fast_done_", True):
            V7T.n_exempt += 1
        return out


c7 = V7T(Xu.shape[1])
c3 = GLM53v3(Xu.shape[1])
Y7 = np.empty_like(Xu)
Y3 = np.empty_like(Xu)
for i in range(len(tu)):
    Y7[i] = c7.process(tu[i], Xu[i])
    Y3[i] = c3.process(tu[i], Xu[i])
print("v7 免责期分支执行帧数 =", V7T.n_exempt)
print("v7 ever_loaded_ =", getattr(c7, "ever_loaded_", None), " fast_on_ =", getattr(c7, "fast_on_", None),
      " fast_done_ =", getattr(c7, "fast_done_", None))
d = np.abs(Y3 - Y7).max(axis=1)
print("max|v3-v7| =", d.max(), " 分歧帧数 =", int((d > 1e-9).sum()))
idx = np.where(d > 1e-9)[0]
if len(idx):
    print("首次分歧 t =", tu[idx[0]], " 差异 =", d[idx[0]])
