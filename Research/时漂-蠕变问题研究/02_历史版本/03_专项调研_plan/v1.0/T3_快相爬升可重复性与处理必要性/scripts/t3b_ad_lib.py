# -*- coding: utf-8 -*-
"""T4-B 专用公共层（复制自 progress/07-v6/scripts/ad_lib.py 后改名改 import）。

改动（相对原文件，均已在报告「方法与口径」中声明）：
  1. `load_rec` 改为**自动定位 `##Data` 标记行**（原文件写死 `skiprows=24`）；
  2. 去掉对 glm53_v3 / ad_v4 的 import（本任务只做通道级因果判据分析，
     算法原型走 `t4b_glm53_v6.py` 副本）；
  3. `med_smooth` 增加 `causal=True` 选项（向右对齐 = 只看过去帧），
     T4-Q5 的因果判据必须用因果版，非因果版只用于事后描述并与既有口径对齐。
"""
import os
import numpy as np
import pandas as pd


def data_start_row(path):
    """返回 ##Data 标记后的列名行号（0-based），供 pd.read_csv(skiprows=...) 使用。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("##Data"):
                return i + 1
    raise RuntimeError("no ##Data marker in " + path)


def load_rec(p):
    """自动定位 ##Data；时间轴取 timestamp 列（禁用 elapsed）。"""
    df = pd.read_csv(p, skiprows=data_start_row(p))
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def med_smooth(x, k, causal=False):
    """中值平滑。causal=False = 居中（既有口径）；causal=True = 只回溯过去 k 帧。"""
    k = max(1, int(k))
    s = pd.Series(x)
    if causal:
        return s.rolling(k, min_periods=1).median().to_numpy()
    return s.rolling(k, center=True, min_periods=1).median().to_numpy()


def to_grid(t, X, fs=100.0):
    """重采样到 100 Hz 均匀网格（np.interp）。"""
    dtm = 1.0 / fs
    span = t[-1] - t[0]
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu, dtm


def prep(path, fs=100.0):
    t, X = load_rec(path)
    tu, Xu, dtm = to_grid(t, X, fs)
    return dict(t=t, X=X, tu=tu, Xu=Xu, dtm=dtm, span=float(t[-1] - t[0]),
                tot=Xu.sum(axis=1), dup=int((np.diff(t) <= 0).sum()),
                pkt_dt=float(np.median(np.diff(t))))
