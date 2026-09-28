# -*- coding: utf-8 -*-
"""T4-A 数据层（复制自 progress/07-v6/scripts/ad_lib.py 的**数据层部分**，改 import 后自用）。

与既有桶的差异（有意为之，报告 §2 已声明）：
  1. `load_rec` 改为**自动定位 `##Data` 标记行**（不写死 skiprows=24，按 00 号文档 §4.2-1）；
  2. 已**移除算法层**（GLM53v3 / ad_v4 / run_algo / make_traced / event_table 等）——
     T4-A 只做形态与机制表征，不需要 v6 原型；这样本脚本目录不依赖其它任务桶。
  3. 新增 `packet_dt`（包周期估计，用于 ±1 包敏感性）。

口径：时间轴一律 `timestamp` 列（禁用 elapsed）；重采样到 100 Hz 由调用方 `to_grid` 完成。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def data_hdr(path):
    """返回 `##Data` 标记的下一行（列名行）在文件中的 0-based 行号，供 skiprows 用。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("##Data"):
                return i + 1
    raise RuntimeError("no ##Data marker in " + str(path))


def load_rec(p):
    """自动定位 ##Data；返回 (t 从 0 起, X[n, nch])；时间轴取 timestamp 列。"""
    hdr = data_hdr(p)
    df = pd.read_csv(p, skiprows=hdr)
    ch = [c for c in df.columns if c.startswith("ch")]
    if not ch:
        raise RuntimeError("no ch* columns in " + str(p))
    t = df["timestamp"].to_numpy(float)
    return t - t[0], df[ch].to_numpy(float)


def to_grid(t, X, fs=100.0):
    """np.interp 重采样到 fs 均匀网格（口径：100 Hz / dt=0.01 s）。"""
    span = float(t[-1] - t[0])
    tu = np.arange(0.0, span, 1.0 / fs)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu


def med_smooth(x, k):
    return pd.Series(x).rolling(max(1, int(k)), center=True, min_periods=1).median().to_numpy()


def packet_dt(t):
    """包周期估计：包内样本时间戳几乎相同（dt~1e-4 s），故取「正 dt 的 p90」当包周期。
    指尖 ~0.0167 s、变载实录 ~0.040 s（07-v6 §2.6）。返回 (p90, p50)。"""
    d = np.diff(np.asarray(t, float))
    d = d[d > 1e-9]
    if d.size == 0:
        return np.nan, np.nan
    return float(np.percentile(d, 90)), float(np.percentile(d, 50))
