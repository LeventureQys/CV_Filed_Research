# -*- coding: utf-8 -*-
"""T3-A 数据层（复制自 T4-A 的 `t4a_ad_lib.py`，只改文件名与注释；源头为 progress/07-v6/scripts/ad_lib.py 的数据层）。

口径（00_共享/指标字典与口径.md 与 00-项目组织文档.md §4.2）：
  1. `load_rec` **自动定位 `##Data` 标记行**（不写死 skiprows=24）；
  2. 时间轴一律 `timestamp` 列（**禁用 elapsed**）；
  3. 重采样到 100 Hz 由调用方 `to_grid` 完成（dt=0.01 s）；
  4. `packet_dt` 给出包周期（指尖 ~0.0167 s、变载实录 ~0.040 s），用于 ±1 包敏感性。

> 本文件**不 import 任何其它任务桶**（00_共享/数据与脚本复用清单.md §3 要求复制而非 import）。
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
