# -*- coding: utf-8 -*-
"""T2 数据层（自 progress/07-v6/scripts/ad_lib.py 的**数据层部分**复制并按 00 号文档 §4.2 改写）。

与既有桶的有意差异（报告 §2 已声明）：
  1. `load_rec` **自动定位 `##Data` 标记行**（不写死 skiprows=24）；
  2. 移除算法层（GLM53v3 / ad_v4 / run_algo / event_table 等）——T2 只做三阶段时间特征实测；
  3. 新增 `load_packets`（包内均值 + 包时刻系列，口径同 07-v6/ce_v6_estimator.load_frames）
     与 `packet_dt`（±1 包敏感性用）。

口径：时间轴一律 `timestamp` 列（禁用 elapsed）；100 Hz 网格由 `to_grid` 完成。
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
    """自动定位 ##Data；返回 (t 自 0 起, X[n, nch])；时间轴取 timestamp 列。"""
    hdr = data_hdr(p)
    df = pd.read_csv(p, skiprows=hdr)
    ch = [c for c in df.columns if c.startswith("ch")]
    if not ch:
        raise RuntimeError("no ch* columns in " + str(p))
    t = df["timestamp"].to_numpy(float)
    return t - t[0], df[ch].to_numpy(float)


def to_grid(t, X, fs=100.0):
    """np.interp 重采样到 fs 均匀网格（与 T4-A / 既有原型链路完全同式，保证 k_on 可对齐）。"""
    span = float(t[-1] - t[0])
    tu = np.arange(0.0, span, 1.0 / fs)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu


def load_packets(t, X, tol=1e-3):
    """包级系列：同包样本（timestamp 差 < tol）取均值，落在包均时刻。
    口径同 07-v6/ce_v6_estimator.load_frames（用于「原始包时间戳」轴）。返回 (tp, Xp)。"""
    grp = np.concatenate([[0], np.cumsum(np.diff(t) >= tol)])
    n = int(grp[-1]) + 1
    cnt = np.bincount(grp, minlength=n)
    Xp = np.column_stack([np.bincount(grp, weights=X[:, c], minlength=n) / cnt
                          for c in range(X.shape[1])])
    tp = np.bincount(grp, weights=t, minlength=n) / cnt
    return tp, Xp


def med_smooth(x, k):
    return pd.Series(x).rolling(max(1, int(k)), center=True, min_periods=1).median().to_numpy()


def packet_dt(t):
    """包周期估计：包内样本时间戳几乎相同，故取「正 dt 的 p90」当包周期。
    指尖 ~0.0167 s、变载实录 ~0.040 s（07-v6 §2.6）。返回 (p90, p50)。"""
    d = np.diff(np.asarray(t, float))
    d = d[d > 1e-9]
    if d.size == 0:
        return np.nan, np.nan
    return float(np.percentile(d, 90)), float(np.percentile(d, 50))
