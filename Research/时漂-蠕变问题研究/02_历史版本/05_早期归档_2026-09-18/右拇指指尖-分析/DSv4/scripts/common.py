# -*- coding: utf-8 -*-
"""DSv4 公共工具：数据加载、空载-负载-空载分段、空间布局映射。

数据目录约定：
    temp/右拇指指尖/数据{1,2,3}/device_001_seg000.csv            原始值
    temp/右拇指指尖/数据{1,2,3}/device_001_seg000_kalman_compensated.csv  主机现有 Kalman 补偿值
CSV 前 24 行为 ##Session 头，第 25 行为列名，之后为数据。
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\右拇指指尖")
OUT = BASE / "DSv4"
FIG = OUT / "figures"
RES = OUT / "results"
DATASETS = ["数据1", "数据2", "数据3"]

# 9x7 物理布局掩码（session.json 中 device.layout_mask），
# 31 个 active 位按行优先顺序对应 CSV 的 ch0~ch30。
LAYOUT_MASK = "110110011011101111110111111011111100000001000000100000010000001"
ROWS, COLS = 9, 7


def channel_columns(df):
    return [c for c in df.columns if c.startswith("ch") and c[2:].isdigit()]


def load_csv(name, compensated=False):
    """读取某一组数据的 CSV 数据表（不含元数据行）。"""
    suffix = "_kalman_compensated" if compensated else ""
    path = BASE / name / f"device_001_seg000{suffix}.csv"
    df = pd.read_csv(path, skiprows=24)
    return df


def load_dataset(name):
    """返回 dict：t、X(raw)、Xk(主机 Kalman 文件)、ch_cols、total。"""
    raw = load_csv(name, compensated=False)
    kal = load_csv(name, compensated=True)
    ch = channel_columns(raw)
    t = raw["elapsed"].to_numpy(dtype=float)
    X = raw[ch].to_numpy(dtype=float)
    Xk = kal[ch].to_numpy(dtype=float)
    if Xk.shape != X.shape:
        raise ValueError(f"{name} 的 Kalman 文件尺寸不一致")
    return {
        "name": name,
        "t": t,
        "X": X,
        "Xk": Xk,
        "ch_cols": ch,
        "total": X.sum(axis=1),
        "fs": 1.0 / np.median(np.diff(t)),
        "n": len(t),
    }


def segment_total(total, thr_ratio=0.15, min_gap=20):
    """用总信号做空载/负载分割。

    返回 s0, s1（负载主段的首尾 index）。负载段定义为 total > thr_ratio*max(total)，
    间隙小于 min_gap 个点的相邻段合并；取最长段。
    """
    thr = thr_ratio * float(np.max(total))
    loaded = total > thr
    # 相邻段合并
    idx = np.where(loaded)[0]
    if len(idx) == 0:
        return 0, len(total)
    runs = np.split(idx, np.where(np.diff(idx) > min_gap)[0] + 1)
    merged = []
    cur = runs[0]
    for nxt in runs[1:]:
        if nxt[0] - cur[-1] <= min_gap:
            cur = np.r_[cur, nxt]
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    best = max(merged, key=len)
    return int(best[0]), int(best[-1] + 1)


def segment_dataset(name):
    d = load_dataset(name)
    s0, s1 = segment_total(d["total"])
    return d, s0, s1


def active_positions():
    """返回与 CSV ch0~ch30 顺序对应的 31 个 (row, col) 数组。"""
    ones = [i for i, b in enumerate(LAYOUT_MASK) if b == "1"]
    if len(ones) != 31:
        raise RuntimeError(f"layout_mask active 位={len(ones)}，应为 31")
    pos = np.array([(i // COLS, i % COLS) for i in ones], dtype=int)
    return pos


def channels_to_map(values):
    """把 per-channel 向量(31,)映射为 9x7 空间图，无传感器处为 NaN。"""
    vals = np.asarray(values, dtype=float)
    if vals.shape != (31,):
        raise ValueError(f"需要 31 个通道值，收到 {vals.shape}")
    grid = np.full((ROWS, COLS), np.nan)
    for j, (r, c) in enumerate(active_positions()):
        grid[r, c] = vals[j]
    return grid


def map_to_channels(grid):
    """9x7 空间图取回 per-channel 向量(31,)。"""
    g = np.asarray(grid, dtype=float)
    out = np.empty(31)
    for j, (r, c) in enumerate(active_positions()):
        out[j] = g[r, c]
    return out


def load_session_meta(name):
    with open(BASE / name / "session.json", encoding="utf-8") as f:
        return json.load(f)


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt
