# -*- coding: utf-8 -*-
"""x1 阶段回落优化实验：共享装载层。

数据根已从 temp/ 迁到
`Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/算法数据&原始数据/`，
本模块把 v30_lib 的装载函数接到新根上，并额外提供"总通道和"视角的
事件切分与回落量测工具。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# v30_lib 在导入时断言 ROOT 下存在 CMakeLists.txt；数据根搬迁后该断言会先炸，
# 这里先把它的 ROOT 补成真实仓库根，再导入。
ROOT = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
assert os.path.isfile(os.path.join(ROOT, "CMakeLists.txt")), ROOT
sys.path.insert(0, HERE)
import v30_lib as L  # noqa: E402

L.ROOT = ROOT  # 数据根已被搬到 Document 下，覆盖 lib 里的旧默认

DOC_ROOT = os.path.join(ROOT, "Document", "Update", "Dev-Version",
                        "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据")
WORKING = os.path.join(DOC_ROOT, "working")
ARCHIVED = os.path.join(DOC_ROOT, "archived")


def sessions(root=WORKING, max_depth=5):
    return L.discover_sessions(root, max_depth)


def load_pre(ds_dir):
    name = ("device_001_pre_seg0.csv"
            if os.path.isfile(os.path.join(ds_dir, "device_001_pre_seg0.csv"))
            else "device_001_seg000.csv")
    return name, L.load_stream(ds_dir, name)


def load_recorded(ds_dir):
    return L.load_stream(ds_dir, "device_001_seg000.csv")


def total(s):
    return s["V"].sum(axis=1)


def edges(el, tot, rel_hi=0.25, rel_lo=0.15):
    """按总量电平找加载/卸载沿（滞回双门限），返回 (ups, downs)。"""
    a, b = float(tot.min()), float(tot.max())
    hi = a + rel_hi * (b - a)
    lo = a + rel_lo * (b - a)
    ups, downs = [], []
    state = tot[0] > hi
    for i in range(1, len(tot)):
        if not state and tot[i] > hi:
            ups.append(i)
            state = True
        elif state and tot[i] < lo:
            downs.append(i)
            state = False
    return ups, downs


def step_levels(el, tot, ups, downs, win=1.5):
    """每个加载沿之后的稳态电平（沿后 win 秒到段末的中位）与沿前基线。"""
    out = []
    for k, i in enumerate(ups):
        j = downs[k] if k < len(downs) else len(el) - 1
        pre = float(np.median(tot[max(0, i - 60):max(1, i - 5)]))
        seg = tot[i:j]
        post = float(np.median(seg[-int(min(len(seg), 200)):]))
        out.append(dict(i=i, j=j, t=float(el[i]), pre=pre, post=post,
                        step=post - pre))
    return out


def fall_metrics(y, el, i, j, base_i=None):
    """一次加载事件（沿 i → 卸载 j）内的"回落"量测。

    定义（相对沿前基线 b）：
      peak   = max(y[i..] )                   沿后峰值
      settle = 段末 200 帧中位                 稳定电平
      overshoot = peak - settle               回落幅度（用户嫌"过大"的量）
      t_peak / t_settle                       对应时刻
      band2   = 首次进入 settle±2%·step 且不再离开的时刻
    """
    peak = float(np.max(y[i:j]))
    ip = i + int(np.argmax(y[i:j]))
    seg = y[i:j]
    settle = float(np.median(seg[-min(len(seg), 200):]))
    step = (settle - y[base_i]) if base_i is not None else settle
    tol = 0.02 * abs(step) if step else 1.0
    band = settle + tol
    tband = None
    below = np.nonzero(seg <= band)[0]
    if len(below):
        tband = float(el[i + below[0]] - el[i])
    return dict(peak=peak, t_peak=float(el[ip] - el[i]), settle=settle,
                overshoot=peak - settle,
                os_frac=(peak - settle) / abs(step) if step else 0.0,
                t_band2=tband, step=step)
