# -*- coding: utf-8 -*-
"""T1-A 数据层（复制自 `progress/07-v6/scripts/ad_lib.py` 的**数据层部分**，改 import 后自用）。

与既有 `ad_lib.py` 的差异（有意为之，报告 §2 已声明）：
  1. `load_rec` 改为**自动定位 `##Data` 标记行**（不写死 `skiprows=24`，按 00 号文档 §4.2-1）；
  2. **移除算法层**（GLM53v3 / ad_v4 / ALGOS / run_algo / event_table / slow_windows /
     mid_load_events）—— T1-A 只做稳定时间的定义与实测；算法由本目录 `t1a_glm53_*.py`
     副本经 `t1a_common.run_plain / run_traced` 驱动，保证本 `scripts/` 可独立跑通；
  3. `to_grid` 固定 **100 Hz**（`np.interp`，与原型链路口径一致）；`prep` 也改用该网格
     （既有 `ad_lib.prep` 用 `span/(n-1)` 自适应步长，与本项目 100 Hz 网格口径不一致）；
  4. 新增 `packet_dt`（包周期估计，±1 包敏感性用）。

逐字保留：`med_smooth` / `find_periods` / `detect_events`（口径不变）。

口径：时间轴一律 `timestamp` 列（禁用 `elapsed`）；域由调用方声明（恒载 9 组 = `processed_display`，
变载实录 4 份 = ADC）。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def data_hdr(path):
    """`##Data` 标记的下一行（列名行）0-based 行号，供 skiprows 用。"""
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
    """np.interp 重采样到 fs 均匀网格（本任务口径：100 Hz / dt=0.01 s）。"""
    span = float(t[-1] - t[0])
    tu = np.arange(0.0, span, 1.0 / fs)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu


def med_smooth(x, k):
    return pd.Series(x).rolling(max(1, int(k)), center=True, min_periods=1).median().to_numpy()


def packet_dt(t):
    """包周期估计：包内时间戳几乎相同，取正 dt 的 p90 当包周期（指尖 ~16.7 ms、实录 ~40 ms）。"""
    d = np.diff(np.asarray(t, float))
    d = d[d > 1e-9]
    if d.size == 0:
        return np.nan, np.nan
    return float(np.percentile(d, 90)), float(np.percentile(d, 50))


def find_periods(tot, dt, frac=0.05, min_s=3.0):
    """受载段（总量超过 frac×峰值且持续 ≥min_s）。逐字复制自 ad_lib。"""
    thr = frac * tot.max()
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    return sorted([(a, b) for a, b in zip(s, e) if (b - a) * dt >= min_s])


def detect_events(tot, dt, rel=0.15, absfrac=0.08):
    """变载事件候选（2 s 窗中位电平差 + 1.5 s 合并 + 局部最陡帧回溯）。逐字复制自 ad_lib。"""
    n = len(tot)
    pn = int(2.0 / dt)
    cand = []
    for i in range(pn, n - pn, max(1, int(0.1 / dt))):
        pre = np.median(tot[i - pn:i])
        post = np.median(tot[i:i + pn])
        if abs(post - pre) > max(rel * abs(pre), absfrac * tot.max()):
            cand.append((i, post - pre))
    ev = []
    for i, dl in cand:
        if ev and i - ev[-1][0] <= int(1.5 / dt):
            if abs(dl) > abs(ev[-1][1]):
                ev[-1] = (i, dl)
        else:
            ev.append((i, dl))
    ref, seen = [], set()
    for i, dl in ev:
        a, b = max(0, i - int(2 / dt)), min(n - 1, i + int(2 / dt))
        sm = med_smooth(tot[a:b], max(3, int(0.15 / dt)))
        k = int(np.argmax(np.abs(np.diff(sm))))
        if a + k in seen:
            continue
        seen.add(a + k)
        ref.append((a + k, dl))
    return ref


def prep(path, fs=100.0):
    """读一份录制并重采样到 100 Hz；返回 dict（口径与 t1a_common.load_grid 一致）。"""
    t, X = load_rec(path)
    tu, Xu = to_grid(t, X, fs)
    p90, p50 = packet_dt(t)
    return dict(t=t, X=X, tu=tu, Xu=Xu, dtm=1.0 / fs, span=float(t[-1] - t[0]),
                tot=Xu.sum(axis=1), dup=int((np.diff(t) <= 0).sum()),
                pkt_p90=p90, pkt_p50=p50)
