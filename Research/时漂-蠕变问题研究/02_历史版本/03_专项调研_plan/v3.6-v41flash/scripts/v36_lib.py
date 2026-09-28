# -*- coding: utf-8 -*-
"""plan v3.6 共享层：数据集发现/装载 + 「加载段」切分 + 补偿一致性口径。

沿用 plan/v3.0 的 `v30_lib`（数据根会被用户重组，只按「有 session.json + pre/seg csv」发现），
在其上补三件事：

1. `load_segments()`：按**算法输入流**（`device_001_pre_seg0.csv`）的总量做滞回切分，
   给出每次「加载段」的 (i_on, i_off, t_on, t_off, 台阶)；
2. `ded_of()`：补偿量 `ded = Σ 输入 − Σ 显示`（逐帧）；
3. `seg_metrics()`：把「段内某窗口的 ded / 台阶」这一类**一致性口径**统一在一处，
   避免每个探针各写一份。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_V30 = os.path.abspath(os.path.join(_HERE, "..", "..", "v3.0", "scripts"))
if _V30 not in sys.path:
    sys.path.insert(0, _V30)

import v30_lib as L  # noqa: E402

ROOT = L.ROOT
DATA_ROOT = L.DATA_ROOT
DS_TARGET = os.path.join(DATA_ROOT, "working", "零基线-反复增减同一负载",
                         "20260919_160854_single_device_7b3977")


def load(ds_dir):
    ds = L.load_dataset(ds_dir)
    tin, tout, traw = L.totals(ds)
    ds["tot_in"] = tin
    ds["tot_out"] = tout
    ds["tot_raw"] = traw
    ds["ded"] = tin - tout
    return ds


def med3(x):
    """3 帧中值（与算法内部同口径），边界退化为邻值。"""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    y = x.copy()
    y[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


def load_segments(tot, el, lo_frac=0.30, hi_frac=0.70, min_frames=30, smooth=5):
    """滞回切分「空载段 / 加载段」（在输入总量上做，抗单帧掉点）。

    返回 [(kind, i0, i1), ...]，kind ∈ {"idle","loaded"}；只保留长度 ≥ min_frames 的段。
    """
    t = np.asarray(tot, dtype=float)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        t = np.convolve(t, k, mode="same")
    a, b = float(t.min()), float(t.max())
    lo = a + lo_frac * (b - a)
    hi = a + hi_frac * (b - a)
    state = "idle" if t[0] < hi else "loaded"
    start = 0
    segs = []
    for i in range(1, len(t)):
        if state == "idle" and t[i] > hi:
            segs.append((state, start, i - 1))
            state, start = "loaded", i
        elif state == "loaded" and t[i] < lo:
            segs.append((state, start, i - 1))
            state, start = "idle", i
    segs.append((state, start, len(t) - 1))
    out = []
    for k, i0, i1 in segs:
        if i1 - i0 + 1 >= min_frames:
            out.append((k, i0, i1))
    return out


def seg_table(ds, segs, w0=10.0, w1=20.0):
    """逐加载段给出「台阶 / 该段静置窗内的补偿量」——一致性分析的主表。

    窗口以**段起点**（该段电平首次越过带上沿的帧）为参考：`[i_on + w0, i_on + w1]`（秒）。
    """
    el = ds["pre"]["el"]
    tin = ds["tot_in"]
    ded = ds["ded"]
    rows = []
    for idx, (kind, i0, i1) in enumerate(segs):
        if kind != "loaded":
            continue
        # 台阶：取前一个空载段的末值与本段起点后 1 s 的输入电平之差
        base = None
        for j in range(idx - 1, -1, -1):
            if segs[j][0] == "idle":
                base = float(np.median(tin[segs[j][1]:segs[j][2] + 1]))
                break
        if base is None:
            base = float(np.median(tin[max(0, i0 - 50):i0 + 1]))
        t0 = float(el[i0])
        m = (el >= t0 + w0) & (el <= t0 + w1)
        if not np.any(m):
            m = (el >= t0) & (el <= t0 + w1)
        lvl = float(np.median(tin[m])) if np.any(m) else float(tin[i0])
        rows.append(dict(seg=idx, i_on=int(i0), i_off=int(i1), t_on=t0,
                         t_off=float(el[i1]), dur=float(el[i1] - el[i0]),
                         base=base, lvl=lvl, step=lvl - base,
                         ded_med=float(np.median(ded[m])) if np.any(m) else 0.0,
                         ded_frac=(float(np.median(ded[m])) / (lvl - base)
                                   if abs(lvl - base) > 1e-9 else 0.0)))
    return rows


def fmt_table(rows, cols=None, header=None):
    cols = cols or ["seg", "t_on", "dur", "base", "step", "ded_med", "ded_frac"]
    header = header or cols
    lines = [" ".join("%10s" % h for h in header)]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append("%10.2f" % v if isinstance(v, float) else "%10s" % v)
        lines.append(" ".join(cells))
    return "\n".join(lines)
