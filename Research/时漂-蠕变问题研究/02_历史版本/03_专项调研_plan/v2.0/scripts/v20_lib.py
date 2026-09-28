# -*- coding: utf-8 -*-
"""v2.0 共享层：数据集装载 + 通用统计工具。

数据集（三条流，帧数与时间轴逐帧对齐）：
  device_001_pre_seg0.csv   算法前读数（= 关闭算法时本该显示的显示域值）
  device_001_seg000.csv     算法结果（v6，参数集 plan-v1.0 A1+A4+A5a）
  device_001_raw_seg000.csv 原始 ADC 逐帧流（未过阈值）

已知交付缺陷（T9 报告 §3.3，本数据集**仍然存在**）：
  pre 流的 `##Data` 表头只写 `timestamp,elapsed,frame_index`，缺 21 个通道名，
  但数据行有 24 列 ⇒ 本模块按位置回退（末 NCH 列即通道）。
"""
import csv
import os

# scripts/ → v2.0/ → plan/ → v4.1flash/ → temp/ → 仓库根
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
assert os.path.isfile(os.path.join(ROOT, "CMakeLists.txt")), ROOT
assert os.path.isfile(os.path.join(ROOT, "src", "domain", "drift_v6",
                                   "drift_v6_compensator.cpp")), ROOT
# 原型（已归档到 progress/archived/ 下）
PROTO_V6 = os.path.join(ROOT, "temp", "v4.1flash", "progress", "archived",
                        "07-v6", "scripts", "glm53_v6.py")
CPP_V6 = os.path.join(ROOT, "src", "domain", "drift_v6",
                      "drift_v6_compensator.cpp")

DATA_ROOT = os.path.join(ROOT, "temp", "算法数据&原始数据")
DS_ZERO = os.path.join(DATA_ROOT, "从零基线开始 - 恒定负载 - 反复加减同一个负载",
                       "20260919_100351_single_device_f9740b")
DS_T9 = os.path.join(DATA_ROOT, "恒定负载下反复加减同一个负载",
                     "20260919_092417_single_device_602c03")
NCH = 21


def open_session(sess_json):
    import json
    with open(sess_json, encoding="utf-8") as fh:
        return json.load(fh)


def read_rows(path):
    """返回 (header_names_or_None, header_cfg dict, data_rows list[list[str]])"""
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    cfg = {}
    for r in rows[1:di]:
        if "," in r:
            k, v = r.split(",", 1)
            cfg[k] = v
    data = [r.split(",") for r in rows[di + 2:] if r.strip()]
    return cfg, data


def load_stream(ds_dir, name):
    """返回 dict(ts, el, fr, V[n, NCH], cfg, has_chnames)

    列位置固定：0=timestamp, 1=elapsed, 2=frame_index, 3..3+NCH-1=通道。
    表头缺通道名与否都按位置取（见模块 docstring 的已知缺陷说明）。
    """
    import numpy as np
    cfg, data = read_rows(os.path.join(ds_dir, name))
    has_names = bool(cfg.get("_hdr_has_ch", False))
    n = len(data)
    ts = np.empty(n)
    el = np.empty(n)
    fr = np.empty(n, dtype=np.int64)
    V = np.empty((n, NCH))
    for i, f in enumerate(data):
        if len(f) < 3 + NCH:
            raise ValueError(f"{name} 行 {i} 列数不足: {len(f)}")
        ts[i] = float(f[0])
        el[i] = float(f[1])
        fr[i] = int(float(f[2]))
        V[i] = [float(x) for x in f[3:3 + NCH]]
    return dict(ts=ts, el=el, fr=fr, V=V, cfg=cfg, n=n, has_names=has_names,
                name=name)


def load_dataset(ds_dir):
    pre = load_stream(ds_dir, "device_001_pre_seg0.csv")
    main = load_stream(ds_dir, "device_001_seg000.csv")
    raw = load_stream(ds_dir, "device_001_raw_seg000.csv")
    for s in (pre, main, raw):
        assert s["n"] == pre["n"], (s["name"], s["n"], pre["n"])
        assert abs(s["el"][0] - pre["el"][0]) < 1e-9
        assert abs(s["el"][-1] - pre["el"][-1]) < 1e-9
    return dict(pre=pre, main=main, raw=raw,
                sess=open_session(os.path.join(ds_dir, "session.json")))


# ── 通用工具 ────────────────────────────────────────────────────────────

def plateau_segments(tot, el, amp=None, hyst_frac=0.25, min_dur=0.25):
    """按滞回把 total 序列切成 idle / loaded 平台段。

    门限：以 pre 的 min/max 定义幅度；hyst_frac 为滞回带宽（占幅度比例）。
    返回 [(kind, i0, i1), ...]，短于 min_dur 的过渡段丢弃。
    """
    import numpy as np
    lo, hi = float(np.min(tot)), float(np.max(tot))
    if amp is None:
        amp = hi - lo
    mid = 0.5 * (lo + hi)
    h = hyst_frac * amp
    state = None
    segs = []
    start = 0
    for i in range(len(tot)):
        s = "loaded" if tot[i] > mid + h else ("idle" if tot[i] < mid - h else state)
        if s != state:
            if state is not None and i - 1 > start:
                segs.append((state, start, i - 1))
            state = s
            start = i
    if state is not None:
        segs.append((state, start, len(tot) - 1))
    out = []
    nmin = int(min_dur * 100)
    for kind, a, b in segs:
        if b - a >= nmin:
            out.append((kind, a, b))
    return out


def smoothstep(x):
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def edges_from_tot(tot, thr):
    """由平滑总量跨门限给出上升/下降沿时刻索引（用于事件定位，不是算法检测器）。"""
    up, dn = [], []
    prev = tot[0] > thr
    for i in range(1, len(tot)):
        cur = tot[i] > thr
        if cur and not prev:
            up.append(i)
        elif prev and not cur:
            dn.append(i)
        prev = cur
    return up, dn


def downsample(el, **series):
    """按 0.1 s 桶均值降采样，返回 (t[], dict[name]->list)"""
    import numpy as np
    step = 0.1
    t0 = el[0]
    keys = list(series)
    tb, acc, cnt = [], {k: [] for k in keys}, []
    i = 0
    n = len(el)
    while i < n:
        j = i
        ti = el[i]
        while j < n and el[j] < ti + step:
            j += 1
        if j > i:
            tb.append(ti)
            for k in keys:
                acc[k].append(float(np.mean(series[k][i:j])))
        i = j
    return tb, acc
