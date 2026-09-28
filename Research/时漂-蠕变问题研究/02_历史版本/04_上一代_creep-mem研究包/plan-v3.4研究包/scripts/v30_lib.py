# -*- coding: utf-8 -*-
"""v3.0 共享层：装载「测试台工况一览」录制 + 平台段切分 + 漂移指标。

数据集三条流（帧数与时间轴逐帧对齐）：
  device_001_pre_seg0.csv    算法前读数（关算法时本该显示的显示域值）
  device_001_seg000.csv      算法结果（主文件）
  device_001_raw_seg000.csv  原始 ADC 逐帧流

与 plan/v2.0/scripts/v20_lib.py 的差异：
  - 数据根目录可指向任意会话目录（本版主数据在 `测试台工况一览/` 下）；
  - pre 流表头缺 21 个通道名（交付缺陷），按位置回退（末 NCH 列即通道）；
  - 新增 `plateaus()`：用「空载带/受载带」双门限的滞回切分，供稳定工况静置段量测。
"""
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
assert os.path.isfile(os.path.join(ROOT, "CMakeLists.txt")), ROOT
CPP_V6 = os.path.join(ROOT, "src", "domain", "drift_v6",
                      "drift_v6_compensator.cpp")
DATA_ROOT = os.path.join(ROOT, "temp", "算法数据&原始数据")
OVERVIEW = os.path.join(DATA_ROOT, "测试台工况一览")
DS_ZERO = os.path.join(DATA_ROOT, "从零基线开始 - 恒定负载 - 反复加减同一个负载",
                       "20260919_100351_single_device_f9740b")
DS_T9 = os.path.join(DATA_ROOT, "恒定负载下反复加减同一个负载",
                     "20260919_092417_single_device_602c03")
NCH = 21


def discover_sessions(root=None, max_depth=4):
    """扫描数据根，返回 [(标签, 会话目录)]。

    用户的 `temp/算法数据&原始数据/` 会在 `working/`（在用）与 `archived/`（归档）
    之间搬动，且会话目录可能在任意层级（甚至直接就是那一层）。因此这里**不写死路径**，
    只认「目录里有 session.json，且有 device_001_pre_seg0.csv 或 device_001_seg000.csv」。
    """
    root = root or DATA_ROOT
    out = []
    if not os.path.isdir(root):
        return out

    def has(dirpath):
        if not os.path.isfile(os.path.join(dirpath, "session.json")):
            return False
        return (os.path.isfile(os.path.join(dirpath, "device_001_pre_seg0.csv"))
                or os.path.isfile(os.path.join(dirpath, "device_001_seg000.csv")))

    def walk(dirpath, depth):
        if has(dirpath):
            rel = os.path.relpath(dirpath, root)
            out.append((rel.replace("\\", "/"), dirpath))
            return
        if depth <= 0:
            return
        try:
            subs = sorted(os.listdir(dirpath))
        except OSError:
            return
        for d in subs:
            q = os.path.join(dirpath, d)
            if os.path.isdir(q):
                walk(q, depth - 1)

    walk(root, max_depth)
    return out


def read_rows(path):
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
    import numpy as np
    cfg, data = read_rows(os.path.join(ds_dir, name))
    n = len(data)
    ts = np.empty(n)
    el = np.empty(n)
    fr = np.empty(n, dtype=np.int64)
    V = np.empty((n, NCH))
    for i, f in enumerate(data):
        if len(f) < 3 + NCH:
            raise ValueError("%s 行 %d 列数不足: %d" % (name, i, len(f)))
        ts[i] = float(f[0])
        el[i] = float(f[1])
        fr[i] = int(float(f[2]))
        V[i] = [float(x) for x in f[3:3 + NCH]]
    return dict(ts=ts, el=el, fr=fr, V=V, n=n, cfg=cfg, name=name)


def load_dataset(ds_dir):
    pre = load_stream(ds_dir, "device_001_pre_seg0.csv")
    main = load_stream(ds_dir, "device_001_seg000.csv")
    raw = load_stream(ds_dir, "device_001_raw_seg000.csv")
    for s in (pre, main, raw):
        assert s["n"] == pre["n"], (s["name"], s["n"], pre["n"])
    with open(os.path.join(ds_dir, "session.json"), encoding="utf-8") as fh:
        sess = json.load(fh)
    return dict(pre=pre, main=main, raw=raw, sess=sess)


def totals(ds):
    import numpy as np
    return (ds["pre"]["V"].sum(axis=1),
            ds["main"]["V"].sum(axis=1),
            ds["raw"]["V"].sum(axis=1))


def hysteresis_segments(tot, lo_frac=0.25, hi_frac=0.75, min_frames=25):
    """按电平把序列切成 plateau 段：tot < lo => idle；tot > hi => loaded。

    lo/hi 由 tot 的 min/max 线性插值给出。返回 [(kind, i0, i1), ...]。
    """
    import numpy as np
    t = np.asarray(tot, dtype=float)
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
    return [(k, i0, i1) for (k, i0, i1) in segs if i1 - i0 >= min_frames]


def slope_per_s(el, y, i0, i1, frac=0.05):
    """段内「首尾 5% 均值」之差 / 时间差 => ADC/s。"""
    import numpy as np
    m = max(1, int(frac * (i1 - i0 + 1)))
    y0 = float(np.mean(y[i0:i0 + m]))
    y1 = float(np.mean(y[i1 - m + 1:i1 + 1]))
    dt = float(el[i1] - el[i0])
    return (y1 - y0) / dt if dt > 0 else 0.0


def band_stats(el, y, i0, i1, dt_flat=1.0):
    """段内按 dt_flat 秒桶取中位，返回 (t[], med[], pk_pk, drift_first_last)。"""
    import numpy as np
    t, med = [], []
    i = i0
    while i <= i1:
        j = i
        while j <= i1 and el[j] < el[i] + dt_flat:
            j += 1
        if j > i:
            t.append(float(el[i]))
            med.append(float(np.median(y[i:j])))
        i = j
    med = np.asarray(med)
    return (np.asarray(t), med,
            float(med.max() - med.min()) if len(med) else 0.0,
            float(med[-1] - med[0]) if len(med) else 0.0)


def smoothstep(x):
    import numpy as np
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def edge_indices(tot, thr):
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
