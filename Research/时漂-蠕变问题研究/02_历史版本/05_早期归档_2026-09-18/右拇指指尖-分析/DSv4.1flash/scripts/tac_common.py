# -*- coding: utf-8 -*-
"""公共数据加载与基础工具（右拇指指尖 三组 空载-恒定负载-空载 数据）。"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                      # DSv4.1flash
ROOT = os.path.dirname(OUT)                      # 右拇指指尖
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

DATASETS = ["数据1", "数据2", "数据3"]
ROWS, COLS = 9, 7
LAYOUT_MASK = "110110011011101111110111111011111100000001000000100000010000001"
MASK_ON = [i for i, b in enumerate(LAYOUT_MASK) if b == "1"]
assert len(MASK_ON) == 31

# ADC 满量程（未标定时上位机显示换算上限）；数据单位 N（display_force_unit）
ADC_MAX = 4095.0


def load_csv(path):
    df = pd.read_csv(path, skiprows=24)
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    return df, ch_cols


def load_dataset(name):
    """返回 dict：t/elapsed(s)、frame_index、X(N×31)、ch_cols、fs、path"""
    path = os.path.join(ROOT, name, "device_001_seg000.csv")
    df, ch_cols = load_csv(path)
    t = df["elapsed"].to_numpy(dtype=float)
    # elapsed 是整数毫秒量化后的秒值 -> 存在重复；用 timestamp 做更精确的时间轴
    ts = df["timestamp"].to_numpy(dtype=float)
    ts = ts - ts[0]
    fi = df["frame_index"].to_numpy(dtype=float)
    X = df[ch_cols].to_numpy(dtype=float)
    dt = np.diff(ts)
    good = dt > 0
    fs = 1.0 / np.median(dt[good]) if good.any() else np.nan
    return dict(name=name, path=path, df=df, ch_cols=ch_cols, t=ts, t_raw=t,
                frame_index=fi, X=X, fs=fs, n=len(ts))


def spatial_map(vec31):
    """把 31 维向量按 layout_mask 放回 9×7 阵列（无效位为 nan）。"""
    m = np.full(ROWS * COLS, np.nan)
    v = np.asarray(vec31, dtype=float)
    for j, pos in enumerate(MASK_ON):
        m[pos] = v[j]
    return m.reshape(ROWS, COLS)


def detect_segments(total, t, fs=100.0, thr_frac=0.10, guard_s=0.5, min_len_s=2.0,
                    quiet_s=0.3):
    """基于总量信号做 空载-负载-空载 分段（绕峰值双向扫阈值法）。

    为什么不用"持续高于阈值"或"最后跌破阈值"：本数据的时漂会让总量在卸载后
    仍高于加载体现在前的水平，任何基于"是否低于某固定阈值"的持续判据都会
    把后空载段误判进负载段。

    做法（最稳健）：
      1. 取总量峰值位置 k* 与峰值 P
      2. 阈值 thr = thr_frac·(P - 静息水平)
      3. 从 k* 向左扫描，遇到第一个 total < thr 即为负载起点
      4. 从 k* 向右扫描，遇到第一个 total < thr 即为负载终点
      5. 两侧留 guard_s 保护带

    返回 dict(pre=(a,b), load=(a,b), post=(a,b), thr, amp, base, edge=(s0,s1))
    """
    total = np.asarray(total, float)
    n = len(total)
    k = int(np.argmax(total))
    P = float(total[k])
    nq = max(1, int(quiet_s * fs))
    base = float(min(np.median(total[:nq]), np.median(total[-nq:])))
    amp = P - base
    if amp <= 0:
        return None
    thr = base + thr_frac * amp
    # 向左
    s0 = 0
    i = k
    while i >= 0:
        if total[i] < thr:
            s0 = i
            break
        i -= 1
    # 向右
    s1 = n - 1
    i = k
    while i < n:
        if total[i] < thr:
            s1 = i
            break
        i += 1
    g = int(guard_s * fs)
    a = max(0, s0 - g)
    b = min(n, s1 + g)
    if b <= a + int(min_len_s * fs):
        b = n
    return dict(pre=(0, a), load=(a, b), post=(b, n),
                thr=thr, amp=amp, base=base, edge=(s0, s1),
                fall_found=(s1 < n - 1))


def polyfit_slope(t, y):
    k, b = np.polyfit(t, y, 1)
    return k, b


def r2(y, yhat):
    y = np.asarray(y, dtype=float)
    v = np.var(y)
    if v <= 0:
        return np.nan
    return 1.0 - np.var(y - yhat) / v


def moving_average(x, w):
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def savefig(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("  [fig]", name)
    return p


def dump_json(obj, name):
    p = os.path.join(RES, name)

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))

    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=conv)
    print("  [json]", name)
    return p
