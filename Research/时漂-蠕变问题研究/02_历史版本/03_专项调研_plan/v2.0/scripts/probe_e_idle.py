# -*- coding: utf-8 -*-
"""v2.0 PROBE-E：空载判据可行性与候选替代判据的多数据集体检（只读统计）。

背景：现役判据（drift_v6_compensator.cpp:317-319）
    idle_now = (ts_smooth < 0.10·max(level_ref, eps)) || (ts_smooth < 1.5·min_ts + eps)
其中 `min_ts` 是**补偿器全生命周期**最小值。在「空载基线远大于负载增量」的工况下两条判据同时退化
（T9 报告 §2.1 已证）。本脚本用与 C++ 相同的 EMA 规则把 `ts_smooth / level_ref / min_ts` 重算一遍，
逐数据集比较「现役判据」与「滚动窗相对判据候选」的可用性。

只做统计，不改任何源码；输出 stdout 供人工判读。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

tmp = os.path.join(L.ROOT, "temp")
LEGACY = os.path.join(tmp, "原始数据only")

CASES = [
    ("新录制·从零基线", os.path.join(tmp, "算法数据&原始数据",
                                     "从零基线开始 - 恒定负载 - 反复加减同一个负载",
                                     "20260919_100351_single_device_f9740b"),
     "device_001_pre_seg0.csv", True),
    ("T9·恒定负载(装夹预载)", os.path.join(tmp, "算法数据&原始数据",
                                       "恒定负载下反复加减同一个负载",
                                       "20260919_092417_single_device_602c03"),
     "device_001_pre_seg0.csv", True),
    ("四指指尖·数据1(力域)", os.path.join(LEGACY, "四指指尖", "数据1"),
     "device_001_seg000.csv", False),
    ("右拇指指尖·数据1(力域)", os.path.join(LEGACY, "右拇指指尖", "数据1"),
     "device_001_seg000.csv", False),
    ("变化负载·切换负载-快相无责", os.path.join(
        LEGACY, "变化负载", "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc"),
     "device_001_seg000.csv", False),
]


def read_any(path, prefer_pre):
    """读任意录制 CSV；返回 (el, V)。表头有通道名就用通道名（取 chNN，排除 raw_ch），
    否则按位置回退（末 NCH 列）。"""
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    idx = [i for i, h in enumerate(hdr) if h.startswith("ch") and not h.startswith("ch_")]
    if not idx:
        ncol = len(rows[di + 2].split(","))
        idx = list(range(ncol - L.NCH, ncol))
    el, vals = [], []
    for r in rows[di + 2:]:
        if not r.strip():
            continue
        f = r.split(",")
        el.append(float(f[1]))
        vals.append([float(f[i]) for i in idx])
    return np.asarray(el), np.asarray(vals)


def ema_series(tot, el, tau=0.30, tau_lvl=10.0, dt_cap=0.1):
    ts_s = np.empty_like(tot)
    lvl = np.empty_like(tot)
    ts_s[0] = lvl[0] = tot[0]
    last = el[0]
    for i in range(1, len(tot)):
        dt = el[i] - last
        last = el[i]
        dt = min(dt, dt_cap) if dt > 0 else 0.0
        if dt > 0:
            ts_s[i] = ts_s[i - 1] + (dt / tau) * (tot[i] - ts_s[i - 1])
            lvl[i] = lvl[i - 1] + (dt / tau_lvl) * (ts_s[i] - lvl[i - 1])
        else:
            ts_s[i], lvl[i] = ts_s[i - 1], lvl[i - 1]
    return ts_s, lvl


def rolling_min(x, w):
    """按帧数的滚动最小值（w 帧），O(n)。"""
    import collections
    dq = collections.deque()
    out = np.empty_like(x)
    for i, v in enumerate(x):
        while dq and x[dq[-1]] >= v:
            dq.pop()
        dq.append(i)
        while dq[0] <= i - w:
            dq.popleft()
        out[i] = x[dq[0]]
    return out


def rolling_range(x, w):
    import collections
    dq_lo, dq_hi = collections.deque(), collections.deque()
    out = np.empty_like(x)
    for i, v in enumerate(x):
        while dq_lo and x[dq_lo[-1]] >= v:
            dq_lo.pop()
        dq_lo.append(i)
        while dq_hi and x[dq_hi[-1]] <= v:
            dq_hi.pop()
        dq_hi.append(i)
        while dq_lo[0] <= i - w:
            dq_lo.popleft()
        while dq_hi[0] <= i - w:
            dq_hi.popleft()
        out[i] = x[dq_hi[0]] - x[dq_lo[0]]
    return out


def main():
    print(f"{'数据集':26s}{'n':>7}{'空载(3%)':>10}{'受载(97%)':>10}{'增量/基线':>10}"
          f"{'现役idle%':>11}{'W30 idle%':>11}{'真空载帧%':>11}")
    for name, d, fn, need_pre in CASES:
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            print(f"{name:26s}  MISSING {p}")
            continue
        el, V = read_any(p, need_pre)
        tot = V.sum(1)
        n = len(el)
        ts_s, lvl = ema_series(tot, el)
        lt = np.minimum.accumulate(ts_s)
        eps = 1e-6 * (1 + abs(tot).max())
        idle_cur = (ts_s < 0.10 * np.maximum(lvl, eps)) | (ts_s < 1.5 * lt + eps)

        # 候选：30 s 滚动窗（按 ~100 Hz → 3000 帧）
        w = int(30.0 * 100)
        w = min(w, max(n // 2, 10))
        wmin = rolling_min(ts_s, w)
        wrng = rolling_range(ts_s, w)
        wmax = wmin + wrng
        # 判据：接近滚动窗谷底 且 窗内幅度足够大（说明确实经历过一次加载）
        near = ts_s < wmin + 0.10 * np.maximum(wrng, eps)
        enough = wrng > 0.15 * max(wmax.max(), eps)
        idle_cand = near & enough

        lo = np.percentile(tot, 3)
        hi = np.percentile(tot, 97)
        thr_true = lo + 0.25 * (hi - lo)
        truth_idle = tot < thr_true
        amp = hi - lo
        print(f"{name:26s}{n:7d}{lo:10.0f}{hi:10.0f}{amp/max(lo,1):10.2f}"
              f"{100*idle_cur.mean():10.1f}%{100*idle_cand.mean():10.1f}%"
              f"{100*truth_idle.mean():10.1f}%")

        # 一致性：候选判据相对"真值空载"的漏报/误报
        tp = (idle_cand & truth_idle).sum()
        fn_ = ((~idle_cand) & truth_idle).sum()
        fp = (idle_cand & ~truth_idle).sum()
        print(f"{'':26s}  候选 vs 真值: 命中={tp} 漏报={fn_} 误报={fp} "
              f"(漏报率={100*fn_/max(tp+fn_,1):.1f}% 误报率={100*fp/max(tp+fp,1):.1f}%)")


if __name__ == "__main__":
    main()
