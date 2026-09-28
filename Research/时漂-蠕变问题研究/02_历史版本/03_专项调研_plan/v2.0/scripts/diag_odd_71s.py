# -*- coding: utf-8 -*-
"""只读诊断：`temp/算法数据&原始数据/各种奇怪工况/20260919_134056_single_device_f40a1b`
在 71~73 s 之间那段"看起来有点怪"的爬升。

不改任何源码、不改算法、不写交付产物，只做统计与打印。
口径：总量 = 21 通道求和（ADC，显示域）；偏移 off = main − pre（逐帧相减）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

DS = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                  "20260919_134056_single_device_f40a1b")


def main():
    pre = L.load_stream(DS, "device_001_pre_seg0.csv")
    main = L.load_stream(DS, "device_001_seg000.csv")
    el = pre["el"]
    tp = pre["V"].sum(1)
    tm = main["V"].sum(1)
    off = tm - tp

    # ── 1. 大尺度：60~85 s 的 1 s 桶，先看这段在整个形态中的位置 ──
    print("── 60~85 s，1 s 桶（pre / main / off）──")
    print(f"{'t':>7}{'pre':>9}{'main':>9}{'off':>8}{'dPre/s':>9}")
    prev = None
    for t in np.arange(60.0, 85.0, 1.0):
        m = (el >= t) & (el < t + 1.0)
        if not m.any():
            continue
        p, q = float(np.mean(tp[m])), float(np.mean(tm[m]))
        d = (p - prev) if prev is not None else 0.0
        print(f"{t:7.1f}{p:9.0f}{q:9.0f}{q-p:8.0f}{d:9.0f}")
        prev = p

    # ── 2. 细尺度：70.5~73.5 s，0.05 s 桶 ──
    print("\n── 70.5~73.5 s，0.05 s 桶（20 Hz）──")
    print(f"{'t':>8}{'pre':>9}{'main':>9}{'off':>8}{'dPre/50ms':>11}{'dMain/50ms':>12}")
    prevp = prevm = None
    for t in np.arange(70.5, 73.5, 0.05):
        m = (el >= t) & (el < t + 0.05)
        if not m.any():
            continue
        p, q = float(np.mean(tp[m])), float(np.mean(tm[m]))
        dp = (p - prevp) if prevp is not None else 0.0
        dq = (q - prevm) if prevm is not None else 0.0
        print(f"{t:8.2f}{p:9.0f}{q:9.0f}{q-p:8.0f}{dp:11.0f}{dq:12.0f}")
        prevp, prevm = p, q

    # ── 3. 逐帧：爬升的起止与形态（用斜率定位）──
    i0 = int(np.searchsorted(el, 70.0))
    i1 = int(np.searchsorted(el, 74.0))
    seg_p = tp[i0:i1]
    seg_t = el[i0:i1]
    # 每帧斜率（总量/秒），用 0.1 s 窗的差
    print("\n── 逐帧爬升率（70~74 s，仅列 |斜率| > 50 ADC/s 的帧，按 0.1 s 聚合）──")
    print(f"{'t':>8}{'pre':>9}{'斜率ADC/s':>11}{'main':>9}{'m斜率':>9}")
    last = -9
    for i in range(i0, i1):
        j = i + 10
        if j >= len(el):
            break
        slope = (tp[j] - tp[i]) / max(el[j] - el[i], 1e-9)
        mslope = (tm[j] - tm[i]) / max(el[j] - el[i], 1e-9)
        if abs(slope) > 50 and el[i] - last > 0.1:
            print(f"{el[i]:8.2f}{tp[i]:9.0f}{slope:11.0f}{tm[i]:9.0f}{mslope:9.0f}")
            last = el[i]

    # ── 4. 该段逐通道：谁在爬 ──
    print("\n── 71.0~73.0 s 的逐通道变化（末−首，按 |Δ| 排序）──")
    a = int(np.searchsorted(el, 71.0))
    b = int(np.searchsorted(el, 73.0))
    d = pre["V"][b - 1] - pre["V"][a]
    order = np.argsort(-np.abs(d))
    print(f"{'ch':>4}{'t=71s':>10}{'t=73s':>10}{'Δpre':>10}{'Δmain':>10}")
    for c in order[:12]:
        print(f"ch{c:<2d}{pre['V'][a][c]:10.0f}{pre['V'][b-1][c]:10.0f}"
              f"{d[c]:10.0f}{main['V'][b-1][c]-main['V'][a][c]:10.0f}")
    print(f"  合计Δpre={d.sum():.0f}  Δmain={(main['V'][b-1]-main['V'][a]).sum():.0f}")

    # ── 5. 与它的上下文比较：这段是不是"孤立"的爬升 ──
    print("\n── 该段前后的稳定性（各 3 s 窗的 pre 极差）──")
    for t in (66.0, 68.0, 70.0, 71.0, 72.0, 73.0, 74.0, 76.0, 78.0):
        m = (el >= t) & (el < t + 3.0)
        if m.any():
            s = tp[m]
            print(f"  t={t:5.1f}~{t+3:5.1f}  pre 中位={np.median(s):7.0f} "
                  f"极差={s.max()-s.min():7.0f} 标准差={s.std():7.1f}")

    # ── 6. 采样率/时间戳是否异常（爬升可能只是时间轴问题）──
    print("\n── 70~74 s 的帧间隔分布 ──")
    m = (el >= 70.0) & (el < 74.0)
    idx = np.where(m)[0]
    dt = np.diff(el[idx])
    print(f"  帧数={len(idx)}  dt 中位={np.median(dt)*1000:.2f}ms  "
          f"p95={np.percentile(dt,95)*1000:.2f}ms  max={dt.max()*1000:.2f}ms  "
          f"dt<=0 的帧数={int((dt<=0).sum())}")
    ts = pre["ts"][idx]
    dts = np.diff(ts)
    print(f"  timestamp 间隔：中位={np.median(dts)*1000:.2f}ms  max={dts.max()*1000:.2f}ms  "
          f"重复时间戳帧数={int((dts<=0).sum())}")


if __name__ == "__main__":
    main()
