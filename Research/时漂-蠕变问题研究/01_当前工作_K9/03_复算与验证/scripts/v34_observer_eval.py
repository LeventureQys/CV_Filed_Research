# -*- coding: utf-8 -*-
"""v3.4 观测器一致性验证（±10% 判据）：三个反复增减录制。

判据（用户口径）：同一负载（满载/半载）反复增减，显示电平相对**首次**电平的
偏差在 ±10%·台阶 以内即合格。同时报保压期平坦度（显示 std）与朴素基线对比。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_observer_core import observe
from v34_law_identify import segments, classify, SESSIONS
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))

HOLDS = {  # 7b3977 手工沿表（类, 沿完成, 段末）
    "目标working_7b3977": [
        ("满载", 17.0, 35.5), ("半载", 38.0, 40.5),
        ("满载", 43.0, 55.5), ("半载", 58.0, 62.0),
        ("满载", 64.5, 215.0), ("满载", 234.0, 238.5),
        ("满载", 247.0, 250.5), ("半载", 253.3, 254.4),
        ("满载", 263.0, 274.0), ("满载", 279.5, 281.2),
        ("半载", 283.0, 288.0), ("满载", 290.5, 294.2),
        ("半载", 295.5, 296.2), ("满载", 304.5, 307.5)],
}


def main():
    for tag, d in SESSIONS:
        s = L.load_stream(d, "device_001_pre_seg0.csv")
        el, ts = s["el"], s["ts"]
        V = s["V"]
        D, X = observe(ts, V)
        din = V.sum(axis=1)
        dout = D.sum(axis=1)
        dx = X.sum(axis=1)
        # 通用段表做安静段平坦度
        segs = classify(segments(el, din))
        flats = []
        for t0, t1, lv, c in segs:
            m = (el >= t0 + 2.0) & (el <= t1 - 1.0)
            if m.sum() < 100:
                continue
            flats.append((t0, t1, float(np.std(din[m])), float(np.std(dout[m]))))
        print("== %s" % tag)
        print("   保压段显示 std：输入std → 显示std（应显著更小）")
        for t0, t1, si, so in flats[:14]:
            print("     t=%6.1f~%6.1f  in_std=%6.0f → out_std=%6.0f" % (t0, t1, si, so))
        if tag in HOLDS:
            print("   ±10% 一致性（相对该类首次电平，10%·满载台阶≈1500 ADC）:")
            first = {}
            ok = 0
            tot = 0
            for cls, t0, t1 in HOLDS[tag]:
                m = (el >= t0 + 1.0) & (el <= t1)
                if m.sum() < 30:
                    continue
                lv = float(np.median(dout[m]))
                if cls not in first:
                    first[cls] = lv
                    print("     %s 首次(基准) = %7.0f" % (cls, lv))
                    continue
                dev = lv - first[cls]
                tot += 1
                passed = abs(dev) <= 1500.0
                ok += passed
                print("     %s t=%6.1f 显示=%7.0f 偏差=%+6.0f %s"
                      % (cls, t0, lv, dev, "OK" if passed else "**超差**"))
            if tot:
                print("     合格 %d/%d" % (ok, tot))


if __name__ == "__main__":
    main()
