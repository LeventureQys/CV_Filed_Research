# -*- coding: utf-8 -*-
"""v3.4 观测器 v2 验证：新恒载录制（衰减问题）+ 目标录制（±10% 一致性）+ 平坦度。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_observer_core2 import observe2
from v34_observer_eval import HOLDS
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [
    ("新恒载_71208f", os.path.join(
        L.DATA_ROOT, "working", "零基线-同一荷载测试")),
    ("目标_7b3977", os.path.join(
        L.DATA_ROOT, "archived", "零基线-反复增减同一负载",
        "20260919_160854_single_device_7b3977")),
    ("归档恒载", os.path.join(
        L.DATA_ROOT, "archived", "持续恒定负载")),
    ("归档_0cb8b6", os.path.join(
        L.DATA_ROOT, "archived", "零基线-反复增减同一负载",
        "20260919_152749_single_device_0cb8b6")),
]


def load_first(root):
    if os.path.isfile(os.path.join(root, "session.json")):
        dirs = [("", root)]
    else:
        dirs = L.discover_sessions(root, 3)
    for label, d in dirs:
        name = ("device_001_pre_seg0.csv"
                if os.path.isfile(os.path.join(d, "device_001_pre_seg0.csv"))
                else "device_001_seg000.csv")
        return label, L.load_stream(d, name)
    raise SystemExit("no session under " + root)


def main():
    for tag, root in SESSIONS:
        label, s = load_first(root)
        el, ts, V = s["el"], s["ts"], s["V"]
        D, X = observe2(ts, V)
        din = V.sum(axis=1)
        dout = D.sum(axis=1)
        print("== %s (%s, %.0fs)" % (tag, label, el[-1]))
        if tag.startswith("新恒载"):
            for t in (20, 60, 120, 180, 240, 290, 330):
                i = int(np.argmin(np.abs(el - t)))
                print("   t=%3ds 输入=%7.0f 显示=%7.0f x=%7.0f"
                      % (t, din[i], dout[i], X[i].sum()))
            m = (el >= 30) & (el <= 340)
            print("   恒载段 [30,340]s: 显示 std=%.0f 漂移=%+0.0f（输入漂移 %+0.0f）"
                  % (np.std(dout[m]),
                     dout[m][:-1][np.searchsorted(el[m], 330)] - dout[m][np.searchsorted(el[m], 30)],
                     din[m][np.searchsorted(el[m], 330)] - din[m][np.searchsorted(el[m], 30)]))
        if tag == "目标_7b3977":
            first, ok, tot = {}, 0, 0
            for cls, t0, t1 in HOLDS["目标working_7b3977"]:
                m = (el >= t0 + 1.0) & (el <= t1)
                if m.sum() < 30:
                    continue
                lv = float(np.median(dout[m]))
                if cls not in first:
                    first[cls] = lv
                    continue
                dev = lv - first[cls]
                ok += abs(dev) <= 1500.0
                tot += 1
                print("   %s t=%6.1f 显示=%7.0f 偏差=%+6.0f %s"
                      % (cls, t0, lv, dev, "OK" if abs(dev) <= 1500 else "**超差**"))
            print("   一致性合格 %d/%d" % (ok, tot) if tot else "")
        if tag.startswith("归档恒载"):
            m = (el >= 30)
            seg = dout[m]
            print("   长保压显示漂移（30s→末尾）: %+0.0f ADC（输入 %+0.0f）"
                  % (seg[-1] - seg[0], din[m][-1] - din[m][0]))


if __name__ == "__main__":
    main()
