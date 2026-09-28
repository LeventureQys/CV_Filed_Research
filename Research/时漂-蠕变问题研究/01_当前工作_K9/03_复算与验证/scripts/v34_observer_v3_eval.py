# -*- coding: utf-8 -*-
"""v3.4 观测器 v3 验证：空载直通（新报障录制）+ 恒载钉平 + 反复增减一致性。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_observer_core3 import observe3
from v34_observer_eval import HOLDS
import v30_lib as L

SESSIONS = [
    ("空载全程_1d4d3b（报障）", os.path.join(
        L.DATA_ROOT, "working", "零基线-同一荷载测试",
        "20260919_190545_single_device_1d4d3b"), "idle"),
    ("恒载_71208f", os.path.join(
        L.DATA_ROOT, "working", "零基线-同一荷载测试",
        "20260919_182059_single_device_71208f"), "const"),
    ("反复增减_7b3977", os.path.join(
        L.DATA_ROOT, "archived", "零基线-反复增减同一负载",
        "20260919_160854_single_device_7b3977"), "repeat"),
    ("长恒载_700s", os.path.join(
        L.DATA_ROOT, "archived", "持续恒定负载"), "const"),
]


def main():
    for tag, d, kind in SESSIONS:
        if not os.path.isfile(os.path.join(d, "session.json")):
            sid = d.split("\\")[-1].split("/")[-1]
            found = False
            for label, dd in L.discover_sessions(L.DATA_ROOT, 4):
                if sid in label.replace("\\", "/"):
                    d = dd
                    found = True
                    break
            assert found, d
        name = ("device_001_pre_seg0.csv"
                if os.path.isfile(os.path.join(d, "device_001_pre_seg0.csv"))
                else "device_001_seg000.csv")
        s = L.load_stream(d, name)
        el, ts, V = s["el"], s["ts"], s["V"]
        D, X = observe3(ts, V)
        din, dout = V.sum(axis=1), D.sum(axis=1)
        print("== %s  (%.0fs)" % (tag, el[-1]))
        if kind == "idle":
            for t in (5, 30, 60, 120, 200):
                i = int(np.argmin(np.abs(el - t)))
                print("   t=%3ds 输入=%7.0f 显示=%7.0f 差=%+5.0f"
                      % (t, din[i], dout[i], dout[i] - din[i]))
            m = (el >= 10)
            print("   [10s,末尾] 显示-输入: 均值=%+0.0f 最大偏差=%0.0f"
                  % (np.mean(dout[m] - din[m]),
                     np.max(np.abs(dout[m] - din[m]))))
        if kind == "const":
            m = (el >= 30)
            print("   [30s,末尾] 显示漂移 %+0.0f（输入 %+0.0f）std=%.0f"
                  % (dout[m][-1] - dout[m][0], din[m][-1] - din[m][0],
                     np.std(dout[m])))
        if kind == "repeat":
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
            print("   一致性合格 %d/%d" % (ok, tot))


if __name__ == "__main__":
    main()
