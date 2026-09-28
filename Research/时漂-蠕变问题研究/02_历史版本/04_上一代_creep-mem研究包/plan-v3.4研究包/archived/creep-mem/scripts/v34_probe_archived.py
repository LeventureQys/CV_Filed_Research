# -*- coding: utf-8 -*-
"""v3.4 附加验证（v2）：discover 定位会话；0cb8b6 差分时间线 + 随机切换工况偏移。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_regression import run
import v30_lib as L


def find(sub):
    for label, d in L.discover_sessions():
        if sub in label.replace("\\", "/"):
            return label, d
    raise SystemExit("not found: " + sub)


def diff_timeline(el, d0, d1, win=5.0):
    """按 win 秒窗聚合 |Δ|，输出变化集中的区间。"""
    diffs = np.abs(d1 - d0)
    t = el[0]
    while t < el[-1]:
        m = (el >= t) & (el < t + win)
        if m.sum():
            mx = diffs[m].max()
            if mx > 50:
                print("    t=%6.1f~%6.1f  max|Δ|=%7.0f  median=%6.0f"
                      % (t, t + win, mx, np.median(diffs[m])))
        t += win


def main():
    for sub, tag in (("0cb8b6", "归档-零基线反复增减"),
                     ("6b2e70", "随机切换A"),
                     ("31b7f9", "随机切换B")):
        label, d = find(sub)
        pre, s0, o0 = run(d, ["--mem", "0"])
        _, s1, o1 = run(d, ["--mem", "120"])
        el = pre["el"]
        print("\n== %s  %s  帧%d" % (tag, label, len(o0)))
        dd = o1 - o0
        print("  Δ(out_ON-out_OFF): p50=%+6.0f p05=%+7.0f p95=%+7.0f max|Δ|=%7.0f"
              % (np.median(dd), *np.percentile(dd, [5, 95]), np.abs(dd).max()))
        print("  变化区间(>50):")
        diff_timeline(el, o0, o1, 10.0)
        for name, oo in (("OFF", o0), ("ON ", o1)):
            x = oo - s0
            print("  %s 显示-输入: p5=%+7.0f p50=%+7.0f p95=%+7.0f"
                  % (name, *np.percentile(x, [5, 50, 95])))


if __name__ == "__main__":
    main()
