# -*- coding: utf-8 -*-
"""T2 步骤1：把每个会话的「输入总量」时序压成 5 秒桶中位序列打印出来，
用于人工确认台阶结构、保压段位置和回落现象，再据此定义事件切分参数。"""
import sys

import numpy as np

import t2_lib as T


def bucket(el, y, dt=5.0):
    out = []
    t = el[0]
    while t < el[-1]:
        m = (el >= t) & (el < t + dt)
        if m.sum():
            out.append((t, float(np.median(y[m]))))
        t += dt
    return out


def main(argv):
    only = argv[1] if len(argv) > 1 else None
    for label, d in T.all_sessions():
        if only and only not in label:
            continue
        s = T.load_pre(d)
        el, tot = s["el"], T.total(s)
        rec = T.load_recorded(d)
        rt = T.total(rec) if rec is not None else None
        print("=" * 110)
        print("%s   n=%d  %.0fs  输入 %.0f~%.0f"
              % (label, s["n"], el[-1], tot.min(), tot.max()))
        b = bucket(el, tot, 5.0)
        line = []
        for t, v in b:
            line.append("%4.0f:%7.0f" % (t, v))
            if len(line) == 6:
                print("   " + " ".join(line))
                line = []
        if line:
            print("   " + " ".join(line))
        if rt is not None:
            b2 = bucket(el, rt, 5.0)
            print("   [录制显示] " + " ".join("%4.0f:%7.0f" % (t, v) for t, v in b2[:18]))
        evs = T.load_events(el, tot)
        print("   事件 %d 个: %s" % (
            len(evs),
            "; ".join("t=%.1f→%.1f step=%.0f" % (e["t_up"], e["t_dn"], e["step"])
                      for e in evs)))


if __name__ == "__main__":
    main(sys.argv)
