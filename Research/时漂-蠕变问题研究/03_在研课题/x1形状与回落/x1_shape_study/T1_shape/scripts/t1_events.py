# -*- coding: utf-8 -*-
"""步骤 1：台阶目录 —— 逐会话切出加载/卸载沿，列出各通道台阶幅度与保压时长。

输出 results/t1_steps_working.txt、results/t1_steps_archived.txt
用法： python t1_events.py
"""
import os
import sys

import numpy as np

import t1_lib as T

MAX_RAMP = 2.5       # 沿宽上限（s）：更慢的"压入"不算台阶，不做快相形状分析
MIN_PLATEAU = 20.0   # 保压时长下限（s）


def catalog(root, tag):
    out = []
    w = out.append
    for label, d in T.discover_sessions(root, 5):
        pre, rec = T.load_session(d)
        t, V = pre["t"], pre["V"]
        tot = T.total(pre)
        steps, sm = T.detect_steps(t, tot)
        ups = [s for s in steps if s["sign"] > 0]
        w("=" * 112)
        w("%s   帧=%d %.0f s  台阶 加载%d/卸载%d  thr=%.0f ADC/s  min_step=%.0f"
          % (label, len(t), t[-1], len(ups), len(steps) - len(ups),
             steps[0]["thr"] if steps else float("nan"),
             steps[0]["min_step"] if steps else float("nan")))
        amp = V.max(axis=0) - np.percentile(V, 5, axis=0)
        chs = T.top_channels(pre, 6)
        w("  通道幅度 top: " + "  ".join("ch%d=%.0f" % (c, amp[c]) for c in chs))
        for k, s in enumerate(steps):
            nxt = steps[k + 1]["t0"] if k + 1 < len(steps) else t[-1]
            plateau = nxt - s["t1"]
            usable = (s["sign"] > 0 and s["dur"] <= MAX_RAMP
                      and plateau >= MIN_PLATEAU)
            w("  %s t=%7.2f 沿宽%5.2f s Δ=%+8.0f 保压%6.1f s %s"
              % ("UP" if s["sign"] > 0 else "DN", s["t1"], s["dur"], s["d"],
                 plateau, "<-- 可用" if usable else ""))
            if s["sign"] > 0:
                cells = []
                for c in chs:
                    y = V[:, c]
                    a = int(np.searchsorted(t, s["t0"]))
                    b = int(np.searchsorted(t, s["t1"]))
                    lv_pre = float(np.median(y[max(0, a - 50):a + 1]))
                    lv_post = float(np.median(y[b:b + 50]))
                    cells.append("ch%d%+7.0f(r%.0f)" % (c, lv_post - lv_pre,
                                                        amp[c]))
                w("        通道台阶: " + " ".join(cells))
    txt = "\n".join(out)
    os.makedirs(T.RESULTS, exist_ok=True)
    fn = os.path.join(T.RESULTS, "t1_steps_%s.txt" % tag)
    with open(fn, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    sys.stdout.write("written %s (%d lines)\n" % (fn, len(out)))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("working", "both"):
        catalog(T.WORKING, "working")
    if which in ("archived", "both"):
        catalog(T.ARCHIVED, "archived")
