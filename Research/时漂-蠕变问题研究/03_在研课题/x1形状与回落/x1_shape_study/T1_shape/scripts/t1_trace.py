# -*- coding: utf-8 -*-
"""步骤 0b：把每个会话的通道总量按 1 s 桶取中位打印出来，看清工况结构。
输出 results/t1_trace.txt
"""
import os
import sys

import numpy as np

import t1_lib as T


def main():
    out = []
    w = out.append
    for label, d in T.discover_sessions(T.WORKING, 5):
        pre, rec = T.load_session(d)
        t, V = pre["t"], pre["V"]
        tot = T.total(pre)
        sm = T.smooth_ma(T.med_smooth(tot, 5), 21)
        w("=" * 108)
        w("%s   帧=%d  %.0f s" % (label, len(t), t[-1]))
        tt, med = T.band_med(t, sm, 1.0)
        k = 0
        while k < len(tt):
            chunk = med[k:k + 12]
            w("  t=%6.0f s : %s" % (tt[k], " ".join("%6.0f" % v for v in chunk)))
            k += 12
        dq = np.percentile(np.abs(np.gradient(sm, t)), [50, 90, 99, 100])
        w("  |d(tot)/dt| ADC/s: p50=%.0f p90=%.0f p99=%.0f max=%.0f"
          % (dq[0], dq[1], dq[2], dq[3]))
    txt = "\n".join(out)
    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "t1_trace.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    sys.stdout.write("written results/t1_trace.txt\n")


if __name__ == "__main__":
    main()
