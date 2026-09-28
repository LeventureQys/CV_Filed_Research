# -*- coding: utf-8 -*-
"""步骤 2c：沿附近 0.1 s 分辨率的原始轨迹与导数，用于判定"机械加载何时结束"。
用法： python t1_edgeview.py
输出： results/t1_edgeview.txt
"""
import os
import sys

import numpy as np

import t1_lib as T

CASES = [
    ("working", "9c3ca5", 3.0),
    ("working", "73032d", 63.3),
    ("working", "3f32c5", 17.0),
    ("archived", "0cb8b6", 121.3),
    ("archived", "持续恒定负载", 4.2),
    ("archived", "7b3977", 16.5),
]


def main():
    out = []
    w = out.append
    for tag, key, t_target in CASES:
        root = T.WORKING if tag == "working" else T.ARCHIVED
        label = pre = None
        for lb, d in T.discover_sessions(root, 5):
            if key in lb:
                label, pre = lb, T.load_session(d)[0]
                break
        if pre is None:
            w("!! miss %s" % key)
            continue
        t, V = pre["t"], pre["V"]
        tot = T.total(pre)
        steps, sm = T.detect_steps(t, tot)
        ups = [s for s in steps if s["sign"] > 0]
        s = min(ups, key=lambda q: abs(q["t1"] - t_target))
        rng = V.max(axis=0) - V.min(axis=0)
        c = int(np.argmax(rng))
        w("=" * 112)
        w("%s  沿 t0=%.2f t1=%.2f 沿宽%.2f Δ=%+.0f  最大通道 ch%d (range %.0f)"
          % (label, s["t0"], s["t1"], s["dur"], s["d"], c, rng[c]))
        w("   %-9s %10s %10s %10s %10s %10s" %
          ("t-t0", "tot", "d(tot)/dt", "ch%d" % c, "d(ch)/dt", "note"))
        for x in np.arange(-1.0, 12.01, 0.2):
            tt = s["t0"] + x
            k = int(np.searchsorted(t, tt))
            if k < 1 or k >= len(t):
                continue
            dtot = (sm[k + 1] - sm[k - 1]) / (t[k + 1] - t[k - 1])
            yc = T.smooth_ma(V[:, c], 21)
            dch = (yc[k + 1] - yc[k - 1]) / (t[k + 1] - t[k - 1])
            note = ""
            if abs(x) < 1e-9:
                note = "<- 沿起"
            elif abs(tt - s["t1"]) < 0.15:
                note = "<- 沿末(门限判据)"
            w("   %+9.2f %10.0f %10.0f %10.1f %10.2f %s"
              % (x, sm[k], dtot, V[k, c], dch, note))
    txt = "\n".join(out)
    os.makedirs(T.RESULTS, exist_ok=True)
    with open(os.path.join(T.RESULTS, "t1_edgeview.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    sys.stdout.write("written results/t1_edgeview.txt\n")


if __name__ == "__main__":
    main()
