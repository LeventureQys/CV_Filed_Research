# -*- coding: utf-8 -*-
"""v3.6 工具：把录制按 0.5 s 网格打成紧凑数值表（用于替代看图）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
from v36_probe_edges import wmed  # noqa: E402


def grid_table(ds, t0, t1, step=0.5, path=None, head=None):
    el = ds["pre"]["el"]
    tin, tout, ded = ds["tot_in"], ds["tot_out"], ds["ded"]
    ts = np.arange(t0, t1 + 1e-9, step)
    lines = []
    lines.append(head or ("== 网格表 [%.1f, %.1f] s  step=%.2f ==" % (t0, t1, step)))
    lines.append("%8s %10s %10s %10s %8s" % ("t", "in", "out", "ded", "out/in"))
    for t in ts:
        a = wmed(el, tin, t, t + step)
        b = wmed(el, tout, t, t + step)
        if np.isnan(a):
            continue
        lines.append("%8.2f %10.0f %10.0f %10.0f %8.4f" % (t, a, b, a - b, b / a if abs(a) > 1 else float("nan")))
    txt = "\n".join(lines)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return txt


if __name__ == "__main__":
    ds = K.load(K.DS_TARGET)
    t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 230.0
    t1 = float(sys.argv[2]) if len(sys.argv) > 2 else 312.0
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    grid_table(ds, t0, t1, step)
