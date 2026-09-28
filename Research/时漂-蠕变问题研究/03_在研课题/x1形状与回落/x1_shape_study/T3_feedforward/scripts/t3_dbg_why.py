# -*- coding: utf-8 -*-
"""T3 诊断：为什么大沿不触发？（逐帧 slope / e_pre / 门限 / 武装状态）

用法：python t3_dbg_why.py <会话关键字> <沿时刻> [臂]
"""
import sys

import numpy as np

import t3_lib as T

KW = sys.argv[1] if len(sys.argv) > 1 else "7b3977"
T0 = float(sys.argv[2]) if len(sys.argv) > 2 else 3.3
ARM = sys.argv[3] if len(sys.argv) > 3 else "A_now_a1.0"

SPECS = {
    "A_now_a1.0": dict(mode="A", pred="now", alpha=1.0),
    "A_lag_a1.0": dict(mode="A", pred="lag", alpha=1.0),
}


def main():
    for tag, label, d in T.all_sessions():
        if KW not in label:
            continue
        s = T.load_input(d)
        el, ts, V = s["el"], s["ts"], s["V"]
        i0 = max(0, int(np.searchsorted(el, T0 - 1.0)))
        i1 = int(np.searchsorted(el, T0 + 3.0))
        ff = dict(T.DET)
        ff.update(SPECS[ARM])
        ff["diag"] = (i0, i1)
        r = T.observe(ts, V, ff=ff)
        print("==", label, " 臂", ARM, " 门限 thr=%.0f rel=%.2f" % (ff["thr"], ff["rel_on"]))
        print("    帧号    t   | 输入总量 | max slope | max thr | max e_pre | med e_pre | 武装数 | slope>thr | 可触发")
        for row in r["diag"]:
            i, smax, tmax, emax, emed, narm, ngt, nfire = row
            print("   %6d %7.2f | %8.0f | %9.1f | %7.1f | %9.0f | %9.0f | %6d | %9d | %6d"
                  % (i, el[i], V[i].sum(), smax, tmax, emax, emed, narm, ngt, nfire))


if __name__ == "__main__":
    main()
