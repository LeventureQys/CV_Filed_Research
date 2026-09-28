# -*- coding: utf-8 -*-
"""v3.6 探针 7：会话级蠕变比 r 的时间轨迹（判断它是否自激/衰减）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
import v36_replay as R  # noqa: E402
from v36_probe_internal import R_arm  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def main():
    ds = K.load(K.DS_TARGET)
    lines = []
    for arm in (sys.argv[1:] or ["v36:--seed-gain:0.7", "v36:--seed-gain:0.7:--v36-b:1"]):
        exe, args = R_arm(arm)
        st = R.run_arm(ds, exe, args)
        t = st["t"]
        ts = np.arange(0, 312, 5.0)
        L = ["== %s：r(t) 与 ded/A 轨迹 ==" % arm,
             "%8s %10s %10s %10s %10s %10s" % ("t", "r(会话)", "ded", "A_sum", "ded/A", "state")]
        for u in ts:
            m = (t >= u) & (t < u + 5.0)
            if not m.any():
                continue
            dd = float(np.median(st["ded"][m]))
            aa = float(np.median(st["A_sum"][m]))
            L.append("%8.1f %10.4f %10.0f %10.0f %10.4f %10.0f"
                     % (u, float(np.median(st["creep_ratio"][m])), dd, aa,
                        dd / aa if abs(aa) > 1 else float("nan"),
                        float(np.median(st["state"][m]))))
        lines.append("\n".join(L))
        print("\n".join(L))
        print()
    with open(os.path.join(RES, "v36_creep_ratio.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(lines) + "\n")
    print("-> %s" % os.path.join(RES, "v36_creep_ratio.txt"))


if __name__ == "__main__":
    main()
