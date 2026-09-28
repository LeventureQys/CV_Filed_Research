# -*- coding: utf-8 -*-
"""v3.6 探针 4：回放臂的**内部状态**按时间网格打印（看图替代）。

用法：
  python v36_probe_internal.py <臂> <t0> <t1> <step> [列名...]
  臂 ∈ v31 / v32 / v36[:参数...]
例：
  python v36_probe_internal.py v36 244 276 1.0 state ev_valid ev_kind g A_sum pct_sum ded
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
import v36_replay as R  # noqa: E402

DEFAULT_COLS = ["state", "ev_valid", "ev_kind", "tau", "g", "A_sum", "pct_sum",
                "creep_ratio", "seed_used", "ded", "sum_in", "sum_out"]

V34_EXE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                       "v3.4", "scripts", "build", "v34_runner.exe"))


def R_arm(name):
    if name == "v31":
        return R.EXE_V31, []
    if name.startswith("v32"):
        p = name.split(":")
        return R.EXE_V32, (["--seed-gain", p[1]] if len(p) > 1 else ["--seed-gain", "0.7"])
    if name.startswith("v34"):
        # plan v3.4（= 2026-09-19 已落地产品源码的 creep-mem）。--mem 0 = 回到 v3.1 行为
        p = name.split(":")
        return V34_EXE, (["--mem", p[1]] if len(p) > 1 else ["--mem", "120"])
    p = name.split(":")
    return R.EXE_V36, p[1:]


def grid(st, cols, t0, t1, step, ds=None):
    t = st["t"]
    ts = np.arange(t0, t1 + 1e-9, step)
    lines = ["%8s" % "t" + "".join("%12s" % c for c in cols)]
    for u in ts:
        m = (t >= u) & (t < u + step)
        if not m.any():
            continue
        cells = []
        for c in cols:
            cells.append("%12.1f" % float(np.median(st[c][m])))
        lines.append("%8.2f" % u + "".join(cells))
    return "\n".join(lines)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "v36"
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 244.0
    t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 276.0
    step = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    cols = sys.argv[5:] or DEFAULT_COLS
    ds = K.load(K.DS_TARGET)
    exe, args = R_arm(arm)
    st = R.run_arm(ds, exe, args)
    print("臂 %s  参数 %s" % (arm, args))
    print(grid(st, [c for c in cols if c in st], t0, t1, step, ds))


if __name__ == "__main__":
    main()
