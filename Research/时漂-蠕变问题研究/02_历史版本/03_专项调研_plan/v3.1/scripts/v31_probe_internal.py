# -*- coding: utf-8 -*-
"""探针 B（v3.1）：273 s 卸载前后 1 s 分辨率的**内部状态**追踪（真实 C++ 本体）。

输出：results/v31_after273.txt + results/v31_trace.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3.0", "scripts")))
import v30_ab as A  # noqa: E402

WORK = os.path.join(L.DATA_ROOT, "working")
DS = os.path.join(WORK, "零基线-反复增减同一负载",
                  "20260919_152749_single_device_0cb8b6")
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def main():
    el, V = A.read_any(os.path.join(DS, "device_001_pre_seg0.csv"))
    main_tot = A.read_any(os.path.join(DS, "device_001_seg000.csv"))[1].sum(1)
    tin = V.sum(1)
    D = A.run_arm(A.RUNNERS["base"], [], el, V)   # 交付默认 = plan-v3.0 PCT
    tout = D["sum_out"]
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("录制时的参数集就是 plan-v3.0 PCT（manifest 实证），复算即当前交付态")
    p("")
    p("=== 回放校验：复算 vs 录制 ===")
    d = tout - main_tot
    p("  max|Δ| %.1f  中位|Δ| %.1f  RMS %.1f" %
      (np.abs(d).max(), np.median(np.abs(d)), float(np.sqrt((d ** 2).mean()))))
    p("")
    p("=== 1 s 分辨率（265~343 s）：输入 / 录制显示 / 复算显示 / 内部状态 ===")
    p("%7s %8s %9s %9s %8s %7s %9s %9s %8s %7s %8s" %
      ("t", "state", "输入", "录制", "复算", "offset", "A_sum", "ded1", "pct", "g", "n_load"))
    for a in np.arange(265.0, 343.0, 1.0):
        m = (el >= a) & (el < a + 1.0)
        if not m.any():
            continue
        f = lambda k: float(np.median(D[k][m]))
        p("%7.1f %8.0f %9.0f %9.0f %8.0f %7.0f %9.0f %9.0f %8.0f %7.4f %8.0f" %
          (a, f("state"), np.median(tin[m]), np.median(main_tot[m]), np.median(tout[m]),
           np.median((tout - tin)[m]), f("A_sum"), f("ded_capped"), f("pct_sum"),
           f("g"), f("n_loaded")))
    p("")
    p("=== 全程 5 s 分辨率：状态与内部量 ===")
    p("%7s %6s %9s %9s %9s %9s %9s %7s" %
      ("t", "state", "输入", "复算", "A_sum", "ded1", "pct", "g"))
    for a in np.arange(0.0, 343.0, 5.0):
        m = (el >= a) & (el < a + 5.0)
        if not m.any():
            continue
        f = lambda k: float(np.median(D[k][m]))
        p("%7.1f %6.0f %9.0f %9.0f %9.0f %9.0f %9.0f %7.4f" %
          (a, f("state"), np.median(tin[m]), np.median(tout[m]), f("A_sum"),
           f("ded_capped"), f("pct_sum"), f("g")))
    with open(os.path.join(RES, "v31_after273.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:80]))
    print("\n-> %s" % os.path.join(RES, "v31_after273.txt"))


if __name__ == "__main__":
    main()
