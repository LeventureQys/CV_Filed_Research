# -*- coding: utf-8 -*-
"""探针 D（v3.1）：卸载沿的**响应形状**（0.1 s 分辨率）+ 回放偏离的时间分布。

输出：results/v31_edge_shape.txt
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


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def main():
    el, pre = A.read_any(os.path.join(DS, "device_001_pre_seg0.csv"))
    _, main = A.read_any(os.path.join(DS, "device_001_seg000.csv"))
    tin, tout = pre.sum(1), main.sum(1)
    lines = []
    p = lines.append

    p("=== ① 回放（plan-v3.0 默认）vs 录制：偏离随时间 ===")
    D = A.run_arm(A.RUNNERS["base"], [], el, pre)
    d = D["sum_out"] - tout
    p("%8s %10s %10s %10s %10s %8s %8s" %
      ("t", "输入", "录制", "复算", "Δ(复算−录制)", "state", "pct"))
    for a in np.arange(0.0, 343.0, 10.0):
        m = (el >= a) & (el < a + 10.0)
        if not m.any():
            continue
        f = lambda k: float(np.median(D[k][m]))
        p("%8.1f %10.0f %10.0f %10.0f %10.0f %8.0f %8.0f" %
          (a, np.median(tin[m]), np.median(tout[m]), np.median(D["sum_out"][m]),
           np.median(d[m]), f("state"), f("pct_sum")))
    p("")
    p("=== ② 卸载沿的响应形状（0.1 s 分辨率，取沿前 1 s 到沿后 8 s）===")
    for t0 in (24.42, 61.55, 97.63, 115.15, 272.81):
        p("--- 沿 t=%.2f ---" % t0)
        p("%8s %10s %10s %10s %10s" % ("Δt", "输入", "录制显示", "offset", "行内Δ"))
        prev_in = None
        for dt in np.arange(-1.0, 8.01, 0.5):
            t = t0 + dt
            iv = wmed(el, tin, t, t + 0.5)
            ov = wmed(el, tout, t, t + 0.5)
            p("%8.1f %10.0f %10.0f %10.0f %10s" %
              (dt, iv, ov, ov - iv,
               "" if prev_in is None else "%+.0f/%+.0f" % (iv - prev_in, ov - prev_in)))
            prev_in = iv
    p("")
    p("=== ③ 上行沿的响应形状（对照）===")
    for t0 in (108.0, 121.02, 284.21):
        p("--- 沿 t=%.2f ---" % t0)
        for dt in np.arange(-1.0, 8.01, 1.0):
            t = t0 + dt
            iv = wmed(el, tin, t, t + 0.5)
            ov = wmed(el, tout, t, t + 0.5)
            p("   Δt %5.1f  输入 %9.0f  录制 %9.0f  offset %9.0f" % (dt, iv, ov, ov - iv))
    with open(os.path.join(RES, "v31_edge_shape.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print("\n-> %s" % os.path.join(RES, "v31_edge_shape.txt"))


if __name__ == "__main__":
    main()
