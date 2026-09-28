# -*- coding: utf-8 -*-
"""探针 H（v3.1）：把**现场录制**与两个回放臂放到同一口径下比（卸载沿）。

指标（同一函数，见 v31_ab.unload_metrics）：
  卸载Gd = 卸载沿的 Δ显示/Δ输入（[3,5] s 窗，|Δin|>1500）—— 越接近 1 越好
  卸载回跳 = (显示在 [0.4,1.0] s 的抬升) − (输入同期抬升) —— 越接近 0 越好
现场 = 录制 main 流（**权威**）；两个回放臂只是候选实现的代理（回放与现场存在已知偏离）。

输出：results/v31_live_vs_replay.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3.0", "scripts")))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402
import v30_ab as A  # noqa: E402
import v31_ab as B  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
WORK = os.path.join(L.DATA_ROOT, "working")
DS = os.path.join(WORK, "零基线-反复增减同一负载",
                  "20260919_152749_single_device_0cb8b6")


def main():
    el, pre = A.read_any(os.path.join(DS, "device_001_pre_seg0.csv"))
    _, live = A.read_any(os.path.join(DS, "device_001_seg000.csv"))
    tin, tlive = pre.sum(1), live.sum(1)
    Dpre = A.run_arm(os.path.join(HERE, "build", "v31_runner_prefix.exe"), [], el, pre)
    Dfix = A.run_arm(os.path.join(HERE, "build", "v30_runner.exe"), [], el, pre)
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("现场 = 录制 main 流（权威，用的是 plan-v3.0 首版代码）")
    p("")
    p("%-16s %10s %12s %12s %10s" % ("臂", "卸载沿数", "卸载Gd中位", "回跳中位", "回跳最大"))
    for name, o in (("现场(录制)", tlive),
                    ("回放-修复前", Dpre["sum_out"]),
                    ("回放-修复后", Dfix["sum_out"])):
        gd, bm, bx, n = B.unload_metrics(el, tin, o)
        p("%-16s %10d %12.3f %12.0f %10.0f" % (name, n, gd, bm, bx))
    p("")
    p("=== 逐沿明细（卸载Gd / 回跳），现场 vs 两臂 ===")
    p("%8s %9s %9s %9s %9s %9s %9s" %
      ("沿t", "Δ输入", "现场Gd", "现场回跳", "前Gd", "后Gd", "前回跳/后回跳"))
    # 复算沿位置
    d = np.array([B.wmed(el, tin, t, t + 1.2) - B.wmed(el, tin, t - 0.4, t) for t in el])
    d = np.nan_to_num(d)
    hot = d < -900.0
    i, n = 0, len(el)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        k = i + int(np.argmin(d[i:j]))
        t = float(el[k])
        di = B.wmed(el, tin, t + 3, t + 5) - B.wmed(el, tin, t - 2.5, t - 0.5)
        if abs(di) > 1500:

            def cell(x):
                do = B.wmed(el, x, t + 3, t + 5) - B.wmed(el, x, t - 2.5, t - 0.5)
                lo = B.wmed(el, x, t, t + 0.15)
                hi = B.wmed(el, x, t + 0.4, t + 1.0)
                i_lo = B.wmed(el, tin, t, t + 0.15)
                i_hi = B.wmed(el, tin, t + 0.4, t + 1.0)
                return do / di, (hi - lo) - (i_hi - i_lo)
            g0, b0 = cell(tlive)
            g1, b1 = cell(Dpre["sum_out"])
            g2, b2 = cell(Dfix["sum_out"])
            p("%8.2f %9.0f %9.3f %9.0f %9.3f %9.3f %9.0f/%9.0f" %
              (t, di, g0, b0, g1, g2, b1, b2))
        i = j
    with open(os.path.join(RES, "v31_live_vs_replay.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % os.path.join(RES, "v31_live_vs_replay.txt"))


if __name__ == "__main__":
    main()
