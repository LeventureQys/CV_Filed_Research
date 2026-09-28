# -*- coding: utf-8 -*-
"""探针 G（v3.1）：录制的 main 流是否与算法自身的硬不变量/口径自洽。

① C 限幅不变量：main_i ≤ pre_i·1.005（plan-v2.0 起是交付算法的硬约束）
② main ≤ pre（扣除非负）
③ elapsed 是否连续（有无掉帧 → 现场算法看到的 dt 与本文件不同）
④ main−pre 的分布（扣除量）

输出：results/v31_consistency.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

WORK = os.path.join(L.DATA_ROOT, "working")
DS = os.path.join(WORK, "零基线-反复增减同一负载",
                  "20260919_152749_single_device_0cb8b6")
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def read(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    data = [r.split(",") for r in rows[di + 2:] if r.strip()]
    el = np.array([float(f[1]) for f in data])
    ts = np.array([float(f[0]) for f in data])
    fr = np.array([int(f[2]) for f in data])
    V = np.array([[float(x) for x in f[3:24]] for f in data])
    return el, V, ts, fr


def main():
    el, pre, ts, fr = read(os.path.join(DS, "device_001_pre_seg0.csv"))
    _, main, _, _ = read(os.path.join(DS, "device_001_seg000.csv"))
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("帧 %d  frame_index %d..%d 连续? %s" %
      (len(el), fr[0], fr[-1], bool(np.all(np.diff(fr) == 1))))
    p("elapsed 跨度 %.4f s  唯一 elapsed %d 个（每包 %.2f 帧）" %
      (el[-1] - el[0], len(np.unique(el)), len(el) / len(np.unique(el))))
    d = np.diff(el)
    p("elapsed 间隔：中位 %.4f s  p99 %.4f  max %.4f  零间隔占比 %.3f" %
      (np.median(d), np.percentile(d, 99), d.max(), float(np.mean(d == 0))))
    big = np.nonzero(d > 0.05)[0]
    p("间隔 > 50 ms 的处数 %d（若有大间隔 ⇒ 现场算法看到的帧与本文件不同）" % len(big))
    if len(big):
        for i in big[:8]:
            p("    t=%.3f 间隔 %.4f s" % (el[i], d[i]))
    if len(big) > 8:
        p("    ...（共 %d 处）" % len(big))
    p("")
    p("=== ① C 限幅不变量 main ≤ pre·1.005 ===")
    lim = pre * 1.005
    viol = main - lim
    p("  越界格点数（>0.01）: %d / %d" % (int(np.sum(viol > 0.01)), viol.size))
    p("  最大越界量 %.3f ADC（0.005 以内可视为 CSV 精度）" % float(viol.max()))
    p("")
    p("=== ② main ≤ pre（扣除非负）===")
    d2 = main - pre
    p("  main > pre 的格点占比 %.4f%%   最大超出 %.1f ADC" %
      (100.0 * float(np.mean(d2 > 0.01)), float(d2.max())))
    p("")
    p("=== ③ 扣除量（pre−main）总量分布 ===")
    tded = (pre - main).sum(1)
    p("  中位 %.0f  p5 %.0f  p95 %.0f  min %.0f  max %.0f" %
      (np.median(tded), np.percentile(tded, 5), np.percentile(tded, 95),
       tded.min(), tded.max()))
    p("")
    p("=== ④ 逐通道：扣除量随时间（每 40 s 中位，21 通道求和为总量）===")
    p("%7s %s" % ("t", " ".join("ch%-4d" % k for k in range(0, 21, 2))))
    for a in np.arange(20, 343, 40.0):
        m = (el >= a) & (el < a + 40)
        if not m.any():
            continue
        vals = [np.median((pre - main)[m, k]) for k in range(0, 21, 2)]
        p("%7.1f %s" % (a, " ".join("%-6.0f" % v for v in vals)))
    with open(os.path.join(RES, "v31_consistency.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % os.path.join(RES, "v31_consistency.txt"))


if __name__ == "__main__":
    main()
