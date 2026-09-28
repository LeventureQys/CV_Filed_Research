# -*- coding: utf-8 -*-
"""探针 A（v3.1）：`working/零基线-反复增减同一负载` 总览 —— 平台段切分 + 算法前后对照。

输出：results/v31_overview.txt
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


def main():
    ds = L.load_dataset(DS)
    el = ds["pre"]["el"]
    pre, main, raw = ds["pre"]["V"], ds["main"]["V"], ds["raw"]["V"]
    tp, tm, tr = pre.sum(1), main.sum(1), raw.sum(1)
    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("帧 %d  时长 %.1f s  采样 %.2f Hz  通道 %d" %
      (len(el), el[-1] - el[0], (len(el) - 1) / (el[-1] - el[0]), pre.shape[1]))
    p("params: %s" % ds["sess"]["algorithm"]["params"])
    p("")
    p("总量：pre min %.0f max %.0f | main min %.0f max %.0f | raw==pre? %s" %
      (tp.min(), tp.max(), tm.min(), tm.max(), bool(np.allclose(tp, tr))))
    p("")
    segs = L.hysteresis_segments(tp, min_frames=50)
    p("平台段切分（基于 pre 总量，滞回 25%%/75%%）共 %d 段：" % len(segs))
    p("%-3s %-7s %8s %8s %7s %10s %10s %10s %10s %10s" %
      ("#", "kind", "t0", "t1", "dur", "pre首", "pre末", "algo首", "algo末", "补末"))
    for k, (kind, i0, i1) in enumerate(segs):
        p("%-3d %-7s %8.2f %8.2f %7.2f %10.0f %10.0f %10.0f %10.0f %10.0f" %
          (k, kind, el[i0], el[i1], el[i1] - el[i0],
           np.median(tp[i0:i0 + 100]), np.median(tp[max(i0, i1 - 100):i1 + 1]),
           np.median(tm[i0:i0 + 100]), np.median(tm[max(i0, i1 - 100):i1 + 1]),
           np.median((tm - tp)[max(i0, i1 - 100):i1 + 1])))
    p("")
    p("=== 5 s 桶轨迹（总量口径）===")
    p("%7s %9s %9s %9s %9s" % ("t", "pre", "main", "offset", "ded_pre(%)"))
    i = 0
    n = len(el)
    step = 5.0
    while i < n:
        j = i
        while j < n and el[j] < el[i] + step:
            j += 1
        p("%7.1f %9.0f %9.0f %9.0f" % (el[i], np.median(tp[i:j]), np.median(tm[i:j]),
                                       np.median((tm - tp)[i:j])))
        i = j
    with open(os.path.join(RES, "v31_overview.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print("...")
    print("\n-> %s" % os.path.join(RES, "v31_overview.txt"))


if __name__ == "__main__":
    main()
