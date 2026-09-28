# -*- coding: utf-8 -*-
"""探针 A：稳定工况总览 —— 平台段切分 + 算法前后漂移对比。

输出：results/v30_overview.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

DS = os.path.join(L.OVERVIEW, "20260919_141824_single_device_110871")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "v30_overview.txt")


def main():
    ds = L.load_dataset(DS)
    el = ds["pre"]["el"]
    tot_pre, tot_main, tot_raw = L.totals(ds)
    n = len(el)
    dt = np.diff(el)
    dt = dt[dt > 0]
    lines = []
    p = lines.append
    p("数据集: %s" % DS)
    p("帧数 %d  时长 %.1f s  采样 %.2f Hz  中位 dt %.4f s" %
      (n, el[-1] - el[0], (n - 1) / (el[-1] - el[0]), float(np.median(dt))))
    p("params: %s" % ds["sess"]["algorithm"]["params"])
    p("")
    p("总量口径（21 通道求和，ADC）：")
    p("  pre   min %10.0f  max %10.0f" % (tot_pre.min(), tot_pre.max()))
    p("  algo  min %10.0f  max %10.0f" % (tot_main.min(), tot_main.max()))
    p("  raw   min %10.0f  max %10.0f" % (tot_raw.min(), tot_raw.max()))
    p("")

    segs = L.hysteresis_segments(tot_pre, min_frames=50)
    p("平台段切分（基于 pre 总量，滞回 25%%/75%%）共 %d 段：" % len(segs))
    p("%-3s %-7s %9s %9s %8s %12s %12s %12s %12s" %
      ("#", "kind", "t0", "t1", "dur", "pre_drift", "algo_drift", "pre_slope", "algo_slope"))
    rows = []
    for k, (kind, i0, i1) in enumerate(segs):
        dur = el[i1] - el[i0]
        _, _, pp_pre, d_pre = L.band_stats(el, tot_pre, i0, i1, 0.5)
        _, _, pp_al, d_al = L.band_stats(el, tot_main, i0, i1, 0.5)
        s_pre = L.slope_per_s(el, tot_pre, i0, i1)
        s_al = L.slope_per_s(el, tot_main, i0, i1)
        rows.append((k, kind, el[i0], el[i1], dur, d_pre, d_al, s_pre, s_al,
                     pp_pre, pp_al, i0, i1))
        p("%-3d %-7s %9.2f %9.2f %8.2f %12.0f %12.0f %12.1f %12.1f" %
          (k, kind, el[i0], el[i1], dur, d_pre, d_al, s_pre, s_al))
    p("")
    p("长静置段（>= 8 s）明细，按 dt=1 s 桶中位：")
    for (k, kind, t0, t1, dur, d_pre, d_al, s_pre, s_al, pp_pre, pp_al, i0, i1) in rows:
        if dur < 8.0:
            continue
        tb, mb, pp_m, df_m = L.band_stats(el, tot_main, i0, i1, 1.0)
        tp, mp, pp_p, df_p = L.band_stats(el, tot_pre, i0, i1, 1.0)
        p("  #%d %-6s t=[%.1f,%.1f] dur %.1f s" % (k, kind, t0, t1, dur))
        p("     pre  桶中位 首 %.0f 末 %.0f  极差 %.0f  首末差 %+.0f" %
          (mp[0], mp[-1], pp_p, df_p))
        p("     algo 桶中位 首 %.0f 末 %.0f  极差 %.0f  首末差 %+.0f  (占电平 %.2f%%)" %
          (mb[0], mb[-1], pp_m, df_m,
           100.0 * df_m / max(abs(float(np.mean(mp))), 1.0)))
        step = max(1, len(mb) // 14)
        p("     algo 轨迹: " + " ".join("%.0f" % v for v in mb[::step]))
        p("     pre  轨迹: " + " ".join("%.0f" % v for v in mp[::step]))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
