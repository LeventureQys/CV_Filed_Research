# -*- coding: utf-8 -*-
"""v3.6 探针 3：沿后「形状」对齐比较 —— 用户观察「245 s 之后本该长得像 15 s 之后」。

对每次加载沿取 [t−1, t+40] s 的输入/显示，减掉沿前基线并除以本段落定电平，
输出归一化形状表（每 0.5 s 采一点），用于判断：
  A. 输入本身各次加载的**快相爬升形状**是否一致（若一致，形状先验可用）；
  B. 显示（补偿后）的形状在哪几次一致、哪几次掉了。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
from v36_probe_edges import find_edges, wmed  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def shape_at(el, x, t, t0, t1, step_s=0.5, base=None):
    ts = np.arange(t + t0, t + t1 + 1e-9, step_s)
    return ts, np.array([wmed(el, x, u, u + step_s) for u in ts])


def main():
    ds = K.load(K.DS_TARGET)
    el = ds["pre"]["el"]
    tin, tout = ds["tot_in"], ds["tot_out"]
    edges = find_edges(el, tin)
    ups = [e for e in edges if e[1] > 0]
    lines = []
    p = lines.append
    p("== 目标录制：逐加载沿的归一化形状（形状 = (x(t) − 沿前基线) / 本段落定增量）==")
    p("%8s %10s %10s %10s   %s" % ("t_on", "基线", "落定增量", "40s末增量", "备注"))
    rows = []
    for (t, _, _) in ups:
        b = wmed(el, tin, t - 2.5, t - 0.5)
        # 落定增量：沿后 [30,40] s（若被下一沿截断则取到下一沿前 1 s）
        nxt = min([e[0] for e in edges if e[0] > t + 1.0] + [el[-1]])
        hi_t = min(t + 40.0, nxt - 1.0)
        if hi_t < t + 6.0:
            continue
        a = wmed(el, tin, max(t + 6.0, hi_t - 6.0), hi_t) - b
        rows.append((t, b, a, nxt, hi_t))
        p("%8.2f %10.0f %10.0f %10s   %s" % (t, b, a, "-", "窗到 %.1f s" % hi_t))
    p("")
    p("== 归一化输入形状（相对本段落定增量）==")
    grid = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0]
    p("%8s" % "t_on" + "".join("%8.2f" % g for g in grid))
    for (t, b, a, nxt, hi_t) in rows:
        vals = []
        for g in grid:
            if t + g > hi_t - 0.5:
                vals.append(float("nan"))
                continue
            v = wmed(el, tin, t + g, t + g + 0.5) - b
            vals.append(v / a if abs(a) > 1 else float("nan"))
        p("%8.2f" % t + "".join("%8.3f" % v for v in vals))
    p("")
    p("== 归一化显示形状（相对本段落定增量）==")
    p("%8s" % "t_on" + "".join("%8.2f" % g for g in grid))
    for (t, b, a, nxt, hi_t) in rows:
        vals = []
        for g in grid:
            if t + g > hi_t - 0.5:
                vals.append(float("nan"))
                continue
            v = wmed(el, tout, t + g, t + g + 0.5) - b
            vals.append(v / a if abs(a) > 1 else float("nan"))
        p("%8.2f" % t + "".join("%8.3f" % v for v in vals))
    p("")
    p("== 逐沿「补偿量 ded 的时间轨迹」（段内，相对沿）==")
    gg = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0]
    p("%8s" % "t_on" + "".join("%8.1f" % g for g in gg))
    for (t, b, a, nxt, hi_t) in rows:
        vals = []
        for g in gg:
            if t + g > hi_t - 0.5:
                vals.append(float("nan"))
                continue
            vals.append(wmed(el, ds["ded"], t + g, t + g + 0.5))
        p("%8.2f" % t + "".join("%8.0f" % v for v in vals))
    txt = "\n".join(lines)
    with open(os.path.join(RES, "v36_shape_align.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)
    print("-> %s" % os.path.join(RES, "v36_shape_align.txt"))


if __name__ == "__main__":
    main()
