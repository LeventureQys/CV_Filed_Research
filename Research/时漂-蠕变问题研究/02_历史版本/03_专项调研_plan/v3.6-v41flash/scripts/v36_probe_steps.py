# -*- coding: utf-8 -*-
"""v3.6 探针 5：完整「电平台阶表」——把输入的每一次变载找全（用于核对沿表与判断同一负载的一致性）。

做法：对输入总量做 3 帧中值 + 0.5 s 滑动中值平滑，再取相邻平台（每 0.5 s 一格的中位）之差，
|Δ| 超过门限且持续 ≥0.4 s 才记为一处台阶；合并相邻（<0.8 s）的台阶。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def plateau_levels(el, x, step=0.5):
    ts = np.arange(el[0], el[-1] + 1e-9, step)
    lv = np.array([np.median(x[(el >= u) & (el < u + step)]) if np.any((el >= u) & (el < u + step))
                   else np.nan for u in ts])
    return ts, lv


def steps(ts, lv, thr=600.0, hold=0.4, merge=0.8):
    out = []
    i = 0
    n = len(lv)
    while i < n:
        # 从 i 起找与当前平台明显不同的电平
        base = lv[i]
        j = i + 1
        while j < n and abs(lv[j] - base) < thr:
            j += 1
        if j >= n:
            break
        # 平台确认：后续 hold 秒都留在新电平附近
        k = min(n - 1, j + int(hold / (ts[1] - ts[0])))
        newlv = float(np.median(lv[j:k + 1]))
        if abs(newlv - base) < thr:
            i = j
            continue
        out.append((float(ts[j]), float(base), newlv, newlv - float(base)))
        i = k
    # 合并
    m = []
    for s in out:
        if m and s[0] - m[-1][0] < merge:
            a = m[-1]
            m[-1] = (a[0], a[1], s[2], s[2] - a[1])
        else:
            m.append(s)
    return m


def main():
    ds = K.load(K.DS_TARGET)
    el = ds["pre"]["el"]
    tin, tout = ds["tot_in"], ds["tot_out"]
    ts, li = plateau_levels(el, tin)
    _, lo = plateau_levels(el, tout)
    si = steps(ts, li)
    lines = ["== 输入电平台阶表（门限 600 ADC，合并 0.8 s）==",
             "%8s %12s %12s %10s %10s %10s" % ("t", "前电平", "后电平", "Δ输入", "前显示", "后显示")]
    for (t, a, b, d) in si:
        m1 = (ts >= t - 1.5) & (ts < t - 0.2)
        m2 = (ts >= t + 0.6) & (ts < t + 2.0)
        lines.append("%8.2f %12.0f %12.0f %10.0f %10.0f %10.0f"
                     % (t, a, b, d, np.nanmedian(lo[m1]), np.nanmedian(lo[m2])))
    ups = [s for s in si if s[3] > 0]
    dns = [s for s in si if s[3] < 0]
    lines.append("")
    lines.append("上行沿 %d 个：Δ 中位 %.0f  范围 %.0f~%.0f" %
                 (len(ups), np.median([s[3] for s in ups]),
                  min(s[3] for s in ups), max(s[3] for s in ups)))
    lines.append("上行沿（前电平 < 5000 的「从零/近零起跳」）：")
    for s in ups:
        if s[1] < 5000:
            lines.append("   t=%8.2f  Δ=%+8.0f  前=%7.0f → 后=%7.0f" % (s[0], s[3], s[1], s[2]))
    lines.append("下行沿 %d 个：Δ 中位 %.0f  范围 %.0f~%.0f" %
                 (len(dns), np.median([s[3] for s in dns]),
                  min(s[3] for s in dns), max(s[3] for s in dns)))
    txt = "\n".join(lines)
    with open(os.path.join(RES, "v36_level_steps.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)
    print("-> %s" % os.path.join(RES, "v36_level_steps.txt"))


if __name__ == "__main__":
    main()
