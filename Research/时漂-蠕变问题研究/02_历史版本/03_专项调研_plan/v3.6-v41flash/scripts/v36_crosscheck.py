# -*- coding: utf-8 -*-
"""v3.6 跨数据集回归核查：候选臂 vs 现役 v3.1（数据根下全部会话）。

判据：
  ① 没有「卸载→重载」的录制应当**逐位不变**（冷启动 r=0 ⇒ 无 seed）；
  ② 有「卸载→重载」的录制，补偿量（ded 中位）应上升、G 允许小幅下降；
  ③ C 限幅硬不变量对每一臂都必须 0 越界。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
import v36_replay as R  # noqa: E402
from v36_probe_internal import R_arm  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def movavg(x, w):
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    out = np.empty(n)
    h = w // 2
    for i in range(n):
        a, b = max(0, i - h), min(n, i + h + 1)
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def find_edges(el, tin, win_s=1.2, thr=900.0):
    sm = movavg(tin, 21)
    d = np.array([np.median(sm[(el >= t) & (el < t + win_s)]) -
                  np.median(sm[(el >= t - 0.4) & (el < t)]) if
                  np.any((el >= t) & (el < t + win_s)) and np.any((el >= t - 0.4) & (el < t))
                  else 0.0 for t in el])
    hot = np.abs(d) > thr
    edges, i, n = [], 0, len(el)
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        k = i + int(np.argmax(np.abs(d[i:j])))
        edges.append((float(el[k]), 1 if d[k] > 0 else -1))
        i = j
    return edges


def gmed(st, edges):
    t, tin, tout = st["t"], st["sum_in"], st["sum_out"]
    gs = []
    for (te, dirn) in edges:
        m1 = (t >= te - 2.5) & (t < te - 0.5)
        m2 = (t >= te + 3.0) & (t < te + 5.0)
        if not m1.any() or not m2.any():
            continue
        di = np.median(tin[m2]) - np.median(tin[m1])
        do = np.median(tout[m2]) - np.median(tout[m1])
        if abs(di) > 1200:
            gs.append(do / di)
    return float(np.median(gs)) if gs else float("nan")


def main():
    arms = sys.argv[1:] or ["v36:--seed-gain:0.7", "v36:--seed-gain:0.7:--v36-b:1:--v36-d:400"]
    sessions = K.L.discover_sessions()
    lines = []
    for a in arms:
        exe, args = R_arm(a)
        L = ["== 跨数据集核查：臂 %s (args=%s) ==" % (a, " ".join(args)),
             "%-58s %7s %7s %8s %10s %8s %8s %6s" %
             ("数据集", "帧数", "差异帧", "max|Δ|", "ded中位(31)", "ded(候选)", "G(31)", "G(候选)")]
        for lab, d in sessions:
            try:
                ds = K.load(d)
            except Exception as exc:  # noqa: BLE001
                L.append("%-58s  载入失败: %s" % (lab[:58], exc))
                continue
            try:
                s1 = R.run_arm(ds, R.EXE_V31, [])
                s2 = R.run_arm(ds, exe, args)
            except Exception as exc:  # noqa: BLE001
                L.append("%-58s  回放失败: %s" % (lab[:58], exc))
                continue
            n = min(len(s1["sum_out"]), len(s2["sum_out"]))
            dif = np.abs(s1["sum_out"][:n] - s2["sum_out"][:n])
            edges = find_edges(ds["pre"]["el"], ds["tot_in"])
            viol = float(np.nanmax(np.abs(s2["max_clamp_viol"])))
            L.append("%-58s %7d %7d %8.0f %10.0f %8.0f %8.3f %7.3f%s"
                     % (lab[:58], n, int((dif > 1e-6).sum()), dif.max(),
                        float(np.median(s1["ded"])), float(np.median(s2["ded"])),
                        gmed(s1, edges), gmed(s2, edges),
                        "  限幅越界=%.3g" % viol if viol > 1e-6 else ""))
        lines.append("\n".join(L))
        print("\n".join(L))
        print()
    with open(os.path.join(RES, "v36_crosscheck.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(lines) + "\n")
    print("-> %s" % os.path.join(RES, "v36_crosscheck.txt"))


if __name__ == "__main__":
    main()
