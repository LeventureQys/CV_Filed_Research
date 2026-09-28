# -*- coding: utf-8 -*-
"""只读实验（续）：把"时间轴口径"与"重采样"两个因素分开。

上一实验发现：同一条输入，喂给同一份算法，只换时间轴口径，**静态偏移能差 2200 ADC**
（安静段 108~118 s：A 原始 −666 / B 包内平均 −278 / C 插值 −2488）。
本实验拆开两个因素，判断到底是哪个在起决定作用：

  A  原始逐帧          ：13160 帧，elapsed 原样
  A1 原始值+包时刻     ：13160 帧（**值不变**），但 elapsed 换成该帧所属包的组时刻
                        ⇒ 只改时间轴，不改数据
  B  包内平均          ：3988 帧（数据变了：4 帧平均），elapsed = 组时刻
  C  插值到 100 Hz     ：13094 帧（数据变了：插值），均匀 10 ms
  C1 插值到 25.6 Hz    ：约 3988 帧（数据变了：插值），与 B **同样帧率/同样时间轴**
                        ⇒ 与 B 比：同样的时间轴，只差"平均 vs 插值"
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

DS = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                  "20260919_134056_single_device_f40a1b")
RUNNER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "build", "batch_runner.exe"))


def run(el, V):
    n = V.shape[1]
    lines = ["%d 100" % n]
    for i in range(len(el)):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    r = subprocess.run([RUNNER], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8")
    rows = r.stdout.splitlines()
    body = rows[1:-1]
    so = np.array([float(x.split()[2]) for x in body])
    si = np.array([float(x.split()[1]) for x in body])
    cnt = dict(zip(["shape_hits", "n_valley_exit", "n_reanchor_idle", "n_g_floor",
                    "n_g_valley_reset", "n_clamp"], [float(x) for x in rows[-1].split()[1:]]))
    return si, so, cnt


def main():
    pre = L.load_stream(DS, "device_001_pre_seg0.csv")
    el, ts, V = pre["el"], pre["ts"], pre["V"]
    n = len(el)

    keys = np.round(el, 3)
    bounds = np.flatnonzero(np.concatenate([[True], np.diff(keys) != 0]))
    ends = np.append(bounds[1:], n)
    elB = el[bounds]
    VB = np.vstack([V[a:b].mean(0) for a, b in zip(bounds, ends)])
    # A1：逐帧值不变，时间轴换成包时刻
    elA1 = np.empty(n)
    for a, b in zip(bounds, ends):
        elA1[a:b] = el[a]

    # C1：插值到与 B 相同的 25.6 Hz 网格
    keep = np.concatenate([[True], np.diff(ts) > 0])
    t, Vs = ts[keep], V[keep]
    g = np.arange(t[0], t[-1], float(np.median(np.diff(elB))))
    VC1 = np.empty((len(g), V.shape[1]))
    for k in range(V.shape[1]):
        VC1[:, k] = np.interp(g, t, Vs[:, k])
    elC1 = g - g[0]

    arms = [("A  原始逐帧", el, V), ("A1 原始值+包时刻", elA1, V),
            ("B  包内平均", elB, VB), ("C1 插值@25.6Hz", elC1, VC1)]
    res = {}
    print("=== 内部量 ===")
    print("%-18s %8s %10s %10s %10s %10s" %
          ("口径", "帧数", "shape_hits", "n_clamp", "g_floor", "valley_rst"))
    for tag, e, X in arms:
        si, so, c = run(e, X)
        res[tag] = (e, si, so, c)
        print("%-18s %8d %10d %10d %10d %10d" %
              (tag, len(e), int(c["shape_hits"]), int(c["n_clamp"]), int(c["n_g_floor"]),
               int(c["n_g_valley_reset"])))

    print("\n=== 显示偏移 (显示−输入) 按区间比较 ===")
    print("%-16s %14s %16s %14s %14s" %
          ("区间", "A 原始", "A1 值不变改时刻", "B 包内平均", "C1 插值"))
    for a, b in ((5, 8), (27, 36), (55, 58), (69.7, 71.7), (72, 74),
                 (79.6, 89.6), (108, 118), (120, 125)):
        row = ["%g~%gs" % (a, b)]
        for tag, e, X in arms:
            _e, si, so, _c = res[tag]
            m = (_e >= a) & (_e < b)
            row.append("%14.0f" % np.median(so[m] - si[m]) if m.any() else "             -")
        print("%-16s %14s %16s %14s %14s" % tuple(row))

    print("\n=== 关键判据 ===")
    eA, siA, soA, _ = res["A  原始逐帧"]
    eA1, siA1, soA1, _ = res["A1 原始值+包时刻"]
    offA, offA1 = soA - siA, soA1 - siA1
    print("① 只改时间轴（A1 vs A）：偏移中位|差|=%.0f  RMS=%.0f  max|差|=%.0f ADC"
          % (np.median(np.abs(offA1 - offA)), np.sqrt(np.mean((offA1 - offA) ** 2)),
             np.abs(offA1 - offA).max()))
    eB, siB, soB, _ = res["B  包内平均"]
    eC1, siC1, soC1, _ = res["C1 插值@25.6Hz"]
    m = (eB >= 108) & (eB < 118)
    mc1 = (eC1 >= 108) & (eC1 < 118)
    print("② 同时刻同帧率、只差平均 vs 插值（B vs C1）在安静段 108~118s："
          "B 偏移=%.0f  C1 偏移=%.0f  ⇒ 差 %.0f ADC"
          % (np.median(soB[m] - siB[m]), np.median(soC1[mc1] - siC1[mc1]),
             abs(np.median(soB[m] - siB[m]) - np.median(soC1[mc1] - siC1[mc1]))))


if __name__ == "__main__":
    main()
