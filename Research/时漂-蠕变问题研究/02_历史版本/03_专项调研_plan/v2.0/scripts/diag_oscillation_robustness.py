# -*- coding: utf-8 -*-
"""只读实验：随机震荡工况下，算法对"时间轴口径"有多敏感？

问题：该录制把 1~4 帧压在同一个 `elapsed` 上（组间 39 ms、组内 timestamp 跨度 0.01 ms），
      而输入在振荡段以 ~4.2 Hz / 9k ADC 峰峰摆动。
      ⇒ 那么"同一时刻塞 4 个不同值"会不会让算法做出错误判断？算法是不是在跟着抖动乱动？

实验：同一条输入，喂给**同一份交付算法**（batch_runner.exe），只改时间轴口径：
  A 原始逐帧      ：13160 帧，elapsed 原样（同一 elapsed 重复出现）
  B 包内平均      ：按 elapsed 分组取通道均值，3988 帧，elapsed = 组时刻
  C 插值到 100 Hz ：用 timestamp 线性插值到 100 Hz 均匀网格
比较三者的输出总量、显示偏移、以及"事件/形状库命中"等内部量。
如果三者接近 ⇒ 算法对包结构不敏感，震荡没有把它搞坏；
如果差异大 ⇒ 问题在时间轴口径，可优化点明确。
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


def run(el, V, mode=""):
    n = V.shape[1]
    lines = ["%d 100" % n]
    for i in range(len(el)):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    r = subprocess.run([RUNNER] + ([mode] if mode else []),
                       input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8")
    rows = r.stdout.splitlines()
    assert rows[0].startswith("OK "), rows[0][:120]
    body = rows[1:-1]
    so = np.array([float(x.split()[2]) for x in body])
    si = np.array([float(x.split()[1]) for x in body])
    cnt = dict(zip(["shape_hits", "n_valley_exit", "n_reanchor_idle", "n_g_floor",
                    "n_g_valley_reset", "n_clamp", "in_event", "in_slow", "g",
                    "clamp_alpha"], [float(x) for x in rows[-1].split()[1:]]))
    return si, so, cnt


def main():
    pre = L.load_stream(DS, "device_001_pre_seg0.csv")
    el, ts, V = pre["el"], pre["ts"], pre["V"]
    n = len(el)

    # A 原始
    siA, soA, cntA = run(el, V)

    # B 包内平均
    keys = np.round(el, 3)
    bounds = np.flatnonzero(np.concatenate([[True], np.diff(keys) != 0]))
    elB = el[bounds]
    VB = np.vstack([V[a:b].mean(0) for a, b in
                    zip(bounds, np.append(bounds[1:], n))])
    siB, soB, cntB = run(elB, VB)

    # C 插值到 100 Hz（按 timestamp）
    keep = np.concatenate([[True], np.diff(ts) > 0])
    t, Vs = ts[keep], V[keep]
    g = np.arange(t[0], t[-1], 0.01)
    VC = np.empty((len(g), V.shape[1]))
    for k in range(V.shape[1]):
        VC[:, k] = np.interp(g, t, Vs[:, k])
    elC = g - g[0]
    siC, soC, cntC = run(elC, VC)

    print("=== 三口径的内部量 ===")
    print("%-6s %8s %10s %10s %10s %10s %8s" %
          ("口径", "帧数", "shape_hits", "n_clamp", "g_floor", "valley_rst", "末g"))
    for tag, c, m in (("A原始", cntA, n), ("B包内平均", cntB, len(elB)),
                      ("C插值100Hz", cntC, len(elC))):
        print("%-6s %8d %10d %10d %10d %10d %8.4f" %
              (tag, m, int(c["shape_hits"]), int(c["n_clamp"]), int(c["n_g_floor"]),
               int(c["n_g_valley_reset"]), c["g"]))

    print("\n=== 输出总量比较（把 B/C 映射回 A 的帧位置后比）===")
    soB_full = np.interp(el, elB, soB)
    soC_full = np.interp(el, elC, soC)
    siB_full = np.interp(el, elB, siB)
    siC_full = np.interp(el, elC, siC)
    for tag, s in (("B", soB_full), ("C", soC_full)):
        d = s - soA
        print("  输出 so_%s − so_A: 中位=%+8.1f  中位|差|=%7.1f  RMS=%7.1f  max|差|=%7.1f ADC"
              % (tag, np.median(d), np.median(np.abs(d)),
                 np.sqrt(np.mean(d ** 2)), np.abs(d).max()))
    for tag, s in (("B", siB_full), ("C", siC_full)):
        d = s - siA
        print("  输入 si_%s − si_A: 中位=%+8.1f  中位|差|=%7.1f  RMS=%7.1f  max|差|=%7.1f ADC"
              % (tag, np.median(d), np.median(np.abs(d)),
                 np.sqrt(np.mean(d ** 2)), np.abs(d).max()))

    # 偏移（补偿量）比较：这才是"算法行为"的口径
    offA, offB, offC = soA - siA, soB_full - siB_full, soC_full - siC_full
    print("\n=== 显示偏移 (显示−输入) 比较 ===")
    print("%-16s %12s %12s %12s" % ("区间", "A 原始", "B 包内平均", "C 插值"))
    for a, b in ((5, 8), (27, 36), (55, 58), (69.7, 71.7), (72, 74), (79.6, 89.6),
                 (108, 118), (120, 125)):
        m = (el >= a) & (el < b)
        if m.any():
            print("%-16s %12.0f %12.0f %12.0f"
                  % ("%g~%gs" % (a, b), np.median(offA[m]), np.median(offB[m]),
                     np.median(offC[m])))
    print("\n偏移的极差（抖动强度）：A=%.0f  B=%.0f  C=%.0f ADC"
          % (offA.max() - offA.min(), offB.max() - offB.min(),
             offC.max() - offC.min()))

    # 三者在振荡段的逐帧一致性
    m = (el >= 69.7) & (el < 71.7)
    print("\n振荡段 69.7~71.7s：A/B/C 的偏移 std = %.0f / %.0f / %.0f ADC"
          % (offA[m].std(), offB[m].std(), offC[m].std()))
    print("振荡段输入总量 std = %.0f ADC（对比：算法输出的偏移 std 越小 = 抗抖越好）"
          % siA[m].std())


if __name__ == "__main__":
    main()
