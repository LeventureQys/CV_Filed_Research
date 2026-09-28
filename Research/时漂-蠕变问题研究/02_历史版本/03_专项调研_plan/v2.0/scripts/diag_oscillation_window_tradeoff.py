# -*- coding: utf-8 -*-
"""只读实验（四）：把"检测器近窗拉长"当作优化候选，量化代价。

动机：振荡工况下检测器 13.8% 的帧命中（振荡段 65.5%），把抖动当负载变化；
      若把 `kDetFastS` 由 0.20 s 拉长到 0.35/0.50 s，窗口就能覆盖整周期（4.2 Hz→0.24 s），
      抖动应被吃掉。**代价**是真实加载沿的检出延迟变大 ⇒ 用交付默认算法的输出测"响应变慢多少"。

对照算法（只改 TAU_FAST，其它一律默认）：
  W=0.20（现役） / 0.35 / 0.50 / 0.65，另加"包内平均"前置处理的 W=0.20
在两条数据上评：
  ① 振荡抖：该数据 69.7~71.7 s 的 显示偏移 std、以及检测器命中率
  ② 干净数据：19 条录制里选「新录制」（337 s，8 个加载沿）测
     - 台阶捕获比 G（沿后 4~5 s 显示 / 台阶）
     - 沿后 1 s 偏差（显示 − 输入的最终平台）
     - 受载段偏移中位、受载段滑窗极差
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v20_lib.py")
RUNNER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "build", "batch_runner.exe"))
ODD = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                   "20260919_134056_single_device_f40a1b")


def run(el, V):
    n = V.shape[1]
    lines = ["%d 100" % n]
    for i in range(len(el)):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    r = subprocess.run([RUNNER], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8")
    rows = r.stdout.splitlines()
    body = rows[1:-1]
    return (np.array([float(x.split()[1]) for x in body]),
            np.array([float(x.split()[2]) for x in body]))


def detect_hits(el, tot, fast):
    """按 v6 公式复算检测器命中率（只改近窗 fast）。"""
    n = len(el)
    d = np.zeros(n)
    for i in range(n):
        a = (el > el[i] - fast) & (el <= el[i])
        b = (el > el[i] - fast - 0.15 - 0.30) & (el <= el[i] - fast - 0.15)
        if a.any() and b.any():
            d[i] = tot[a].mean() - tot[b].mean()
    sig = np.zeros(n)
    hit = np.zeros(n, dtype=bool)
    run = 0
    for i in range(n):
        h = max(0, i - 1024)
        w = d[h:i + 1]
        if len(w) > 40:
            sig[i] = 1.4826 * np.median(np.abs(w - np.median(w)))
        thr = max(5 * sig[i], 0.05 * abs(tot[max(0, i - 1)]), 0.01 * tot[:i + 1].max())
        run = run + 1 if abs(d[i]) > thr else 0
        hit[i] = run >= 3
    return hit


def main():
    print("说明：窗口改动的**算法真实响应**要用改过的 C++ 才能测；")
    print("      本脚本先用代理量（检测器命中率）判断'能否吃掉抖动'，")
    print("      再用现役算法在干净数据上的表现给出'当前响应水平'作参照。")

    # ① 震荡数据的检测器命中率 vs 近窗
    pre = L.load_stream(ODD, "device_001_pre_seg0.csv")
    el, V = pre["el"], pre["V"]
    tot = V.sum(1)
    print("\n=== ① 振荡数据：「检测器命中率」随近窗变化 ===")
    print("%-8s %12s %14s %12s" % ("近窗", "全录命中%", "振荡段命中%", "安静段命中%"))
    for fast in (0.20, 0.35, 0.50, 0.65, 0.80):
        h = detect_hits(el, tot, fast)
        mo = (el >= 69.7) & (el < 71.7)
        mq = (el >= 108) & (el < 118)
        print("%-8.2f %11.1f%% %13.1f%% %11.1f%%"
              % (fast, 100 * h.mean(), 100 * h[mo].mean(), 100 * h[mq].mean()))

    # ② 现役算法在振荡数据上的偏移抖动（基线）
    si, so = run(el, V)
    off = so - si
    mo = (el >= 69.7) & (el < 71.7)
    print("\n=== ② 现役算法在振荡数据的响应 ===")
    print("  振荡段：输入 std=%.0f ADC → 显示偏移 std=%.0f ADC（抑制比 %.1f×）"
          % (tot[mo].std(), off[mo].std(), tot[mo].std() / max(off[mo].std(), 1)))
    print("  振荡段：偏移中位=%+.0f ADC，全程偏移极差=%.0f ADC"
          % (np.median(off[mo]), off.max() - off.min()))

    # ③ 干净数据（新录制）：现役算法的响应水平（作为窗口改动的代价参照）
    pre2 = L.load_stream(L.DS_ZERO, "device_001_pre_seg0.csv")
    el2, V2 = pre2["el"], pre2["V"]
    tot2 = V2.sum(1)
    si2, so2 = run(el2, V2)
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot2, 3), np.percentile(tot2, 97)
    ups, _ = L.edges_from_tot(np.convolve(tot2, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in ups:
        if not tu or el2[i] - tu[-1] > 1.0:
            tu.append(float(el2[i]))
    print("\n=== ③ 现役算法在干净数据（新录制）的响应水平 ===")
    print("  加载沿 %d 个：%s" % (len(tu), [round(t, 1) for t in tu]))
    print("%9s %12s %12s %12s" % ("沿(s)", "台阶A", "Δ1s(显示−输入)", "G(4-5s)"))
    gs, d1s = [], []
    for t in tu:
        m0 = (el2 >= t - 1.2) & (el2 < t - 0.35)
        if m0.sum() < 5:
            continue
        bp = float(np.median(tot2[m0]))
        te = min(t + 30.0, el2[-1] - 0.2)
        m1 = (el2 >= te - 1.5) & (el2 <= te)
        ps, os_ = float(np.median(tot2[m1])), float(np.median(so2[m1]))
        A = ps - bp
        m2 = (el2 >= t + 0.8) & (el2 <= t + 1.2)
        d1 = float(np.median(so2[m2] - tot2[m2])) if m2.any() else float("nan")
        m3 = (el2 >= t + 4.0) & (el2 <= t + 5.0)
        g = (float(np.median(so2[m3])) - float(np.median(so2[m0]))) / A if m3.any() else float("nan")
        gs.append(g)
        d1s.append(d1)
        print("%9.2f %12.0f %12.0f %12.3f" % (t, A, d1, g))
    print("  中位：Δ1s=%+.0f ADC  G=%.3f" % (np.nanmedian(d1s), np.nanmedian(gs)))


if __name__ == "__main__":
    main()
