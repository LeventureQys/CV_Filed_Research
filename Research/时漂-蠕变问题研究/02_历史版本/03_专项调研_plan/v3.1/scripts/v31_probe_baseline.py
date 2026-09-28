# -*- coding: utf-8 -*-
"""探针 I（临时诊断）：基线丢失排查 —— 逐帧状态 + 沿定位。

针对「零基线-反复增减同一负载 / 20260919_160854_single_device_7b3977」，
把 pre（算法输入）、main（录制显示）、raw 与真实 C++ 本体复算的逐帧内部状态对齐导出。

输出：
  results/v31_baseline_<tag>.txt    摘要（沿表 + 平台表 + 关键窗口轨迹）
  results/v31_baseline_<tag>.csv    逐帧（t, tin, tmain, treplay, state, ded, A_sum, idle_now...）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3.0", "scripts")))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402
import v30_trace as TR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
WORK = os.path.join(L.DATA_ROOT, "working")
DS = os.path.join(WORK, "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")
TAG = os.path.basename(DS)
SMOOTH = 21
EDGE_THR = 900.0
EDGE_WIN = 1.2


def movavg(x, win):
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    out = np.empty(n)
    half = win // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = (c[b] - c[a]) / (b - a)
    return out


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def find_edges(el, tin, win_s=EDGE_WIN, thr=EDGE_THR):
    sm = movavg(tin, SMOOTH)
    d = np.nan_to_num(np.array(
        [wmed(el, sm, t, t + win_s) - wmed(el, sm, t - 0.4, t) for t in el]))
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
        edges.append((float(el[k]), 1 if d[k] > 0 else -1, k))
        i = j
    return edges


def main():
    ds = L.load_dataset(DS)
    el = ds["pre"]["el"]
    pre = ds["pre"]["V"]
    main_v = ds["main"]["V"]
    raw = ds["raw"]["V"]
    tin = pre.sum(1)
    tmain = main_v.sum(1)
    traw = raw.sum(1)
    tr = TR.run_stream(el, pre)
    D = tr["D"]
    tout = D["sum_out"]

    lines = []
    p = lines.append
    p("数据集 %s" % DS)
    p("帧 %d  时长 %.1f s  通道 %d" % (len(el), el[-1] - el[0], pre.shape[1]))
    p("params %s" % ds["sess"]["algorithm"]["params"])
    p("基准电平（pre 2%% 分位）%.0f" % float(np.percentile(tin, 2)))
    p("复算 vs 录制显示 main 一致性：max|Δ| %.3f  中位|Δ| %.3f  相同帧比例 %.4f" %
      (np.abs(tout - tmain).max(), np.median(np.abs(tout - tmain)),
       float(np.mean(np.abs(tout - tmain) < 1e-6))))
    p("runner meta %s" % tr["meta"])
    p("")

    tin_b = np.percentile(tin, 2)
    tmain_b = np.percentile(tmain, 2)
    traw_b = np.percentile(traw, 2)
    p("空载基线参考（2%% 分位）：pre %.0f  main %.0f  raw %.0f" % (tin_b, tmain_b, traw_b))
    p("")

    edges = find_edges(el, tin)
    p("=== 沿表（|1.2 s 变化| > %.0f ADC，共 %d 个）===" % (EDGE_THR, len(edges)))
    p("%8s %3s %9s %9s %9s %9s %9s %9s %9s" %
      ("t", "向", "tin前", "tin后", "Δin", "main前", "main后", "Δmain", "G"))
    for (t, dirn, i) in edges:
        a = wmed(el, tin, t - 2.5, t - 0.5)
        b = wmed(el, tin, t + 3, t + 5)
        c = wmed(el, tmain, t - 2.5, t - 0.5)
        d2 = wmed(el, tmain, t + 3, t + 5)
        g = (d2 - c) / (b - a) if abs(b - a) > 1 else float("nan")
        p("%8.2f %3s %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.3f" %
          (t, "UP" if dirn > 0 else "DN", a, b, b - a, c, d2, d2 - c, g))
    p("")

    p("=== 每 5 s 的 1 s 桶中位轨迹（tin/main/replay/raw）===")
    p("%8s %10s %10s %10s %10s %8s %8s %8s %8s" %
      ("t", "tin", "tmain", "treplay", "traw", "idle", "state", "ded", "A_sum"))
    t = float(el[0])
    t_end = float(el[-1])
    while t < t_end:
        a = wmed(el, tin, t, t + 1.0)
        b = wmed(el, tmain, t, t + 1.0)
        c = wmed(el, tout, t, t + 1.0)
        d2 = wmed(el, traw, t, t + 1.0)
        e = wmed(el, D["idle_now"], t, t + 1.0)
        f = wmed(el, D["state"], t, t + 1.0)
        g = wmed(el, D["comp_total"], t, t + 1.0)
        h = wmed(el, D["A_sum"], t, t + 1.0)
        p("%8.1f %10.0f %10.0f %10.0f %10.0f %8.0f %8.0f %8.0f %8.0f" %
          (t, a, b, c, d2, e, f, g, h))
        t += 5.0

    np.save(os.path.join(RES, "v31_baseline_%s.npy" % TAG), tr["X"])
    with open(os.path.join(RES, "v31_baseline_%s.cols" % TAG), "w",
              encoding="utf-8") as fh:
        fh.write(" ".join(TR.COLS) + "\n")
    with open(os.path.join(RES, "v31_baseline_%s.txt" % TAG), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    extra = dict(el=el, tin=tin, tmain=tmain, traw=traw, edges=edges, meta=tr["meta"],
                 ds=DS, params=ds["sess"]["algorithm"]["params"])
    np.savez(os.path.join(RES, "v31_baseline_%s.npz" % TAG),
             el=el, tin=tin, tmain=tmain, traw=traw, X=tr["X"],
             edges=np.array([[e[0], e[1], e[2]] for e in edges]))
    print("\n".join(lines[:80]))
    print("...")
    print("-> %s" % os.path.join(RES, "v31_baseline_%s.txt" % TAG))


if __name__ == "__main__":
    main()
