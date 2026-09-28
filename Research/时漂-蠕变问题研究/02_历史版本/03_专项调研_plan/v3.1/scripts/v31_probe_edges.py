# -*- coding: utf-8 -*-
"""探针 C（v3.1）：逐加载沿的**台阶保真**与平台电平（录制口径，不依赖回放）。

对每个沿：Δ 取「沿后 [3,5] s 中位 − 沿前 [−2.5,−0.5] s 中位」，G = Δ显示/Δ输入。
再列出每个下行沿之后的偏移轨迹，看「同一负载的台阶幅度/落点是否随时间漂移」。

输出：results/v31_edges.txt + results/v31_edges.csv
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
    V = np.array([[float(x) for x in f[3:24]] for f in data])
    return el, V


def wmed(el, x, t0, t1):
    """时间窗 [t0,t1) 内的中位。"""
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


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


def find_edges(el, tin, win_s=1.2, thr=900.0):
    """按「1.2 s 内的变化量」找沿；同一次跳变只报一个（取 |Δ| 最大处）。"""
    sm = movavg(tin, 21)
    d = np.array([wmed(el, sm, t, t + win_s) - wmed(el, sm, t - 0.4, t) for t in el])
    d = np.nan_to_num(d)
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
    el, pre = read(os.path.join(DS, "device_001_pre_seg0.csv"))
    _, main = read(os.path.join(DS, "device_001_seg000.csv"))
    tin, tout = pre.sum(1), main.sum(1)
    lines = []
    p = lines.append

    edges = find_edges(el, tin)
    p("数据集 %s（录制口径；pre=算法输入, main=算法结果）" % DS)
    p("帧 %d 时长 %.1f s 通道 %d 空载 %.0f" %
      (len(el), el[-1] - el[0], pre.shape[1], float(np.percentile(tin, 2))))
    p("沿数 %d（判据：1.2 s 内变化 > 900 ADC）" % len(edges))
    p("")
    p("=== 逐沿台阶保真（Δ 取沿后 [3,5] s − 沿前 [−2.5,−0.5] s）===")
    p("%8s %4s %9s %9s %7s %9s %9s %9s" %
      ("t", "向", "Δ输入", "Δ显示", "G", "沿前in", "沿后in", "沿后off"))
    rows = []
    for (t, dirn, i) in edges:
        di = wmed(el, tin, t + 3, t + 5) - wmed(el, tin, t - 2.5, t - 0.5)
        do = wmed(el, tout, t + 3, t + 5) - wmed(el, tout, t - 2.5, t - 0.5)
        g = do / di if abs(di) > 1 else float("nan")
        rows.append((t, dirn, di, do, g))
        p("%8.2f %4s %9.0f %9.0f %7.3f %9.0f %9.0f %9.0f" %
          (t, "↑" if dirn > 0 else "↓", di, do, g,
           wmed(el, tin, t - 2.5, t - 0.5), wmed(el, tin, t + 3, t + 5),
           wmed(el, tout - tin, t + 3, t + 5)))
    p("")
    big = [r for r in rows if abs(r[2]) > 1200 and np.isfinite(r[4])]
    up = [r for r in big if r[1] > 0]
    dn = [r for r in big if r[1] < 0]
    for lbl, grp in (("上行", up), ("下行", dn)):
        if grp:
            gs = [r[4] for r in grp]
            p("%s %d 个（|Δin|>1200）：G 中位 %.3f  范围 %.3f~%.3f" %
              (lbl, len(grp), float(np.median(gs)), min(gs), max(gs)))
    p("")
    p("=== 每个下行沿之后：显示−输入 的偏移轨迹（看落点是否一致）===")
    p("%8s %9s %9s %9s %9s %9s" % ("沿t", "+5s", "+10s", "+20s", "+40s", "+60s"))
    for (t, dirn, i) in edges:
        if dirn > 0:
            continue
        vals = [wmed(el, tout - tin, t + k, t + k + 2.0) for k in (5, 10, 20, 40, 60)]
        p("%8.2f %9.0f %9.0f %9.0f %9.0f %9.0f" % (t, *vals))
    p("")
    p("=== 指定沿的逐通道台阶 ===")
    for t0 in (25.0, 271.5, 284.0, 299.5, 332.5):
        cand = min(edges, key=lambda e: abs(e[0] - t0))
        t = cand[0]
        p("  --- 最接近 %.1f 的沿：t=%.2f（%s）---" % (t0, t, "↑" if cand[1] > 0 else "↓"))
        dic = np.array([wmed(el, pre[:, k], t + 3, t + 5) - wmed(el, pre[:, k], t - 2.5, t - 0.5)
                        for k in range(pre.shape[1])])
        doc = np.array([wmed(el, main[:, k], t + 3, t + 5) - wmed(el, main[:, k], t - 2.5, t - 0.5)
                        for k in range(pre.shape[1])])
        p("      合计 Δin %+.0f  Δout %+.0f  G %.3f" %
          (dic.sum(), doc.sum(), doc.sum() / dic.sum() if abs(dic.sum()) > 1 else float("nan")))
        o = np.argsort(-np.abs(dic))
        for k in o[:6]:
            gi = doc[k] / dic[k] if abs(dic[k]) > 1 else float("nan")
            p("      ch%-3d Δin %+8.0f  Δout %+8.0f  G %7.3f" % (k, dic[k], doc[k], gi))
    with open(os.path.join(RES, "v31_edges.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "v31_edges.csv"), "w", encoding="utf-8") as fh:
        fh.write("t,dir,din,dout,G\n")
        for r in rows:
            fh.write("%.3f,%d,%.3f,%.3f,%.6f\n" % r)
    print("\n".join(lines))
    print("\n-> %s" % os.path.join(RES, "v31_edges.txt"))


if __name__ == "__main__":
    main()
