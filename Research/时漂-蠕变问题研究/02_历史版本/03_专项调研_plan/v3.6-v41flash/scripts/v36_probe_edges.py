# -*- coding: utf-8 -*-
"""v3.6 探针 2：把目标录制切成「逐次加载事件」，量每次事件的**台阶**与**补偿量**，
给出「一致性」这一口径的主表（后续所有候选改法都用这张表对比）。

沿检测：输入总量的 1.2 s 变化量 > thr 即记为沿；同一次跳变只报 |Δ| 最大处
（沿用 plan/v3.1 `v31_probe_edges.find_edges` 的口径，保证与该桶可比）。

每次加载事件量：
  Δin    = 沿后 [2,4] s 中位 − 沿前 [−2.5,−0.5] s 中位
  段内落定窗 [t+8, t+18] s（若事件提前结束则截断到下一沿前 0.5 s）
  ded    = 该窗内 (Σ输入 − Σ显示) 的中位
  frac   = ded / Δin            ← 「一致性」判据：各次加载应当接近
  out_in = 该窗内 显示/输入     ← 同样是一致性判据
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


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


def find_edges(el, tin, win_s=1.2, thr=900.0):
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


def event_rows(ds, edges, thr_step=900.0):
    el = ds["pre"]["el"]
    tin, tout, ded = ds["tot_in"], ds["tot_out"], ds["ded"]
    ups = [e for e in edges if e[1] > 0]
    rows = []
    for n_, (t, _, i) in enumerate(ups):
        nxt = None
        for e in edges:
            if e[0] > t + 1.0:
                nxt = e[0]
                break
        t_hi_end = (nxt - 0.5) if nxt is not None else (el[-1])
        din = wmed(el, tin, t + 2.0, t + 4.0) - wmed(el, tin, t - 2.5, t - 0.5)
        if abs(din) < thr_step:
            continue
        w0, w1 = t + 8.0, min(t + 18.0, t_hi_end)
        if w1 - w0 < 2.0:
            w0, w1 = t + 4.0, min(t + 8.0, t_hi_end)
        m = (el >= w0) & (el < w1)
        if not m.any():
            continue
        rows.append(dict(
            k=n_ + 1, t=t, din=din,
            in_lvl=float(np.median(tin[m])), out_lvl=float(np.median(tout[m])),
            ded=float(np.median(ded[m])),
            frac=float(np.median(ded[m])) / din if abs(din) > 1 else float("nan"),
            ratio=float(np.median(tout[m])) / float(np.median(tin[m]))
            if abs(np.median(tin[m])) > 1 else float("nan"),
            w0=w0, w1=w1, prev_edge=nxt))
    return rows


def dump(rows, path, title):
    lines = [title]
    lines.append("%4s %9s %10s %10s %10s %9s %9s   %s" %
                 ("#", "t_on", "Δin", "in_lvl", "out_lvl", "ded", "ded/Δin", "窗(s)"))
    for r in rows:
        lines.append("%4d %9.2f %10.0f %10.0f %10.0f %9.0f %9.3f   [%.1f,%.1f]" %
                     (r["k"], r["t"], r["din"], r["in_lvl"], r["out_lvl"],
                      r["ded"], r["frac"], r["w0"], r["w1"]))
    fr = [r["frac"] for r in rows]
    dd = [r["ded"] for r in rows]
    if len(fr) > 1:
        lines.append("")
        lines.append("ded/Δin  中位 %.3f  极差 %.3f  相对散布(极差/|中位|) %.2f"
                     % (np.median(fr), max(fr) - min(fr),
                        (max(fr) - min(fr)) / abs(np.median(fr)) if abs(np.median(fr)) > 1e-9 else float("nan")))
        lines.append("ded      中位 %.0f  极差 %.0f  min %.0f  max %.0f"
                     % (np.median(dd), max(dd) - min(dd), min(dd), max(dd)))
    txt = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)
    print("-> %s\n" % path)
    return txt


def main():
    ds = K.load(K.DS_TARGET)
    el = ds["pre"]["el"]
    edges = find_edges(el, ds["tot_in"])
    print("沿数 %d" % len(edges))
    rows = event_rows(ds, edges)
    dump(rows, os.path.join(RES, "v36_event_table_live.txt"),
         "== 目标录制（现场显示）· 逐加载事件：台阶与补偿量 ==")


if __name__ == "__main__":
    main()
