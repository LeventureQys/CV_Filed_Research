# -*- coding: utf-8 -*-
"""v2.0 PROBE-D：44 s 附近的偏移明细与偏移分解（MainAgent 初判用）。

目的：把「main − pre」在关键平台上拆成两路：
  · 事件路（滑行/交接继承）：事件期内与交接后短时间内；
  · 慢相路（γ·A·g）：稳态保压段。
并用「关闭算法即 pre」这一孪生对照，给出每平台的可归因量。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402


def med(tot, el, t0, t1):
    m = (el >= t0) & (el <= t1)
    return float(np.median(tot[m])) if m.any() else float("nan")


def main():
    d = L.load_dataset(L.DS_ZERO)
    pre, main = d["pre"], d["main"]
    el = pre["el"]
    tp, tm = pre["V"].sum(1), main["V"].sum(1)
    off = tm - tp

    # ── 1. 100 ms 分辨率：41.0~48.0 s ──
    print("── 41~48 s（100 ms 桶）──")
    print(f"{'t':>7}{'pre':>8}{'main':>8}{'off':>8}{'d(pre)/dt':>11}{'d(off)/dt':>11}")
    ts = np.arange(41.0, 48.0, 0.1)
    prev_p = prev_o = None
    for t in ts:
        m = (el >= t) & (el < t + 0.1)
        if not m.any():
            continue
        p, q = float(np.mean(tp[m])), float(np.mean(tm[m]))
        dp = (p - prev_p) / 0.1 if prev_p is not None else 0.0
        do = ((q - p) - prev_o) / 0.1 if prev_o is not None else 0.0
        print(f"{t:7.2f}{p:8.0f}{q:8.0f}{q-p:8.0f}{dp:11.0f}{do:11.0f}")
        prev_p, prev_o = p, q - p

    # ── 2. 关键平台偏移（按 17 段平台）──
    print("\n── 平台级偏移与相对量 ──")
    segs = L.plateau_segments(tp, el)
    print(f"{'#':>3}{'kind':7}{'t0':>8}{'t1':>8}{'dur':>7}{'pre':>8}{'main':>8}"
          f"{'off':>8}{'off/pre':>9}{'off/amp(14552)':>15}")
    for i, (kind, a, b) in enumerate(segs):
        p = med(tp, el, el[a], el[b])
        q = med(tm, el, el[a], el[b])
        print(f"{i:>3}{kind:7}{el[a]:8.2f}{el[b]:8.2f}{el[b]-el[a]:7.1f}"
              f"{p:8.0f}{q:8.0f}{q-p:8.0f}{100*(q-p)/p:9.2f}%{100*(q-p)/14552:15.2f}%")

    # ── 3. 偏移的极性与"符号翻转"时刻 ──
    print("\n── 偏移符号段（连续同号段 > 1 s）──")
    sgn = np.sign(off)
    i0 = 0
    for i in range(1, len(el)):
        if sgn[i] != sgn[i0] and abs(off[i]) > 50:
            if el[i] - el[i0] > 1.0:
                print(f"  sign={int(sgn[i0]):+d}  t={el[i0]:7.2f}~{el[i-1]:7.2f}  "
                      f"off 中位={np.median(off[i0:i]):+8.0f}  最大|off|={np.max(np.abs(off[i0:i])):7.0f}")
            i0 = i

    # ── 4. 逐平台：pre 是否达到真实稳态（用 pre 自身 5 s 滑窗最大判断平台是否还在爬）──
    print("\n── 每平台 pre 的稳定性（该段内 pre 首末差 / 段内极差）──")
    for i, (kind, a, b) in enumerate(segs):
        if b - a < 100:
            continue
        seg = tp[a:b + 1]
        print(f"  #{i:<3d} {kind:7s} t={el[a]:7.2f}~{el[b]:7.2f}  "
              f"首={seg[:50].mean():7.0f} 末={seg[-50:].mean():7.0f} "
              f"差={seg[-50:].mean()-seg[:50].mean():+7.0f} 极差={seg.max()-seg.min():7.0f}")


if __name__ == "__main__":
    main()
