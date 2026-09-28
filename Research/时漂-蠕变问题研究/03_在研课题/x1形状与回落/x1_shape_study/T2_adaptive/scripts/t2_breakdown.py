# -*- coding: utf-8 -*-
"""T2 步骤8：回落成分分解——基线在 [2s,10s] 窗口里到底补了多少、缺多少。

对每个门槛内事件，量：
  Δy        输入去零点后的载荷响应 y 在 [2s,10s] 的涨幅（真实需求）
  Δx1/Δx2   观察器在同期给出的补偿增量
  Δzero     零点跟踪在同期吸收掉的部分（是"隐形补偿"）
  Δy−Δx1−Δx2  显示净变化；若 Δzero>0 说明显示被零点漂移垫高
  缺口       Δy − (Δx1+Δx2+Δzero) = 显示净变化（应与 Δ显示 一致，用于自检）
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

OUT = os.path.join(T.OUT_ROOT, "results")


def main():
    lines = []
    lines.append("%-46s %6s %8s %8s %8s %8s %8s %8s %8s"
                 % ("会话", "沿t", "台阶", "Δy2-10", "Δx1", "Δx2", "Δzero",
                    "Δ显示", "缺口"))
    tot_gap, tot_dy, tot_dx = [], [], []
    cov, exc = [], []
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el, V, tot = s["el"], s["V"], T.total(s)
        evs = T.load_events(el, tot, win=60.0)
        if not evs:
            continue
        r = O.observe(el.copy(), V, "base")
        ylp = T.lowpass(r["D"], el, 1.0)
        xlp = T.lowpass(tot, el, 5.0)
        zero = tot - r["Y"]
        for ev in evs:
            m = T.event_metrics(r["D"], tot, el, ev, xlp=xlp, ylp=ylp, x1=r["X1"])
            if not m or not T.event_gate(m, ev):
                continue
            tt = el - ev["t_up"]
            sel = (tt >= 2.0) & (tt <= 10.0)
            sel[:ev["i_up"]] = False
            sel[ev["i_win"]:] = False
            if sel.sum() < 50:
                continue

            def dl(arr):
                v = arr[sel]
                k = max(5, len(v) // 5)
                return float(np.mean(v[-k:]) - np.mean(v[:k]))

            dy, dx1, dx2 = dl(r["Y"]), dl(r["X1"]), dl(r["X2"])
            dz, dd = dl(zero), dl(r["D"])
            gap = dy - (dx1 + dx2 + dz)
            lines.append("%-46s %6.1f %8.0f %+8.0f %+8.0f %+8.0f %+8.0f %+8.0f %+8.0f"
                         % (label[-46:], ev["t_up"], ev["step"], dy, dx1, dx2,
                            dz, dd, gap - dd))
            tot_gap.append(gap)
            tot_dy.append(dy)
            tot_dx.append(dx1 + dx2 + dz)
            cov.append((dx1 + dx2 + dz) / dy if abs(dy) > 1 else np.nan)
            exc.append(m["x1_exc"])
    lines.append("")
    lines.append("宏平均：Δy=%+.0f  Δ(x1+x2+zero)=%+.0f  缺口=%+.0f  补偿覆盖 %.0f%%  "
                 "x1 超调 %+.0f ADC"
                 % (np.mean(tot_dy), np.mean(tot_dx), np.mean(tot_gap),
                    100 * np.nanmean(cov), np.nanmean(exc)))
    lines.append("  Δy            = [2s,10s] 内 y（去零点载荷响应）涨幅 = 这段时间的真实需求")
    lines.append("  Δ(x1+x2+zero) = 同期观察器给出的补偿增量 + 零点跟踪吸收量")
    lines.append("  缺口          = Δy − 补偿增量 − Δ显示（应≈0，用于自检）")
    lines.append("  x1 超调       = Δx1 − Δy：<0 表示 x1 涨得比实测蠕变慢（拖不住）、"
                 ">0 表示涨得比蠕变快（把显示往下拽）")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_breakdown.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
