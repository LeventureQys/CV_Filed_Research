# -*- coding: utf-8 -*-
"""T2 步骤6：固定形状的权衡曲线粗扫（r1 × τc1 二维）。

目的：在"原形状"空间里先确定「回落 — 长保压偏差」的最优前沿，作为自适应变体的对照：
如果自适应只是落在固定参数的折中上，就没有进 C++ 的价值。
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

OUT = os.path.join(T.OUT_ROOT, "results")
R1S = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30, 0.34]
TCS = [4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0]


def main():
    sess, evc = {}, {}
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el, tot = s["el"], T.total(s)
        keep = []
        for e in T.load_events(el, tot, win=60.0):
            m = T.event_metrics(np.zeros_like(tot), tot, el, e)
            if m and T.event_gate(m, e):
                keep.append(e)
        if keep:
            evc[label] = keep
            sess[label] = dict(el=el, V=s["V"], tot=tot,
                               xlp=T.lowpass(tot, el, 5.0))
    lines = ["固定形状二维扫描（r1 × τc1），只统计门槛内事件；每格：绝对回落均 / 保证段偏差均",
             ""]
    hdr = "%-7s" % "τc1\\r1" + "".join("%22s" % ("%.2f" % r) for r in R1S)
    lines.append(hdr)
    grid = {}
    for tc in TCS:
        row = "%-7.0f" % tc
        for r1 in R1S:
            drops, errs, stds, ids, tss, tdrop = [], [], [], [], [], []
            for label, c in sess.items():
                r = O.observe(c["el"].copy(), c["V"], "base",
                              p=dict(r1=r1, tc1=tc))
                ylp = T.lowpass(r["D"], c["el"], 1.0)
                sm = T.summarize(label, r["D"], c["tot"], c["el"], evc[label],
                                 subset=True, xlp=c["xlp"], ylp=ylp)
                if sm["n_ev"]:
                    drops.append(sm["decline_mean"])
                    errs.append(sm["seg_err_mean"])
                    stds.append(sm["std_mean"])
                    ids.append(sm["idle"])
                    tss.append(sm["t_settle_mean"])
                    tdrop.append(sm["tail_mean"])
            if drops:
                grid[(tc, r1)] = (np.mean(drops), np.mean(errs),
                                  np.mean(stds), np.mean(ids), np.mean(tss),
                                  np.mean(tdrop))
                row += "%22s" % ("%6.0f /%6.0f" % (np.mean(drops),
                                                   np.mean(errs)))
            else:
                row += "%22s" % "-"
        lines.append(row)
    lines.append("")
    lines.append("%-7s %-7s %9s %10s %9s %9s %9s %9s"
                 % ("τc1", "r1", "绝对回落", "保证段偏差", "保压std", "空载偏差",
                    "稳时", "尾段爬升"))
    for tc in TCS:
        for r1 in R1S:
            g = grid.get((tc, r1))
            if g:
                lines.append("%-7.0f %-7.2f %9.1f %10.1f %9.1f %9.1f %9.1f %9.1f"
                             % (tc, r1, g[0], g[1], g[2], g[3], g[4], g[5]))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_grid.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
