# -*- coding: utf-8 -*-
"""T2 步骤9：每次加载事件应有的 x1 幅度（由构造关系反解）与固定 r1 的差异。

构造关系（单指数快态、τc1=8）：x1(t) = R·e_now·(1−e^{−t/8})，故
  x1(10s) ≈ 0.713·R·s，s = 台阶（早期 e_now≈s），
  显示回落 ≈ x1(10s) − x1(平台) ≈ 0.25·R·s（实测斜率）。
因此由实测回落反解 R_need = 回落 /（0.25·台阶），与固定 r1=0.12 比较：
若 R_need 在事件间差 3~10 倍，就说明"固定幅度"本身是形状失配的根源，
自适应幅度有明确目标量。
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

OUT = os.path.join(T.OUT_ROOT, "results")


def main():
    lines = []
    lines.append("%-44s %7s %8s %9s %9s %10s %8s"
                 % ("会话", "沿t", "台阶", "回落", "R_need", "R_need/r1", "x1超调"))
    rn, x1_at10_ratio = [], []
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el, tot = s["el"], T.total(s)
        evs = T.load_events(el, tot, win=60.0)
        if not evs:
            continue
        r = O.observe(el.copy(), s["V"], "base")
        ylp = T.lowpass(r["D"], el, 1.0)
        xlp = T.lowpass(tot, el, 5.0)
        for ev in evs:
            m = T.event_metrics(r["D"], tot, el, ev, xlp=xlp, ylp=ylp, x1=r["X1"])
            if not m or not T.event_gate(m, ev):
                continue
            need = m["drop"] / (0.25 * abs(ev["step"])) if ev["step"] else np.nan
            rn.append(need)
            x1_at10_ratio.append(m["dx1_in"] / 0.713 / abs(ev["step"])
                                 if ev["step"] else np.nan)
            lines.append("%-44s %7.1f %8.0f %9.1f %9.3f %10.2f %8.0f"
                         % (label[-44:], ev["t_up"], ev["step"], m["drop"],
                            need, need / 0.12, m["x1_exc"]))
    lines.append("")
    lines.append("R_need（由实测回落反解的快态幅度比）：均值 %.3f 中位 %.3f "
                 "范围 %.3f~%.3f（max/min = %.1f 倍）"
                 % (np.nanmean(rn), np.nanmedian(rn), np.nanmin(rn),
                    np.nanmax(rn), np.nanmax(rn) / max(np.nanmin(rn), 1e-6)))
    lines.append("独立交叉核对：Δx1[2,10s] / 0.713 / 台阶 = 实测等效幅度比："
                 "均值 %.3f 中位 %.3f"
                 % (np.nanmean(x1_at10_ratio), np.nanmedian(x1_at10_ratio)))
    lines.append("（固定参数给的是 0.12；以上两个估计都以「单指数 τc1=8、R·s 为目标」"
                 "为前提）")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_rneed.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
