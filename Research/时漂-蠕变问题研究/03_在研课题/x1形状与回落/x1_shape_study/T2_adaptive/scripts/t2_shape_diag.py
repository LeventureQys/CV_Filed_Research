# -*- coding: utf-8 -*-
"""T2 步骤5：形状诊断——x1 上升轨迹的可预测性。

对每个门槛内事件，用基线回放测：
  (a) 显示回落 drop 与 x1 在 [2,10]s 内涨幅的相关系数 r；
  (b) 比值 drop / x1涨幅 的会话内变异系数（低=只要控住 x1 涨幅就能控住回落）；
  (c) x1(10s)/x1(平台) 与 e 的形态特征（当前 r1·e 目标的解释力）；
  (d) τc1 取不同值时 drop 的理论响应（用单指数闭式核对）。
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

OUT = os.path.join(T.OUT_ROOT, "results")


def main():
    lines = []
    rows = []
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
            m = T.event_metrics(r["D"], tot, el, ev, xlp=xlp, ylp=ylp)
            if not m or not T.event_gate(m, ev):
                continue
            i, j = ev["i_up"], ev["i_win"]
            g = lambda arr, t0, t1: float(np.median(  # noqa: E731
                arr[(el - ev["t_up"] >= t0) & (el - ev["t_up"] <= t1) &
                    (np.arange(len(el)) >= i) & (np.arange(len(el)) < j)]))
            x1_2 = g(r["X1"], 1.5, 2.5)
            x1_10 = g(r["X1"], 9.5, 10.5)
            x1_pl = float(np.median(r["X1"][(el >= ev["t_up"] + 20) &
                                            (el <= ev["t_up"] + 45)]))
            e_2 = g(r["E"], 1.5, 2.5)
            e_10 = g(r["E"], 9.5, 10.5)
            rows.append(dict(label=label, t=ev["t_up"], step=ev["step"],
                             drop=m["drop"], drop_rel=m["drop_rel"],
                             x1_gain=x1_10 - x1_2, x1_10=x1_10, x1_pl=x1_pl,
                             e_2=e_2, e_10=e_10, err=m["seg_err"],
                             cres=x1_10 / max(e_2, 1.0),
                             cres_pl=x1_pl / max(e_10, 1.0),
                             ratio=m["drop"] / max(x1_10 - x1_2, 1.0),
                             dratio=m["drop"] / max(e_10 - e_2, 1.0)))
    lines.append("门槛内事件 %d 个" % len(rows))
    lines.append("")
    lines.append("%-44s %7s %8s %8s %8s %8s %8s %8s %7s %7s"
                 % ("会话", "t_up", "台阶", "回落", "x1涨幅2-10s", "x1@10s",
                    "x1@平台", "回落/x1涨", "x1@10/e", "保证段偏差"))
    for r in rows:
        lines.append("%-44s %7.1f %8.0f %8.1f %8.0f %8.0f %8.0f %8.3f %7.3f %7.0f"
                     % (r["label"][-44:], r["t"], r["step"], r["drop"],
                        r["x1_gain"], r["x1_10"], r["x1_pl"], r["ratio"],
                        r["cres"], r["err"]))
    drops = np.array([r["drop"] for r in rows])
    gains = np.array([r["x1_gain"] for r in rows])
    crs = np.array([r["cres"] for r in rows])
    ratios = np.array([r["ratio"] for r in rows])
    lines.append("")
    lines.append("相关系数 corr(回落, x1 涨幅)=%.3f   corr(回落, 台阶)=%.3f   "
                 "corr(回落/台阶, x1@10s/e@2s)=%.3f"
                 % (np.corrcoef(drops, gains)[0, 1],
                    np.corrcoef(drops, [r["step"] for r in rows])[0, 1],
                    np.corrcoef([r["drop_rel"] for r in rows], crs)[0, 1]))
    lines.append("回落/x1涨幅: 均值 %.3f 中位 %.3f 标准差 %.3f（变异系数 %.2f）"
                 % (ratios.mean(), np.median(ratios), ratios.std(),
                    ratios.std() / max(abs(ratios.mean()), 1e-9)))
    lines.append("x1@10s/e@2s 均值 %.3f（固定形状理论值 = r1·(1−e^{−10/8}) = %.3f）"
                 % (crs.mean(), 0.12 * (1 - np.exp(-10 / 8))))
    # 会话内变异
    lines.append("")
    lines.append("会话内（≥2 事件）回落/x1涨幅 的变异：")
    bys = {}
    for r in rows:
        bys.setdefault(r["label"], []).append(r)
    for label, rs in bys.items():
        if len(rs) < 2:
            continue
        rr = np.array([x["ratio"] for x in rs])
        lines.append("  %-52s n=%d 均值%.3f 标准差%.3f 变异系数%.2f"
                     % (label[-52:], len(rs), rr.mean(), rr.std(),
                        rr.std() / max(abs(rr.mean()), 1e-9)))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_shape_diag.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
