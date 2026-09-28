# -*- coding: utf-8 -*-
"""T2 步骤10：把输入自身的漂移扣掉后，观察器留下的"包"有多大、变体是否压平了它。

isoresidual = ylp(D) − xlp(输入)，因果一阶低通（显示 τ=1s、输入 τ=5s）。
它 = zero − x1 − x2 + (输入低通的滞后残差)，即观察器施加的补偿量（含零点）的相反数。
指标：
  hump_ir  = max[2,10]s isoresidual − median[20,45]s isoresidual
             （正 = 观察器在这段时间多补了，形成冲高后回落）
  dip_ir   = min[2,10]s − median[20,45]s（负 = 观察器补得不够/滞后，显示被输入抬起）
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

OUT = os.path.join(T.OUT_ROOT, "results")


def main(argv):
    names = argv[1:] or ["base", "A_store", "A_amp", "B_tau_k4f", "B_bi",
                         "B_bi_light", "B_bi2", "B_bi_slow", "C_lp_t3",
                         "S_r1_0.06", "S_tc1_32", "S_tc1_64", "S_r1_0.08_tc1_32"]
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
    lines = ["%-12s %10s %10s %10s %10s %10s %10s"
             % ("变体", "hump_ir均", "hump_ir峰", "dip_ir均", "dip_ir峰",
                "保证段偏差", "稳时")]
    for name in names:
        variant, p_over, desc = O.VARIANTS[name]
        hs, ds, errs, tss = [], [], [], []
        for label, c in sess.items():
            r = O.observe(c["el"].copy(), c["V"], variant, p=p_over)
            ylp = T.lowpass(r["D"], c["el"], 1.0)
            ir = ylp - c["xlp"]
            for ev in evc[label]:
                m = T.event_metrics(r["D"], c["tot"], c["el"], ev,
                                    xlp=c["xlp"], ylp=ylp, x1=r["X1"])
                if not m:
                    continue
                t = c["el"] - ev["t_up"]
                i, j = ev["i_up"], ev["i_win"]

                def sl(t0, t1):
                    k = (t >= t0) & (t <= t1)
                    k[:i] = False
                    k[j:] = False
                    return ir[k] if k.sum() else None

                w = sl(2.0, 10.0)
                pl = sl(20.0, 45.0)
                if w is None or pl is None or len(w) < 20 or len(pl) < 20:
                    continue
                base_pl = float(np.median(pl))
                hs.append(float(np.max(w) - base_pl))
                ds.append(float(np.min(w) - base_pl))
                errs.append(m["seg_err"])
                tss.append(m["t_settle"])
        if hs:
            lines.append("%-12s %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f"
                         % (name, np.mean(hs), np.max(hs), np.mean(ds),
                            np.min(ds), np.mean(errs), np.mean(tss)))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_isoresidual.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
