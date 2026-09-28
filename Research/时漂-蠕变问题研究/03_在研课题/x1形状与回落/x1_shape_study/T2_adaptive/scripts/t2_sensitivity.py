# -*- coding: utf-8 -*-
"""T2 步骤7：变体新参数 ±50% 敏感性。

对每个变体的每个新参数分别做 ×0.5 与 ×1.5 扰动（其余保持标称），
统计干净保压事件宏平均的回落/保证段偏差/保压std 变化，确认最优不是尖点。
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

OUT = os.path.join(T.OUT_ROOT, "results")

NEWPARAMS = {
    "A_amp": ["tau_r1", "r1_c", "tau_res", "k_acc", "r1_max"],
    "A_store": ["tau_r1", "r1_c", "r1_max"],
    "B_tau_k2": ["k_tau", "k_ref"],
    "B_tau_k4": ["k_tau", "k_ref"],
    "B_bi": ["r1a", "tc1a", "r1b", "tc1b"],
    "B_bi_light": ["r1a", "tc1a", "r1b", "tc1b"],
    "B_bi2": ["r1a", "tc1a", "r1b", "tc1b", "tau_lp_x"],
    "C_lp_t3": ["tau_lp"],
    "C_lp_t6": ["tau_lp"],
}


def metrics(variant, p_over, sess, evc):
    drops, errs, stds, ios = [], [], [], []
    for label, c in sess.items():
        r = O.observe(c["el"].copy(), c["V"], variant, p=p_over)
        ylp = T.lowpass(r["D"], c["el"], 1.0)
        sm = T.summarize(label, r["D"], c["tot"], c["el"], evc[label],
                         subset=True, xlp=c["xlp"], ylp=ylp, x1=r["X1"])
        if sm["n_ev"]:
            drops.append(sm["drop_mean"])
            errs.append(sm["seg_err_mean"])
            stds.append(sm["std_mean"])
            ios.append(sm["idle"])
    return (float(np.mean(drops)), float(np.mean(errs)), float(np.mean(stds)),
            float(np.mean(ios)))


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
    base = metrics("base", {}, sess, evc)
    lines = ["敏感性（干净保压事件宏平均；基准=基线 v3.4 r1=0.12/τc1=8）",
             "基准: 回落 %.1f  保证段偏差 %.1f  保压std %.1f  空载偏差 %.1f"
             % base, ""]
    lines.append("%-12s %-10s %6s %10s %10s %10s %10s %10s"
                 % ("变体", "参数", "倍数", "回落均", "Δ回落", "保证段偏差",
                    "Δ保证段", "保压std"))
    for name, params in NEWPARAMS.items():
        variant, p_over, desc = O.VARIANTS[name]
        nom = metrics(variant, p_over, sess, evc)
        lines.append("%-12s %-10s %6s %10.1f %10s %10.1f %10s %10.1f"
                     % (name, "(标称)", "1.0", nom[0], "-", nom[1], "-", nom[2]))
        for pname in params:
            for f in (0.5, 1.5):
                pp = dict(p_over)
                pp[pname] = p_over[pname] * f
                if pname in ("r1_c", "r1_max", "r1_min") or pname.startswith("r1"):
                    pp[pname] = min(pp[pname], 0.6)
                m = metrics(variant, pp, sess, evc)
                lines.append("%-12s %-10s %6.1f %10.1f %+10.1f %10.1f %+10.1f %10.1f"
                             % (name, pname, f, m[0], m[0] - nom[0], m[1],
                                m[1] - nom[1], m[2]))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "t2_sensitivity.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
