# -*- coding: utf-8 -*-
"""t1a_08_minhold：D1-ev 的"最小可用窗 `min_hold`"敏感性 —— restep 工况的可测性边界。

问题：在多次变载的实录里，事件之后常常很快又发生下一次真实变载，于是"读数必须停住 30 s"（D1）
以及"停住 ≥5 s"（D1-ev）都判不出来 ⇒ restep 的稳定时间**结构性不可测**。
本脚本把 `min_hold` 从 1 s 扫到 5 s，回答"要多短的静默窗才能测出 restep"，并给出该窗口下的数值。

产出：results/t1a_minhold_sweep.csv、results/_t1a_08_minhold.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t1a_common as C

MIN_HOLDS = [1.0, 2.0, 3.0, 5.0]


def main():
    C.start_log("08_minhold")
    ev = pd.read_csv(os.path.join(C.RES, "t1_events.csv"), encoding="utf-8-sig")
    ev["kind3"] = ev["kind"].replace({"partial_unload": "unload"})
    print("事件 %d 个；min_hold 扫描 %s s" % (len(ev), MIN_HOLDS))
    rows = []
    for ds, g in ev.groupby("ds", sort=False):
        z = np.load(os.path.join(C.CACHE, "t1a_%s.npz" % ds.replace("/", "_")))
        tu = z["tu"]
        peak = float(np.max(z["yraw_tot"]))
        times = [(float(t), float(j)) for t, j in zip(g.t_on, g.J) if np.isfinite(j)]
        for _, e in g.iterrows():
            k0 = int(e["k_on"])
            t_on = float(e["t_on"])
            J_ch, _, _ = C.amp_5s(z["yraw_ch"], k0)
            cut = C.next_event_cut(times, t_on, peak)
            cut_k = None if cut is None else int(round(cut / C.DT))
            r = dict(ds=ds, ev_id=e["ev_id"], family=e["family"], dom=e["dom"], kind=e["kind3"],
                     t_on=t_on, J_tot=float(e["J"]), J_ch=J_ch,
                     win_to_next_s=np.nan if cut is None else float(cut - t_on))
            for mh in MIN_HOLDS:
                for arm in C.ARMS:
                    v, cens, win = C.t_stable_ev(tu, z["%s_ch" % arm], k0, J_ch, cut_k,
                                                 min_hold=mh)
                    r["T_ev_ch5_mh%.0f_%s" % (mh, arm)] = v
                    r["cens_ev_ch5_mh%.0f_%s" % (mh, arm)] = cens
            rows.append(r)
        del z
    df = pd.DataFrame(rows)
    p = os.path.join(C.RES, "t1a_minhold_sweep.csv")
    df.round(4).to_csv(p, index=False, encoding="utf-8-sig")

    print("\n== 可测事件数（分母 = 各类事件数）与中位值 ==")
    print("  %-6s %-16s %s" % ("kind", "arm", "  ".join(["mh=%.0fs" % m for m in MIN_HOLDS])))
    for kd in ("onset", "restep", "unload"):
        s = df[df.kind == kd]
        n = len(s)
        for arm in C.ARMS:
            cells = []
            for mh in MIN_HOLDS:
                c = "T_ev_ch5_mh%.0f_%s" % (mh, arm)
                v = pd.to_numeric(s[c], errors="coerce")
                cells.append("%2d/%2d (中位 %6.2f)" % (int(v.notna().sum()), n,
                                                       float(v.median()) if v.notna().any() else np.nan))
            print("  %-6s %-16s %s" % (kd, arm, "  ".join(cells)))
    print("\n== restep（n=%d）到下一真实事件的可用窗分布 ==" % int((df.kind == "restep").sum()))
    w = df[df.kind == "restep"]["win_to_next_s"]
    print("  中位 %.2f s, p10~p90 %.2f~%.2f s, <5 s 的 %d/%d, <2 s 的 %d/%d"
          % (w.median(), w.quantile(.1), w.quantile(.9), int((w < 5).sum()), int(w.notna().sum()),
             int((w < 2).sum()), int(w.notna().sum())))
    w2 = df[df.kind == "onset"]["win_to_next_s"]
    print("  onset  中位 %.2f s, <5 s 的 %d/%d" % (w2.median(), int((w2 < 5).sum()),
                                                    int(w2.notna().sum())))
    print("\n-> %s" % p)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
