# -*- coding: utf-8 -*-
"""临时诊断脚本（T7-A）：核对卸载沿候选为何漏检。用完即删。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402


def main(tag="再切换负载"):
    d = C.prep(tag)
    tu, dtm = d["tu"], d["dtm"]
    ds = C.zbar(d["tot"], dtm)
    acc, rows, edges, _ = C.detect_unloads(tu, d["tot"], dtm, d["pkt_dt"])
    print(f"{tag}: 严格集 {len(acc)}")
    print("edges(cand):", [(round(e["t_cand"], 2), round(e["dlt"], 1)) for e in edges])
    for r in rows:
        print(f"  t0={r['t0']:8.2f} t_cand={r['t_cand']:8.2f} next_edge={r['t_next_edge']:8.2f} "
              f"pre={r['pre']:10.1f} post={r['post']:10.1f} post_adapt={r['post_adapt']:10.1f} "
              f"sf={r['step_frac']:.4f} sfa={r['step_frac_adapt']:.4f} "
              f"stpre={r['stab_pre']:.4f} stpost={r['stab_post']:.4f} "
              f"stposta={r['stab_post_adapt']:.4f} win={r['post_win_s']:.2f} "
              f"kind={r['kind']} kindA={r['kind_adapt']} ok={r['stable']}/{r['stable_adapt']}")
    gaps, eps, floor, peak = C.parse_plateaus(ds, tu, dtm)
    print(f"floor={floor:.2f} peak={peak:.1f} thr={floor+0.10*(peak-floor):.2f}")
    print("idle gaps:", [(round(g["t0"], 2), round(g["t1"], 2), round(g["lvl"], 2)) for g in gaps])
    print("episodes :", [(round(e["t0"], 2), round(e["t1"], 2), e["i0"], e["i1"]) for e in eps])
    for r in rows:
        ep = next((e for e in eps if e["i0"] <= r["i0"] <= e["i1"]), None)
        gp = None
        for g in gaps:
            if g["i1"] <= (ep["i0"] if ep else r["i0"]):
                gp = g
        z0 = C.zero_level(ds, gp, dtm, 0.5, 3.0)[0] if gp is not None else float(np.percentile(ds, 5))
        print(f"  event t0={r['t0']:.2f} i0={r['i0']} ep={'None' if ep is None else (round(ep['t0'],2), round(ep['t1'],2), ep['i0'], ep['i1'])}"
              f" z0={z0:.4f} pre={r['pre']:.4f} Jt={z0-r['pre']:.4f}")


def trace(tag, t0, half=0.8):
    import t7a_common as C
    d = C.prep(tag)
    tu, dtm = d["tu"], d["dtm"]
    tot = d["tot"]
    ds = C.zbar(tot, dtm)
    dst = C.med_smooth(tot, int(round(0.1 / dtm)))
    t0 = float(t0)
    i0 = int(round(t0 / dtm))
    print(f"== {tag} @ {t0}  （原始 tot / Z̄(中心0.5s) / 因果0.1s；包间隔 {d['pkt_dt']*1000:.3f} ms）")
    n = len(tu)
    for j in range(i0 - int(half / dtm), i0 + int(half / dtm) + 1, max(1, int(0.02 / dtm))):
        if 0 <= j < n:
            print(f"  t={tu[j]:9.3f} (τ={tu[j]-t0:+7.3f})  tot={tot[j]:12.3f}  "
                  f"Zc={ds[j]:12.3f}  Zt={dst[j]:12.3f}")
    print("  原始 timestamp 相邻差（前 12 个正差）:")
    dt = np.diff(d["t"])
    print("   ", np.round(dt[dt > 0][:12] * 1000, 4), " ms")


def tscheck():
    import t7a_common as C
    for tag, path in C.ALL:
        d = C.prep(tag)
        t = d["t"]
        dt = np.diff(t)
        big = np.where(dt > 0.5)[0]
        small = dt[(dt > 0) & (dt <= 0.5)]
        print(f"{tag:<18} n={len(t):>6} span={t[-1]-t[0]:>7.2f}s  "
              f"小步长(<0.5s) 中位={np.median(small)*1000:>7.3f}ms p10={np.percentile(small,10)*1000:>7.3f} "
              f"p90={np.percentile(small,90)*1000:>7.3f} 零差占比={(dt<=0).mean()*100:>5.1f}%  "
              f"大跳({'>'}0.5s) n={len(big)} 合计={dt[big].sum():>7.2f}s")
        if len(big):
            print("     大跳时刻/大小:", [(round(float(t[i]), 2), round(float(dt[i]), 2))
                                          for i in big[:12]])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trace":
        trace(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "ts":
        tscheck()
    else:
        main(*sys.argv[1:])
