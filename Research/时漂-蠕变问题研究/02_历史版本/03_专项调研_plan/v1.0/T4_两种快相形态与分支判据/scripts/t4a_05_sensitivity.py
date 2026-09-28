# -*- coding: utf-8 -*-
"""t4a_05_sensitivity：口径敏感性（armed 门限 10/20/30% + 时间戳 ±1 包）。

需求书硬性要求：armed 判定不得挂在单门限上；凡结论落在包时间戳量化量级内必须给 ±1 包敏感性。

产出：results/t4a_sensitivity.csv
日志：results/_t4a_05_sensitivity.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t4a_common as C
import t4a_ad_lib as L

KEYF = ["t50", "t80", "t90", "z_at_005", "z_at_01", "z_at_02", "z_at_10",
        "n_frames_rise", "rise_frames_10_90", "step_frame_frac"]


def main():
    C.start_log()
    mor = pd.read_csv(os.path.join(C.RES, "t4a_morphology.csv"))
    ir = pd.read_csv(os.path.join(C.RES, "t4a_input_recover.csv"))
    mor["t_on_r"] = mor["t_on"].round(3)
    ir["t_on_r"] = ir["t_on"].round(3)
    mor = mor.merge(ir[["key", "t_on_r", "T_ramp"]], on=["key", "t_on_r"], how="left")
    rows = []

    # ── ① armed 门限 ──
    for thr in (10, 20, 30):
        ld = mor[mor.jump > 0].copy()
        ld["kind_t"] = np.where(ld.pre_over_peak >= thr / 100.0, "restep", "onset")
        ld = ld[ld.clean]
        for f in ["t90", "z_at_02", "z_at_10", "T_ramp"]:
            o = ld[ld.kind_t == "onset"][f].to_numpy(float)
            r = ld[ld.kind_t == "restep"][f].to_numpy(float)
            o, r = o[np.isfinite(o)], r[np.isfinite(r)]
            mo, o10, o90, no = C.band(o)
            mr, r10, r90, nr = C.band(r)
            rows.append(dict(variant="armed_thr_%d" % thr, feature=f, n_onset=no, n_restep=nr,
                             med_onset=mo, med_restep=mr, auc=C.auc(o, r),
                             band_overlap=C.overlap_ratio((o10, o90), (r10, r90)),
                             note="与 armed_20 标签不一致 %d 个"
                                  % int((ld.kind_t != ld.kind).sum())))
    ld0 = mor[mor.jump > 0]
    for thr in (10, 30):
        k = np.where(ld0.pre_over_peak >= thr / 100.0, "restep", "onset")
        rows.append(dict(variant="armed_thr_%d" % thr, feature="label_flip_count",
                         n_onset=int((k == "onset").sum()), n_restep=int((k == "restep").sum()),
                         med_onset=np.nan, med_restep=np.nan, auc=np.nan, band_overlap=np.nan,
                         note="全部加载类事件中与 armed_20 不一致 %d 个"
                              % int((k != ld0.kind).sum())))

    # ── ② ±1 包 ──
    recs = {}
    for r in C.RECS:
        tu, Xu, pkt, nraw = C.load_grid(r)
        Z = Xu.sum(axis=1)
        recs[r["key"]] = dict(tu=tu, Z=Z, Zs=L.med_smooth(Z, C.KSM), pkt=pkt,
                              amp=float(np.percentile(Z, 99.5) - np.percentile(Z, 0.5)),
                              peak=float(L.med_smooth(Z, C.KSM).max()))
    base = {}
    for shift, tag in ((0, "base"), (1, "pkt_plus1"), (-1, "pkt_minus1")):
        tbl = []
        for _, q in mor[mor.jump.notna()].iterrows():
            R = recs[q.key]
            st = max(1, int(round(R["pkt"] / C.DT))) if np.isfinite(R["pkt"]) else 1
            k = int(q.k_on) + shift * st
            if k < 1 or k >= len(R["Z"]) - int(6.0 / C.DT):
                continue
            m = C.event_metrics(R["tu"], R["Z"], R["Zs"], k, R["peak"], R["amp"], R["pkt"], [k])
            m.update(key=q.key, rec=q.rec, kind0=q.kind, clean0=bool(q["clean"]), t_on0=q.t_on)
            tbl.append(m)
        t = pd.DataFrame(tbl)
        for f in KEYF:
            if f not in t.columns:
                continue
            a = t[(t.kind0 == "onset") & t.clean0][f].to_numpy(float)
            b = t[(t.kind0 == "restep") & t.clean0][f].to_numpy(float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            rows.append(dict(variant=tag, feature=f, n_onset=len(a), n_restep=len(b),
                             med_onset=float(np.median(a)) if len(a) else np.nan,
                             med_restep=float(np.median(b)) if len(b) else np.nan,
                             auc=C.auc(a, b), band_overlap=np.nan,
                             note="包周期：指尖 0.0167 s / 变载实录 0.040 s"))
        base[tag] = t.set_index(["key", "t_on0"])[KEYF]
    for tag in ("pkt_plus1", "pkt_minus1"):
        j = base[tag].join(base["base"], lsuffix="_s", rsuffix="_b", how="inner")
        for f in KEYF:
            if f + "_s" not in j.columns:
                continue
            d = (j[f + "_s"] - j[f + "_b"]).abs()
            rows.append(dict(variant=tag + "_delta", feature=f, n_onset=int(d.notna().sum()),
                             n_restep=0, med_onset=float(d.median()), med_restep=np.nan,
                             auc=np.nan, band_overlap=np.nan,
                             note="逐事件 |Δ| 中位（相对 base；±1 包 = 16.7/40 ms）"))
    out = pd.DataFrame(rows)
    out.round(5).to_csv(os.path.join(C.RES, "t4a_sensitivity.csv"), index=False,
                        encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    print("== armed 门限敏感性（clean 子集）==")
    print(out[out.variant.str.startswith("armed")].round(4).to_string(index=False))
    print("\n== ±1 包敏感性（clean 子集；delta 行为逐事件位移量）==")
    print(out[out.variant.str.startswith("pkt")].round(4).to_string(index=False))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
