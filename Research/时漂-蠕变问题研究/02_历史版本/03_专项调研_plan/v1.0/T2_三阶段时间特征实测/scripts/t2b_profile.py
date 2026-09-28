# -*- coding: utf-8 -*-
"""T2-B：固定 τ 网格的形状剖面（**全项目复用接口**）+ 时间轴口径敏感性（T2-Q1 / T2-Q6）。

产物：
  results/t2_shape_profile.csv          ← 接口表（列名固定，不得改）
        ds, kind, event_t, tau, f_median, f_p10, f_p90, n
        * 逐事件行：event_t = 该事件 t_on，n = 1，f_median=f_p10=f_p90=该事件的 f(τ)
        * 池化行  ：event_t = 空，n = 池内事件数，f_* = **事件级**分位数（指标字典 §4 Spread(τ) 口径）
        * ds ∈ {RT1..SW4, ALL}；kind ∈ {onset, restep, unload, partial_unload, ALL}
  results/t2_shape_profile_variants.csv ← 各口径变体（grid/pkt × raw/z3/bar/clean/±1 包）
  results/t2_timeaxis_compare.csv       ← 逐事件「100 Hz 网格 vs 原始包时刻」并排
  results/t2_timeaxis_summary.csv       ← 两口径差异汇总（Q6）
"""
import numpy as np
import pandas as pd

import t2_common as C

TAU = C.TAU_FIXED
KINDS = ["onset", "restep", "unload", "partial_unload", "ALL"]
DS_LIST = [r["key"] for r in C.RECS] + ["ALL"]


def _f_event(d, t0, pre, J, axis, taus):
    t_ax, y, ybar, _ = C.axis_series(d, axis)
    return np.array([C.f_at(t_ax, y, t0, pre, J, tau) for tau in taus], float)


def main():
    C.start_log("t2b_profile")
    print("== T2-B 形状剖面（τ 网格）与时间轴口径 ==")
    all_d = C.load_all()
    ev = C.load_events()
    taus = np.array(TAU, float)

    # ── 逐事件 f(τ)（多种口径）──
    per = {}          # (key, t_on) -> DataFrame
    rows_long = []
    # 主口径 = Z3（原始读数 3 帧中值）；其余为对照
    VA = ["grid", "grid_raw", "grid_bar", "grid_clean", "grid_shift_p1", "grid_shift_m1",
          "pkt", "pkt_raw", "pkt_shift_p1", "pkt_shift_m1"]
    VAR_AXIS = {"grid": "grid3", "grid_raw": "grid", "grid_bar": "gridbar",
                "grid_clean": "grid3", "grid_shift_p1": "grid3", "grid_shift_m1": "grid3",
                "pkt": "pkt3", "pkt_raw": "pkt", "pkt_shift_p1": "pkt3", "pkt_shift_m1": "pkt3"}
    for _, e in ev.iterrows():
        d = all_d[e["key"]]
        t0 = float(d["tu"][int(e["k_on"])])
        pkt = d["pkt"]
        t_ax, y, ybar, _ = C.axis_series(d, "grid")
        pre, post, J = C.j_of_idx(ybar, int(e["k_on"]))
        rec = dict(key=e["key"], fam=e["fam"], dom=e["dom"], kind=e["kind"],
                   t_on=round(t0, 3), clean=bool(e["clean"]), J=J, pre=pre, post=post)
        for v in VA:
            ax = VAR_AXIS[v]
            f = _f_event(d, t0, pre, J, ax, taus)
            if v.endswith("shift_p1"):
                f = _f_event(d, t0 + pkt, pre, J, ax, taus)
            elif v.endswith("shift_m1"):
                f = _f_event(d, t0 - pkt, pre, J, ax, taus)
            rec[v] = f
        per[(e["key"], rec["t_on"])] = rec
        for i, tau in enumerate(taus):
            rows_long.append(dict(ds=e["key"], kind=e["kind"], event_t=rec["t_on"], tau=float(tau),
                                  f_median=rec["grid"][i], f_p10=rec["grid"][i],
                                  f_p90=rec["grid"][i], n=1, fam=e["fam"],
                                  clean=rec["clean"]))
    long_df = pd.DataFrame(rows_long)

    # ── 池化：事件级分位数 ──
    def pooled(sub, ds_tag, kind_tag):
        out = []
        for i, tau in enumerate(TAU):
            v = sub[:, i]
            v = v[np.isfinite(v)]
            if v.size == 0:
                out.append(dict(ds=ds_tag, kind=kind_tag, event_t=np.nan, tau=float(tau),
                                f_median=np.nan, f_p10=np.nan, f_p90=np.nan, n=0))
            else:
                out.append(dict(ds=ds_tag, kind=kind_tag, event_t=np.nan, tau=float(tau),
                                f_median=float(np.median(v)),
                                f_p10=float(np.percentile(v, 10)),
                                f_p90=float(np.percentile(v, 90)), n=int(v.size)))
        return out

    keys = list(per.keys())
    mat = np.vstack([per[k]["grid"] for k in keys])
    kinds = np.array([per[k]["kind"] for k in keys])
    dss = np.array([per[k]["key"] for k in keys])
    pooled_rows = []
    for dsk in DS_LIST:
        m0 = np.ones(len(keys), bool) if dsk == "ALL" else (dss == dsk)
        for kd in KINDS:
            m = m0 & (np.ones(len(keys), bool) if kd == "ALL" else (kinds == kd))
            if m.sum() == 0:
                continue
            pooled_rows += pooled(mat[m], dsk, kd)
    p_df = pd.DataFrame(pooled_rows)

    interface = pd.concat([long_df[["ds", "kind", "event_t", "tau", "f_median", "f_p10",
                                    "f_p90", "n"]], p_df], ignore_index=True)
    interface = interface.sort_values(["ds", "kind", "event_t", "tau"],
                                      na_position="last").reset_index(drop=True)
    interface.to_csv(C.os.path.join(C.RES, "t2_shape_profile.csv"), index=False,
                     encoding="utf-8-sig")
    print("  -> results/t2_shape_profile.csv (%d 行；逐事件 %d + 池化 %d)" %
          (len(interface), len(long_df), len(p_df)))

    # ── 变体表（含逐事件与池化）──
    vrows = []
    for v in VA:
        m = np.vstack([per[k][v] for k in keys])
        if v == "grid_clean":
            cm = np.array([per[k]["clean"] for k in keys])
            m = m[cm]
            kk = kinds[cm]
        else:
            kk = kinds
        for kd in ("onset", "restep", "unload", "partial_unload", "ALL"):
            sel = np.ones(len(kk), bool) if kd == "ALL" else (kk == kd)
            if sel.sum() == 0:
                continue
            for i, tau in enumerate(TAU):
                vv = m[sel, i]
                vv = vv[np.isfinite(vv)]
                if vv.size == 0:
                    continue
                vrows.append(dict(variant=v, kind=kd, tau=float(tau), n=int(vv.size),
                                  f_median=float(np.median(vv)),
                                  f_p10=float(np.percentile(vv, 10)),
                                  f_p90=float(np.percentile(vv, 90))))
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(C.os.path.join(C.RES, "t2_shape_profile_variants.csv"), index=False,
               encoding="utf-8-sig")
    print("  -> results/t2_shape_profile_variants.csv (%d 行，%d 个变体)" % (len(vdf), len(VA)))

    # ── Q6：100 Hz 网格 vs 原始包时刻（并排）──
    # 统一用「网格口径」的 pre/J（只让时间轴变、不让幅度窗变），J_pkt 单列作诊断。
    trows = []
    for k in keys:
        d = all_d[per[k]["key"]]
        t0 = per[k]["t_on"]
        pkt = d["pkt"]
        jp = C.j_of(d["Zpbar"], d["tp"], t0)
        for ax in ("grid3", "pkt3"):
            t_ax, y, ybar, _ = C.axis_series(d, ax)
            pre, post, J = per[k]["pre"], per[k]["post"], per[k]["J"]
            row = dict(key=per[k]["key"], kind=per[k]["kind"], t_on=t0, axis=("grid" if ax=="grid3" else "pkt"),
                       pkt_dt=pkt, pre=pre, post=post, J=J, J_pkt=jp[2])
            for fr in (0.25, 0.50, 0.80, 0.90, 0.95):
                row["t%02d" % int(fr * 100)] = C.cross_time(t_ax, y, t0, pre, J, fr)
            for tau, c in ((0.05, "f_005"), (0.10, "f_010"), (0.20, "f_020"),
                           (1.00, "f_100"), (5.00, "f_500")):
                row[c] = C.f_at(t_ax, y, t0, pre, J, tau)
            trows.append(row)
        # ±1 包：只移动事件原点，轴线不变
        for off, tag in ((+pkt, "grid_p1"), (-pkt, "grid_m1"), (+pkt, "pkt_p1"), (-pkt, "pkt_m1")):
            ax = "pkt3" if tag.startswith("pkt") else "grid3"
            t_ax, y, ybar, _ = C.axis_series(d, ax)
            pre, post, J = per[k]["pre"], per[k]["post"], per[k]["J"]
            row = dict(key=per[k]["key"], kind=per[k]["kind"], t_on=t0, axis=tag,
                       pkt_dt=pkt, pre=pre, post=post, J=J, J_pkt=jp[2])
            for fr in (0.25, 0.50, 0.80, 0.90, 0.95):
                row["t%02d" % int(fr * 100)] = C.cross_time(t_ax, y, t0 + off, pre, J, fr)
            for tau, c in ((0.05, "f_005"), (0.10, "f_010"), (0.20, "f_020"),
                           (1.00, "f_100"), (5.00, "f_500")):
                row[c] = C.f_at(t_ax, y, t0 + off, pre, J, tau)
            trows.append(row)
    tdf = pd.DataFrame(trows)
    tdf.to_csv(C.os.path.join(C.RES, "t2_timeaxis_compare.csv"), index=False,
               encoding="utf-8-sig")
    print("  -> results/t2_timeaxis_compare.csv (%d 行)" % len(tdf))

    # ── Q6 汇总 ──
    srows = []
    for kind in ("onset", "restep", "unload", "partial_unload", "ALL"):
        s = tdf if kind == "ALL" else tdf[tdf.kind == kind]
        if not len(s):
            continue
        for a, b in (("grid", "pkt"), ("grid", "grid_p1"), ("grid", "grid_m1"),
                     ("grid", "pkt_p1"), ("grid", "pkt_m1"), ("pkt", "pkt_p1")):
            sa = s[s.axis == a].set_index("key")
            sb = s[s.axis == b].set_index("key")
            ix = sa.index.intersection(sb.index)
            for col in ("t25", "t50", "t90", "t95", "f_005", "f_020", "f_100", "J"):
                da = sa.loc[ix, col].to_numpy(float)
                db = sb.loc[ix, col].to_numpy(float)
                d = db - da
                d = d[np.isfinite(d)]
                if d.size == 0:
                    continue
                srows.append(dict(kind=kind, axis_a=a, axis_b=b, metric=col, n=int(d.size),
                                  med_delta=float(np.median(d)),
                                  med_abs_delta=float(np.median(np.abs(d))),
                                  p90_abs_delta=float(np.percentile(np.abs(d), 90)),
                                  max_abs_delta=float(np.max(np.abs(d)))))
    sdf = pd.DataFrame(srows)
    sdf.to_csv(C.os.path.join(C.RES, "t2_timeaxis_summary.csv"), index=False,
               encoding="utf-8-sig")
    print("  -> results/t2_timeaxis_summary.csv")

    # ── 控制台：主剖面 + 时间轴差异 ──
    print("\n== 池化形状剖面 f(τ)（原始读数，事件级中位 [p10,p90]）==")
    hdr = "%-14s" % "kind"
    for tau in (0.05, 0.2, 1.0, 5.0, 30.0):
        hdr += "%22s" % ("τ=%.2gs" % tau)
    print(hdr)
    for kind in ("onset", "restep", "unload", "partial_unload"):
        r = p_df[(p_df.ds == "ALL") & (p_df.kind == kind)]
        line = "%-14s" % kind
        for tau in (0.05, 0.2, 1.0, 5.0, 30.0):
            q = r[r.tau == tau]
            line += "%22s" % ("%.3f [%.3f,%.3f]" % (q.f_median.iloc[0], q.f_p10.iloc[0],
                                                    q.f_p90.iloc[0]) if len(q) else "n/a")
        print(line + "  (n=%d)" % (r.n.max() if len(r) else 0))
    print("\n== Q6 时间轴口径差异（median |Δ|，s）==")
    q6 = sdf[(sdf.axis_a == "grid") & (sdf.axis_b == "pkt")]
    for kind in ("onset", "restep", "unload", "ALL"):
        q = q6[q6.kind == kind]
        if not len(q):
            continue
        g = {r.metric: r for _, r in q.iterrows()}
        print("   %-14s n=%2d  Δt90=%.3f  Δt95=%.3f  Δf(0.05)=%.3f  Δf(0.2)=%.3f  Δf(1)=%.3f" %
              (kind, int(g["t90"].n), g["t90"].med_abs_delta, g["t95"].med_abs_delta,
               g["f_005"].med_abs_delta, g["f_020"].med_abs_delta, g["f_100"].med_abs_delta))
    print("\n完成。")


if __name__ == "__main__":
    main()
