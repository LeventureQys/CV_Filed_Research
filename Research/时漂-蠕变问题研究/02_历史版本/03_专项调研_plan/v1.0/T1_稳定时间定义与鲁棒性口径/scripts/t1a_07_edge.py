# -*- coding: utf-8 -*-
"""t1a_07_edge：`t_on`（真沿）定义对 T_stable 的影响 —— 口径自由度 ③ 的定量化。

背景（00_共享/接口冲突登记.md C-1）：T_stable 的三个自由度是 ①主通道/总通道 ②参考电平 5 s/含蠕变
③`t_on` 取"最大单帧跳变"（T4-A：检出沿 ±0.20 s 内最大单帧跳变）还是别的定义。
第一轮 `r4_filter_tradeoff.py::onset_index` 用的是**全录制 0.2 s 中心差分最大处**，与本任务沿用的
T4-A `t_on` 不是同一帧 —— 本脚本把这一差值与其对 T_stable 的影响逐份算出来。

产出：results/t1a_settle_edge_sensitivity.csv、results/_t1a_07_edge.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t1a_ad_lib as L
import t1a_common as C


def r4_onset_index(y, dt):
    """逐字复制 13-v6-assessment/r4_filter_tradeoff.py::onset_index（全录制最陡上升帧）。"""
    w = max(1, int(0.10 / dt))
    ys = L.med_smooth(y, w)
    d = np.zeros_like(ys)
    d[w:-w] = ys[2 * w:] - ys[:-2 * w]
    return int(np.argmax(d))


def main():
    C.start_log("07_edge")
    ev = pd.read_csv(os.path.join(C.RES, "t1_events.csv"), encoding="utf-8-sig")
    on = ev[(ev.kind == "onset") & (ev.dom == "显示域")]
    print("对拍对象：恒载 9 组的最早 onset（n=%d）" % len(on))
    rows = []
    for _, e in on.iterrows():
        ds = e["ds"]
        z = np.load(os.path.join(C.CACHE, "t1a_%s.npz" % ds.replace("/", "_")))
        tu = z["tu"]
        ych = z["yraw_ch"]
        k4 = r4_onset_index(ych, C.DT)
        kt = int(e["k_on"])
        J4, _, _ = C.amp_5s(ych, k4)
        Jt, _, _ = C.amp_5s(ych, kt)
        r = dict(ds=ds, ev_id=e["ev_id"], t_on_t4a=float(e["t_on"]), t_on_r4=float(tu[k4]),
                 d_edge_s=float(tu[k4] - tu[kt]), J_ch_t4a=Jt, J_ch_r4=J4,
                 J_tot=e["J"], clean=bool(e["clean"]))
        for arm in C.ARMS:
            a4, _ = C.t_stable(tu, z["%s_ch" % arm], k4, J4)
            at, _ = C.t_stable(tu, z["%s_ch" % arm], kt, Jt)
            r["T_ch5_r4edge_" + arm] = a4
            r["T_ch5_t4aedge_" + arm] = at
            r["d_edge_" + arm] = (at - a4) if np.isfinite(a4) and np.isfinite(at) else np.nan
            b4, _ = C.t_stable(tu, z["%s_tot" % arm], k4, C.amp_5s(z["%s_tot" % arm], k4)[0])
            bt, _ = C.t_stable(tu, z["%s_tot" % arm], kt, C.amp_5s(z["%s_tot" % arm], kt)[0])
            r["T_tot5_r4edge_" + arm] = b4
            r["T_tot5_t4aedge_" + arm] = bt
            # 含蠕变口径（与第一轮同名口径）：参考幅度 = 卸载沿前 5 s 中位 − pre
            for tag, kk, pp in (("r4edge", k4, J4), ("t4aedge", kt, Jt)):
                ju = C.unload_index(ych, kk)
                jc = np.nan
                if ju is not None:
                    a = max(kk, ju - int(5.0 / C.DT))
                    if ju > a:
                        jc = float(np.median(ych[a:ju]) - C.amp_5s(ych, kk)[1])
                cv, _ = C.t_stable(tu, z["%s_ch" % arm], kk, jc)
                r["T_chcreep_%s_%s" % (tag, arm)] = cv
        rows.append(r)
        print("  %-22s t_on: T4A %7.2f s / r4 %7.2f s（Δ=%+.2f s → %s）J_ch %7.4f vs %7.4f"
              % (ds, r["t_on_t4a"], r["t_on_r4"], r["d_edge_s"],
                 "r4 更晚" if r["d_edge_s"] > 0 else "r4 更早", r["J_ch_t4a"], r["J_ch_r4"]))
        del z
    df = pd.DataFrame(rows)
    p = os.path.join(C.RES, "t1a_settle_edge_sensitivity.csv")
    df.round(4).to_csv(p, index=False, encoding="utf-8-sig")

    print("\n== t_on 差值（r4 最陡帧 − T4-A 最大单帧跳变）：中位 %.3f s，p10~p90 %.3f~%.3f s，n=%d =="
          % (df.d_edge_s.median(), df.d_edge_s.quantile(.1), df.d_edge_s.quantile(.9), len(df)))
    print("\n== 同一口径下的 T_stable 中位（主通道+5 s，恒载 9 组） ==")
    print("  %-6s %-12s %-12s %s" % ("arm", "r4 边(r4 的 J)", "T4-A 边", "差（T4A−r4）"))
    for arm in C.ARMS:
        a = df["T_ch5_r4edge_" + arm].median()
        b = df["T_ch5_t4aedge_" + arm].median()
        print("  %-6s %12.2f %12.2f %+12.2f" % (arm, a, b, b - a))
    print("\n== 同一口径下的 T_stable 中位（总通道+5 s，恒载 9 组） ==")
    for arm in C.ARMS:
        a = df["T_tot5_r4edge_" + arm].median()
        b = df["T_tot5_t4aedge_" + arm].median()
        print("  %-6s %12.2f %12.2f %+12.2f" % (arm, a, b, b - a))
    print("\n== 同一口径下的 T_stable 中位（含蠕变口径，恒载 9 组） ==")
    for arm in C.ARMS:
        a = df["T_chcreep_r4edge_" + arm].median()
        b = df["T_chcreep_t4aedge_" + arm].median()
        print("  %-6s %12.2f %12.2f %+12.2f" % (arm, a, b, b - a))

    # ── 网格口径：第一轮 ad_lib.prep 用 dtm=span/(n−1)≈0.009949 s（≈100.5 Hz），本任务用 100 Hz ──
    print("\n== 网格口径敏感性（第一轮 ad_lib.prep 网格 vs 本任务 100 Hz 网格） ==")
    grid_rows = []
    for _, e in on.iterrows():
        rec = e["ds"]
        t, X = L.load_rec(C.REC_BY_KEY[rec]["path"])
        span = t[-1] - t[0]
        dtm = span / (len(t) - 1)
        tu2 = np.arange(0.0, span, dtm)
        Xu2 = np.vstack([np.interp(tu2, t, X[:, c]) for c in range(X.shape[1])]).T
        ch = C.REC_BY_KEY[rec]["ch"]
        y2 = Xu2[:, ch]
        k2 = r4_onset_index(y2, dtm)
        r = dict(ds=rec, n_grid_r4=len(tu2), dtm_r4=dtm, t_on_r4grid=float(tu2[k2]))
        for arm in C.ARMS:
            out = C.run_arm(rec, arm, tu2, Xu2)
            J2, _, _ = C.amp_5s(y2, k2)
            v, c = C.t_stable(tu2, out["ch"], k2, J2)
            r["T_ch5_r4grid_" + arm] = v
            v0, c0 = C.t_stable(tu2, out["ch"], k2, C.amp_5s(y2, k2)[0])
            del out, v0, c0
        mine = df[df.ds == rec]
        r["T_ch5_100hzgrid_r4edge_v6"] = float(mine["T_ch5_r4edge_v6"].iloc[0])
        grid_rows.append(r)
        print("  %-22s n=%5d dtm=%.6f | v6 第一轮网格 %7.2f s vs 本任务 100 Hz 网格 %7.2f s（第一轮公布 %s）"
              % (rec, len(tu2), dtm, r["T_ch5_r4grid_v6"], r["T_ch5_100hzgrid_r4edge_v6"],
                 ("%.2f" % float(pd.read_csv(
                     os.path.join(C.PROG, "13-v6-assessment", "results", "settle_arms.csv"),
                     encoding="utf-8-sig").query("arm=='v6' and rec=='%s'" % rec)["T_stable"].iloc[0]))))
        del Xu2, tu2, X
    gdf = pd.DataFrame(grid_rows)
    pg = os.path.join(C.RES, "t1a_grid_sensitivity.csv")
    gdf.round(4).to_csv(pg, index=False, encoding="utf-8-sig")
    print("  v6 中位：第一轮网格 %.2f s vs 本任务 100 Hz 网格 %.2f s（差 %+.2f s）"
          % (gdf["T_ch5_r4grid_v6"].median(), gdf["T_ch5_100hzgrid_r4edge_v6"].median(),
             gdf["T_ch5_r4grid_v6"].median() - gdf["T_ch5_100hzgrid_r4edge_v6"].median()))
    print("  逐份最大差 %.2f s（%s）" % ((gdf["T_ch5_r4grid_v6"] - gdf["T_ch5_100hzgrid_r4edge_v6"]).abs().max(),
                                        gdf.loc[(gdf["T_ch5_r4grid_v6"] - gdf["T_ch5_100hzgrid_r4edge_v6"])
                                                .abs().idxmax(), "ds"]))
    print("-> %s" % pg)
    print("\n== 与第一轮 settle_arms.csv 的逐份对照（v6 主通道） ==")
    r1 = pd.read_csv(os.path.join(C.PROG, "13-v6-assessment", "results", "settle_arms.csv"),
                     encoding="utf-8-sig")
    r1 = r1[(r1.arm == "v6") & (r1.rec.isin(df.ds))]
    for _, e in r1.iterrows():
        mine = df[df.ds == e["rec"]]
        if not len(mine):
            continue
        print("  %-22s 第一轮 %8.2f s | 本任务(r4 边) %8.2f s | 本任务(T4-A 边) %8.2f s"
              % (e["rec"], e["T_stable"], float(mine["T_ch5_r4edge_v6"].iloc[0]),
                 float(mine["T_ch5_t4aedge_v6"].iloc[0])))
    print("\n-> %s（%d 行）" % (p, len(df)))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
