# -*- coding: utf-8 -*-
"""b1：卸载（卸载沿）后的时漂规律 + 补偿器在卸载沿的行为（问题 1）。

臂的逐帧输出**直接复用** 11-paper-v6/results/cache/*.npz（25 Hz，逐通道原始输出，与
metrics_all.csv 同一次运行）；原始读数与事件几何用 CSV 全速率（≈100 Hz）。

产出：
  results/b_unload_events.csv        每（卸载沿 × 臂）一行：下冲/恢复/残余
  results/b_unload_raw.csv           仅原始读数：几何 + 三阶段幅度 + τ + 恢复时间
  results/b_unload_arm_summary.csv   按家族 × 臂汇总（中位/最大额外下冲、恢复、残余、贴零时长）
  results/b_unload_corr.csv          下冲/恢复 与 幅度/保压/蠕变 的相关性（按子集）
  results/b_unload_traj.csv          卸载沿对齐的归一化轨迹（家族 × 臂 × 时间栅格）
  results/_b1_unload.log             日志
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_common as B                                          # noqa: E402

LOG = []
TRAJ_T = np.array([-2.0, -1.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0,
                   8.0, 12.0, 20.0, 30.0, 45.0, 60.0])


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def _t_cross(tu, x, i0, lvl_up, i_end=None):
    """i0 之后首次 x ≤ lvl_up 的时刻（相对 tu[i0]）。"""
    i_end = len(tu) - 1 if i_end is None else i_end
    w = np.where(x[i0:i_end + 1] <= lvl_up)[0]
    return float(tu[i0 + w[0]] - tu[i0]) if len(w) else np.nan


def _t_hold(tu, x, i0, lo, hi, dt, keep_s=5.0):
    """i0 之后首次进入 [lo,hi] 且此后 keep_s 内不再离开（相对时刻）。"""
    n = len(tu)
    K = max(1, int(keep_s / dt))
    inside = (x >= lo) & (x <= hi)
    for i in range(i0, n):
        if inside[i] and i + K < n and inside[i:i + K].all():
            return float(tu[i] - tu[i0])
    return np.nan


def _stats(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan, int(m.sum())
    rs, ps = stats.spearmanr(x[m], y[m])
    rp, pp = stats.pearsonr(x[m], y[m])
    return float(rs), float(ps), float(rp), float(pp), int(m.sum())


def main():
    B.log_reconfigure()
    ev_rows, raw_rows, traj_rows = [], [], []
    for tag, path in B.ALL:
        d = B.load_csv(tag, path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        tot = Xu.sum(axis=1)
        evs, gfloor, gpeak, ds = B.unload_events(d)
        ts1 = B.med(tot, dt, 0.1)
        z = np.load(B.cache_path(tag), allow_pickle=True)
        ds_step = int(z["ds"][0])
        tuc, dtc = z["tu"], float(z["dtm"][0]) * ds_step
        dis = {"raw": tot[::ds_step], "e3s": z["Yraw_e3s"].sum(axis=1),
               "v6": z["Yraw_v6"].sum(axis=1), "v6trim": z["Yraw_v6trim"].sum(axis=1)}
        for e in evs:
            i_f, i_u = e["i_fall"], e["i_rise"]
            drop = e["drop"]
            # ── 三阶段幅度（对齐 phases.csv 的口径，但在总电平域）──
            y10 = e["pre"] + 0.10 * drop
            w = np.where(ts1[i_u:i_f + 1] >= y10)[0]
            i_e = i_u + int(w[0]) if len(w) else i_u
            a_step = float(ts1[min(len(tu) - 1, i_e + int(0.2 / dt))] - e["pre"])
            a_5s = float(ts1[min(len(tu) - 1, i_e + int(5.0 / dt))] - e["pre"])
            frac1s = float((ts1[min(len(tu) - 1, i_e + int(1.0 / dt))] - e["pre"])
                           / max(abs(drop), 1e-12))
            fast = frac1s >= 0.40
            # ── 卸载后：下冲 / 恢复（原始读数，100 Hz）──
            j0 = i_f + int(0.05 / dt)
            j1 = min(len(tu) - 1, i_f + int(8.0 / dt))
            kk = j0 + int(np.argmin(ds[j0:j1]))
            post, pre = e["post"], e["pre"]
            row = dict(rec=tag, kind=d["kind"], t_dn=e["t_dn"], ev_class=e["ev_class"],
                       hold_s=e["hold_s"], frac1s=frac1s, fast_load=bool(fast),
                       pre=pre, plat=e["plat"], post=post, drop=drop,
                       A_step=a_step if fast else np.nan,
                       A_5s=a_5s if fast else np.nan,
                       creep_fast=(a_5s - a_step) if fast else np.nan,
                       creep_slow=(e["plat"] - a_5s) if fast else np.nan,
                       creep_total=(e["plat"] - e["pre"] - a_step) if fast else np.nan,
                       creep_slow_pct=(100.0 * (e["plat"] - a_5s) / a_5s) if (fast and abs(a_5s) > 1e-9) else np.nan,
                       fall_t=e["fall_t"], post_src=e["post_src"], pre_src=e["pre_src"])
            row["min_raw"] = float(ds[kk])
            row["trough_s"] = float(tu[kk] - tu[i_f])
            row["dip_post"] = post - float(ds[kk])
            row["dip_post_pct"] = 100.0 * row["dip_post"] / max(abs(drop), 1e-9)
            row["dip_pre"] = (pre - float(ds[kk])) if e["ev_class"] == "full" else np.nan
            row["dip_pre_pct"] = (100.0 * row["dip_pre"] / max(abs(drop), 1e-9)
                                  if e["ev_class"] == "full" else np.nan)
            row["resid"] = post - pre
            row["resid_pct_drop"] = 100.0 * (post - pre) / max(abs(drop), 1e-9)
            for f, nm in ((0.10, "t_to10"), (0.05, "t_to5"), (0.01, "t_to1")):
                row[nm] = _t_cross(tu, ds, i_f, post + f * abs(drop))
            for f, nm in ((0.05, "t_rec_keep5"), (0.01, "t_rec_keep1")):
                row[nm] = _t_hold(tu, ds, i_f, post - f * abs(drop), post + f * abs(drop), dt)
            row["below_post_s"] = float(np.sum(ds[j0:j1] < post) * dt)
            # 残余是否衰减：三参数尾拟合 x = c + a·exp(−τ)（窗口用未被 18 s 截断的上界）
            t_end = min(e["post_limit"], e["t_dn"] + 60.0)
            row["post_span"] = t_end - e["t_dn"]
            if t_end - e["t_dn"] >= 8.0:
                c, aa, tau, rms = B.fit_tail3(tu, ds, e["t_dn"] + 1.0, t_end)
                row.update(resid_inf=c - pre if np.isfinite(c) else np.nan,
                           tail_a=aa, tau_tail=tau, rms_tail=rms,
                           tail_ok=bool(np.isfinite(tau) and tau < 0.35 * (t_end - e["t_dn"])
                                        and np.isfinite(rms) and rms < 0.10 * abs(drop)))
            else:
                row.update(resid_inf=np.nan, tail_a=np.nan, tau_tail=np.nan, rms_tail=np.nan,
                           tail_ok=False)
            if row["post_span"] >= 15.0:
                row["resid_late"] = float(np.median(ds[(tu >= t_end - 10.0) & (tu <= t_end)])) - pre
                row["resid_late_pct"] = 100.0 * row["resid_late"] / max(abs(drop), 1e-9)
            else:
                row["resid_late"] = np.nan
                row["resid_late_pct"] = np.nan
            if min(e["post_end"], e["t_dn"] + 60.0) - e["t_dn"] >= 6.0:
                t_end2 = min(e["post_end"], e["t_dn"] + 60.0)
                tau1, rms1, tauf, taus, rms2 = B.fit_taus(tu, ds, e["t_dn"] + 0.1, t_end2, post)
                row.update(tau1=tau1, rms1=rms1, tau_f=tauf, tau_s=taus, rms2=rms2)
                row["tau_ok"] = bool(np.isfinite(tau1) and tau1 < 4000.0
                                     and np.isfinite(rms1) and rms1 < 0.10 * abs(drop))
            else:
                row.update(tau1=np.nan, rms1=np.nan, tau_f=np.nan, tau_s=np.nan, rms2=np.nan,
                           tau_ok=False)
            raw_rows.append(row)
            # ── 各臂在卸载沿的响应 ──
            for arm in B.ARMS:
                ec = dict(e)
                ec["i_fall"], ec["i_rise"] = i_f // ds_step, i_u // ds_step
                a = B.event_axes(ec, tuc, dis[arm], dtc, keep_s=5.0)
                if a is None:
                    continue
                jj0 = ec["i_fall"] + int(0.05 / dtc)
                jj1 = min(len(tuc) - 1, ec["i_fall"] + int(2.0 / dtc))
                xa = B.med(dis[arm], dtc, 0.5)
                pin = float(np.sum(xa[jj0:jj1] <= 0.002 * abs(drop)) * dtc)
                t_to10 = _t_cross(tuc, xa, ec["i_fall"], a["post_dis"] + 0.10 * abs(a["drop_dis"]))
                t_to5 = _t_cross(tuc, xa, ec["i_fall"], a["post_dis"] + 0.05 * abs(a["drop_dis"]))
                t_rec1 = _t_hold(tuc, xa, ec["i_fall"], a["post_dis"] - 0.01 * abs(a["drop_dis"]),
                                 a["post_dis"] + 0.01 * abs(a["drop_dis"]), dtc)
                ev_rows.append(dict(rec=tag, kind=d["kind"], arm=arm, t_dn=e["t_dn"],
                                    ev_class=e["ev_class"], hold_s=e["hold_s"],
                                    frac1s=frac1s, fast_load=bool(fast),
                                    drop_ev=drop, creep_slow_pct=row["creep_slow_pct"],
                                    post_src=e["post_src"], pin0_s=pin,
                                    t_to10=t_to10, t_to5=t_to5,
                                    t_rec_keep_1pct=t_rec1, **a))
                # 轨迹（按家族累积；只在卸载后稳态窗内取样，避免落进下一段负载）
                tj = tuc - e["t_dn"]
                for tv in TRAJ_T:
                    if tv > (e["post_end"] - e["t_dn"]) and tv > 0:
                        continue
                    k = int(np.searchsorted(tj, tv))
                    if k < len(xa):
                        traj_rows.append(dict(kind=d["kind"], arm=arm, t=tv,
                                              norm=(xa[k] - a["post_dis"]) / max(abs(drop), 1e-9)))
    ev = pd.DataFrame(ev_rows)
    raw = pd.DataFrame(raw_rows)
    tr = pd.DataFrame(traj_rows)
    ev.to_csv(os.path.join(B.RES, "b_unload_events.csv"), index=False, encoding="utf-8-sig")
    raw.to_csv(os.path.join(B.RES, "b_unload_raw.csv"), index=False, encoding="utf-8-sig")

    # 额外下冲（相对原始）与汇总
    base = ev[ev.arm == "raw"][["rec", "t_dn", "dip_vs_post", "t_to5", "t_rec_keep_1pct",
                                "resid", "pin0_s"]].rename(
        columns={"dip_vs_post": "dip_raw", "t_to5": "t_to5_raw",
                 "t_rec_keep_1pct": "t_rec_raw", "resid": "resid_raw", "pin0_s": "pin0_raw"})
    ev = ev.merge(base, on=["rec", "t_dn"], how="left")
    ev["dip_extra"] = ev.dip_vs_post - ev.dip_raw
    ev["dip_extra_pct"] = 100.0 * ev.dip_extra / ev.drop_ev.abs().clip(lower=1e-9)
    ev["extra_pin_s"] = ev.pin0_s - ev.pin0_raw
    ev.to_csv(os.path.join(B.RES, "b_unload_events.csv"), index=False, encoding="utf-8-sig")

    srows = []
    for (knd, arm), g in ev.groupby(["kind", "arm"]):
        srows.append(dict(kind=knd, arm=arm, n=len(g),
                          dip_post_med=g.dip_vs_post.median(),
                          dip_post_pct_med=g.dip_post_pct_med if "dip_post_pct_med" in g else
                          100.0 * g.dip_vs_post.median() / max(g.drop_ev.abs().median(), 1e-9),
                          dip_extra_med=g.dip_extra.median(), dip_extra_max=g.dip_extra.max(),
                          dip_extra_pct_max=g.dip_extra_pct.max(),
                          t_to5_med=g.t_to5.median(), t_rec1_med=g.t_rec_keep_1pct.median(),
                          resid_med=g.resid.median(), pin0_med=g.pin0_s.median(),
                          pin0_max=g.pin0_s.max(), extra_pin_max=g.extra_pin_s.max(),
                          n_long=int((g.post_src == "long").sum())))
    sm = pd.DataFrame(srows)
    sm.to_csv(os.path.join(B.RES, "b_unload_arm_summary.csv"), index=False, encoding="utf-8-sig")
    tt = tr.groupby(["kind", "arm", "t"]).norm.agg(["median", "count"]).reset_index()
    tt.to_csv(os.path.join(B.RES, "b_unload_traj.csv"), index=False, encoding="utf-8-sig")

    # 相关性（原始读数；按 家族 × 子集）
    crows = []
    for knd in ("恒载", "实采", "全部"):
        r0 = raw if knd == "全部" else raw[raw.kind == knd]
        subsets = {"全部事件": r0, "仅full": r0[r0.ev_class == "full"],
                   "仅full且长窗": r0[(r0.ev_class == "full") & (r0.post_src == "long")],
                   "快加载|frac1s≥0.4": r0[r0.fast_load]}
        for sname, r1 in subsets.items():
            for tgt in ("dip_post_pct", "dip_pre_pct", "t_rec_keep5", "t_to5",
                        "resid_pct_drop", "tau1"):
                for feat in ("drop", "hold_s", "creep_total", "creep_slow", "creep_slow_pct",
                             "frac1s", "plat"):
                    rs, ps, rp, pp, n = _stats(r1[feat], r1[tgt])
                    crows.append(dict(family=knd, subset=sname, target=tgt, feature=feat,
                                      n=n, spearman=rs, p_spearman=ps, pearson=rp, p_pearson=pp))
    cor = pd.DataFrame(crows)
    cor.to_csv(os.path.join(B.RES, "b_unload_corr.csv"), index=False, encoding="utf-8-sig")

    # ── 打印 ──
    p("=" * 130)
    p("表 1  原始读数：卸载沿几何 + 三阶段幅度 + 卸载后行为（dip=低于卸载后稳态的最小值）")
    p("=" * 130)
    cols = ["rec", "t_dn", "ev_class", "hold_s", "frac1s", "pre", "plat", "post", "drop",
            "A_step", "A_5s", "creep_slow", "creep_slow_pct", "fall_t", "trough_s",
            "dip_post", "dip_post_pct", "dip_pre", "resid", "resid_pct_drop",
            "resid_inf", "tau_tail", "tail_ok", "post_span", "resid_late", "resid_late_pct",
            "t_to10", "t_to5", "t_to1", "t_rec_keep5", "t_rec_keep1", "tau1", "tau_ok",
            "post_src"]
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        p(raw[cols].round(3).to_string(index=False))
    p("")
    p("=" * 130)
    p("表 2  家族 × 臂：卸载沿附近显示行为（dip_extra=该臂下冲 − 原始下冲；pin0=显示贴零时长）")
    p("=" * 130)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        p(sm.round(3).to_string(index=False))
    p("")
    p("=" * 130)
    p("表 3  相关性（原始读数；* = p<0.05）")
    p("=" * 130)
    show = cor[(cor.subset.isin(["仅full且长窗", "快加载|frac1s≥0.4", "全部事件"]))
               & (cor.target.isin(["dip_post_pct", "t_rec_keep5", "t_to5", "resid_pct_drop"]))
               & (cor.feature.isin(["drop", "hold_s", "creep_total", "creep_slow_pct"]))].copy()
    show["sig"] = np.where(show.p_spearman < 0.05, "*s", "") + \
                  np.where(show.p_pearson < 0.05, "*p", "")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        p(show[["family", "subset", "target", "feature", "n", "spearman", "p_spearman",
                "pearson", "p_pearson", "sig"]].round(3).to_string(index=False))
    p("")
    p("=" * 130)
    p("表 4  卸载沿对齐的归一化轨迹（显示−post)/drop，中位；t<0 为卸载前")
    p("=" * 130)
    for knd in ("恒载", "实采"):
        sub = tt[tt.kind == knd].pivot_table(index="t", columns="arm", values="median")
        p(f"  [{knd}]")
        with pd.option_context("display.width", 200):
            p(sub[list(B.ARMS)].round(4).to_string())
    with open(os.path.join(B.RES, "_b1_unload.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print("\n-> results/b_unload_events.csv / b_unload_raw.csv / b_unload_arm_summary.csv / "
          "b_unload_corr.csv / b_unload_traj.csv / _b1_unload.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
