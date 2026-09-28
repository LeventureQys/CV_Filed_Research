# -*- coding: utf-8 -*-
"""T3-A 01：形状散布（Q1）、形状参数化（Q2）、形状库泛化与乐观性折扣（Q4）。

口径（本脚本全部遵守，报告 §2 逐条复述）：
  * 形状 f(τ) = (Z̄(t_on+τ) − base)/J，**base = T4-A 的 pre 窗中位**（C-2：T4-A 已冻结）；
  * 主 τ 网格 = T4-A 的 z_at_* 采样点 [0.05,0.10,0.20,0.30,0.50,1.00,2.00,5.00]（grid 列 = T4A_z_at_grid）；
    加密网格 dτ=0.01 s（grid 列 = fine_0p01s）只用于变异带曲线；
  * **onset 与 restep 分开算，绝不混算**（需求文档 §5-2）；
  * n<20 一律报「中位 + p10~p90」（指标字典 §6-2）；
  * **CV 的诚实处理**：CV=std/mean 在 mean 趋 0（小 τ 的 f、含负值的事件）时会病态放大甚至变号。
    因此每行同时给：`cv`（指标字典 §4 定义，仅当 mean>0.05 标 cv_valid=True）、
    `cqv`（四分位变异系数 (p75−p25)/(p75+p25)，对离群与非正定稳健）、`spread`(p90−p10)、
    `norm_spread`=(p90−p10)/med。**报告以 cqv / spread 为主，cv 作口径对照**（已在报告 §2.3 声明）。
  * 样本三档（`results/t3a_inlier_audit.csv` 冻结）：all（全部装载事件）→ J_ok（|J|≥2% 记录峰值）
    → usable（再加 clean 与 f(τ)∈[0,1.5] 的非病态判据）。**报告以 usable 为主样本**，
    all 用于暴露分布尾部（病态事件的存在本身是结论）。

泛化（Q4）——四个场景 × 两个反演口径 × 两个参考电平：
  场景 own（**乐观**：形状库含本事件自己）、loo_event（同族留一事件）、
       loo_rec（同族留一录制）、cross_fam（跨族）、cross_kind（跨形态 onset↔restep）
  口径 single@τ（单点 Â = Z(τ)/f̄(τ)）、winLS(0.2→τ)（窗内最小二乘，对单点病态更稳健）
  参考电平 post（T4-A 口径，J 本身）与 5s（07-v6 ROM 口径）。

产物：results/t3a_shape_spread.csv, t3a_shape_curves.csv, t3a_param_cv.csv, t3a_param_events.csv,
      t3a_generalization.csv, t3a_generalization_raw.csv, t3a_shape_trajectory.csv,
      results/_t3a_01_shape.log
运行：python scripts/t3a_01_shape.py   （约 60~120 s；需先跑 t3a_00_recon.py 与 t3a_02f_audit.py）
"""
import os

import numpy as np
import pandas as pd

import t3a_common as C

TAU_D = [0.20, 0.30, 0.50, 1.00]          # 决策时刻 τ_d（反演报告用）
TAU_D_FOR_PLOT = [0.20, 0.50, 1.00]


def p75p25(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return np.nan, np.nan
    return float(np.percentile(v, 75)), float(np.percentile(v, 25))


def qstats(v):
    """一行统计量：n/mean/std/cv/cv_valid/med/p10/p90/p25/p75/spread/norm_spread/cqv。"""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    cv, mu, sd, n = C.cv_of(v)
    med, p10, p90, nn = C.band(v)
    p75, p25 = p75p25(v)
    cqv = ((p75 - p25) / (p75 + p25)) if (np.isfinite(p75) and np.isfinite(p25)
                                          and abs(p75 + p25) > 1e-9) else np.nan
    return dict(n=nn, mean=mu, std=sd, cv=cv,
                cv_valid=bool(np.isfinite(cv) and np.isfinite(mu) and mu > 0.05),
                med=med, p10=p10, p90=p90, p25=p25, p75=p75,
                spread=(p90 - p10), norm_spread=((p90 - p10) / med if med else np.nan),
                cqv=cqv, lo=(float(np.min(v)) if nn else np.nan),
                hi=(float(np.max(v)) if nn else np.nan),
                rng=(float(np.max(v) - np.min(v)) if nn else np.nan))


def shape_at(grid_fine, key, t_on, taus):
    g = grid_fine[(grid_fine["key"] == key) & np.isclose(grid_fine["t_on"], t_on)].sort_values("tau")
    if g.empty:
        return np.full(len(taus), np.nan)
    return np.interp(taus, g["tau"].to_numpy(float), g["f"].to_numpy(float))


def main():
    C.start_log("t3a_01_shape")
    grid = pd.read_csv(os.path.join(C.RES, "t3a_shape_grid.csv"))
    grid["uid"] = grid["key"] + "@" + grid["t_on"].map(lambda x: "%.2f" % x)
    aud = pd.read_csv(os.path.join(C.RES, "t3a_inlier_audit.csv"))
    ev = C.load_t4a_events()
    ev["uid"] = ev["key"] + "@" + ev["t_on"].map(lambda x: "%.2f" % x)
    load = ev[ev["is_load"]].merge(
        aud[["uid", "J_ok", "posdef", "usable", "f_min", "f_max", "J_over_peak"]], on="uid", how="left")
    grid_f = grid[grid["grid"] == C.TAU_FINE_LABEL]
    grid_m = grid[grid["grid"] == C.TAU_MAIN_LABEL]
    print("装载类事件 n=%d（onset %d / restep %d）" %
          (len(load), (load["kind"] == "onset").sum(), (load["kind"] == "restep").sum()))
    strata = [("all", load),
              ("J_ok", load[load["J_ok"]]),
              ("usable", load[load["usable"]])]
    for sname, sub in strata:
        print("  样本 %-7s n=%-3d（onset %d / restep %d）" %
              (sname, len(sub), (sub["kind"] == "onset").sum(), (sub["kind"] == "restep").sum()))
    print("  （usable = |J|≥2%%峰值 & clean & f(τ)∈[0,1.5]；判据冻结在 results/t3a_inlier_audit.csv）")
    print("主 τ 网格 = T4-A 的 z_at_* 采样点；base = T4-A 的 pre 窗中位\n")

    # ══════════════ Q1 形状散布 ══════════════
    print("══ Q1 归一化快相形状的同类稳定性（CV / CQV / p10~p90）══")
    rows = []
    for kind in ["onset", "restep"]:
        for sname, sub in strata:
            uids = set(sub[sub["kind"] == kind]["uid"])
            for t in C.TAU_MAIN:
                v = grid_m[(grid_m["kind"] == kind) & np.isclose(grid_m["tau"], t)
                           & (grid_m["uid"].isin(uids))]["f"].to_numpy(float)
                r = qstats(v)
                r.update(kind=kind, sample=sname, grid=C.TAU_MAIN_LABEL, tau=t)
                rows.append(r)
    spread_df = pd.DataFrame(rows)[
        ["kind", "sample", "grid", "tau", "n", "mean", "std", "cv", "cv_valid",
         "med", "p10", "p90", "p25", "p75", "spread", "norm_spread", "cqv", "lo", "hi", "rng"]]
    spread_df.to_csv(os.path.join(C.RES, "t3a_shape_spread.csv"), index=False, encoding="utf-8-sig")
    for kind in ["onset", "restep"]:
        for sname, _ in strata:
            s = spread_df[(spread_df["kind"] == kind) & (spread_df["sample"] == sname)]
            if s.empty:
                continue
            print("  %s / %s（n=%d）:" % (kind, sname, int(s["n"].iloc[0])))
            print("    τ(s)     " + "".join("%10.2f" % t for t in C.TAU_MAIN))
            print("    中位     " + "".join("%10.3f" % x for x in s["med"]))
            print("    p10~p90  " + "".join("%10s" % ("%.3f~%.3f" % (a, b))
                                           for a, b in zip(s["p10"], s["p90"])))
            print("    spread   " + "".join("%10.3f" % x for x in s["spread"]))
            print("    CQV      " + "".join("%10s" % ("%.3f" % x if np.isfinite(x) else "n/a")
                                           for x in s["cqv"]))
            print("    CV       " + "".join("%10s" % ("%.3f" % x if np.isfinite(x) else "n/a")
                                           for x in s["cv"]))
            print("    极差     " + "".join("%10.3f" % x for x in s["rng"]))
    print("\n  写加密网格分位曲线（onset/restep × 样本）…")
    crow = []
    for kind in ["onset", "restep"]:
        for sname, sub in strata:
            uids = set(sub[sub["kind"] == kind]["uid"])
            gg = grid_f[grid_f["uid"].isin(uids)]
            for t, g in gg.groupby("tau"):
                v = g["f"].to_numpy(float)
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                crow.append(dict(kind=kind, sample=sname, tau=float(t), n=int(v.size),
                                 med=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                 p90=float(np.percentile(v, 90)),
                                 p25=float(np.percentile(v, 25)),
                                 p75=float(np.percentile(v, 75))))
    curves = pd.DataFrame(crow)
    curves.to_csv(os.path.join(C.RES, "t3a_shape_curves.csv"), index=False, encoding="utf-8-sig")
    print("  -> results/t3a_shape_spread.csv (%d 行)、t3a_shape_curves.csv (%d 行)\n"
          % (len(spread_df), len(curves)))

    # ══════════════ Q2 形状参数化 ══════════════
    print("══ Q2 形状参数化的参数 CV 与推荐参数 ══")
    prm = []
    for _, e in load.iterrows():
        f5 = shape_at(grid_f, e["key"], e["t_on"], C.FIT_TAU)
        f4a = shape_at(grid_f, e["key"], e["t_on"], C.FIT_TAU_T4A)
        t1, r1 = C.fit_single_tau(C.FIT_TAU, f5)
        te, be, re = C.fit_stretched_exp(C.FIT_TAU, f5)
        pp, tp, rp = C.fit_power(C.FIT_TAU, f5)
        ta, tb, w, r2t, ok2 = C.fit_two_tau(C.FIT_TAU, f5)
        _, r1a = C.fit_single_tau(C.FIT_TAU_T4A, f4a)
        tea, bea, rea = C.fit_stretched_exp(C.FIT_TAU_T4A, f4a)
        prm.append(dict(uid=e["uid"], key=e["key"], kind=e["kind"], dom=e["dom"], fam=e["fam"],
                        clean=bool(e["clean"]), usable=bool(e["usable"]),
                        jump=float(e["jump"]), J_over_peak=float(e["J_over_peak"]),
                        tau1=t1, tau1_rms=r1, tau_e=te, beta=be, exp_rms=re,
                        p_pow=pp, tau_pow=tp, pow_rms=rp,
                        tau_a=ta, tau_b=tb, w_fast=w, two_tau_rms=r2t, two_tau_ok=ok2,
                        tau1_rms_t4awin=r1a, tau_e_t4awin=tea, beta_t4awin=bea,
                        exp_rms_t4awin=rea,
                        f_at_03=float(e["z_at_03"]), f_at_05=float(e["z_at_05"]),
                        f_at_10=float(e["z_at_10"])))
    pdf = pd.DataFrame(prm)
    pdf.to_csv(os.path.join(C.RES, "t3a_param_events.csv"), index=False, encoding="utf-8-sig")

    prow = []
    params = ["tau1", "tau_e", "beta", "p_pow", "tau_pow", "tau_a", "tau_b", "w_fast"]
    for kind in ["onset", "restep"]:
        for sname, sub in strata:
            uids = set(sub[sub["kind"] == kind]["uid"])
            psub = pdf[pdf["uid"].isin(uids)]
            for p in params:
                r = qstats(psub[p].to_numpy(float))
                r.update(kind=kind, sample=sname, param=p, ptype="拟合参数")
                prow.append(r)
            for col in ["tau1_rms", "exp_rms", "pow_rms", "two_tau_rms"]:
                r = qstats(psub[col].to_numpy(float))
                r.update(kind=kind, sample=sname, param=col, ptype="拟合RMS(模型选择用)")
                prow.append(r)
            for col, lab in [("f_at_03", "f(0.30s)"), ("f_at_05", "f(0.50s)"), ("f_at_10", "f(1.00s)")]:
                r = qstats(psub[col].to_numpy(float))
                r.update(kind=kind, sample=sname, param=lab, ptype="分布型(非拟合)")
                prow.append(r)
    pcv = pd.DataFrame(prow)[["kind", "sample", "param", "ptype", "n", "mean", "std", "cv",
                              "cv_valid", "med", "p10", "p90", "p25", "p75",
                              "spread", "norm_spread", "cqv", "lo", "hi", "rng"]]
    pcv.to_csv(os.path.join(C.RES, "t3a_param_cv.csv"), index=False, encoding="utf-8-sig")
    for kind in ["onset", "restep"]:
        s = pcv[(pcv["kind"] == kind) & (pcv["sample"] == "usable")].sort_values("cqv")
        print("  %s / usable 样本（按 CQV 升序；n=%d）:" % (kind, int(s["n"].max())))
        for _, r in s.iterrows():
            print("    %-16s %-16s CQV=%-7s CV=%-8s 中位=%-9.4f p10~p90=%.3f~%.3f  n=%d"
                  % (r["param"], r["ptype"], ("%.3f" % r["cqv"]) if np.isfinite(r["cqv"]) else "n/a",
                     ("%.3f" % r["cv"]) if np.isfinite(r["cv"]) else "n/a", r["med"],
                     r["p10"], r["p90"], int(r["n"])))
    print("  -> results/t3a_param_cv.csv (%d 行)、t3a_param_events.csv (%d 行)\n"
          % (len(pcv), len(pdf)))

    # ══════════════ Q4 泛化 ══════════════
    print("══ Q4 形状库泛化：乐观（自身标定）vs 悲观（留一 / 跨族）══")
    fine_tau = C.TAU_FINE
    curve_cache = {}

    def lib_of(uids):
        k = tuple(sorted(uids))
        if k in curve_cache:
            return curve_cache[k]
        gg = grid_f[grid_f["uid"].isin(uids)]
        if gg.empty:
            curve_cache[k] = None
            return None
        piv = gg.pivot_table(index="tau", values="f", aggfunc="median").sort_index()
        lib = np.interp(fine_tau, piv.index.to_numpy(float), piv.iloc[:, 0].to_numpy(float))
        curve_cache[k] = lib
        return lib

    def invert(fobs_fn, lib, tau_d, mode):
        if lib is None:
            return np.nan
        if mode == "single":
            taus = np.array([tau_d])
        else:
            taus = fine_tau[(fine_tau >= 0.2 - 1e-9) & (fine_tau <= tau_d + 1e-9)]
            if taus.size < 3:
                return np.nan
        fo = np.array([fobs_fn(t) for t in taus], float)
        fl = np.array([np.interp(t, fine_tau, lib) for t in taus], float)
        m = np.isfinite(fo) & np.isfinite(fl)
        if m.sum() < 1:
            return np.nan
        if mode == "single":
            return float(fo[m][0] / fl[m][0]) if abs(fl[m][0]) > 1e-9 else np.nan
        den = float(np.sum(fl[m] ** 2))
        return float(np.sum(fo[m] * fl[m]) / den) if den > 1e-12 else np.nan

    gen_rows = []
    for _, e in load.iterrows():
        tu, Xu, Z, Zs_obs, pkt, _ = C.load_grid(e["key"])
        k = int(e["k_on"])
        pre, J = float(e["pre"]), float(e["jump"])
        i5 = k + int(round(5.0 / C.DT))
        J5 = (Zs_obs[i5] - pre) if (i5 < len(Zs_obs) and np.isfinite(J)) else np.nan

        def fobs(t, _k=k, _pre=pre, _J=J):
            i = _k + int(round(t / C.DT))
            if i >= len(Zs_obs) or not np.isfinite(_J) or abs(_J) < 1e-12:
                return np.nan
            return float((Zs_obs[i] - _pre) / _J)

        same_fam = set(load[(load["fam"] == e["fam"]) & (load["kind"] == e["kind"])]["uid"])
        same_rec = set(load[(load["key"] == e["key"]) & (load["kind"] == e["kind"])]["uid"])
        other_fam = set(load[(load["fam"] != e["fam"]) & (load["kind"] == e["kind"])]["uid"])
        kinds_other = set(load[load["kind"] != e["kind"]]["uid"])
        # 只用 usable 事件构造库（避免病态形状污染"别人的形状"）
        use_ok = set(load[load["usable"]]["uid"])
        scen = {
            "own": same_fam & use_ok,
            "loo_event": (same_fam & use_ok) - {e["uid"]},
            "loo_rec_broad": same_fam - same_rec,
            "loo_rec": (same_fam & use_ok) - same_rec,
            "cross_fam": other_fam & use_ok,
            "cross_kind": kinds_other & use_ok,
        }
        for sname, uids in scen.items():
            if len(uids) == 0:
                continue
            lib = lib_of(uids)
            for tau_d in TAU_D:
                for mode, mname in [("single", "single@tau"), ("win", "winLS_0.2_tau")]:
                    a_hat = invert(fobs, lib, tau_d, mode)
                    for ref, jref in [("post", J), ("5s", J5)]:
                        if not np.isfinite(a_hat) or not np.isfinite(jref) or abs(jref) < 1e-12:
                            continue
                        gen_rows.append(dict(uid=e["uid"], key=e["key"], kind=e["kind"], dom=e["dom"],
                                             fam=e["fam"], clean=bool(e["clean"]),
                                             usable=bool(e["usable"]), J_ok=bool(e["J_ok"]),
                                             scenario=sname, n_lib=len(uids), tau_d=tau_d,
                                             inv=mname, ref=ref, A_hat=a_hat, J=J, J_ref=jref,
                                             err_pct=(a_hat * J / jref - 1.0) * 100.0))
    gdf = pd.DataFrame(gen_rows)
    gdf.to_csv(os.path.join(C.RES, "t3a_generalization_raw.csv"), index=False, encoding="utf-8-sig")

    srows = []
    for sname, ssel in [("all", gdf), ("usable", gdf[gdf["usable"]]), ("J_ok", gdf[gdf["J_ok"]])]:
        for (sc, kd, td, iv, rf), g in ssel.groupby(["scenario", "kind", "tau_d", "inv", "ref"]):
            srows.append(_sum(g, sname, sc, kd, td, iv, rf))
        for (sc, td, iv, rf), g in ssel.groupby(["scenario", "tau_d", "inv", "ref"]):
            srows.append(_sum(g, sname, sc, "both", td, iv, rf))
    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(C.RES, "t3a_generalization.csv"), index=False, encoding="utf-8-sig")

    for sname in ["all", "usable"]:
        print("  【%s 样本】单点反演 @τ_d=0.20 s，参考 = post 窗（T4-A 口径）:" % sname)
        for kind in ["onset", "restep"]:
            print("   %s:" % kind)
            for sc in ["own", "loo_event", "loo_rec", "cross_fam", "cross_kind"]:
                r = sdf[(sdf["sample"] == sname) & (sdf["scenario"] == sc) & (sdf["kind"] == kind)
                        & (sdf["tau_d"] == 0.20) & (sdf["inv"] == "single@tau") & (sdf["ref"] == "post")]
                if r.empty:
                    continue
                r = r.iloc[0]
                print("     %-11s n=%-3d 中位%+8.2f%%  |误差|中位%7.2f%%  p90%7.2f%%  最大%9.2f%%  >5%%占比 %3.0f%%"
                      % (sc, int(r["n"]), r["med_pct"], r["med_abs_pct"], r["p90_abs_pct"],
                         r["max_abs_pct"], r["share_over_5pct"]))
    print("  -> results/t3a_generalization.csv (%d 行)、t3a_generalization_raw.csv (%d 行)\n"
          % (len(sdf), len(gdf)))

    tra = []
    for _, e in load.iterrows():
        taus = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00])
        f = shape_at(grid_f, e["key"], e["t_on"], taus)
        for t, v in zip(taus, f):
            tra.append(dict(uid=e["uid"], key=e["key"], kind=e["kind"], dom=e["dom"], fam=e["fam"],
                            clean=bool(e["clean"]), usable=bool(e["usable"]), t_on=float(e["t_on"]),
                            jump=float(e["jump"]), J_over_peak=float(e["J_over_peak"]),
                            pre_over_peak=float(e["pre_over_peak"]), peak=float(e["peak"]),
                            tau=t, f=v))
    pd.DataFrame(tra).to_csv(os.path.join(C.RES, "t3a_shape_trajectory.csv"), index=False,
                             encoding="utf-8-sig")
    print("  -> results/t3a_shape_trajectory.csv (%d 行)" % len(tra))
    print("\n完成。下一步：python scripts/t3a_02_strata.py")


def _sum(g, sname, sc, kd, td, iv, rf):
    v = g["err_pct"].to_numpy(float)
    v = v[np.isfinite(v)]
    med, p10, p90, n = C.band(v)
    av = np.abs(v)
    return dict(sample=sname, scenario=sc, kind=kd, tau_d=td, inv=iv, ref=rf, n=n,
                med_pct=med, p10_pct=p10, p90_pct=p90,
                med_abs_pct=float(np.median(av)) if n else np.nan,
                p90_abs_pct=float(np.percentile(av, 90)) if n else np.nan,
                max_abs_pct=float(np.max(av)) if n else np.nan,
                share_over_5pct=float(np.mean(av > 5.0) * 100.0) if n else np.nan)


if __name__ == "__main__":
    main()
