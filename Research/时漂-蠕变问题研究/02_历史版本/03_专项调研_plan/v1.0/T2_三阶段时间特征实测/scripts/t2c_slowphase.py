# -*- coding: utf-8 -*-
"""T2-C：快相→慢相「交接点」三口径对照（T2-Q3）+ 慢相爬升形状的模型比较与泛化（T2-Q4）。

口径：
  A `t90`        —— 原始读数首次达到 pre+0.90·J（指标字典 §2.2 主判）
  B `t95`        —— 同上 0.95
  C `t_knee_bi`  —— **双指数速率交汇点**：把 f(τ) 拟合为 A1(1−e^{−τ/τ1}) + A2(1−e^{−τ/τ2})（τ1<τ2），
                    交接点取两分量**速率相等**处 (A1/τ1)e^{−τ/τ1} = (A2/τ2)e^{−τ/τ2}，
                    解析解 τ = ln(A2·τ1/(A1·τ2)) / (1/τ2 − 1/τ1)

慢相模型（4 种，逐事件拟合 + **跨录制留一泛化**）：
  lin  h = a·u          log  h = a·ln(1+u/b)      pow  h = a·u^b      exp  h = A(1−e^{−u/τ})
  h(u) = (Z̄(t90+u) − Z̄(t90))/J，u 从 t90 起算；拟合窗 [t90, t90+30 s]（统一窗，跨事件可比）
  泛化评估：训练集 = **其它录制**同 kind 的事件（其 h 曲线的中位），预测留出事件的 h 曲线

产物：results/t2_slowphase_fits.csv、t2_slowphase_summary.csv、t2_phase_boundary_conventions.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import t2_common as C

U_FIT = C.SLOW_UNIF          # 30 s 统一慢相窗
U_MIN = 20.0                 # 可用性下限
GRID_U = np.round(np.arange(0.0, U_FIT + 1e-9, 0.1), 3)          # 统一 30 s 窗（0.1 s 步长）
GRID_LONG = np.round(np.arange(0.0, 120.0 + 1e-9, 0.5), 3)       # 段级窗（≤120 s，0.5 s 步长）
MODELS = ("lin", "log", "pow", "exp")


def mod(model):
    if model == "lin":
        return (lambda u, a: a * u), [0.01], ([0.0], [10.0])
    if model == "log":
        return (lambda u, a, b: a * np.log1p(u / b)), [0.05, 5.0], ([-10.0, 0.01], [10.0, 1000.0])
    if model == "pow":
        return (lambda u, a, b: a * np.power(np.maximum(u, 1e-6), b)), [0.05, 0.5], ([0.0, 0.0], [10.0, 3.0])
    if model == "exp":
        return (lambda u, a, t: a * (1.0 - np.exp(-u / t))), [0.05, 10.0], ([0.0, 0.2], [10.0, 100000.0])
    raise ValueError(model)


def fit(model, u, h):
    g, p0, (lo, hi) = mod(model)
    u = np.asarray(u, float)
    h = np.asarray(h, float)
    m = np.isfinite(u) & np.isfinite(h)
    if m.sum() < 4:
        return None
    try:
        p, _ = curve_fit(g, u[m], h[m], p0=p0, bounds=(lo, hi), maxfev=40000)
    except Exception:
        return None
    r = g(u[m], *p) - h[m]
    return dict(params=[float(x) for x in p], rms_pct=100 * float(np.sqrt(np.mean(r ** 2))),
                r2_in=float(1 - np.sum(r ** 2) / max(np.sum((h[m] - h[m].mean()) ** 2), 1e-18)),
                r2_zero=float(1 - np.sum(r ** 2) / max(np.sum(h[m] ** 2), 1e-18)),
                gm=g, u=u, h=h)


def predict(model, params, u):
    g, _, _ = mod(model)
    return g(np.asarray(u, float), *params)


def knee_bi(tau, f):
    """双指数速率交汇点（口径 C）。返回 (t_knee, tau1, tau2, A1, A2, rms)。"""
    def g(t, A1, t1, A2, t2):
        return A1 * (1.0 - np.exp(-t / t1)) + A2 * (1.0 - np.exp(-t / t2))

    m = np.isfinite(f) & (tau > 0)
    if m.sum() < 6:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        p, _ = curve_fit(g, tau[m], f[m], p0=[0.6, 0.2, 0.4, 50.0],
                         bounds=([0.0, 0.005, 0.0, 1.0], [2.0, 5.0, 2.0, 5000.0]), maxfev=60000)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    A1, t1, A2, t2 = [float(x) for x in p]
    if t1 >= t2:                     # 保证 τ1 < τ2（否则交换，交汇点定义不变）
        A1, t1, A2, t2 = A2, t2, A1, t1
    r = float(np.sqrt(np.mean((g(tau[m], *p) - f[m]) ** 2)))
    try:
        num = np.log(A2 * t1 / (A1 * t2))
        den = (1.0 / t2 - 1.0 / t1)
        tk = float(num / den)
        if not np.isfinite(tk) or tk <= 0 or tk > 1e4:
            tk = np.nan
    except Exception:
        tk = np.nan
    return tk, t1, t2, A1, A2, r


def main():
    C.start_log("t2c_slowphase")
    print("== T2-C 交接点三口径 + 慢相形状模型 ==")
    all_d = C.load_all()
    ev = C.load_events()
    ph = pd.read_csv(C.os.path.join(C.RES, "t2_phase_times.csv"))
    ph["_k"] = ph.key + "|" + ph.t_on.round(2).astype(str)

    # ── 逐事件：细网格 f(τ)（原始读数）与慢相 h(u)（Z̄）──
    recs = []
    hcurves = {}
    for _, e in ev.iterrows():
        d = all_d[e["key"]]
        k = int(e["k_on"])
        t0 = float(d["tu"][k])
        t_ax, y_raw, ybar, _ = C.axis_series(d, "grid")
        y3 = d["Z3"]
        pre, post, J = C.j_of_idx(ybar, k)
        t_rec = float(t_ax[-1])
        nxt = ev[(ev.key == e["key"]) & (ev.t_on > t0 + 0.1)].t_on
        t_next = float(nxt.min()) if len(nxt) else np.inf
        t_end = min(t_next - C.GUARD_NEXT, t0 + C.SLOW_CAP, t_rec - C.GUARD_NEXT)
        # 段级末端：下一「卸载/部分卸载」沿（= 第一轮 phases.csv 的段末口径）；没有则到录制末
        un = ev[(ev.key == e["key"]) & (ev.kind.isin(["unload", "partial_unload"]))
                & (ev.t_on > t0 + 0.1)].t_on
        t_seg_end = min(float(un.min()) - C.GUARD_NEXT, t_rec - C.GUARD_NEXT) if len(un) \
            else (t_rec - C.GUARD_NEXT)
        # 口径 A/B（主信号 = Z3，与主表一致）
        t90 = C.cross_time(t_ax, y3, t0, pre, J, 0.90)
        t95 = C.cross_time(t_ax, y3, t0, pre, J, 0.95)
        # 口径 C：双指数速率交汇（τ ∈ [0.05, min(30, t_end−t0)]）
        tmax_c = min(30.0, max(0.0, t_end - t0))
        tau_c = np.round(np.arange(0.05, tmax_c + 1e-9, 0.05), 3)
        f_c = np.array([C.f_at(t_ax, y3, t0, pre, J, x) for x in tau_c], float)
        tk, t1, t2, A1, A2, rbi = knee_bi(tau_c, f_c)
        r = dict(key=e["key"], rec=e["rec"], fam=e["fam"], dom=e["dom"], kind=e["kind"],
                 clean=bool(e["clean"]), t_on=round(t0, 3), jump=J, t90=t90, t95=t95,
                 t90_sus=C.cross_time(t_ax, y3, t0, pre, J, 0.90, sustain=0.10),
                 t_knee_bi=tk, bi_tau1=t1, bi_tau2=t2, bi_A1=A1, bi_A2=A2, bi_rms=rbi,
                 knee_win_s=tmax_c, t_seg_end=t_seg_end, seg_win_s=max(0.0, t_seg_end - t0 - t90))
        # 慢相 h(u)：Z̄ 上，从 t90 起算；两套窗（段级 segend / 统一 30 s）
        if np.isfinite(t90):
            z0 = float(np.interp(t0 + t90, t_ax, ybar))
            for win, cap, umin, step in (("segend", 120.0, 30.0, 0.5), ("win30", U_FIT, U_MIN, 0.1)):
                lim = min(t_seg_end, t0 + t90 + cap) if win == "segend" else \
                    min(t_end, t0 + t90 + cap)
                u = (GRID_LONG if win == "segend" else GRID_U) + t90
                ok = (t0 + u) <= lim
                if ok.sum() >= int(umin / step):
                    uu = u[ok] - t90
                    h = (np.interp(t0 + t90 + uu, t_ax, ybar) - z0) / J
                    hcurves[(e["key"], round(t0, 3), win)] = dict(
                        u=uu, h=h, kind=e["kind"], key=e["key"], fam=e["fam"], dom=e["dom"],
                        clean=bool(e["clean"]), t90=t90, window=win)
                    if win == "segend":
                        r["U_segend"] = float(uu[-1])
                        r["slow_amp_segend_pct"] = 100 * float(h[-1])
                        r["slow_rate_segend_pct_s"] = 100 * float(h[-1]) / float(uu[-1])
                    else:
                        r["U_slow"] = float(uu[-1])
                        r["slow_amp_30_pct"] = 100 * float(h[-1])
                        r["slow_rate_30_pct_s"] = 100 * float(h[-1]) / float(uu[-1])
        recs.append(r)
    bdf = pd.DataFrame(recs)

    # 与主表的 t90/t95 对表（自检；主表 canonical t90 亦为 Z3 首次穿越）
    mg = bdf.merge(ph[["key", "t_on", "t90", "t95", "kind"]].rename(
        columns={"t90": "t90_main", "t95": "t95_main", "kind": "kind_main"}),
        on=["key", "t_on"], how="left")
    d90 = np.nanmax(np.abs(mg.t90 - mg.t90_main))
    print("[自检] 与主表 t90 最大差 = %.2e s（应≈0）" % d90)

    # ── Q3 汇总 ──
    print("\n== Q3 交接点三口径（s）==")
    sumrows = []
    for kind in ("onset", "restep", "unload", "partial_unload", "ALL"):
        s = bdf if kind == "ALL" else bdf[bdf.kind == kind]
        if not len(s):
            continue
        for col in ("t90", "t95", "t_knee_bi", "bi_tau1", "bi_tau2", "bi_rms"):
            m, p10, p90, n = C.band(s[col])
            sumrows.append(dict(kind=kind, metric=col, n=n, med=m, p10=p10, p90=p90))
        print("   %-15s n=%2d  t90=%s   t95=%s   t_knee_bi=%s   (bi_rms 中位 %.3f)" %
              (kind, len(s), C.fmt_band(s.t90, 3), C.fmt_band(s.t95, 3),
               C.fmt_band(s.t_knee_bi, 3), C.band(s.bi_rms)[0]))
    # 三口径两两差
    for kind in ("onset", "restep", "ALL"):
        s = bdf if kind == "ALL" else bdf[bdf.kind == kind]
        dd = (s.t95 - s.t90).dropna()
        dk = (s.t_knee_bi - s.t90).dropna()
        sq = (s.t_knee_bi / s.t90).replace([np.inf, -np.inf], np.nan).dropna()
        if len(dd):
            print("   [%s] t95−t90 中位 %+.2f s（p90 |Δ| %.2f）；t_knee−t90 中位 %+.2f s（n=%d，比值中位 %.2f）"
                  % (kind, np.median(dd), np.percentile(np.abs(dd), 90), np.median(dk) if len(dk) else np.nan,
                     len(dk), np.median(sq) if len(sq) else np.nan))
    pd.DataFrame(sumrows).to_csv(C.os.path.join(C.RES, "t2_phase_boundary_conventions.csv"),
                                 index=False, encoding="utf-8-sig")
    bdf.to_csv(C.os.path.join(C.RES, "t2_boundary_conventions_events.csv"), index=False,
               encoding="utf-8-sig")
    print("  -> results/t2_phase_boundary_conventions.csv, t2_boundary_conventions_events.csv")

    # ── Q4 慢相模型拟合 + 跨录制留一 ──
    print("\n== Q4 慢相模型（窗 A = 段级 [t90, 段末] ≤120 s；窗 B = 统一 [t90, t90+30 s]）==")
    keys = list(hcurves.keys())
    kinds = np.array([hcurves[k]["kind"] for k in keys])
    recs_k = np.array([hcurves[k]["key"] for k in keys])
    wins = np.array([hcurves[k]["window"] for k in keys])
    frows = []
    for i, kk in enumerate(keys):
        cur = hcurves[kk]
        u, h = cur["u"], cur["h"]
        base = dict(key=cur["key"], fam=cur["fam"], dom=cur["dom"], kind=cur["kind"],
                    clean=cur["clean"], t_on=round(kk[1], 3), t90=cur["t90"], window=cur["window"],
                    U=float(u[-1]), amp_end_pct=100 * float(h[-1]),
                    amp_ok=bool(abs(100 * float(h[-1])) <= 100.0))
        for mdl in MODELS:
            r = fit(mdl, u, h)
            if r is None:
                frows.append(dict(base, model=mdl, a=np.nan, b=np.nan, rms_pct_in=np.nan,
                                  r2_in=np.nan, r2_zero=np.nan, rms_pct_loo=np.nan,
                                  n_train=np.nan, rms_pct_val=np.nan))
                continue
            p = r["params"]
            # 跨录制留一：训练 = 其它录制同 kind 同窗的 h 曲线中位
            tr = [j for j in range(len(keys)) if recs_k[j] != cur["key"] and kinds[j] == cur["kind"]
                  and wins[j] == cur["window"]]
            rms_loo = np.nan
            val = np.nan
            if len(tr) >= 3:
                umin = min(hcurves[keys[j]]["u"][-1] for j in tr)
                ggrid = GRID_LONG if cur["window"] == "segend" else GRID_U
                ug = ggrid[ggrid <= umin]
                if len(ug) >= 4:
                    Hm = np.vstack([np.interp(ug, hcurves[keys[j]]["u"], hcurves[keys[j]]["h"])
                                    for j in tr])
                    rt = fit(mdl, ug, np.median(Hm, axis=0))
                    if rt is not None:
                        ug2 = ug[ug <= u[-1]]
                        pred = predict(mdl, rt["params"], ug2)
                        obs = np.interp(ug2, u, h)
                        val = 100 * float(np.sqrt(np.mean((pred - obs) ** 2)))
                        rms_loo = val
            frows.append(dict(base, model=mdl, a=(p[0] if len(p) > 0 else np.nan),
                              b=(p[1] if len(p) > 1 else np.nan), rms_pct_in=r["rms_pct"],
                              r2_in=r["r2_in"], r2_zero=r["r2_zero"], rms_pct_loo=rms_loo,
                              n_train=len(tr), rms_pct_val=val))
    fdf = pd.DataFrame(frows)
    fdf.to_csv(C.os.path.join(C.RES, "t2_slowphase_fits.csv"), index=False, encoding="utf-8-sig")
    print("  -> results/t2_slowphase_fits.csv (%d 行)" % len(fdf))

    srows = []
    for win in ("segend", "win30"):
        for kind in ("onset", "restep", "unload", "partial_unload", "ALL"):
            s0 = fdf[fdf.window == win]
            s0 = s0 if kind == "ALL" else s0[s0.kind == kind]
            for mdl in MODELS:
                q0 = s0[s0.model == mdl]
                if not len(q0):
                    continue
                q = q0[q0.amp_ok]                      # 幅度门：慢相窗内变化 ≤100%·J
                n_excl = int(len(q0) - len(q))
                if not len(q):
                    continue
                m1, p10, p90, n = C.band(q.rms_pct_in)
                m2, p10b, p90b, n2 = C.band(q.rms_pct_loo)
                srows.append(dict(window=win, kind=kind, model=mdl, n=n, n_excl=n_excl,
                                  rms_in_med=m1,
                                  rms_in_p10=p10, rms_in_p90=p90, rms_loo_med=m2,
                                  rms_loo_p10=p10b, rms_loo_p90=p90b, n_loo=n2,
                                  amp_end_med=float(np.median(q.amp_end_pct)),
                                  U_med=float(np.median(q.U)),
                                  a_med=float(np.median(q.a)), b_med=float(np.median(q.b))))
    sdf = pd.DataFrame(srows)
    sdf.to_csv(C.os.path.join(C.RES, "t2_slowphase_summary.csv"), index=False,
               encoding="utf-8-sig")
    print("  -> results/t2_slowphase_summary.csv")
    print("\n   模型对比（rms 单位 = %·J；in=自身数据，loo=跨录制泛化）")
    for win in ("segend", "win30"):
        for kind in ("onset", "restep", "ALL"):
            q0 = sdf[(sdf.window == win) & (sdf.kind == kind)]
            if not len(q0):
                continue
            print("   [%s / %s]  U 中位 %.1f s" % (win, kind, q0.U_med.iloc[0]))
            for mdl in MODELS:
                q = q0[q0.model == mdl]
                if not len(q):
                    continue
                q = q.iloc[0]
                print("      %-4s n=%2d  rms_in 中位 %6.2f%% [%6.2f, %6.2f]   rms_loo 中位 %6.2f%%  "
                      "a≈%.4f b≈%.3f" % (mdl, int(q.n), q.rms_in_med, q.rms_in_p10, q.rms_in_p90,
                                         q.rms_loo_med, q.a_med, q.b_med))
    print("\n完成。")


if __name__ == "__main__":
    main()
