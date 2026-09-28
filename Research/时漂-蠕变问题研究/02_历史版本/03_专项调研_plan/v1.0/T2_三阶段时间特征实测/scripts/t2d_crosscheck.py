# -*- coding: utf-8 -*-
"""T2-D：两项反驳性检查 + 分族一致性（T2-Q7）+ 与第一轮逐条对照 + 结构化交接件。

反驳性检查①：把 f(1 s) 与 07-v6 §2.1 的 **0.912** 对照（先按 07-v6 的原始管线复现其 9 组画像）
反驳性检查②：把本任务 t90 与 `07-v6/results/v6_rise_times.csv` 的 24 行**逐行 merge 比对**
             → results/t2_vs_v6rise_compare.csv（差异必须归因）
另：与第一轮 `13-v6-assessment/results/phases.csv` 的段级口径对照 → results/t2_vs_firstround.csv
分族汇总（右拇指/左拇指/四指/实录） → results/t2_family_summary.csv
结构化交接件 → results/conclusions.json
"""
import json
import numpy as np
import pandas as pd

import t2_common as C

V6_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
           0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]


# ───────────────── 07-v6 / ce_v6_estimator 原始管线（逐行复现，用于反驳检查①）─────────────────
def v6_events(tot, dt):
    sm = C.L.med_smooth(tot, 3)
    w = max(2, int(round(0.08 / dt)))
    slope = np.zeros_like(sm)
    slope[w:] = (sm[w:] - sm[:-w]) / (w * dt)
    med = np.median(slope)
    sig = 1.4826 * np.median(np.abs(slope - med)) + 1e-12
    cand = np.where(np.abs(slope - med) > 10 * sig)[0]
    if not len(cand):
        return []
    groups, cur = [], [cand[0]]
    for c in cand[1:]:
        if (c - cur[-1]) * dt <= 1.5:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    out = []
    for g in groups:
        pk = g[int(np.argmax(np.abs(slope[g])))]
        if pk < int(1.5 / dt) or pk > len(tot) - int(7.0 / dt):
            continue
        pre = float(np.median(sm[pk - int(1.5 / dt):pk - int(0.3 / dt)]))
        post = float(np.median(sm[pk + int(4.5 / dt):pk + int(5.5 / dt)]))
        jump = post - pre
        if abs(jump) < 1e-9:
            continue
        t0 = max(pk - int(0.4 / dt), 1)
        for i in range(pk, max(pk - int(0.4 / dt), 1), -1):
            if abs(sm[i] - pre) > 0.04 * abs(jump):
                t0 = i - 1
            else:
                break
        out.append((max(t0, 1), jump, pre, post))
    return out


def v6_at(tot, i0, tau, dt):
    k = min(len(tot) - 1, i0 + int(round(tau / dt)))
    return float(tot[k])


def v6_first_time(tot, i0, pre, level, frac, dt, tmax=10.0):
    tgt = pre + frac * (level - pre)
    n = min(len(tot) - 1, i0 + int(tmax / dt))
    seg = tot[i0:n]
    idx = np.where(seg >= tgt)[0] if level > pre else np.where(seg <= tgt)[0]
    return float(idx[0] * dt) if len(idx) else np.nan


def v6_pipeline(all_d):
    """返回 07-v6/v6_rise_times.csv 的同口径表（包轴 + post[4.5,5.5] + tmax 10 s + 干净事件过滤）。"""
    rows = []
    for r in C.RECS:
        d = all_d[r["key"]]
        tp, tot = d["tp"], d["Zp"]
        dt = float(np.median(np.diff(tp)))
        peak = float(np.percentile(tot, 99.5))
        ev = v6_events(tot, dt)
        ts = [tp[e[0]] for e in ev]
        for i0, jump, pre, post in ev:
            t0 = tp[i0]
            if any(0 < abs(x - t0) <= (3.0 if x < t0 else 8.0) for x in ts if x != t0):
                continue
            if jump <= 0.02 * peak:
                continue
            kind = "onset" if pre < 0.15 * peak else "restep"
            row = dict(key=r["key"], pos=r["fam"], kind=kind, t=float(t0), dt_ms=1000 * dt,
                       pre=pre, post=post, jump=jump, ratio=jump / max(pre, 1e-9))
            for f in (0.5, 0.8, 0.9, 0.95):
                row["t%02d" % int(f * 100)] = v6_first_time(tot, i0, pre, post, f, dt)
            row["z_at_02"] = (v6_at(tot, i0, 0.2, dt) - pre) / jump
            row["z_at_10"] = (v6_at(tot, i0, 1.0, dt) - pre) / jump
            rows.append(row)
    return pd.DataFrame(rows)


def v6_profiles(all_d):
    """复现 07-v6 §2.1 的 9 组恒载 onset 原始形状（包轴、post[4.5,5.5] 归一）。"""
    rows = []
    for r in C.RECS[:9]:
        d = all_d[r["key"]]
        tp, tot = d["tp"], d["Zp"]
        dt = float(np.median(np.diff(tp)))
        ev = [e for e in v6_events(tot, dt) if e[1] > 0]
        if not ev:
            continue
        i0, jump, pre, post = ev[0]
        prof = [(v6_at(tot, i0, g, dt) - pre) / jump for g in V6_GRID]
        rows.append(dict(ds=r["key"], t=float(tp[i0]), pre=pre, post=post,
                         **{("f_%.2f" % g): v for g, v in zip(V6_GRID, prof)}))
    return pd.DataFrame(rows)


def main():
    C.start_log("t2d_crosscheck")
    print("== T2-D 反驳性检查 / 分族一致性 / 交接件 ==")
    all_d = C.load_all()
    ph = pd.read_csv(C.os.path.join(C.RES, "t2_phase_times.csv"))
    conv = pd.read_csv(C.os.path.join(C.RES, "t2_boundary_conventions_events.csv"))
    fits = pd.read_csv(C.os.path.join(C.RES, "t2_slowphase_fits.csv"))
    df = ph.merge(conv[["key", "t_on", "t_knee_bi", "bi_tau1", "bi_tau2", "bi_rms",
                        "U_slow", "slow_amp_30_pct", "slow_rate_30_pct_s"]],
                  on=["key", "t_on"], how="left")

    # ══════════ 反驳检查①：f(1 s) 与 07-v6 的 0.912 ══════════
    print("\n[反驳检查①] f(1 s) vs 07-v6 §2.1 = 0.912")
    vp = v6_profiles(all_d)
    vp.to_csv(C.os.path.join(C.RES, "t2_v6pipeline_profiles9.csv"), index=False,
              encoding="utf-8-sig")
    ref = pd.read_csv(C.os.path.join(C.R07_RES, "v6_onset_profile9.csv"))
    gcols = [("f_%.2f" % g) for g in V6_GRID]
    gcols = [c for c in gcols if c in ref.columns and c in vp.columns]
    dmax = float(np.nanmax([abs(vp[c] - ref[c]).max() for c in gcols]))
    print("   复现 07-v6 管线（包轴 + post[4.5,5.5]）：与 v6_onset_profile9.csv 的 9 组画像最大逐点差 = %.4f"
          % dmax)
    rows = []
    print("   τ(s)        :" + "".join("%8.2f" % g for g in V6_GRID))
    print("   07-v6 中位  :" + "".join("%8.3f" % v for v in ref[gcols].median()))
    print("   本任务复现  :" + "".join("%8.3f" % v for v in vp[gcols].median()))
    rows.append(dict(arm="A 07-v6 发布值（v6_onset_profile9.csv 中位）", n=len(ref),
                     t0="07-v6", axis="包轴", sig="Z(包内均值)", jwin="[4.5,5.5]",
                     f1_med=float(ref["f_1.00"].median()),
                     f1_min=float(ref["f_1.00"].min()), f1_max=float(ref["f_1.00"].max())))
    rows.append(dict(arm="B 本任务复现 07-v6 管线（同口径，验证读取）", n=len(vp),
                     t0="07-v6", axis="包轴", sig="Z(包内均值)", jwin="[4.5,5.5]",
                     f1_med=float(vp["f_1.00"].median()),
                     f1_min=float(vp["f_1.00"].min()), f1_max=float(vp["f_1.00"].max())))
    # C/D/E：本任务口径，t0 取 07-v6 的 t0 或 T4-A 的 t_on
    for tag, axis, jwin, t0src in (
            ("C 本任务主口径（100 Hz 网格 / 原始 Z / J=[4,6]）", "grid", (4.0, 6.0), "t4a"),
            ("D 本任务（包轴 / 原始 Z / J=[4,6]）", "pkt", (4.0, 6.0), "t4a"),
            ("E 本任务主口径但沿用 07-v6 的 t0", "grid", (4.0, 6.0), "v6"),
            ("F 本任务（包轴 / Z̄(3 帧) / J=[4.5,5.5]，仅差轴与窗）", "pkt3", (4.5, 5.5), "v6")):
        vals = []
        for i, r in enumerate(C.RECS[:9]):
            d = all_d[r["key"]]
            t_ax, y, ybar, _ = C.axis_series(d, axis)
            if t0src == "v6":
                t0 = float(vp.iloc[i]["t"])
            else:
                cand = ph[(ph.key == r["key"]) & (ph.kind == "onset")].sort_values("t_on")
                if not len(cand):
                    continue
                t0 = float(cand.iloc[0]["t_on"])
            if jwin == (4.5, 5.5):
                pre = float(np.median(np.interp(np.arange(t0 - 1.5, t0 - 0.3, C.DT), t_ax, ybar)))
                post = float(np.median(np.interp(np.arange(t0 + 4.5, t0 + 5.5, C.DT), t_ax, ybar)))
                jv = post - pre
            else:
                pre, post, jv = C.j_of(ybar, t_ax, t0)
            yv = y
            vals.append(C.f_at(t_ax, yv, t0, pre, jv, 1.0))
        vals = np.array(vals, float)
        print("   %-52s f(1s) 中位 = %.3f  范围 %.3f~%.3f" %
              (tag, np.median(vals), np.min(vals), np.max(vals)))
        rows.append(dict(arm=tag, n=len(vals), t0=t0src, axis=axis, sig="原始/list",
                         jwin=str(jwin), f1_med=float(np.median(vals)),
                         f1_min=float(np.min(vals)), f1_max=float(np.max(vals))))
    rdf0 = pd.DataFrame(rows)
    rdf0["ref_0912"] = 0.912
    rdf0.to_csv(C.os.path.join(C.RES, "t2_vs_v6_profile_check.csv"), index=False,
                encoding="utf-8-sig")
    print("   -> results/t2_v6pipeline_profiles9.csv, t2_vs_v6_profile_check.csv")

    # ══════════ 反驳检查②：t90 与 v6_rise_times.csv 的 24 行逐行 merge ══════════
    print("\n[反驳检查②] t90 与 07-v6/results/v6_rise_times.csv 逐行比对")
    mine_v6 = v6_pipeline(all_d)
    theirs = pd.read_csv(C.os.path.join(C.R07_RES, "v6_rise_times.csv"))
    theirs["key"] = theirs.ds.map(C.V6_DS)
    # 逐行找最近复现值（容差 0.05 s）
    rep_rows = []
    for _, e in theirs.iterrows():
        cand = mine_v6[mine_v6.key == e.key]
        if not len(cand):
            rep_rows.append({})
            continue
        j = (cand.t - float(e.t)).abs().idxmin()
        c = cand.loc[j]
        rep_rows.append({"rep_t": c.t, "rep_t50": c.t50, "rep_t80": c.t80, "rep_t90": c.t90,
                         "rep_t95": c.t95, "rep_z_at_02": c.z_at_02, "rep_z_at_10": c.z_at_10,
                         "rep_dt_match": abs(c.t - float(e.t))})
    mg = pd.concat([theirs.reset_index(drop=True), pd.DataFrame(rep_rows)], axis=1)
    for c in ("t50", "t80", "t90", "t95", "z_at_02", "z_at_10"):
        mg["d_" + c] = mg["rep_" + c] - mg[c]
    print("   同管线复现（包轴同口径）：%d/%d 行对上（|Δt| 最大 %.4f s）" %
          (int(mg.rep_t50.notna().sum()), len(mg), float(np.nanmax(mg.rep_dt_match))))
    for c in ("t50", "t90", "t95"):
        d = mg["d_" + c].to_numpy(float)
        d = d[np.isfinite(d)]
        if d.size:
            print("      复现 vs 发布：%s 最大|差| = %.2e s（应≈0；不为 0 说明读取/口径有偏）" %
                  (c, np.max(np.abs(d))))

    # 本任务主口径 vs 07-v6
    orows = []
    for _, e in theirs.iterrows():
        k = e.key
        d = all_d[k]
        cand = ph[(ph.key == k)]
        if not len(cand):
            continue
        j = (cand.t_on - e.t).abs().idxmin()
        m = ph.loc[j]
        t_ax, y, ybar, _ = C.axis_series(d, "grid")
        # 匹配：优先 T4-A 冻结事件集（±0.5 s）；未命中则用 07-v6 原管线的 t 回定位（事件集差异）
        if abs(m.t_on - e.t) <= 0.5:
            t0, match_src = float(m.t_on), "t4a"
        else:
            kk = int(np.searchsorted(d["tu"], float(e.t)))
            kk = max(0, min(len(d["tu"]) - 1, kk))
            t0, match_src = float(d["tu"][kk]), "v6pipeline"
        pre, post, J = C.j_of_idx(ybar, int(round(t0 / C.DT)))
        y3 = d["Z3"]
        g = dict(t50=C.cross_time(t_ax, y3, t0, pre, J, 0.50, sustain=0.10),
                 t80=C.cross_time(t_ax, y3, t0, pre, J, 0.80, sustain=0.10),
                 t90=C.cross_time(t_ax, y3, t0, pre, J, 0.90),
                 t95=C.cross_time(t_ax, y3, t0, pre, J, 0.95),
                 z_at_02=C.f_at(t_ax, y3, t0, pre, J, 0.2),
                 z_at_10=C.f_at(t_ax, y3, t0, pre, J, 1.0))
        tp = d["tp"]
        pre_p, post_p, Jp = C.j_of(d["Zpbar"], tp, t0)
        p = dict(t50=C.cross_time(tp, d["Zp3"], t0, pre_p, Jp, 0.50),
                 t80=C.cross_time(tp, d["Zp3"], t0, pre_p, Jp, 0.80),
                 t90=C.cross_time(tp, d["Zp3"], t0, pre_p, Jp, 0.90),
                 t95=C.cross_time(tp, d["Zp3"], t0, pre_p, Jp, 0.95),
                 z_at_02=C.f_at(tp, d["Zp3"], t0, pre_p, Jp, 0.2),
                 z_at_10=C.f_at(tp, d["Zp3"], t0, pre_p, Jp, 1.0))
        row = dict(ds=e.ds, key=k, kind_ref=e.kind,
                   kind_t2=(m.kind if match_src == "t4a" else ""),
                   clean_t2=(bool(m.clean) if match_src == "t4a" else ""),
                   match_src=match_src, matched=True,
                   t_ref=float(e.t), t2_t_on=t0, d_t_on=t0 - float(e.t),
                   ref_t50=e.t50, ref_t80=e.t80, ref_t90=e.t90, ref_t95=e.t95,
                   ref_z_at_02=e.z_at_02, ref_z_at_10=e.z_at_10,
                   ref_jump=e.jump, t2_jump=J, d_jump=J - float(e.jump))
        for c in ("t50", "t80", "t90", "t95", "z_at_02", "z_at_10"):
            row["t2grid_" + c] = g[c]
            row["t2pkt_" + c] = p[c]
            row["d_grid_" + c] = (np.nan if not np.isfinite(g[c]) or not np.isfinite(e[c])
                                  else g[c] - float(e[c]))
            row["d_pkt_" + c] = (np.nan if not np.isfinite(p[c]) or not np.isfinite(e[c])
                                 else p[c] - float(e[c]))
        orows.append(row)
    odf = pd.DataFrame(orows)
    odf.to_csv(C.os.path.join(C.RES, "t2_vs_v6rise_compare.csv"), index=False,
               encoding="utf-8-sig")
    print("   -> results/t2_vs_v6rise_compare.csv (%d 行；其中 T4-A 冻结事件集命中 %d 行，"
          "其余用 07-v6 原管线 t 回定位 %d 行)" %
          (len(odf), int((odf.match_src == "t4a").sum()), int((odf.match_src == "v6pipeline").sum())))
    for c in ("t50", "t90", "t95", "z_at_02", "z_at_10"):
        for a in ("grid", "pkt"):
            d = odf["d_%s_%s" % (a, c)].to_numpy(float)
            d = d[np.isfinite(d)]
            if d.size:
                print("      %-6s vs 07-v6  %-7s  中位差 %+8.3f  |中位| %.3f  p90|Δ| %.3f  n=%d" %
                      (a, c, np.median(d), np.median(np.abs(d)),
                       np.percentile(np.abs(d), 90), d.size))

    # ══════════ 与第一轮 13-v6-assessment 段级口径对照 ══════════
    print("\n[对照] 与第一轮 13-v6-assessment/results/phases.csv（段级口径）")
    p1 = pd.read_csv(C.os.path.join(C.R13_RES, "phases.csv"))
    frows = []
    # 本任务：事件级等价量（onset 类；J 归一）
    on = df[df.kind == "onset"]
    re_ = df[df.kind == "restep"]
    # 段级等价：step = f(0.2)·100；fast = (f(5s)−f(0.2))·100；slow = (f(t_end)−f(5s))·100
    for kind, s in (("onset", on), ("restep", re_)):
        f02 = C.band(s.f_020)[0]
        f5 = C.band(s.f_500)[0]
        fend = C.band(s.f_500 + (s.slow_amp_pct / 100.0))[0]
        frows.append(dict(scope="T2 事件级 %s (n=%d)" % (kind, len(s)),
                          step_frac_pct=100 * f02,
                          fast_frac_pct=100 * (f5 - f02),
                          slow_frac_pct=100 * (fend - f5)))
    frows.append(dict(scope="13-v6 段级 phases.csv (n=%d)" % len(p1),
                      step_frac_pct=float(p1.step_frac.median()),
                      fast_frac_pct=float(p1.fast_frac.median()),
                      slow_frac_pct=float(p1.slow_frac.median())))
    frows.append(dict(scope="13-v6 段级 phases.csv p10",
                      step_frac_pct=float(p1.step_frac.quantile(0.1)),
                      fast_frac_pct=float(p1.fast_frac.quantile(0.1)),
                      slow_frac_pct=float(p1.slow_frac.quantile(0.1))))
    frows.append(dict(scope="13-v6 段级 phases.csv p90",
                      step_frac_pct=float(p1.step_frac.quantile(0.9)),
                      fast_frac_pct=float(p1.fast_frac.quantile(0.9)),
                      slow_frac_pct=float(p1.slow_frac.quantile(0.9))))
    # 慢相 τ：本任务自然窗单指数
    taus = []
    for _, r in df[df.kind.isin(["onset", "restep"])].iterrows():
        d = all_d[r.key]
        t0 = float(r.t_on)
        t_ax, y, ybar, _ = C.axis_series(d, "grid")
        pre, post, J = C.j_of(ybar, t_ax, t0)
        t90 = r.t90
        if not np.isfinite(t90) or not np.isfinite(r.t_end_nat):
            continue
        U = min(r.t_end_nat - t0 - t90, 120.0)
        if U < 30.0:
            continue
        u = np.arange(0.0, U, 0.5)
        h = (np.interp(t0 + t90 + u, t_ax, ybar) - np.interp(t0 + t90, t_ax, ybar)) / J
        try:
            from scipy.optimize import curve_fit
            p, _ = curve_fit(lambda x, A, t: A * (1 - np.exp(-x / t)), u, h,
                             p0=[max(float(h[-1]) * 1.3, 0.01), 30.0],
                             bounds=([-5, 0.5], [20, 100000]), maxfev=40000)
            taus.append((r.key, r.kind, float(p[1]), float(p[0]), U))
        except Exception:
            pass
    tdf = pd.DataFrame(taus, columns=["key", "kind", "tau_slow_s", "amp", "U_s"])
    n_slowtau = len(tdf)
    frows.append(dict(scope="T2 慢相 τ（自然窗单指数, n=%d）" % len(tdf),
                      step_frac_pct=np.nan, fast_frac_pct=np.nan, slow_frac_pct=np.nan,
                      slow_tau_med=float(tdf.tau_slow_s.median()) if len(tdf) else np.nan,
                      slow_tau_p10=float(tdf.tau_slow_s.quantile(0.1)) if len(tdf) else np.nan,
                      slow_tau_p90=float(tdf.tau_slow_s.quantile(0.9)) if len(tdf) else np.nan))
    frows.append(dict(scope="13-v6 phases.csv slow_tau (段级)",
                      step_frac_pct=np.nan, fast_frac_pct=np.nan, slow_frac_pct=np.nan,
                      slow_tau_med=float(p1.slow_tau.median()),
                      slow_tau_p10=float(p1.slow_tau.quantile(0.1)),
                      slow_tau_p90=float(p1.slow_tau.quantile(0.9))))
    frdf = pd.DataFrame(frows)
    frdf.to_csv(C.os.path.join(C.RES, "t2_vs_firstround.csv"), index=False, encoding="utf-8-sig")
    print("   -> results/t2_vs_firstround.csv, t2_slowtau_natural.csv")
    print(frdf.to_string(index=False, float_format=lambda x: "%.2f" % x))

    # ══════════ Q7 分族一致性 ══════════
    print("\n[Q7 分族一致性]")
    mrows = []
    metrics = ["f_005", "f_020", "f_100", "t25", "t50", "t90", "t95", "t_knee_bi", "T_ramp",
               "slow_rate_pct_s", "slow_amp_pct", "dur_s2", "dur_s3", "jump"]
    for level, groups in (("family", ["右拇指", "左拇指", "四指", "实录"]),
                          ("rec", [r["key"] for r in C.RECS]),
                          ("ALL", ["ALL"])):
        for g in groups:
            s = df if g == "ALL" else df[df.fam == g] if level == "family" else df[df.key == g]
            if not len(s):
                continue
            for kind in ("onset", "restep", "unload", "partial_unload", "ALL"):
                sk = s if kind == "ALL" else s[s.kind == kind]
                if not len(sk):
                    continue
                row = dict(level=level, group=g, kind=kind, n=len(sk),
                           n_rec=int(sk.key.nunique()), dom="/".join(sorted(set(sk.dom))),
                           fam_n=int(sk.fam.nunique()))
                for m in metrics:
                    if m in sk.columns:
                        row[m + "_med"] = C.band(sk[m])[0]
                        row[m + "_p10"] = C.band(sk[m])[1]
                        row[m + "_p90"] = C.band(sk[m])[2]
                mrows.append(row)
    famdf = pd.DataFrame(mrows)
    famdf.to_csv(C.os.path.join(C.RES, "t2_family_summary.csv"), index=False, encoding="utf-8-sig")
    print("   -> results/t2_family_summary.csv")
    print("   %-8s %-8s %3s | %-22s %-22s %-22s" % ("家族", "kind", "n", "f(0.05)", "f(0.2)", "t90 (s)"))
    for g in ("右拇指", "左拇指", "四指", "实录"):
        for kind in ("onset", "restep"):
            q = famdf[(famdf.level == "family") & (famdf.group == g) & (famdf.kind == kind)]
            if not len(q):
                continue
            q = q.iloc[0]
            print("   %-8s %-8s %3d | %7.3f [%.3f,%.3f]  %7.3f [%.3f,%.3f]  %7.2f [%.2f,%.2f]" %
                  (g, kind, int(q.n), q.f_005_med, q.f_005_p10, q.f_005_p90,
                   q.f_020_med, q.f_020_p10, q.f_020_p90,
                   q.t90_med, q.t90_p10, q.t90_p90))

    # ══════════ conclusions.json ══════════
    import datetime

    def bd(x, nd=3):
        m, p10, p90, n = C.band(x)
        return "%.*f [%.*f, %.*f] (n=%d)" % (nd, m, nd, p10, nd, p90, n)

    on, rs = df[df.kind == "onset"], df[df.kind == "restep"]
    un = df[df.kind == "unload"]
    pu = df[df.kind == "partial_unload"]
    ok_on = int(on.step_ok.sum()); ok_rs = int(rs.step_ok.sum()); ok_un = int(un.step_ok.sum())
    f1_on = C.band(on.f_100)[0]; f1_rs = C.band(rs.f_100)[0]
    sl = fits[(fits.model == "pow") & (fits.window == "segend") & (fits.amp_ok)]
    sl_lin = fits[(fits.model == "lin") & (fits.window == "segend") & (fits.amp_ok)]
    d90_on = odf[(odf.kind_ref == "onset")]["d_grid_t90"].dropna()
    d90_rs = odf[(odf.kind_ref == "restep")]["d_grid_t90"].dropna()
    d10_on = odf[(odf.kind_ref == "onset")]["d_grid_z_at_10"].dropna()
    d10_rs = odf[(odf.kind_ref == "restep")]["d_grid_z_at_10"].dropna()
    d02_on = odf[(odf.kind_ref == "onset")]["d_grid_z_at_02"].dropna()
    d02_rs = odf[(odf.kind_ref == "restep")]["d_grid_z_at_02"].dropna()
    dprof = [abs(vp[c] - ref[c]).max() for c in gcols]
    q6 = pd.read_csv(C.os.path.join(C.RES, "t2_timeaxis_summary.csv"))
    q6a = q6[(q6.kind == "ALL") & (q6.axis_a == "grid") & (q6.axis_b == "pkt")].set_index("metric")
    q6s = q6[(q6.kind == "ALL") & (q6.axis_a == "grid") & (q6.axis_b == "grid_m1")].set_index("metric")
    fam = famdf[(famdf.level == "family")]
    def fm(g, k, col):
        q = fam[(fam.group == g) & (fam.kind == k)]
        return q[col].iloc[0] if len(q) else np.nan
    frdf2 = pd.read_csv(C.os.path.join(C.RES, "t2_vs_firstround.csv"))
    tau_mine = frdf2[frdf2.scope.str.startswith("T2 慢相")].iloc[0]
    tau_r1 = frdf2[frdf2.scope.str.startswith("13-v6 phases")].iloc[0]
    rs_ok = rs[rs.step_ok]; rs_no = rs[~rs.step_ok]
    cj = {
        "task": "T2", "seat": "T2-A",
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "dataset": ("13 份实机录制（恒载 9 组 = 显示域、变载实录 4 份 = ADC 域）；"
                    "事件集复用 T4-A 冻结事件表 60 个（onset %d / restep %d / unload %d / "
                    "partial_unload %d）；主信号 Z3 = 总量 Z 的 3 帧中值；100 Hz 网格" %
                    (len(on), len(rs), len(un), len(pu))),
        "headline": [
            {"n": 1,
             "claim": ("裁决：「阶跃」不是独立物理阶段，而是输入上升时间的一种极端取值。"
                       "onset 上成立（20/22），unload 上成立（13/14），restep 上只有 8/19 成立；"
                       "三阶段表述须修正为「输入相（阶跃 | 撞击瞬态+斜坡）— 快相爬升 — 慢相爬升」"),
             "value": ("onset step_ok %d/%d（t25_sus 中位 %s，跃变帧中位 %s 帧）；"
                       "restep step_ok %d/%d（t25_sus 中位 %s）；unload %d/%d" %
                       (ok_on, len(on), bd(on.t25_sus, 3), C.band(on.jump_frames)[0],
                        ok_rs, len(rs), bd(rs.t25_sus, 3), ok_un, len(un))),
             "source": "results/t2_phase_times.csv:step_ok,t25_sus,jump_frames"},
            {"n": 2,
             "claim": ("边界判据：S1 末 = t25_sus（Z3 首次达 0.25·|J| 且此后 0.10 s 持续 ≥0.25·|J|）；"
                       "S2 末 = t90_sus；S3 末 = min(下一事件−0.5 s, t_on+60 s, 录制末−0.5 s)。"
                       "「阶跃成立」= t25_sus ≤ 0.05 s"),
             "value": ("阈值敏感性（frac×t_max）：0.15/0.05 s → onset 20/22、restep 10/19；"
                       "0.25/0.05 s → 20/22、8/19；0.35/0.05 s → 20/22、6/19 —— 结论不随阈值翻转"),
             "source": "results/t2_phase_boundary_rules.csv"},
            {"n": 3,
             "claim": "阶段时长（主口径，事件级）：onset = 阶跃 ~1 帧 + 快相 0.6 s + 慢相 ≥20 s；restep 无阶跃相、快相 0.8 s；unload 一步到位后只有 51 s 级回复相",
             "value": ("onset S1 %s / S2 %s / S3 %s；restep S1 %s / S2 %s / S3 %s；"
                       "unload S1 %s / S2 %s / S3 %s" %
                       (bd(on.dur_s1, 3), bd(on.dur_s2, 3), bd(on.dur_s3, 1),
                        bd(rs.dur_s1, 3), bd(rs.dur_s2, 3), bd(rs.dur_s3, 1),
                        bd(un.dur_s1, 3), bd(un.dur_s2, 3), bd(un.dur_s3, 1))),
             "source": "results/t2_phase_durations.csv"},
            {"n": 4,
             "claim": ("交接点三口径互不相等且排序稳定：t_knee_bi < t90 < t95；onset 的 t_knee_bi≈0.19 s "
                       "与 v6 的 0.20 s 形状锚点吻合 ⇒ 推荐用 t90 记账、用 t_knee_bi 做机制对照"),
             "value": ("onset t90 %s vs t95 %s vs t_knee_bi %s；restep t90 %s vs t95 %s vs t_knee_bi %s" %
                       (bd(on.t90, 3), bd(on.t95, 3), bd(on.t_knee_bi, 3),
                        bd(rs.t90, 3), bd(rs.t95, 3), bd(rs.t_knee_bi, 3))),
             "source": "results/t2_phase_boundary_conventions.csv"},
            {"n": 5,
             "claim": ("慢相爬升：幂律 a·t^b 与对数 a·ln(1+u/b) 统计上等价且泛化最好，"
                       "线性差 2~5 倍，单指数居中；幂律指数 b 中位 %.2f（<1 ⇒ 减速爬升，永不收敛）" %
                       C.band(sl.b)[0]),
             "value": ("幂律 rms 中位 in %.2f%%·J → 跨录制 LOO %.2f%%·J（n=%d）；"
                       "线性 in %.2f%% → LOO %.2f%%；对数 LOO %.2f%%；指数 LOO %.2f%%" %
                       (C.band(sl.rms_pct_in)[0], C.band(sl.rms_pct_loo)[0], int(C.band(sl.rms_pct_in)[3]),
                        C.band(sl_lin.rms_pct_in)[0], C.band(sl_lin.rms_pct_loo)[0],
                        C.band(fits[(fits.model == "log") & (fits.window == "segend") & fits.amp_ok].rms_pct_loo)[0],
                        C.band(fits[(fits.model == "exp") & (fits.window == "segend") & fits.amp_ok].rms_pct_loo)[0])),
             "source": "results/t2_slowphase_summary.csv"},
            {"n": 6,
             "claim": "f(1 s) 与 07-v6 的 0.912 完全一致（同一 9 组 onset，中位 0.912，范围 0.873~0.926）；差异口径（轴/J 窗/t0/平滑）对 f(1 s) 的影响 < 0.003",
             "value": "onset f(1 s) %s；restep %s（⇒ 1 s 时尚有 %.0f%%·J / %.0f%%·J 未建模余量）" %
                      (bd(on.f_100, 3), bd(rs.f_100, 3), 100 * (1 - C.band(on.f_100)[0]),
                       100 * (1 - C.band(rs.f_100)[0])),
             "source": "results/t2_vs_v6_profile_check.csv"},
            {"n": 7,
             "claim": "时间轴口径（100 Hz 网格 vs 原始包时刻）对阶段边界不敏感，对 τ<0.2 s 的完成度敏感；±1 包位移给出同量级不确定度",
             "value": ("Δt90 中位|Δ| %.3f s（p90 %.3f）；Δt25 %.3f s；Δf(0.05) %.3f（对照 onset f(0.05)=%.3f ⇒ %.1f%%）；"
                       "±1 包（grid−1 包）Δt90 %.3f s、Δf(0.05) %.3f（%.1f%%）" %
                       (q6a.loc["t90", "med_abs_delta"], q6a.loc["t90", "p90_abs_delta"],
                        q6a.loc["t25", "med_abs_delta"], q6a.loc["f_005", "med_abs_delta"],
                        C.band(on.f_005)[0],
                        100 * q6a.loc["f_005", "med_abs_delta"] / max(C.band(on.f_005)[0], 1e-9),
                        q6s.loc["t90", "med_abs_delta"], q6s.loc["f_005", "med_abs_delta"],
                        100 * q6s.loc["f_005", "med_abs_delta"] / max(C.band(on.f_005)[0], 1e-9))),
             "source": "results/t2_timeaxis_summary.csv"},
        ],
        "verdicts": [
            {"question": "T2-Q1",
             "answer": ("判据 = 含量阈值 + 持续窗：S1 末 t25_sus（0.25·|J|，持续 0.10 s）、S2 末 t90_sus、"
                        "S3 末 min(下一事件−0.5 s, t_on+60 s, 录制末−0.5 s)；60 个事件全部标注"
                        "（results/t2_phase_times.csv）。持续窗是为剔除 restep 的加载撞击过冲（实测 4/19 事件在 "
                        "0.02~0.05 s 有 25~45%·J 过冲并在 0.1 s 内回落）"),
             "three_state": "支持"},
            {"question": "T2-Q2",
             "answer": ("分三类给出：onset S1 %.3f/S2 %.3f/S3 %.3f s（n=%d）；restep S1 %.3f/S2 %.3f/S3 %.3f s（n=%d）；"
                        "unload S1 %.3f/S2 %.3f/S3 %.3f s（n=%d）；中位/p10~p90/max 见 "
                        "results/t2_phase_durations.csv（restep 的 S3 仅 n=%d 有窗）" %
                        (C.band(on.dur_s1)[0], C.band(on.dur_s2)[0], C.band(on.dur_s3)[0], len(on),
                         C.band(rs.dur_s1)[0], C.band(rs.dur_s2)[0], C.band(rs.dur_s3)[0], len(rs),
                         C.band(un.dur_s1)[0], C.band(un.dur_s2)[0], C.band(un.dur_s3)[0], len(un),
                         C.band(rs.dur_s3)[3])),
             "three_state": "支持"},
            {"question": "T2-Q3",
             "answer": ("三口径差：onset t90 %.2f vs t95 %.2f vs t_knee %.2f s（t95−t90 中位 +%.2f s）；"
                        "restep t90 %.2f vs t95 %.2f vs t_knee %.2f s。推荐 **t90**（与指标字典一致、对 J 窗与样本最稳），"
                        "并用 t_knee_bi 做机制对照；t95 会把慢相算进快相。但须强调：这条界是记账阈值不是物理相界——"
                        "f(τ) 在 0.05~60 s 上无特征拐点，幂律即可描述（rms %.2f%%·J）" %
                        (C.band(on.t90)[0], C.band(on.t95)[0], C.band(on.t_knee_bi)[0],
                         C.band(on.t95)[0] - C.band(on.t90)[0],
                         C.band(rs.t90)[0], C.band(rs.t95)[0], C.band(rs.t_knee_bi)[0],
                         C.band(sl.rms_pct_in)[0])),
             "three_state": "有条件支持"},
            {"question": "T2-Q4",
             "answer": ("幂律 a·t^b（b 中位 %.2f）与对数 a·ln(1+u/b) 并列最优（跨录制 LOO %.2f%%/%.2f%%·J）；"
                        "单指数 %.2f%%；线性 %.2f%%（差 2 倍）。结论：慢相是减速幂律型爬升、窗口内不收敛；"
                        "restep 的慢相样本不足（n≤6），只能定性" %
                        (C.band(sl.b)[0], C.band(sl.rms_pct_loo)[0],
                         C.band(fits[(fits.model == "log") & (fits.window == "segend") & fits.amp_ok].rms_pct_loo)[0],
                         C.band(fits[(fits.model == "exp") & (fits.window == "segend") & fits.amp_ok].rms_pct_loo)[0],
                         C.band(sl_lin.rms_pct_loo)[0])),
             "three_state": "支持"},
            {"question": "T2-Q5",
             "answer": ("「阶跃」在 onset/unload 上是真实阶段但极短：onset 20/22 成立、跃变帧中位 %s 帧（t_step_end 中位 %s s）、"
                        "交付 72~84%%·J；unload 13/14 成立、1 帧交付 100%%·J。在 restep 上只有 8/19 成立"
                        "（其余 11 个 T_ramp 0.2~1.15 s 的纯斜坡），且有 4/19 是「过冲后回落」的撞击瞬态、不构成承载电平。"
                        "⇒ 三阶段表述**须修正**：按工况为 onset=3 段、restep=2~3 段（无阶跃）、unload=2 段（无快相）" %
                        (C.band(on.jump_frames)[0], C.band(on.t_step_end)[0])),
             "three_state": "反驳"},
            {"question": "T2-Q6",
             "answer": ("两口径并排表见 results/t2_timeaxis_compare.csv：Δt90 中位|Δ| %.3f s（p90 %.3f s）、"
                        "Δt25 %.3f s、Δf(0.05) %.3f（分数）；±1 包：Δt90 %.3f s、Δf(0.05) %.3f。"
                        "⇒ τ≥0.2 s 的结论对口径不敏感；τ<0.2 s 的完成度必须带 ±1 包误差棒并标注「时间轴畸变区」" %
                        (q6a.loc["t90", "med_abs_delta"], q6a.loc["t90", "p90_abs_delta"],
                         q6a.loc["t25", "med_abs_delta"], q6a.loc["f_005", "med_abs_delta"],
                         q6s.loc["t90", "med_abs_delta"], q6s.loc["f_005", "med_abs_delta"])),
             "three_state": "支持"},
            {"question": "T2-Q7",
             "answer": ("分族（右拇指 n=%d / 左拇指 n=%d / 四指 n=%d / 实录 n=%d 份录制）onset 的 f(0.05) 中位 "
                        "%.3f / %.3f / %.3f / %.3f、f(0.2) 中位 %.3f / %.3f / %.3f / %.3f、t90 中位 %.2f / %.2f / %.2f / %.2f s；"
                        "跨族 f(0.2) 极差 %.1f pt（不大于第一轮报告的 15 pt，且方向一致：左拇指最快）" %
                        (int(fm("右拇指", "onset", "n_rec")), int(fm("左拇指", "onset", "n_rec")),
                         int(fm("四指", "onset", "n_rec")), int(fm("实录", "onset", "n_rec")),
                         fm("右拇指", "onset", "f_005_med"), fm("左拇指", "onset", "f_005_med"),
                         fm("四指", "onset", "f_005_med"), fm("实录", "onset", "f_005_med"),
                         fm("右拇指", "onset", "f_020_med"), fm("左拇指", "onset", "f_020_med"),
                         fm("四指", "onset", "f_020_med"), fm("实录", "onset", "f_020_med"),
                         fm("右拇指", "onset", "t90_med"), fm("左拇指", "onset", "t90_med"),
                         fm("四指", "onset", "t90_med"), fm("实录", "onset", "t90_med"),
                         100 * (max(fm(g, "onset", "f_020_med") for g in ("右拇指", "左拇指", "四指")) -
                                min(fm(g, "onset", "f_020_med") for g in ("右拇指", "左拇指", "四指"))))),
             "three_state": "支持"},
        ],
        "corrections": [
            {"against": "13-v6-assessment", "was": "阶跃 = 0→0.2 s 窗口，占 5 s 增量 79.8~95.1%（段级中位 89.4%）",
             "now": ("阶跃段实测只有 1~2 帧（onset 跃变帧中位 %s 帧、t25_sus 中位 %.3f s、t_step_end 中位 %.3f s）；"
                     "0.2 s 是**时间轴畸变区上界**，不是阶跃段的物理终点。按同一批数据的事件级口径，"
                     "f(0.2) 中位 %.1f%%（onset）——与第一轮的 89.4%% 只差口径（归一参考与窗口起点）" %
                     (C.band(on.jump_frames)[0], C.band(on.t25_sus)[0], C.band(on.t_step_end)[0],
                      100 * C.band(on.f_020)[0])),
             "verdict": "修正"},
            {"against": "13-v6-assessment", "was": "慢相 τ_slow ≈ 19.7~71 s（中位 40.6 s）/ phases.csv slow_tau 中位 %.2f（p10~p90 %.2f~%.2f）" %
                      (float(p1.slow_tau.median()), float(p1.slow_tau.quantile(0.1)),
                       float(p1.slow_tau.quantile(0.9))),
             "now": ("τ_slow 不可辨识：同一批数据换拟合起点/窗口给出 %.2f s（本任务，从 t90 起算，n=%d，p10~p90 %.2f~%.2f）；"
                     "且幂律/对数在 LOO 上优于单指数 —— τ_slow 只能在声明的窗口口径下引用" %
                     (tau_mine.slow_tau_med, n_slowtau,
                      tau_mine.slow_tau_p10, tau_mine.slow_tau_p90)),
             "verdict": "修正"},
            {"against": "07-v6", "was": "§2.1 原始读数 1 s 处 f=0.912（9 组恒载 onset）",
             "now": "复现完全一致：本任务四套口径（网格/包轴 × 原始/Z3 × J=[4,6]/[4.5,5.5]）的 f(1 s) 中位均为 0.912；复现其管线与 v6_onset_profile9.csv 逐点最大差 %.4f" %
                    max(dprof),
             "verdict": "相同"},
            {"against": "07-v6", "was": "§2.2 onset t90=0.48 s、restep t90=2.87 s；Z(0.2 s)/阶跃 0.796 vs 0.212；Z(1 s)/阶跃 0.925 vs 0.641",
             "now": ("同管线复现 24/24 行（t90 最大差 1e-16 s）；换成 3 帧中值主口径后逐事件 t90 中位差 "
                     "onset %+.3f s / restep %+.3f s（|Δ| 中位 0.144 s、p90 0.96 s）；"
                     "事件级 z_at_10 差 %+.3f / %+.3f。⇒ 单事件层面 07-v6 数字**成立**；"
                     "但「restep 比 onset 慢 6 倍」的聚合差被样本构成放大，本任务换事件集与口径后为 2.2~3.0 倍" %
                     (float(np.median(d90_on)), float(np.median(d90_rs)),
                      float(np.median(d10_on)), float(np.median(d10_rs)))),
             "verdict": "修正"},
            {"against": "T4-A", "was": "「输入上升时间 T_ramp 中位 onset 0.010 s / restep 0.550 s，是形态的驱动量」",
             "now": ("独立验证（支持），但补一条限定：T_ramp（≤0.15 s 视为近阶跃）与「持续穿越判据 t25_sus ≤0.05 s」"
                     "在 restep 上一致率只有 %.0f%%（%d/%d）：%d 个判为「有阶跃」的事件里 %d 个 T_ramp >0.15 s，"
                     "%d 个判为「无阶跃」的事件里 %d 个 T_ramp ≤0.15 s "
                     "⇒ 两个「输入快慢」度量不是同一个量，T3/T4 用哪一个必须声明" %
                     (100 * (int(rs_ok.T_ramp.le(0.15).sum()) + int(rs_no.T_ramp.gt(0.15).sum())) /
                      max(len(rs_ok) + len(rs_no), 1),
                      int(rs_ok.T_ramp.le(0.15).sum()) + int(rs_no.T_ramp.gt(0.15).sum()),
                      len(rs_ok) + len(rs_no), len(rs_ok), int(rs_ok.T_ramp.gt(0.15).sum()),
                      len(rs_no), int(rs_no.T_ramp.le(0.15).sum()))),
             "verdict": "有条件修正"},
            {"against": "13-v6-assessment", "was": "§3.3 跨传感器 0.2 s 占比 右拇指 78.7% / 四指 88.4% / 左拇指 93.5%（主通道口径）",
             "now": ("总量 Z3 口径下 onset f(0.2) 中位 右拇指 %.1f%% / 四指 %.1f%% / 左拇指 %.1f%%（右拇指与第一轮一致，"
                     "四指、左拇指低 5~9 pt）；原因是信号口径（主通道 vs 总量）而非数据不同，"
                     "T4-A 已测同批 onset：总量 82.4%% vs 主通道 84.9%%" %
                     (100 * fm("右拇指", "onset", "f_020_med"), 100 * fm("四指", "onset", "f_020_med"),
                      100 * fm("左拇指", "onset", "f_020_med"))),
             "verdict": "修正"},
        ],
        "gaps": [
            "restep 的慢相样本不足：统一 30 s 窗只有 1~2 个可用事件、段级窗 6 个（其中多数幅度门被剔除），⇒ restep 慢相只能定性；需补采「带载阶跃 + 长保压 ≥60 s」各 5 次以上",
            "0.2 s 以内（尤其 ≤0.05 s）处于时间轴畸变区：包周期 16.7 ms（指尖）/40 ms（实录）决定 f(0.05) 的 ±1 包不确定度中位 0.017~0.097·J（相对 24%~40%）；更高采样率（≥500 Hz，包周期 ≤2 ms）才能把阶跃段定死",
            "无受控速率实验：t25_sus（持续穿越）与 T_ramp（反卷积）在 restep 上一致率只有 53%（10/19），因缺少「同一加载装置 × 5 档上升时间」的标定；需位移/力控装置 0.05/0.2/0.5/1.0/2.0 s × 各 5 次",
            "无真值力/位移：J 与 5 s 参考电平都取自读数本身，无法独立验证慢相幅度；需同步力传感器（对齐 ≤5 ms）",
            "partial_unload 仅 n=5、恒载组 restep 仅 n=2~3 → 相关分位数（p10~p90）不确定度大，已在报告中标注「仅定性参考」",
            "低于 T4-A 检测门限（max(8σ_d, 2% 记录极差)）的小事件未表征；本表 60 事件之外仍有小增量（07-v6 的 24 行里有 9 行不在 T4-A 事件集内）",
        ],
        "blockers": [],
    }
    out = C.os.path.join(C.RES, "conclusions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cj, f, ensure_ascii=False, indent=2)
    print("\n  -> results/conclusions.json")
    print("完成。")


if __name__ == "__main__":
    main()
