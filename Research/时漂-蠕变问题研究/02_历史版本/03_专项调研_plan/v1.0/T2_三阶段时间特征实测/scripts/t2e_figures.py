# -*- coding: utf-8 -*-
"""T2-E：图件（全部英文标签，避免中文字体缺失出方框）。

产物：figures/T2_01_phases_overview.png、T2_02_shape_profile.png、T2_03_slowphase_fit.png、
      figures/T2_04_timeaxis.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import t2_common as C

CIDX = {"onset": 0, "restep": 1, "unload": 2, "partial_unload": 3}
CCOL = {"onset": "#1f6fb4", "restep": "#d1495b", "unload": "#2a9d8f", "partial_unload": "#8d6e63"}


def strip(ax, values, kinds, ylabel, title, logy=False):
    for i, k in enumerate(["onset", "restep", "unload", "partial_unload"]):
        v = np.asarray(values, float)[np.asarray(kinds) == k]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        x = i + (np.random.RandomState(0).rand(v.size) - 0.5) * 0.36
        ax.plot(x, v, "o", ms=3.4, alpha=0.65, color=CCOL[k])
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, "-", lw=2.0, color="k")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["onset", "restep", "unload", "part.unl"], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)


def main():
    C.start_log("t2e_figures")
    print("== T2-E 图件 ==")
    all_d = C.load_all(verbose=False)
    ph = pd.read_csv(C.os.path.join(C.RES, "t2_phase_times.csv"))
    prof = pd.read_csv(C.os.path.join(C.RES, "t2_shape_profile.csv"))
    fits = pd.read_csv(C.os.path.join(C.RES, "t2_slowphase_fits.csv"))
    tax = pd.read_csv(C.os.path.join(C.RES, "t2_timeaxis_compare.csv"))
    kind = ph.kind.to_numpy()

    # ══════════ 图 1：三阶段总览 ══════════
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    ax = axes[0, 0]
    for key, t0, lab in (("RT1", 8.27, "onset (RT1 @8.27 s)"),
                         ("SW1", 110.45, "restep (SW1 @110.45 s)"),
                         ("SW1", 249.63, "unload (SW1 @249.63 s)")):
        d = all_d[key]
        k = int(round(t0 / C.DT))
        e = ph[(ph.key == key) & (ph.t_on == t0)]
        t_ax, y, ybar, _ = C.axis_series(d, "grid")
        pre, post, J = C.j_of_idx(ybar, k)
        tau = np.round(np.arange(-0.1, 3.0, 0.01), 3)
        f = (np.interp(t0 + tau, t_ax, d["Z3"]) - pre) / J
        ax.plot(tau, f, lw=1.2, label=lab)
        if len(e):
            e = e.iloc[0]
            ax.axvline(e.t25_sus, color="k", ls=":", lw=1.0)
            ax.axvline(e.t90_sus, color="k", ls="--", lw=1.0)
        if t0 == 8.27:
            ax.annotate("S1 end (t25$_{sus}$)", xy=(0.06, 0.30), fontsize=7.5)
            ax.annotate("S2/S3 bound (t90$_{sus}$)", xy=(0.75, 0.35), fontsize=7.5)
    ax.axhline(0.25, color="gray", lw=0.8)
    ax.axhline(0.90, color="gray", lw=0.8)
    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(-0.3, 1.35)
    ax.set_xlabel("time from $t_{on}$ (s)")
    ax.set_ylabel("$f(\\tau) = (Z_3(t_{on}+\\tau)-pre)/J$")
    ax.set_title("(a) three-phase structure, raw 3-frame-median $Z_3$", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    strip(axes[0, 1], ph.dur_s1, kind, "duration (s)", "(b) S1 step duration $t25_{sus}$ (log)", logy=True)
    strip(axes[1, 0], ph.dur_s2, kind, "duration (s)", "(c) S2 fast-phase duration $t90_{sus}-t25_{sus}$", logy=True)
    ax = axes[1, 1]
    on = ph[ph.kind == "onset"]
    rs = ph[ph.kind == "restep"]
    un = ph[ph.kind == "unload"]
    ax.scatter(on.T_ramp, on.f_005, s=16, color=CCOL["onset"], label="onset")
    ax.scatter(rs.T_ramp, rs.f_005, s=16, color=CCOL["restep"], label="restep")
    ax.scatter(un.T_ramp, un.f_005, s=16, color=CCOL["unload"], label="unload")
    ax.axhline(0.25, color="k", ls="--", lw=1.0)
    ax.axvline(0.05, color="k", ls=":", lw=1.0)
    ax.set_xlabel("input rise time $T_{ramp}$ from T4-A (s)")
    ax.set_ylabel("$f(0.05)$ (fraction of J delivered in 50 ms)")
    ax.set_title("(d) step-existence test: $t25_{sus}\\leq$0.05 s $\\Leftrightarrow$ $f(0.05)\\geq$0.25", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.05, 1.25)
    C.savefig(fig, "T2_01_phases_overview.png")

    # ══════════ 图 2：形状剖面 ══════════
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    for k in ("onset", "restep", "unload"):
        q = prof[(prof.ds == "ALL") & (prof.kind == k)].sort_values("tau")
        ax.plot(q.tau, q.f_median, "-o", ms=3.2, color=CCOL[k], label="%s (n=%d)" % (k, q.n.max()))
        ax.fill_between(q.tau, q.f_p10, q.f_p90, color=CCOL[k], alpha=0.15)
    ref = pd.read_csv(C.os.path.join(C.R07_RES, "v6_onset_profile9.csv"))
    ax.plot([0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 3.0, 5.0],
            [0.429, 0.743, 0.769, 0.802, 0.857, 0.912, 0.949, 0.969, 1.0],
            "s--", ms=3.2, color="k", lw=1.0, label="07-v6 raw onset median")
    ax.plot([0.15, 0.3, 0.5, 0.75, 1, 2, 3, 4, 5],
            [0.070, 0.131, 0.208, 0.296, 0.384, 0.646, 0.813, 0.923, 1.000],
            "^:", ms=3.2, color="gray", lw=1.0, label="doc profile ($\\tau$=2 s artifact)")
    ax.set_xscale("log")
    ax.set_xlabel("$\\tau$ after $t_{on}$ (s)")
    ax.set_ylabel("$f(\\tau)$ (median, p10-p90 band)")
    ax.set_title("(a) pooled shape profile by event kind ($Z_3$)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")
    ax.set_ylim(-0.2, 1.6)

    ax = axes[1]
    for k in ("onset", "restep", "unload"):
        q = prof[(prof.ds == "ALL") & (prof.kind == k)].sort_values("tau")
        ax.plot(q.tau, q.f_p90 - q.f_p10, "-o", ms=3.2, color=CCOL[k], label=k)
    ax.axvline(0.2, color="k", ls=":", lw=1.0)
    ax.annotate("0.2 s: v6 shape-window anchor", xy=(0.21, 2.6), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\tau$ after $t_{on}$ (s)")
    ax.set_ylabel("cross-event spread $p90-p10$")
    ax.set_title("(b) spread of $f(\\tau)$ vs $\\tau$ (quantization zone at small $\\tau$)", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
    C.savefig(fig, "T2_02_shape_profile.png")

    # ══════════ 图 3：慢相拟合 ══════════
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    f_seg = fits[(fits.window == "segend") & (fits.amp_ok)]
    keys = f_seg[(f_seg.kind == "onset") & (f_seg.model == "pow")].sort_values("U", ascending=False)
    src = None
    for _, r in keys.iterrows():
        c = all_d[r.key]
        t_ax, y, ybar, _ = C.axis_series(c, "grid")
        k = int(round(r.t_on / C.DT))
        pre, post, J = C.j_of_idx(ybar, k)
        u = np.arange(0.0, r.U, 0.5)
        h = (np.interp(r.t_on + r.t90 + u, t_ax, ybar) - np.interp(r.t_on + r.t90, t_ax, ybar)) / J
        src = (u, h, r)
        break
    ax = axes[0]
    if src:
        u, h, r = src
        ax.plot(u, 100 * h, "k-", lw=1.4, label="measured (%s @%.1f s, U=%.0f s)" % (r.key, r.t_on, r.U))
        for mdl, sty in (("lin", ":"), ("log", "--"), ("pow", "-"), ("exp", "-.")):
            rr = f_seg[(f_seg.key == r.key) & (f_seg.t_on == r.t_on) & (f_seg.model == mdl)]
            if not len(rr):
                continue
            rr = rr.iloc[0]
            p = [rr.a] + ([rr.b] if np.isfinite(rr.b) else [])
            if mdl == "lin":
                yv = p[0] * u
            elif mdl == "log":
                yv = p[0] * np.log1p(u / p[1])
            elif mdl == "pow":
                yv = p[0] * np.power(np.maximum(u, 1e-6), p[1])
            else:
                yv = p[0] * (1 - np.exp(-u / p[1]))
            ax.plot(u, 100 * yv, sty, lw=1.2, label="%s (rms %.2f%% J)" % (mdl, rr.rms_pct_in))
    ax.set_xlabel("$u$ from $t90$ (s)")
    ax.set_ylabel("slow-phase rise (% of J)")
    ax.set_title("(a) slow-phase shape and 4 candidate models", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    sdf = pd.read_csv(C.os.path.join(C.RES, "t2_slowphase_summary.csv"))
    q = sdf[(sdf.window == "segend") & (sdf.kind == "ALL")]
    x = np.arange(len(q))
    ax.bar(x - 0.18, q.rms_in_med, width=0.36, label="in-sample", color="#1f6fb4")
    ax.bar(x + 0.18, q.rms_loo_med, width=0.36, label="cross-recording (LOO)", color="#d1495b")
    ax.set_xticks(x)
    ax.set_xticklabels(q.model, fontsize=8)
    for i, (a, b) in enumerate(zip(q.rms_in_med, q.rms_loo_med)):
        ax.text(i - 0.18, a, "%.1f" % a, ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.18, b, "%.1f" % b, ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("RMS residual (% of J)")
    ax.set_title("(b) model comparison, segend window (n=%d)" % int(q.n.max()), fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    v = f_seg[f_seg.model == "pow"].dropna(subset=["rms_pct_in", "rms_pct_loo"])
    ax.scatter(v.rms_pct_in, v.rms_pct_loo, s=20,
               c=[CCOL.get(k, "k") for k in v.kind], label=None)
    lim = [0.3, max(30, float(np.nanmax(v.rms_pct_in)) * 1.2)]
    ax.plot(lim, lim, "k--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("in-sample RMS (% of J)")
    ax.set_ylabel("cross-recording LOO RMS (% of J)")
    ax.set_title("(c) power-law: in-sample vs LOO (power-law model)", fontsize=9)
    for k in ("onset", "restep"):
        ax.plot([], [], "o", color=CCOL[k], label=k)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
    C.savefig(fig, "T2_03_slowphase_fit.png")

    # ══════════ 图 4：时间轴口径 ══════════
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    g = tax[tax.axis == "grid"].set_index("key")
    p = tax[tax.axis == "pkt"].set_index("key")
    ix = g.index.intersection(p.index)
    kmap = ph.drop_duplicates("key").set_index("key")["kind"].to_dict()
    ax = axes[0]
    for k in ("onset", "restep", "unload", "partial_unload"):
        sel = [i for i in ix if kmap.get(i) == k]
        if not sel:
            continue
        ax.scatter(g.loc[sel, "t90"], p.loc[sel, "t90"], s=18, color=CCOL[k], label=k)
    lim = [0.005, max(6, float(np.nanmax(g.t90)) * 1.1)]
    ax.plot(lim, lim, "k--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t90$ on 100 Hz grid (s)")
    ax.set_ylabel("$t90$ on raw packet axis (s)")
    ax.set_title("(a) two time axes: 100 Hz grid vs packet timestamps", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")
    ax = axes[1]
    s = pd.read_csv(C.os.path.join(C.RES, "t2_timeaxis_summary.csv"))
    s = s[(s.kind == "ALL")]
    lbl = ["%s-%s" % (a, b) for a, b in zip(s.axis_a, s.axis_b)]
    mets = ["t90", "f_005", "f_020"]
    x = np.arange(len(lbl))
    for i, m in enumerate(mets):
        q = s[s.metric == m].copy()
        q["lab"] = q.axis_a + "-" + q.axis_b
        q = q.set_index("lab")
        ax.bar(x + (i - 1) * 0.26, [q.med_abs_delta.get(l, np.nan) for l in lbl],
               width=0.25, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(lbl, fontsize=7, rotation=20)
    ax.set_yscale("log")
    ax.set_ylabel("median |difference| (s or fraction)")
    ax.set_title("(b) pairwise axis/shift differences (all events)", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")
    C.savefig(fig, "T2_04_timeaxis.png")
    print("完成。")


if __name__ == "__main__":
    main()
