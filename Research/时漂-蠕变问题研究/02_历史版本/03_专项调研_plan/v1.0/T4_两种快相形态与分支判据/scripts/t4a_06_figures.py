# -*- coding: utf-8 -*-
"""t4a_06_figures：T4-A 图件（英文标签，避免中文字体缺失出方框）。

产出：figures/T4A_01_two_morphology.png、T4A_02_mechanism.png、T4A_03_continuum.png
日志：results/_t4a_06_figures.log
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import t4a_common as C
import t4a_ad_lib as L

TAU = np.arange(0.02, 5.0001, 0.02)
CO = {"onset": "#1f77b4", "restep": "#d62728", "unload": "#2ca02c"}


def load_all():
    mor = pd.read_csv(os.path.join(C.RES, "t4a_morphology.csv"))
    ir = pd.read_csv(os.path.join(C.RES, "t4a_input_recover.csv"))
    mor["t_on_r"] = mor["t_on"].round(3)
    ir["t_on_r"] = ir["t_on"].round(3)
    return mor.merge(ir[["key", "t_on_r", "T_ramp", "rmse_ramp_pct", "rmse_step_pct",
                         "tau0", "gain_ramp"]], on=["key", "t_on_r"], how="left")


def traces(mor):
    """返回 {row_id: (tau, f(τ))}（归一化轮廓）与原始 Zs 缓存。"""
    recs, out = {}, {}
    for r in C.RECS:
        tu, Xu, pkt, nraw = C.load_grid(r)
        Z = Xu.sum(axis=1)
        recs[r["key"]] = dict(tu=tu, Z=Z, Zs=L.med_smooth(Z, C.KSM))
    for i, q in mor.iterrows():
        if not np.isfinite(q.jump) or abs(q.jump) < 1e-12:
            continue
        R = recs[q.key]
        k = int(q.k_on)
        idx = k + (TAU / C.DT).astype(int)
        idx = idx[idx < len(R["Zs"])]
        out[i] = (TAU[:len(idx)], (R["Zs"][idx] - q.pre) / q.jump)
    return recs, out


def band_curve(keys, tr):
    M = []
    for k in keys:
        if k in tr:
            M.append(tr[k][1])
    n = min(len(m) for m in M) if M else 0
    M = np.array([m[:n] for m in M])
    return np.median(M, axis=0), np.percentile(M, 10, axis=0), np.percentile(M, 90, axis=0), M


def fig1(mor, tr, recs):
    ld = mor[(mor.jump > 0) & mor.clean]
    ko = ld[ld.kind == "onset"].index.tolist()
    kr = ld[ld.kind == "restep"].index.tolist()
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.4))

    a = ax[0, 0]
    for keys, lab in ((ko, "onset (n=%d)" % len(ko)), (kr, "restep (n=%d)" % len(kr))):
        m, p10, p90, _ = band_curve(keys, tr)
        t = TAU[:len(m)]
        a.fill_between(t, p10, p90, color=CO[lab.split()[0]], alpha=0.18, lw=0)
        a.plot(t, m, color=CO[lab.split()[0]], lw=2, label=lab)
    a.set_xscale("log")
    a.set_xlim(0.03, 5)
    a.set_ylim(-0.2, 1.15)
    a.axhline(1.0, color="k", ls=":", lw=1)
    a.set_xlabel("time since load edge $t-t_{on}$ (s, log)")
    a.set_ylabel("normalized rise  $f(\\tau)=(\\bar Z(t_{on}+\\tau)-pre)/J$")
    a.set_title("(a) Two fast-phase morphologies (median + p10-p90 band)")
    a.legend(loc="lower right", fontsize=9)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for keys, lab in ((ko, "onset"), (kr, "restep")):
        m, p10, p90, _ = band_curve(keys, tr)
        t = TAU[:len(m)]
        a.fill_between(t, p10, p90, color=CO[lab], alpha=0.18, lw=0)
        a.plot(t, m, color=CO[lab], lw=2, label=lab)
    a.set_xlim(0, 1.0)
    a.set_ylim(-0.2, 1.15)
    a.set_xlabel("time since load edge (s)")
    a.set_ylabel("normalized rise")
    a.set_title("(b) First second: the whole difference lives here")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    for keys, lab in ((ko, "onset"), (kr, "restep")):
        prof = []
        for i in keys:
            q = mor.loc[i]
            Z = recs[q.key]["Z"]
            k = int(q.k_on)
            seg = Z[k:k + int(1.0 / C.DT)]
            if len(seg) < 100 or abs(q.jump) < 1e-12:
                continue
            prof.append(np.abs(np.diff(seg)) / abs(q.jump))
        if not prof:
            continue
        n = min(len(p) for p in prof)
        P = np.array([p[:n] for p in prof])
        a.plot(np.arange(n) * C.DT, np.median(P, axis=0) * 100, color=CO[lab], lw=2, label=lab)
    a.set_xlabel("time since load edge (s)")
    a.set_ylabel("median per-frame increment  $|\\Delta Z|/J$  (%, raw Z, 100 Hz)")
    a.set_title("(c) Frame-by-frame differential signature")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    feats = ["z_at_005", "z_at_01", "z_at_02"]
    xs = np.arange(len(feats))
    for j, (keys, lab) in enumerate(((ko, "onset"), (kr, "restep"))):
        v = [ld.loc[keys, f].to_numpy(float) for f in feats]
        pos = xs + (j - 0.5) * 0.28
        bp = a.boxplot(v, positions=pos, widths=0.22, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(CO[lab])
            b.set_alpha(0.6)
        a.plot(pos, [np.median(x) for x in v], "k_", ms=14)
    a.set_xticks(xs)
    a.set_xticklabels(["0.05 s", "0.10 s", "0.20 s"])
    a.set_ylabel("completion ratio $Z(\\tau)/J$")
    a.set_xlabel("time since load edge")
    a.set_title("(d) Early completion ratio (onset vs restep)")
    a.grid(alpha=0.3)
    fig.suptitle("T4-A  Fig.1  Two fast-phase morphologies (13 recordings, total Z, 100 Hz grid, "
                 "clean events)", fontsize=12)
    C.savefig(fig, "T4A_01_two_morphology.png")


def fig2(mor, tr, recs):
    ld = mor[(mor.jump > 0) & mor.clean]
    ko = ld[ld.kind == "onset"].index.tolist()
    kr = ld[ld.kind == "restep"].index.tolist()
    un = mor[(mor.jump < 0) & mor.clean & (mor.jump.abs() >= 0.5 * mor.pre)]
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.4))

    a = ax[0, 0]
    for keys, lab in ((ko, "onset"), (kr, "restep")):
        curves = []
        for i in keys:
            q = mor.loc[i]
            Z = recs[q.key]["Z"]
            k = int(q.k_on)
            seg = Z[k:min(len(Z) - 1, k + int(1.5 / C.DT))]
            if len(seg) < 20 or abs(q.jump) < 1e-12:
                continue
            d = np.sort(np.abs(np.diff(seg)))[::-1] / abs(q.jump)
            curves.append(np.cumsum(d)[:150])
        n = min(len(c) for c in curves)
        M = np.array([c[:n] for c in curves])
        a.plot(np.arange(n) * C.DT, np.clip(np.median(M, axis=0), 0, 1.2), color=CO[lab], lw=2,
               label="%s (n=%d)" % (lab, len(curves)))
    a.axhline(0.5, color="gray", ls=":", lw=1)
    a.set_xlabel("number of largest frames accumulated (s, 100 Hz)")
    a.set_ylabel("cumulative share of $J$")
    a.set_xlim(0, 1.5)
    a.set_title("(a) Rise concentration: how many frames carry the rise")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for j, (g, lab) in enumerate(((ld[ld.kind == "onset"], "onset"),
                                  (ld[ld.kind == "restep"], "restep"), (un, "unload"))):
        v = g.z_at_02.to_numpy(float)
        v = v[np.isfinite(v)]
        a.scatter(np.full(len(v), j) + np.random.default_rng(3).normal(0, 0.05, len(v)), v,
                  color=CO[lab], s=26, alpha=0.8, label=lab)
        a.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color="k", lw=2)
    a.set_xticks([0, 1, 2])
    a.set_xticklabels(["onset", "restep", "unload"])
    a.set_ylabel("$Z(0.2\\,s)/J$")
    a.set_title("(b) E2: unload (same loaded state) is instantaneous")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    ch = pd.read_csv(os.path.join(C.RES, "t4a_channel_consistency.csv"))
    ch = ch[ch.clean & (ch.jump > 0) if "jump" in ch.columns else ch.clean]
    for j, (lab, col) in enumerate((("onset", "onset"), ("restep", "restep"))):
        v = ch[ch.kind == lab].t50_ch_iqr.to_numpy(float)
        v = v[np.isfinite(v)]
        a.scatter(np.full(len(v), j) + np.random.default_rng(5).normal(0, 0.05, len(v)),
                  np.maximum(v, 1e-4), color=CO[col], s=26, alpha=0.8)
        a.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color="k", lw=2)
    a.set_yscale("log")
    a.set_xticks([0, 1])
    a.set_xticklabels(["onset", "restep"])
    a.set_ylabel("within-event IQR of per-channel $t_{50}$ (s, log)")
    a.set_title("(c) E3: channel synchrony (onset = simultaneous)")
    a.grid(alpha=0.3, which="both")

    a = ax[1, 1]
    ir = mor[mor.jump > 0]
    for lab, col in (("onset", "onset"), ("restep", "restep")):
        v = ir[ir.kind == lab].T_ramp.to_numpy(float)
        v = v[np.isfinite(v)]
        a.hist(np.maximum(v, 0.005), bins=np.logspace(-2.31, 0.1, 16), alpha=0.55,
               color=CO[col], label="%s (median %.2f s)" % (lab, np.median(v)))
    a.set_xscale("log")
    a.set_xlabel("recovered input ramp duration $T_{ramp}$ (s, log)")
    a.set_ylabel("events")
    a.set_title("(d) E4: deconvolved input rise time")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)
    fig.suptitle("T4-A  Fig.2  Mechanism evidence (E1 differential / E2 unload symmetry / "
                 "E3 channel synchrony / E4 deconvolution)", fontsize=12)
    C.savefig(fig, "T4A_02_mechanism.png")


def fig3(mor):
    ld = mor[mor.jump > 0].copy()
    ld["Tp"] = ld.T_ramp.clip(lower=0.005)
    cont = pd.read_csv(os.path.join(C.RES, "t4a_continuum.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.4))

    a = ax[0, 0]
    for lab in ("onset", "restep"):
        s = ld[ld.kind == lab]
        a.scatter(s.Tp, s.z_at_02, s=26 + 260 * s.pre_over_peak.clip(0, 1),
                  color=CO[lab], alpha=0.75, label=lab)
    a.set_xscale("log")
    a.set_xlabel("recovered input ramp duration $T_{ramp}$ (s, log)")
    a.set_ylabel("0.2 s completion  $z_{at,02}$")
    a.set_title("(a) Early shape vs input rise time (marker size = preload ratio)")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    a.axvspan(0.05, 0.30, color="gold", alpha=0.25, label="continuum overlap zone")
    for lab in ("onset", "restep"):
        s = ld[ld.kind == lab]
        a.scatter(s.Tp, s.pre_over_peak, color=CO[lab], s=30, alpha=0.8, label=lab)
    for y in (0.10, 0.20, 0.30):
        a.axhline(y, color="gray", ls=":", lw=1)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("$T_{ramp}$ (s, log)")
    a.set_ylabel("preload ratio  pre / record peak")
    a.set_title("(b) Preload does NOT fix the input style (dotted = 10/20/30% armed)")
    a.legend(fontsize=9)
    a.grid(alpha=0.3, which="both")

    a = ax[1, 0]
    for j, lab in enumerate(("onset", "restep")):
        v = ld[ld.kind == lab].Tp.to_numpy(float)
        a.scatter(np.full(len(v), j) + np.random.default_rng(11).normal(0, 0.06, len(v)),
                  v, color=CO[lab], s=30, alpha=0.8)
        a.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color="k", lw=2)
    ov = ld[(ld.T_ramp >= 0.05) & (ld.T_ramp <= 0.30)]
    for _, q in ov.iterrows():
        a.annotate("%s %.2fs" % (q.kind[:3], q.t_on), (1.35, q.Tp), fontsize=7, va="center")
    a.set_yscale("log")
    a.set_xticks([0, 1])
    a.set_xticklabels(["onset\n(n=%d)" % int((ld.kind == "onset").sum()),
                       "restep\n(n=%d)" % int((ld.kind == "restep").sum())])
    a.set_ylabel("$T_{ramp}$ (s, log)")
    a.set_title("(c) Continuum: both ends populated, %d events in the overlap zone" % len(ov))
    a.grid(alpha=0.3, which="both")

    a = ax[1, 1]
    c2 = cont[(cont.block == "C2") & (cont["sample"] == "clean") &
              (cont.term == "R2_model")]
    if len(c2) == 0:
        c2 = cont[(cont.block == "C2") & (cont["sample"] == "clean")]
        lab = ["armed_20\nonly", "log10 T_ramp\nonly", "both"]
        vals = []
        for m in ("z_at_02 ~ armed_20", "z_at_02 ~ log10 T_ramp",
                  "z_at_02 ~ log10 T_ramp + armed_20"):
            s = c2[c2.model == m]
            vals.append(float(s.r2.iloc[0]) if len(s) else np.nan)
        a.bar(lab, vals, color=["#7f7f7f", "#1f77b4", "#9467bd"], alpha=0.85)
        for i, v in enumerate(vals):
            a.text(i, v + 0.01, "%.2f" % v, ha="center", fontsize=10)
        a.set_ylim(0, 0.8)
        a.set_ylabel("$R^2$ of OLS on $z_{at,02}$ (clean, n=25)")
        a.set_title("(d) Input rise time explains more than the preload flag")
        a.grid(alpha=0.3, axis="y")
    fig.suptitle("T4-A  Fig.3  Two forms are the two ends of one continuum (input rise time)",
                 fontsize=12)
    C.savefig(fig, "T4A_03_continuum.png")


def main():
    C.start_log()
    mor = load_all()
    recs, tr = traces(mor)
    fig1(mor, tr, recs)
    fig2(mor, tr, recs)
    fig3(mor)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
