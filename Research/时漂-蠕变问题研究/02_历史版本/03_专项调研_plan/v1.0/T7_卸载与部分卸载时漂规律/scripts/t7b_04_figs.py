# -*- coding: utf-8 -*-
"""T7-B 步骤 4：图件（T7B_01~03）。

只读 results/*.csv，不再做分析。图内标签一律英文（避免中文字体缺失出方块）。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t7b_ad_lib import RESULTS, FIGURES, ensure_dirs  # noqa: E402

ensure_dirs()
LOG = os.path.join(RESULTS, "_t7b_04_figs.log")
COL = {"onset": "#1f77b4", "restep": "#ff7f0e", "unload": "#d62728",
       "partial_unload": "#2ca02c", "decrement_small": "#9467bd",
       "scan_decrement": "#8c564b"}
LAB = {"onset": "load onset (n=%d)", "restep": "load restep (n=%d)",
       "unload": "full unload (n=%d)", "partial_unload": "partial unload (n=%d)",
       "decrement_small": "small decrement (n=%d)"}


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def med_band(df, grp, col="f", tau_col="tau"):
    g = df[df["group"] == grp]
    taus, m, lo, hi = [], [], [], []
    for tau in sorted(g[tau_col].unique()):
        v = g[g[tau_col] == tau][col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        taus.append(tau); m.append(np.median(v))
        lo.append(np.percentile(v, 10)); hi.append(np.percentile(v, 90))
    return np.array(taus), np.array(m), np.array(lo), np.array(hi)


def fig1():
    sm = pd.read_csv(os.path.join(RESULTS, "t7b_shape_summary.csv"))
    amp = pd.read_csv(os.path.join(RESULTS, "t7b_partial_amp_response.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    a = ax[0]
    for grp in ("onset", "restep", "unload", "partial_unload", "decrement_small"):
        g = sm[sm["group"] == grp].sort_values("tau")
        if len(g) == 0:
            continue
        n = int(g["n"].max())
        a.plot(g["tau"], g["f_med"], "-o", ms=3.5, color=COL[grp], label=LAB[grp] % n)
        if n >= 5:
            a.fill_between(g["tau"], g["f_p10"], g["f_p90"], color=COL[grp],
                           alpha=0.15, lw=0)
    a.axhline(1.0, color="k", lw=0.8, ls=":")
    a.axvline(0.2, color="gray", lw=0.8, ls="--")
    a.text(0.205, 0.05, "tau=0.2 s (>=5 packets)", fontsize=8, color="gray")
    a.set_xscale("log")
    a.set_xlim(0.01, 8.5)
    a.set_ylim(-0.05, 1.35)
    a.set_xlabel("tau after event start  [s]")
    a.set_ylabel("normalized progress f(tau) = (Z(t0+tau)-pre)/D")
    a.set_title("T7B_01a  Direction-unified profile: load vs unload\n"
                "(median line, p10-p90 band for n>=5)")
    a.legend(fontsize=7.5, loc="lower right")
    a.grid(alpha=0.3)

    b = ax[1]
    for grp in ("unload", "partial_unload", "decrement_small"):
        g = amp[amp["kind_eff"] == grp]
        if len(g) == 0:
            continue
        yerr = np.vstack([np.maximum(g["t90_ad"] - g["t90_ad_m1"], 0).fillna(0),
                          np.maximum(g["t90_ad_p1"] - g["t90_ad"], 0).fillna(0)])
        b.errorbar(100 * g["drop_frac_eff"], g["t90_ad"] + 1e-3, yerr=yerr + 1e-3,
                   fmt="o", ms=6, capsize=3, color=COL[grp],
                   label=LAB[grp] % len(g))
    b.set_yscale("log")
    b.set_xlabel("decrement amplitude  |J| / pre-level  [%]")
    b.set_ylabel("t90 (direction-unified)  [s]")
    b.set_title("T7B_01b  Decrement amplitude vs recovery speed\n"
                "(error bars = +/-1 packet sensitivity)")
    b.grid(alpha=0.3, which="both")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "T7B_01_partial_shape.png"), dpi=150)
    plt.close(fig)
    print("  wrote T7B_01_partial_shape.png")


def fig2():
    met = pd.read_csv(os.path.join(RESULTS, "t7b_events_metrics.csv"))
    amp = pd.read_csv(os.path.join(RESULTS, "t7b_partial_amp_response.csv"))
    pr = pd.read_csv(os.path.join(RESULTS, "t7b_asymmetry_pairs.csv"))
    asym = pd.read_csv(os.path.join(RESULTS, "t7b_asymmetry.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    a = ax[0]
    groups = ["onset", "restep", "unload", "partial_unload"]
    vals = [met[met["kind"] == g]["t90"].dropna().to_numpy(float) + 1e-3 for g in groups]
    bp = a.boxplot(vals, tick_labels=[LAB[g] % len(v) for g, v in zip(groups, vals)],
                   widths=0.5, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(COL[g]); patch.set_alpha(0.6)
    for i, (g, v) in enumerate(zip(groups, vals), start=1):
        x = np.random.RandomState(0).normal(i, 0.06, len(v))
        a.plot(x, v, "k.", ms=4, alpha=0.7)
    a.axhline(0.019, color="gray", ls="--", lw=0.8)
    a.text(0.6, 0.021, "1 packet (display domain)", fontsize=7.5, color="gray")
    a.axhline(0.040, color="gray", ls=":", lw=0.8)
    a.text(0.6, 0.043, "1 packet (ADC domain)", fontsize=7.5, color="gray")
    a.set_yscale("log")
    a.set_ylabel("t90  [s]")
    a.set_title("T7B_02a  t90 by event kind\n(direction-unified; dots = events)")
    a.grid(alpha=0.3, which="both")
    plt.setp(a.get_xticklabels(), rotation=18, ha="right", fontsize=7.5)

    b = ax[1]
    b.plot([1e-3, 100], [1e-3, 100], "k--", lw=1, label="symmetric (ratio=1)")
    b.plot([1e-3, 100], [1e-3, 100 / 10], "k:", lw=1, label="ratio=10")
    for dom, mk in (("显示域", "o"), ("ADC域", "s")):
        g = pr[pr["domain"] == dom]
        b.scatter(g["t90_unload"] + 1e-3, g["t90_load"] + 1e-3, marker=mk, s=48,
                  label="%s (n=%d)" % ("display domain" if dom == "显示域" else "ADC domain",
                                       len(g)))
    b.plot(pr["t90_unload"] + 1e-3, pr["t90_ratio_lo"] * (pr["t90_unload"] + 1e-3),
           "x", color="red", ms=6, label="conservative lo (+/-1 pkt)")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("t90 unload  [s]"); b.set_ylabel("t90 load  [s]")
    b.set_title("T7B_02b  Matched pairs (same recording, |J| within 15%)")
    b.legend(fontsize=7.5, loc="lower right"); b.grid(alpha=0.3, which="both")

    c = ax[2]
    st = asym[asym["contrast_id"].isin(["A2_byTramp"]) & (asym["metric"] == "t90")]
    st = st[st["stratum_var"] == "T_ramp_load"]
    lab = list(st["stratum"]); x = np.arange(len(lab))
    c.bar(x - 0.18, st["load_med"], width=0.36, color="#1f77b4",
          label="load t90 (median, by T_ramp stratum)")
    unl_med = float(met[met["kind"] == "unload"]["t90"].median())
    pkt = float(met[met["kind"] == "unload"]["pkt_dt"].median())
    c.bar(x + 0.18, [unl_med] * len(lab), width=0.36, color="#d62728",
          label="unload t90 (median)")
    c.errorbar(x + 0.18, [unl_med] * len(lab), yerr=[[0] * len(lab), [pkt] * len(lab)],
               fmt="none", ecolor="k", capsize=4)
    c.set_xticks(x)
    c.set_xticklabels([s.replace("(近阶跃)", "\n(step-like)").replace("(人工慢压)",
                                                                     "\n(slow hand press)")
                       for s in lab], fontsize=7.5)
    c.set_yscale("log")
    c.set_ylabel("t90  [s]")
    c.set_title("T7B_02c  Load stratified by input ramp T_ramp\n(unload error bar = +1 packet)")
    c.legend(fontsize=7.5)
    c.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "T7B_02_asymmetry.png"), dpi=150)
    plt.close(fig)
    print("  wrote T7B_02_asymmetry.png")


def fig3():
    bb = pd.read_csv(os.path.join(RESULTS, "t7b_baseline_bias.csv"))
    inj = pd.read_csv(os.path.join(RESULTS, "t7b_inject_partial_bias.csv"))
    jmp = pd.read_csv(os.path.join(RESULTS, "t7b_jump7561_check.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))
    a = ax[0]
    allb = pd.concat([bb.assign(set="real"), inj.assign(set="injected(linear)")],
                     ignore_index=True)
    for arm, mk in (("v51", "o"), ("v6", "s")):
        g = allb[(allb["arm"] == arm) & (allb["src"].isin(["t4a_frozen", "t7b_scan",
                                                           "inject_linear"]))]
        a.scatter(100 * g["drop_frac_ad"], g["bias_pct_J"], marker=mk, s=42,
                  label="%s (n=%d)" % ("v5.1" if arm == "v51" else "v6", len(g)))
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("decrement amplitude  |J| / pre-level  [%]")
    a.set_ylabel("post-event baseline bias  (display - raw) / |J|  [%]")
    a.set_title("T7B_03a  Baseline bias after decrement\n"
                "(>0 = display too high, <0 = frozen over-deduction)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    b = ax[1]
    for arm, mk in (("v51", "o"), ("v6", "s")):
        g = bb[(bb["arm"] == arm)]
        b.scatter(g["absJ_ad"], np.abs(g["induced_jump"]), marker=mk, s=42,
                  label="%s real decrements" % ("v5.1" if arm == "v51" else "v6"))
        g2 = inj[inj["arm"] == arm]
        b.scatter(g2["absJ_ad"], np.abs(g2["induced_jump"]), marker=mk, s=46,
                  facecolors="none", edgecolors="k",
                  label="%s injected partials" % ("v5.1" if arm == "v51" else "v6"))
    v2 = jmp[jmp["alg"].isin(["CompV1", "CompV2"])]
    b.scatter(v2["jump_raw_at_max"] * 0 + 25000, np.abs(v2["induced_at_max"]), marker="*",
              s=150, color="purple",
              label="v2 prototype (raw frames, x-position arbitrary)")
    b.axhline(7561, color="purple", ls="--", lw=1)
    b.text(6e3, 7900, "changelog 2026-10-26: 7561 ADC", fontsize=7.5, color="purple")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("decrement amplitude |J|  [ADC or N]")
    b.set_ylabel("max algorithm-induced single-frame jump  [same unit]")
    b.set_title("T7B_03b  Single-frame jump introduced by the compensator")
    b.legend(fontsize=7.5, loc="lower right"); b.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "T7B_03_baseline_bias.png"), dpi=150)
    plt.close(fig)
    print("  wrote T7B_03_baseline_bias.png")


if __name__ == "__main__":
    tee = Tee(LOG)
    _o = sys.stdout
    sys.stdout = tee
    try:
        print("CMD: python scripts/t7b_04_figs.py")
        fig1(); fig2(); fig3()
    finally:
        sys.stdout = _o
        tee.flush()
        tee.f.close()
