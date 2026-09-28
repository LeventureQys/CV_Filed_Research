# -*- coding: utf-8 -*-
"""T7-A / step7：出图（全部英文标签，避免中文字体缺失变成方框）。

T7A_01_unload_profile.png   卸载沿归一化完成度轮廓（中位 + p10~p90）+ 4 个代表事件的原始轨迹
T7A_02_residual_offset.png  卸载后残余偏移随时间（raw / v5.1 / v6 / v5 四臂）
T7A_03_zero_drift.png       多轮零点漂移（原始 vs 补偿后两条线；v5 的伪零点单独一栏）
T7A_04_event_traces.png     t0 对齐的原始读数轨迹（含 ±1 包敏感性）
另产出底层 csv：results/t7_event_traces.csv

用法：python scripts/t7a_7_figs.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402

LOG = C.Log("7_figs")
TRACE_CASES = [("右拇指指尖/数据1", 115.37), ("左拇指指尖/数据1", 147.38),
               ("中途切换-13ffca", 68.90), ("切换负载-快相无责", 249.61)]
ASCII = {"右拇指指尖/数据1": "RThumb-1", "右拇指指尖/数据2": "RThumb-2",
         "右拇指指尖/数据3": "RThumb-3", "左拇指指尖/数据1": "LThumb-1",
         "左拇指指尖/数据2": "LThumb-2", "左拇指指尖/数据3": "LThumb-3",
         "四指指尖/数据1": "FourF-1", "四指指尖/数据2": "FourF-2",
         "四指指尖/数据3": "FourF-3", "切换负载-快相无责": "Switch-fast",
         "再切换负载": "Reswitch", "中途切换-1d9493": "MidSwitch-1d9493",
         "中途切换-13ffca": "MidSwitch-13ffca"}


def an(tag):
    return ASCII.get(tag, tag)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 130, "savefig.bbox": "tight"})


def main():
    C.log_reconfigure()
    style()
    ev = pd.read_csv(os.path.join(C.RES, "t7_unload_events.csv"))
    sh = pd.read_csv(os.path.join(C.RES, "t7_unload_shape.csv"))
    fo = pd.read_csv(os.path.join(C.RES, "t7_frozen_offset.csv"))
    zd = pd.read_csv(os.path.join(C.RES, "t7_zero_drift.csv"))
    rd = pd.read_csv(os.path.join(C.RES, "t7_zero_rounds.csv"))

    # ── 底层轨迹 csv（图 1/4 的支撑数据）─────────────────────────────
    trows = []
    for tag, t0 in TRACE_CASES:
        d = C.prep(tag)
        tu, dtm = d["tu"], d["dtm"]
        ds = C.zbar(d["tot"], dtm)
        dst = C.med_smooth(d["tot"], max(1, int(round(0.1 / dtm))))
        i0 = int(round(t0 / dtm))
        for k in range(i0 - int(0.6 / dtm), min(len(tu), i0 + int(3.0 / dtm))):
            trows.append(dict(rec=tag, t0=t0, t=float(tu[k]), tau=float(tu[k] - t0),
                              tot=float(d["tot"][k]), zbar=float(ds[k]), zcausal=float(dst[k]),
                              pkt_dt=float(d["pkt_dt"])))
    tr = pd.DataFrame(trows)
    C.save(tr, "t7_event_traces.csv")

    # ── 图 1 ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    a = ax[0, 0]
    for nm, sub in (("all (n=14)", sh), ("force domain (n=9)", sh[sh["kind_rec"] == "恒载"]),
                    ("ADC domain (n=5)", sh[sh["kind_rec"] == "实采"])):
        g = sub.groupby("tau")["prog"]
        md, lo, hi = g.median(), g.quantile(0.10), g.quantile(0.90)
        a.plot(md.index, md.values, marker="o", ms=3, label=nm)
        a.fill_between(md.index, lo.values, hi.values, alpha=0.15)
    a.axhline(1.0, color="k", lw=0.8, ls="--")
    a.axvline(0.2, color="r", lw=0.8, ls=":")
    a.text(0.21, 0.15, "0.2 s time-axis distortion region", color="r", fontsize=7)
    a.set_xscale("symlog", linthresh=0.05)
    a.set_xlabel("time after unload edge t0 (s)")
    a.set_ylabel("unload completion  (Zbar(pre)-Zbar(t0+tau))/|J|")
    a.set_title("(a) Unloading-edge normalized profile (raw readings)")
    a.legend(fontsize=7)
    a.set_ylim(-0.05, 1.15)

    a = ax[0, 1]
    for tag, t0 in TRACE_CASES:
        s = tr[tr["rec"] == tag]
        y = s["tot"] - s["tot"].iloc[0]
        a.plot(s["tau"], y / abs(np.nanmin(y)) if np.nanmin(y) < 0 else y, lw=1.0,
               label=f"{an(tag)} @{t0:.1f}s")
    a.set_xlim(-0.62, 1.2)
    a.set_xlabel("tau (s)")
    a.set_ylabel("raw total, normalized to edge amplitude")
    a.set_title("(b) Raw readings aligned at t0 (edge = 1-2 packets)")
    a.legend(fontsize=6)

    a = ax[1, 0]
    for nm, sub in (("force", ev[ev["rec"].str.contains("指尖")]),
                    ("ADC", ev[~ev["rec"].str.contains("指尖")])):
        x = np.linspace(0.01, 1.0, 60)
        a.plot(sub["t50"], np.full(len(sub), 0.75 if nm == "force" else 0.25), "o",
               alpha=0.7, label=f"t50 {nm}")
        a.plot(sub["t90"], np.full(len(sub), 0.5 if nm == "force" else 0.0), "s",
               alpha=0.7, label=f"t90 {nm}")
        del x
    for nm, row in (("force packet", 17.6e-3), ("ADC packet", 40.2e-3)):
        a.axvline(row, color="k", ls="--", lw=0.8)
        a.text(row, 0.95, nm, rotation=90, fontsize=6, va="top")
    a.set_xscale("log")
    a.set_ylim(-0.3, 1.1)
    a.set_xlabel("time constant (s)")
    a.set_title("(c) t50/t90 vs packet interval")
    a.legend(fontsize=6, ncol=2)
    a.set_yticks([])

    a = ax[1, 1]
    a.scatter(ev["dip_zero_pct"] * 100, ev["resid_late_pct"] * 100,
              c=["#ff7f0e" if "指尖" in r else "#1f77b4" for r in ev["rec"]], s=25)
    a.axhline(0, color="k", lw=0.8)
    a.axvline(0.5, color="k", lw=0.8, ls=":")
    a.set_xlabel("undershoot below pre-load zero (% of load)")
    a.set_ylabel("steady residual at window end (% of load)")
    a.set_title("(d) Q4: overshoot vs no-return (orange=force, blue=ADC)")
    for _, r in ev.iterrows():
        if r["dip_zero_pct"] * 100 > 0.8 or abs(r["resid_late_pct"] * 100) > 0.8:
            a.annotate(an(r["rec"]), (r["dip_zero_pct"] * 100, r["resid_late_pct"] * 100),
                       fontsize=6)
    fig.suptitle("T7-A Fig 1: unloading-edge time characteristics (raw data only, n=14)", y=1.0)
    fig.savefig(os.path.join(C.FIG, "T7A_01_unload_profile.png"))
    plt.close(fig)
    LOG("-> figures/T7A_01_unload_profile.png")

    # ── 图 2：残余偏移随时间 ─────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for k, (nm, sel) in enumerate((("force domain (n=9)", fo[fo["kind_rec"] == "恒载"]),
                                   ("ADC domain (n=5)", fo[fo["kind_rec"] == "实采"]))):
        a = ax[k]
        for arm in C.ARMS:
            g = sel.groupby("tau")[f"resid_{arm}_pct"]
            md = g.median() * 100
            a.plot(md.index, md.values, marker="o", ms=3, color=C.ARM_COLOR[arm],
                   label=C.ARM_LABEL[arm])
            if arm == "raw":
                a.fill_between(md.index, g.quantile(0.10).values * 100,
                               g.quantile(0.90).values * 100, color="0.6", alpha=0.2)
        a.axhline(0, color="k", lw=0.8)
        a.set_xlabel("tau after unload edge (s)")
        a.set_ylabel("residual offset (% of |J|)")
        a.set_title(f"({'ab'[k]}) {nm}")
        a.legend(fontsize=6)
    a = ax[2]
    s = pd.read_csv(os.path.join(C.RES, "t7_arm_unload.csv"))
    g = s.groupby("arm")[["ded_late_pct", "dip_extra_pct", "extra_pin", "dfrozen_late_pct"]].median()
    x = np.arange(len(g))
    a.bar(x - 0.2, g["ded_late_pct"] * 100, width=0.4, label="residual deduction (%|J|)")
    a.bar(x + 0.2, g["dip_extra_pct"] * 100, width=0.4, label="extra undershoot (%|J|)")
    a.set_xticks(x)
    a.set_xticklabels([C.ARM_LABEL[i] for i in g.index], rotation=15, fontsize=6)
    a.set_ylabel("% of |J|")
    a.set_title("(c) compensation residual after unload (median, n=14)")
    a.legend(fontsize=6)
    fig.suptitle("T7-A Fig 2: residual offset vs time and algorithm coupling", y=1.02)
    fig.savefig(os.path.join(C.FIG, "T7A_02_residual_offset.png"))
    plt.close(fig)
    LOG("-> figures/T7A_02_residual_offset.png")

    # ── 图 3：零基线漂移 ────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))
    a = ax[0]
    for tag, sub in zd[zd["kind_rec"] == "实采"].groupby("rec"):
        sub = sub.sort_values("t0")
        a.plot(sub["t0"], sub["z_raw"], marker="o", ms=4, lw=1.0, label=an(tag))
    a.set_xlabel("recording time (s)")
    a.set_ylabel("zero-load reading, raw (ADC)")
    a.set_title("(a) ADC domain: raw zero drifts 32-41% of its own value")
    a.legend(fontsize=6)
    a = ax[1]
    for tag, sub in zd[zd["kind_rec"] == "恒载"].groupby("rec"):
        sub = sub.sort_values("t0")
        a.plot(sub["t0"], sub["z_raw"], marker="o", ms=4, lw=1.0, label=an(tag))
    a.set_xlabel("recording time (s)")
    a.set_ylabel("zero-load reading, raw (N)")
    a.set_title("(b) force domain: right thumb / four-finger rise, left thumb clamped at 0")
    a.legend(fontsize=6)
    a = ax[2]
    w = 0.2
    lbl = ["v5.1 (e3s)", "v6", "v5 (with auto-zero)"]
    for i, arm in enumerate(("e3s", "v6", "v5")):
        x = zd[f"z_{arm}_minus_raw"].to_numpy(float)
        x = x[np.isfinite(x)]
        a.bar(i, np.median(x), width=0.5, color=["#ff7f0e", "#2ca02c", "#9467bd"][i],
              label=f"{lbl[i]} (median)")
        a.plot(np.full(min(len(x), 40), i) + np.linspace(-0.15, 0.15, min(len(x), 40)),
               np.sort(x)[:40], "k.", ms=2, alpha=0.5)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(range(3))
    a.set_xticklabels(lbl, fontsize=7)
    a.set_ylabel("compensated zero - raw zero")
    a.set_title("(c) does the algorithm touch the zero? (31 idle plateaus)")
    a.legend(fontsize=6)
    fig.suptitle("T7-A Fig 3: zero-baseline drift (P3) - raw vs compensated lines", y=1.03)
    fig.savefig(os.path.join(C.FIG, "T7A_03_zero_drift.png"))
    plt.close(fig)
    LOG("-> figures/T7A_03_zero_drift.png")

    # ── 图 4：代表事件轨迹 ──────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    for k, (tag, t0) in enumerate(TRACE_CASES):
        a = ax.flat[k]
        s = tr[tr["rec"] == tag]
        a.plot(s["tau"], s["tot"], lw=1.0, label="raw total")
        a.plot(s["tau"], s["zbar"], lw=1.0, color="0.5", label="Zbar (centered 0.5 s)")
        a.plot(s["tau"], s["zcausal"], lw=1.0, color="#d62728", ls="--",
               label="causal 0.1 s median")
        e = ev[(ev["rec"] == tag) & (np.abs(ev["t_on"] - t0) < 0.5)]
        if len(e):
            r = e.iloc[0]
            a.axhline(r["z0_ref"], color="#1f77b4", lw=1.0, ls=":",
                      label=f"pre-load zero = {r['z0_ref']:.3g}")
            a.axhline(r["post"], color="#2ca02c", lw=1.0, ls=":",
                      label=f"post level = {r['post']:.3g}")
            a.plot([r["trough_s"]], [r["min_raw"]], "kv", ms=5, label="trough")
        a.axvline(0, color="k", lw=0.8)
        a.set_xlim(-0.6, 3.0)
        a.set_title(f"{an(tag)} @ t0={t0:.2f}s (packet {s['pkt_dt'].iloc[0]*1000:.1f} ms)",
                    fontsize=8)
        a.set_xlabel("tau (s)")
        a.set_ylabel("reading")
        a.legend(fontsize=6)
    fig.suptitle("T7-A Fig 4: representative unloading events (raw readings)", y=1.0)
    fig.savefig(os.path.join(C.FIG, "T7A_04_event_traces.png"))
    plt.close(fig)
    LOG("-> figures/T7A_04_event_traces.png")

    # 图 3 的支撑：把两条线的差也写一份小表（便于引用）
    d2 = zd[["rec", "kind_rec", "gap_idx", "t0", "z_raw", "z_e3s", "z_v6", "z_v5",
             "z_e3s_minus_raw", "z_v6_minus_raw", "z_v5_minus_raw", "dz_raw"]]
    C.save(d2, "t7_zero_lines.csv")
    LOG("")
    LOG("图件支撑：results/t7_event_traces.csv、t7_zero_lines.csv、t7_unload_shape.csv、"
        "t7_frozen_offset.csv、t7_zero_drift.csv、t7_arm_unload.csv")
    LOG.close("python scripts/t7a_7_figs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
