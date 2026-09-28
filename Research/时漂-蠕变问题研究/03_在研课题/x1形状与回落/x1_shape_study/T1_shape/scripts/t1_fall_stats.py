# -*- coding: utf-8 -*-
"""步骤 5：显示回落/过扣统计与归因出图（T1 形状研究 D 部分）。

输入 results/t1_replay_events.csv（t1_replay.py 产出）
输出 results/t1_fall_stats.txt + figure/T1_回落与归因*.png
用法： python t1_fall_stats.py
"""
import csv
import io
import os
import sys

import numpy as np
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import t1_lib as T  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NUM = ("W", "t0", "t1", "ramp", "plateau", "step", "dch", "win", "parity", "peak",
       "t_peak", "end_disp", "end_v", "fall", "fall_frac", "v_creep", "x1_end",
       "x2_end", "d_x1", "d_x2")


def load():
    with io.open(os.path.join(T.RESULTS, "t1_replay_events.csv"),
                 encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in NUM:
            try:
                r[k] = float(r[k]) if r[k] != "" else float("nan")
            except (ValueError, KeyError):
                r[k] = float("nan")
    return rows


def valid(r):
    """窗口有效性门：窗口内不得含卸载/大幅反向，台阶为正，峰值在窗内。"""
    if not np.isfinite(r["step"]) or r["step"] <= 0:
        return False
    if not np.isfinite(r["v_creep"]) or abs(r["v_creep"]) > 0.4 * abs(r["step"]):
        return False
    if not np.isfinite(r["fall"]) or abs(r["fall"]) > 0.6 * abs(r["step"]):
        return False
    if not np.isfinite(r["t_peak"]) or r["t_peak"] < 0.05:
        return False
    return True


def key_of(r):
    return (r["ses"], round(r["t0"], 2), r["W"], r["sig"], r["ch"])


def main():
    rows = load()
    tot = [r for r in rows if r["sig"] == "tot"]
    w = []
    w.append("=" * 104)
    w.append("T1 显示回落/过扣统计（fall = D(t_peak) − D(t_end)；ideal = 0）")
    w.append("量测口径：D(t_peak) 后 0.5 s 中位 − 窗口末 0.3 s 中位；"
             "v_creep = 同窗口输入爬升；fall = Δx1 + Δx2 − v_creep")
    w.append("")

    # ---------- 1. 事件台账（录制的实机显示 = 真值） ----------
    recs = [r for r in tot if r["cas"] == "recorded" and valid(r)]
    recs.sort(key=lambda r: r["step"])
    w.append("-" * 104)
    w.append("[1] 实机录制显示的回落（device_001_seg000.csv = 真值，与回放无关）")
    w.append("    窗口 10 s（保压 ≥ 12 s 的全部加载事件，n=%d）" % len(recs))
    w.append("    %-46s %8s %8s %8s %8s %8s %8s"
             % ("会话/事件", "台阶", "回落", "占台阶", "输入爬升", "t_peak", "保压"))
    for r in recs:
        w.append("    %-46s %+8.0f %+8.0f %8s %+8.0f %8.1f %8.1f"
                 % ((r["ses"].split("/")[-1][:32] + " t=%.1f" % r["t0"]),
                    r["step"], r["fall"],
                    "%.1f%%" % (100 * r["fall_frac"]) if np.isfinite(r["fall_frac"]) else "-",
                    r["v_creep"], r["t_peak"], r["plateau"]))
    big = [r for r in recs if r["step"] >= 6000]
    small = [r for r in recs if r["step"] < 6000]
    for name, g in (("大台阶(≥6000 ADC，首次从零加载)", big),
                    ("小台阶(<6000 ADC，受载态增量)", small)):
        if not g:
            continue
        f = np.array([r["fall"] for r in g])
        fr = np.array([r["fall_frac"] for r in g if np.isfinite(r["fall_frac"])])
        w.append("    %s：n=%d  回落中位 %+.0f ADC（p5/p95 %+.0f/%+.0f）  "
                 "占台阶中位 %+.1f%%  为正(过扣) %d/%d"
                 % (name, len(g), float(np.median(f)),
                    float(np.percentile(f, 5)), float(np.percentile(f, 95)),
                    100 * float(np.median(fr)), int(np.sum(f > 0)), len(g)))
    if len(recs) >= 6:
        st = np.array([r["step"] for r in recs])
        fa = np.array([r["fall"] for r in recs])
        rho, p = sps.spearmanr(st, fa)
        w.append("    Spearman ρ(台阶, 回落) = %+.3f (p=%.3f, n=%d)  "
                 "→ 台阶越大回落越小" % (rho, p, len(recs)))
        rho2, p2 = sps.spearmanr(st, np.array([r["fall_frac"] for r in recs]))
        w.append("    Spearman ρ(台阶, 回落/台阶) = %+.3f (p=%.3f)" % (rho2, p2))

    # ---------- 2. 归因（仅取 parity 达标的回放） ----------
    w.append("")
    w.append("-" * 104)
    w.append("[2] 回落归因（反事实回放）。parity = 回放与录制逐通道 max|Δ| ADC")
    base = {}
    for r in tot:
        if r["cas"] == "base":
            base[key_of(r)] = r
    all_ses = sorted({r["ses"] for r in tot if r["cas"] == "base"})
    for thr in (50.0, 250.0):
        par_ok = sorted({r["ses"] for r in tot
                         if r["cas"] == "base" and np.isfinite(r["parity"])
                         and r["parity"] <= thr})
        w.append("")
        w.append("  parity ≤ %.0f ADC 的会话：%s"
                 % (thr, "；".join(s.split("/")[-1][-14:] for s in par_ok) or "无"))
        ks = [k for k in base if base[k]["ses"] in par_ok and valid(base[k])]
        ks.sort(key=lambda k: (base[k]["ses"], base[k]["t0"], base[k]["W"]))
        w.append("    %-38s %6s %7s %7s %7s %7s %7s %7s %7s"
                 % ("会话/事件/W", "台阶", "回落", "Δx1", "Δx2", "输入爬升",
                    "x1份额", "x2份额", "回落/台阶"))
        shares1, shares2 = [], []
        for k in ks:
            r = base[k]
            dxs = r["d_x1"] + r["d_x2"]
            s1 = r["d_x1"] / dxs if abs(dxs) > 1e-9 else np.nan
            s2 = r["d_x2"] / dxs if abs(dxs) > 1e-9 else np.nan
            shares1.append(s1)
            shares2.append(s2)
            w.append("    %-38s %+6.0f %+7.0f %+7.0f %+7.0f %+7.0f %7s %7s %7s"
                     % ((r["ses"].split("/")[-1][-14:] + " t=%.1f W=%.0f"
                         % (r["t0"], r["W"])), r["step"], r["fall"], r["d_x1"],
                        r["d_x2"], r["v_creep"],
                        "%.0f%%" % (100 * s1) if np.isfinite(s1) else "-",
                        "%.0f%%" % (100 * s2) if np.isfinite(s2) else "-",
                        "%.1f%%" % (100 * r["fall_frac"])
                        if np.isfinite(r["fall_frac"]) else "-"))
        if shares1:
            w.append("    补偿增长中 x1 份额中位 %.0f%%，x2 份额中位 %.0f%%（n=%d）"
                     % (100 * float(np.nanmedian(shares1)),
                        100 * float(np.nanmedian(shares2)), len(shares1)))
        if thr == 50.0:
            k50 = ks
    ks = k50
    w.append("")
    w.append("[3] 反事实：改 τc1 / 改 r1 / 关 x1 / 关 x2 后的回落变化（parity ≤ 50 事件）")
    variants = ["no_x1", "no_x2", "tc1_2", "tc1_20", "r1_005", "r1_020", "shape_fit"]
    alt = {}
    for r in tot:
        if r["cas"] in variants:
            alt[(r["cas"],) + key_of(r)] = r
    w.append("    %-12s %10s %10s %10s %10s" %
             ("方案", "回落中位", "Δ回落中位", "回落/台阶中位", "n"))
    bf = np.array([base[k]["fall"] for k in ks])
    w.append("    %-12s %+10.0f %10s %10s %10d"
             % ("base(现状)", float(np.median(bf)), "-",
                "%.1f%%" % (100 * float(np.median([base[k]["fall_frac"] for k in ks]))),
                len(ks)))
    for v in variants:
        f = [alt[(v,) + k]["fall"] for k in ks if (v,) + k in alt]
        fr = [alt[(v,) + k]["fall_frac"] for k in ks if (v,) + k in alt]
        if not f:
            continue
        w.append("    %-12s %+10.0f %+10.0f %10s %10d"
                 % (v, float(np.median(f)),
                    float(np.median(f)) - float(np.median(bf)),
                    "%.1f%%" % (100 * float(np.median(fr))), len(f)))
    w.append("")
    w.append("    逐事件回落（parity ≤ 50，单位 ADC）：")
    hdr = "    %-30s" % "事件"
    for v in ["base"] + variants:
        hdr += "%10s" % v
    w.append(hdr)
    for k in ks:
        row = "    %-30s" % ("%s t=%.1f W=%.0f"
                             % (base[k]["ses"].split("/")[-1][-10:],
                                base[k]["t0"], base[k]["W"]))
        row += "%10.0f" % base[k]["fall"]
        for v in variants:
            row += "%10s" % ("%.0f" % alt[(v,) + k]["fall"]
                             if (v,) + k in alt else "-")
        w.append(row)
    w.append("")
    w.append("    说明：Δx1 是 base 回放里快态在窗口内的增长量；x1 份额 = Δx1/(Δx1+Δx2)。")
    w.append("    'no_x1' 关掉快态（r1=0）后 x2 会接手更多 ⇒ 回落不一定变小，")
    w.append("    这正是「只动 x1 参数能否解决回落」的判据。")

    txt = "\n".join(w)
    with open(os.path.join(T.RESULTS, "t1_fall_stats.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt.encode("ascii", "replace").decode("ascii"))
    make_figures(rows, recs, base, alt, ks)


def make_figures(rows, recs, base, alt, ks):
    os.makedirs(T.FIGURE, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    # 图：回落 vs 台阶
    ax = axes[0]
    st = np.array([r["step"] for r in recs])
    fa = np.array([r["fall"] for r in recs])
    w10 = np.array([r["W"] for r in recs])
    for Wv, col, mk in ((10.0, "#1f77b4", "o"), (60.0, "#ff7f0e", "s")):
        m = w10 == Wv
        ax.scatter(st[m], fa[m], s=42, alpha=0.75, color=col, marker=mk,
                   label="窗口 %.0f s" % Wv)
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("加载台阶 Δ (ADC, 通道总和)")
    ax.set_ylabel("显示回落 fall (ADC)")
    ax.set_title("图3a 实机显示回落 vs 台阶幅度\n(fall>0 = 过扣，显示下沉)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # 图：Δx1 / Δx2 / Δv 分解（parity 达标）
    ax = axes[1]
    if ks:
        idx = np.arange(len(ks))
        d1 = np.array([base[k]["d_x1"] for k in ks])
        d2 = np.array([base[k]["d_x2"] for k in ks])
        dv = np.array([base[k]["v_creep"] for k in ks])
        ax.bar(idx - 0.28, d1, 0.26, color="#d62728", label="Δx1 快态增长")
        ax.bar(idx, d2, 0.26, color="#2ca02c", label="Δx2 慢态增长")
        ax.bar(idx + 0.28, dv, 0.26, color="#7f7f7f", label="Δv 输入自身爬升")
        ax.set_xticks(idx)
        ax.set_xticklabels(["%.0fs/%.0fs" % (base[k]["t0"], base[k]["W"])
                            for k in ks], rotation=90, fontsize=7)
        ax.set_ylabel("窗口内增长量 (ADC)")
        ax.set_title("图3b 回落来源分解（回放 parity ≤ 50 ADC）\n回落 = Δx1+Δx2−Δv")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    # 图：反事实对比
    ax = axes[2]
    if ks:
        variants = [("base", None), ("no_x1", "关 x1(r1=0)"),
                    ("no_x2", "关 x2(r2max=0)"), ("tc1_2", "τc1=2 s"),
                    ("tc1_20", "τc1=20 s"), ("r1_005", "r1=0.05"),
                    ("r1_020", "r1=0.20")]
        vals, labs = [], []
        for v, lab in variants:
            if v == "base":
                vals.append([base[k]["fall"] for k in ks])
                labs.append("base\n(r1=.12,τ=8)")
            else:
                g = [alt[(v,) + k]["fall"] for k in ks if (v,) + k in alt]
                vals.append(g)
                labs.append(lab)
        bp = ax.boxplot(vals, tick_labels=labs, showfliers=False, patch_artist=True)
        for i, p in enumerate(bp["boxes"]):
            p.set_facecolor("#cfe2f3" if i else "#f4cccc")
        ax.axhline(0, color="k", lw=1.2)
        ax.set_ylabel("显示回落 fall (ADC)")
        ax.set_title("图3c 反事实：改 τc1 / r1 / 关态 后的回落")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(T.FIGURE, "T1_回落与归因.png"), dpi=120)
    plt.close(fig)
    print("figure -> %s" % os.path.join(T.FIGURE, "T1_回落与归因.png"))


if __name__ == "__main__":
    main()
