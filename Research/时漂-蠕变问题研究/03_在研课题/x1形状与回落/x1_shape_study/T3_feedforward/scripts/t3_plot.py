# -*- coding: utf-8 -*-
"""T3 步骤7：出图（中文图注）。

fig1  代表工况全程叠画（输入 / 基线显示 / 前馈显示）+ 小台阶沿后 5 s 放大
fig2  四个代表事件的沿后 6 s 放大（输入 + 各臂显示，标注回落与稳定）
fig3  各臂指标对比条形图（Δ回落 / Δ自回弹下冲 / T±5% / 电平偏置）
fig4  误触发与抖动风险（输入 + 触发标记 + 与基线显示偏差）
fig5  参数敏感性（门限 × α、τ_boost）
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import t3_lib as T
import t3_replay as R

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figure")
RES = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

C_IN = "#8c8c8c"
C_BASE = "#1f77b4"
C_A = "#d62728"
C_ALAG = "#ff7f0e"
C_C = "#2ca02c"


def find(kw):
    for tag, label, d in T.all_sessions():
        if kw in label:
            return label, d
    raise SystemExit("no session " + kw)


def arms_displays(label, d, specs):
    s = T.load_input(d)
    el, ts, V = s["el"], s["ts"], s["V"]
    out = {}
    for name, spec in specs:
        ff = None
        if spec:
            ff = dict(R.DET_BASE)
            ff.update(spec)
        out[name] = T.observe(ts, V, ff=ff)["D"].sum(axis=1)
    return el, ts, V, s["V"].sum(axis=1), out


SPECS = [("基线", None),
         ("F-A now α=1.0", dict(mode="A", pred="now", alpha=1.0)),
         ("F-A lag α=1.0", dict(mode="A", pred="lag", alpha=1.0)),
         ("F-C α=0.8 τb=2s", dict(mode="C", pred="now", alpha=0.8,
                                  tau_boost=2.0, boost_s=1.5))]


def fig1():
    label, d = find("192141")
    el, ts, V, din, disp = arms_displays(label, d, SPECS)
    edges, _ = T.find_edges(el, din)
    tgt = min(edges, key=lambda e: abs(e["t"] - 30.08))
    fig = plt.figure(figsize=(15, 8.5))
    g = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.22)
    ax = fig.add_subplot(g[0, 0])
    ax.plot(el, din, color=C_IN, lw=0.7, label="输入（算法前读数，通道总量）")
    ax.plot(el, disp["基线"], color=C_BASE, lw=0.9, label="基线显示 = v−x1−x2")
    ax.plot(el, disp["F-A lag α=1.0"], color=C_ALAG, lw=0.9, label="F-A（lag 预测，α=1.0）")
    ax.plot(el, disp["F-A now α=1.0"], color=C_A, lw=0.7, alpha=0.8, label="F-A（当前 e）")
    for e in edges:
        ax.axvline(e["t"], color="k", lw=0.4, alpha=0.25)
    ax.set_title("(a) 全程：受载态反复小台阶 —— 输入 / 基线 / 前馈显示", fontsize=11)
    ax.set_ylabel("通道总量 ADC")
    ax.set_xlabel("时间 s")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(g[0, 1])
    rr = [r for r in csv.DictReader(open(os.path.join(RES, "t3_edges.csv"), encoding="utf-8"))
          if r["session"] == label and abs(float(r["t"]) - tgt["t"]) < 0.5]
    names, falls, dips = [], [], []
    for arm in R.ARMS:
        row = [r for r in rr if r["arm"] == arm[0]]
        if not row:
            continue
        names.append(arm[0])
        falls.append(float(row[0]["fall"]))
        dips.append(float(row[0]["dip"]))
    x = np.arange(len(names))
    ax.bar(x - 0.2, falls, 0.4, color=C_BASE, label="回落 fall（ADC）")
    ax.bar(x + 0.2, dips, 0.4, color="#d62728", label="自回弹下冲 dip（ADC）")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title("(b) 该沿各臂：回落幅度与下冲", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(g[1, :])
    m = (el >= tgt["t"] - 1.0) & (el <= tgt["t"] + 5.0)
    ax.plot(el[m] - tgt["t"], din[m], color=C_IN, lw=1.0, label="输入")
    for name, col in (("基线", C_BASE), ("F-A now α=1.0", C_A),
                      ("F-A lag α=1.0", C_ALAG), ("F-C α=0.8 τb=2s", C_C)):
        ax.plot(el[m] - tgt["t"], disp[name][m], color=col, lw=1.4, label=name)
        ax.axhline(float(np.median(disp[name][tgt["j"] - 200:tgt["j"]])), color=col,
                   ls=":", lw=0.8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title("(c) 沿后 5 s 放大（点线 = 各臂段末电平）：前馈把 x1 扣除提前，显示更早落到终值附近",
                 fontsize=11)
    ax.set_xlabel("沿后时间 s")
    ax.set_ylabel("ADC")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    fig.suptitle("图 1  F-A 前馈对「沿后显示」的影响（会话 20260919_192141，沿 t=%.1f s，台阶 %.0f ADC）"
                 % (tgt["t"], tgt["step"]), fontsize=12)
    fig.savefig(os.path.join(OUT, "fig1_代表工况叠画.png"), dpi=110)
    plt.close(fig)
    print("fig1 ok", flush=True)


def fig2():
    """四个代表事件：基线 vs F-A 的沿后 6 s 放大（含回落/下冲标注）。"""
    picks = [("192141", 4.578),
             ("192141", 30.08),
             ("193320", 13.97),
             ("160854", 15.11)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for ax, (kw, t0) in zip(axes.ravel(), picks):
        label, d = find(kw)
        el, ts, V, din, disp = arms_displays(label, d, SPECS)
        edges, _ = T.find_edges(el, din)
        tgt = min(edges, key=lambda e: abs(e["t"] - t0))
        m = (el >= tgt["t"] - 1.0) & (el <= tgt["t"] + 6.0)
        ax.plot(el[m] - tgt["t"], din[m], color=C_IN, lw=1.0, label="输入")
        ax.plot(el[m] - tgt["t"], disp["基线"][m], color=C_BASE, lw=1.3, label="基线")
        ax.plot(el[m] - tgt["t"], disp["F-A lag α=1.0"][m], color=C_ALAG, lw=1.3,
                label="F-A（lag，α=1.0）")
        ax.plot(el[m] - tgt["t"], disp["F-C α=0.8 τb=2s"][m], color=C_C, lw=1.0,
                label="F-C（α=0.8+τboost）")
        ax.set_title("%s  t=%.1f s  台阶%.0f ADC" % (label.split("/")[-1][-18:], tgt["t"], tgt["step"]),
                     fontsize=10)
        ax.set_xlabel("沿后时间 s")
        ax.set_ylabel("ADC")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("图 2  沿后 6 s 局部放大：基线 vs F-A / F-C（前馈把扣除提前，回落更快收敛）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig2_沿后放大.png"), dpi=110)
    plt.close(fig)
    print("fig2 ok", flush=True)


def fig3():
    with open(os.path.join(RES, "t3_summary.json"), encoding="utf-8") as fh:
        js = json.load(fh)
    blk = js["blocks"]["cls_受载态小台阶"]
    arms = [a for a, _ in R.ARMS if a in blk]
    base = blk["base"]
    fall = [blk[a]["fall"] - base["fall"] for a in arms]
    dip = [blk[a]["dip"] - base["dip"] for a in arms]
    lvl = [blk[a]["lvl"] for a in arms]
    t5 = [blk[a]["t5"] for a in arms]
    x = np.arange(len(arms))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    ax.bar(x - 0.2, fall, 0.4, color="#1f77b4", label="Δ回落 fall（负 = 比基线回落更小）")
    ax.bar(x + 0.2, dip, 0.4, color="#d62728", label="Δ自回弹下冲 dip（正 = 比基线更差）")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("相对基线 ADC")
    ax.set_title("(a) 受载态小台阶：相对基线的回落与下冲变化（中位）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax = axes[1]
    ax.bar(x, lvl, 0.5, color="#9467bd", label="电平偏置 lvl（负 = 比基线扣得更多）")
    ax.axhline(0, color="k", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, t5, "o--", color="#2ca02c", label="T±5% 中位（s，仅定义者）")
    ax2.set_ylabel("稳定时间 s")
    ax.set_ylabel("ADC")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=25, fontsize=8, ha="right")
    ax.set_title("(b) 稳态电平偏置与稳定时间", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax2.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("图 3  各前馈臂相对基线的指标对比（全数据集，受载态小台阶类）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig3_指标对比.png"), dpi=110)
    plt.close(fig)
    print("fig3 ok", flush=True)


def fig4():
    label, d = find("134726")     # 零负载-随机切换工况
    el, ts, V, din, disp = arms_displays(label, d, SPECS[:3])
    s = T.load_input(d)
    ff = dict(R.DET_BASE)
    ff.update(dict(mode="A", pred="lag", alpha=1.0))
    trig = T.observe(ts, V, ff=ff)["trig"]
    ev, ms = None, None
    rng = float(din.max() - din.min())
    rise = din - np.concatenate([[din[0]] * 100, din[:-100]])
    ms = max(500.0, 0.05 * rng)
    ev, last = [], -1e9
    for i in range(len(el)):
        if rise[i] > ms and el[i] - last > 2.0:
            ev.append(el[i])
            last = el[i]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 2.2, 1.2]})
    ax = axes[0]
    ax.plot(el, din, color=C_IN, lw=0.7, label="输入")
    ax.plot(el, disp["基线"], color=C_BASE, lw=0.9, label="基线显示")
    ax.plot(el, disp["F-A lag α=1.0"], color=C_ALAG, lw=0.9, label="F-A（lag，α=1.0）")
    for t0 in ev:
        ax.axvline(t0, color="#2ca02c", lw=0.6, alpha=0.5)
    ax.set_title("(a) 随机切换工况：输入 / 基线 / 前馈显示（绿线 = 独立检测的真实加载事件）", fontsize=11)
    ax.set_ylabel("ADC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    tt = np.array([float(el[i]) for c, i, sl, ep, xb in trig])
    ee = np.array([ep for c, i, sl, ep, xb in trig])
    onmask = T.event_mask(el, ev, 0.5, 2.0)
    on = onmask[np.clip(np.searchsorted(el, tt), 0, len(el) - 1)] if len(tt) else np.array([])
    ax.stem(tt[on], ee[on] * 0.12 * 21, linefmt="C3-", markerfmt="C3.", basefmt=" ",
            label="事件窗内触发（x1 置位增量，总通道）")
    if len(tt):
        ax.stem(tt[~on], ee[~on] * 0.12 * 21, linefmt="C0-", markerfmt="C0.", basefmt=" ",
                label="事件窗外触发（误触发影响力）")
    ax.set_ylabel("x1 置位增量 ADC")
    ax.set_title("(b) 触发点与置位幅度（事件窗外触发 = 误触发，幅度小才无害）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(el, disp["F-A lag α=1.0"] - disp["基线"], color="#d62728", lw=0.7)
    ax.axhline(0, color="k", lw=0.6)
    for t0 in ev:
        ax.axvspan(t0 - 0.5, t0 + 20.0, color="#eeeeee", zorder=0)
    ax.set_ylabel("前馈−基线 ADC")
    ax.set_xlabel("时间 s")
    ax.set_title("(c) 与基线显示的差（灰底 = 事件后 20 s 窗口；窗外差 = 误触发/状态残留影响）", fontsize=11)
    ax.grid(alpha=0.25)
    fig.suptitle("图 4  误触发与抖动风险（会话 20260919_134726 零负载-随机切换）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "fig4_误触发风险.png"), dpi=110)
    plt.close(fig)
    print("fig4 ok", flush=True)


def fig5():
    path = os.path.join(RES, "t3_sensitivity.csv")
    if not os.path.isfile(path):
        print("fig5 跳过（无 t3_sensitivity.csv）")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        for k in ("thr", "rel", "alpha", "tau_boost", "boost_s", "fall", "dip",
                  "dip_max", "t5", "lvl", "n_trig", "n_trig_off", "dev_p95", "dev_idle"):
            r[k] = float(r[k])
    base = [r for r in rows if r["arm"] == "base"]
    b = base[0] if base else None
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    g1 = [r for r in rows if r["arm"].startswith("thr")]
    ax = axes[0]
    ax.plot([r["thr"] for r in g1], [r["fall"] for r in g1], "o-", color=C_BASE, label="回落 fall")
    ax.plot([r["thr"] for r in g1], [r["dip"] for r in g1], "s-", color="#d62728", label="自回弹下冲 dip")
    if b:
        ax.axhline(b["fall"], color=C_BASE, ls=":", lw=0.9, label="基线回落")
        ax.axhline(b["dip"], color="#d62728", ls=":", lw=0.9, label="基线下冲")
    ax.set_xlabel("绝对门限 thr（ADC/s）")
    ax.set_ylabel("ADC")
    ax.set_title("(a) 沿门限敏感性（F-A lag, α=1.0）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    g2 = [r for r in rows if r["arm"].startswith("A_lag_a")]
    g2.sort(key=lambda r: r["alpha"])
    ax.plot([r["alpha"] for r in g2], [r["fall"] for r in g2], "o-", color=C_BASE, label="回落 fall")
    ax.plot([r["alpha"] for r in g2], [-r["lvl"] for r in g2], "^-", color="#9467bd",
            label="−电平偏置（正值=扣得更多）")
    ax.plot([r["alpha"] for r in g2], [r["dip"] for r in g2], "s-", color="#d62728", label="下冲 dip")
    ax.set_xlabel("置位系数 α")
    ax.set_title("(b) α 敏感性（F-A lag）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    g3 = [r for r in rows if r["arm"].startswith("B_tb") or r["arm"].startswith("C_a0.8_lag_tb2.0_bs")]
    tb = [r for r in g3 if r["arm"].startswith("B_tb")]
    tb.sort(key=lambda r: r["tau_boost"])
    if tb:
        ax.plot([r["tau_boost"] for r in tb], [r["fall"] for r in tb], "o-", color=C_BASE,
                label="回落 fall（F-B）")
        ax.plot([r["tau_boost"] for r in tb], [r["n_trig"] for r in tb], "x--",
                color="#7f7f7f", label="触发次数（F-B）")
    ax.set_xlabel("τ_boost（s）")
    ax.set_title("(c) τ_boost 敏感性（F-B，boost 1.5 s）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle("图 5  参数敏感性（代表会话子集：回落/下冲/触发数）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "fig5_参数敏感性.png"), dpi=110)
    plt.close(fig)
    print("fig5 ok", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    if "1" in which:
        fig1()
    if "2" in which:
        fig2()
    if "3" in which:
        fig3()
    if "4" in which:
        fig4()
    if "5" in which:
        fig5()
