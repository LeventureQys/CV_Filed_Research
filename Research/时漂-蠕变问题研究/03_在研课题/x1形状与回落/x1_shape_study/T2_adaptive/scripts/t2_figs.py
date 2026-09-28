# -*- coding: utf-8 -*-
"""T2 步骤4：出图（中文图注）。

图1  代表性工况：输入 / 基线显示 / 各变体显示 叠画 + 下冲局部放大
图2  基线剖面：y、e、x1、x2、zero 与显示，标出「回落」的组成
图3  权衡曲线：回落深度 vs 保压偏差（各变体/扫描点），并标出基线
图4  逐事件回落柱状对比（全部门槛内事件）
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import t2_lib as T  # noqa: E402
import t2_observer as O  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(T.OUT_ROOT, "figure")
os.makedirs(FIG, exist_ok=True)

REP = ["193320", "持续恒定负载", "7b3977"]  # 代表工况（干净保压 / 长保压 / 反复增减）
SHOW = ["base", "A_amp", "B_tau_k4", "B_bi_light", "B_bi2", "C_lp_t3"]


def load_one(key):
    for label, d in T.all_sessions():
        if key in label:
            s = T.load_pre(d)
            rec = T.load_recorded(d)
            return label, s, (T.total(rec) if rec is not None else None)
    raise SystemExit("no session " + key)


def fig1():
    for key in REP:
        label, s, drec = load_one(key)
        el, tot = s["el"], T.total(s)
        evs = T.load_events(el, tot, win=60.0)
        ev = None
        for e in evs:
            m = T.event_metrics(np.zeros_like(tot), tot, el, e)
            if m and T.event_gate(m, e):
                ev = e
                break
        if ev is None:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                                 gridspec_kw={"height_ratios": [2.4, 1.0]})
        ax = axes[0]
        ax.plot(el, tot, color="#999999", lw=0.9, label="输入（pre，算法前读数）")
        for name in SHOW:
            variant, p_over, _d = O.VARIANTS[name]
            r = O.observe(el.copy(), s["V"], variant, p=p_over)
            ax.plot(el, r["D"], lw=0.9,
                    label=("基线 v3.4" if name == "base" else name))
        ax.axvline(ev["t_up"], color="#d62728", ls="--", lw=0.8)
        ax.axvline(ev["t_up"] + 10.0, color="#d62728", ls=":", lw=0.8)
        ax.axvline(ev["t_up"] + 20.0, color="#2ca02c", ls=":", lw=0.8)
        ax.axvline(ev["t_up"] + 45.0, color="#2ca02c", ls=":", lw=0.8)
        ax.set_xlim(max(0, ev["t_up"] - 15), min(el[-1], ev["t_up"] + 90))
        ax.set_ylabel("通道总量 ADC")
        ax.set_title("图1  %s：输入与各变体显示（红虚线=加载沿/取峰窗，绿点线=平台窗）"
                     % label, fontsize=11)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        ax.grid(alpha=0.25)
        ax = axes[1]
        t0, t1 = ev["t_up"] - 1.0, ev["t_up"] + 40.0
        m = (el >= t0) & (el <= t1)
        ax.plot(el[m], tot[m], color="#999999", lw=1.0, label="输入")
        for name in SHOW:
            variant, p_over, _d = O.VARIANTS[name]
            r = O.observe(el.copy(), s["V"], variant, p=p_over)
            ax.plot(el[m], r["D"][m], lw=1.1,
                    label=("基线 v3.4" if name == "base" else name))
        ax.set_xlabel("时间 s")
        ax.set_ylabel("放大：显示 ADC")
        ax.set_title("下冲局部放大（沿后 0–40 s，显示自身）", fontsize=10)
        ax.legend(loc="lower right", fontsize=8, ncol=3)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fn = "T2_图1_%s.png" % T.session_key(label)[:40]
        fig.savefig(os.path.join(FIG, fn), dpi=110)
        plt.close(fig)
        print("图1 ->", fn)


def fig2():
    label, s, drec = load_one(REP[0])
    el, tot = s["el"], T.total(s)
    evs = T.load_events(el, tot, win=60.0)
    ev = None
    for e in evs:
        m = T.event_metrics(np.zeros_like(tot), tot, el, e)
        if m and T.event_gate(m, e):
            ev = e
            break
    r = O.observe(el.copy(), s["V"], "base")
    zero = tot - r["Y"]
    m = (el >= ev["t_up"] - 2) & (el <= ev["t_up"] + 40)
    t = el[m] - ev["t_up"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.4]})
    ax = axes[0]
    ax.plot(t, tot[m], color="#999999", lw=1.0, label="输入 v（pre）")
    ax.plot(t, r["D"][m], color="#1f77b4", lw=1.2, label="显示 = v−x1−x2")
    ax.plot(t, r["Y"][m], color="#8c564b", lw=1.0, ls="--", label="y = v−zero")
    ax.plot(t, zero[m], color="#7f7f7f", lw=0.9, ls=":", label="zero 零点")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylabel("通道总量 ADC")
    ax.grid(alpha=0.25)
    res = T.event_metrics(r["D"], tot, el, ev, ylp=T.lowpass(r["D"], el, tau=1.0),
                          xlp=T.lowpass(tot, el, tau=5.0))
    ax.set_title("图2  %s：基线剖面（回落=峰值%.0f − 平台%.0f = %.0f ADC，"
                 "占台阶 %.2f%%）"
                 % (label, res["peak"], res["plateau"], res["drop"],
                    100 * res["drop_rel"]), fontsize=11)
    ax = axes[1]
    ax.plot(t, r["X1"][m], color="#d62728", lw=1.1, label="x1 快态")
    ax.plot(t, r["X2"][m], color="#2ca02c", lw=1.1, label="x2 慢态")
    ax.plot(t, r["E"][m], color="#9467bd", lw=1.0, label="e = max(y−x1−x2,0)")
    ax.plot(t, (zero - zero[0])[m], color="#7f7f7f", lw=0.9, ls=":",
            label="zero 漂移（相对沿）")
    ax.set_xlabel("相对加载沿的时间 s")
    ax.set_ylabel("ADC")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fn = "T2_图2_基线剖面_%s.png" % T.session_key(label)[:32]
    fig.savefig(os.path.join(FIG, fn), dpi=110)
    plt.close(fig)
    print("图2 ->", fn)


def fig3():
    """权衡曲线：回落深度（越小越好） vs 保证段偏差（越大越接近 0 越好）。"""
    sess, evc = {}, {}
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el, tot = s["el"], T.total(s)
        evc[label] = T.load_events(el, tot, win=60.0)
        sess[label] = dict(el=el, V=s["V"], tot=tot, xlp=T.lowpass(tot, el, 5.0))
    pts = []
    for name, (variant, p_over, desc) in O.VARIANTS.items():
        drops, errs, stds, exc = [], [], [], []
        for label, c in sess.items():
            evs = evc[label]
            if not evs:
                continue
            r = O.observe(c["el"].copy(), c["V"], variant, p=p_over)
            ylp = T.lowpass(r["D"], c["el"], 1.0)
            sm = T.summarize(label, r["D"], c["tot"], c["el"], evs,
                             subset=True, xlp=c["xlp"], ylp=ylp, x1=r["X1"])
            if sm["n_ev"]:
                drops.append(sm["drop_mean"])
                errs.append(sm["seg_err_mean"])
                stds.append(sm["std_mean"])
                exc.append(sm["x1_exc_mean"])
        if drops:
            pts.append((name, float(np.mean(drops)), float(np.mean(errs)),
                        float(np.mean(stds)), float(np.mean(exc))))
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, dr, er, sd, ex in pts:
        c = "#d62728" if name == "base" else ("#1f77b4" if name.startswith("S_")
                                              else "#2ca02c")
        ax.scatter(dr, er, s=70, color=c, zorder=3)
        ax.annotate(name, (dr, er), fontsize=8, xytext=(4, 3),
                    textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    bx = [p for p in pts if p[0] == "base"][0]
    ax.axvline(bx[1], color="#d62728", lw=0.8, ls=":")
    ax.axhline(bx[2], color="#d62728", lw=0.8, ls=":")
    ax.annotate("基线 v3.4", (bx[1], bx[2]), fontsize=10, color="#d62728",
                xytext=(8, -16), textcoords="offset points")
    ax.set_xlabel("回落深度（[2,10]s 显示峰值 − [20,45]s 平台，ADC；越小越好）")
    ax.set_ylabel("保证段偏差 = 显示−输入低通（ADC；越接近 0 = 显示掉落越少）")
    ax.set_title("图3  权衡曲线（干净保压事件宏平均，13 会话）\n"
                 "红=基线 v3.4；蓝=仅改 r1/τc1 的参数扫描；绿=形状自适应变体；"
                 "左下角方向=两个指标同时改善", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "T2_图3_权衡曲线.png"), dpi=110)
    plt.close(fig)
    print("图3 -> T2_图3_权衡曲线.png")
    with open(os.path.join(T.OUT_ROOT, "results", "t2_tradeoff.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("%-16s %12s %14s %10s %12s\n"
                 % ("变体", "回落ADC", "保证段偏差", "保压std", "x1超调"))
        for name, dr, er, sd, ex in pts:
            fh.write("%-16s %12.1f %14.1f %10.1f %12.1f\n" % (name, dr, er, sd, ex))


def fig4():
    sess, evc = {}, {}
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el, tot = s["el"], T.total(s)
        evs = T.load_events(el, tot, win=60.0)
        keep = []
        for e in evs:
            m = T.event_metrics(np.zeros_like(tot), tot, el, e)
            if m and T.event_gate(m, e):
                keep.append(e)
        if keep:
            evc[label] = keep
            sess[label] = dict(el=el, V=s["V"], tot=tot,
                               xlp=T.lowpass(tot, el, 5.0))
    names = [n for n in SHOW if n in O.VARIANTS]
    labels = list(sess)
    fig, ax = plt.subplots(figsize=(15, 7))
    width = 0.8 / (len(names) + 1)
    xs = np.arange(len(labels))
    for k, name in enumerate(["base"] + [n for n in names if n != "base"]):
        variant, p_over, _d = O.VARIANTS[name]
        vals = []
        for label in labels:
            c = sess[label]
            r = O.observe(c["el"].copy(), c["V"], variant, p=p_over)
            ylp = T.lowpass(r["D"], c["el"], 1.0)
            ds = []
            for e in evc[label]:
                m = T.event_metrics(r["D"], c["tot"], c["el"], e, xlp=c["xlp"],
                                    ylp=ylp)
                if m:
                    ds.append(m["drop"])
            vals.append(float(np.mean(ds)) if ds else np.nan)
        ax.bar(xs + k * width, vals, width=width,
               label=("基线 v3.4" if name == "base" else name))
    ax.set_xticks(xs + 0.4)
    ax.set_xticklabels([l[-34:].replace("/", "\n") for l in labels],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("回落深度 ADC（峰值−平台）")
    ax.set_title("图4  逐会话回落深度对比（负值=10s 窗内显示尚未见峰、未回落）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "T2_图4_逐会话回落.png"), dpi=110)
    plt.close(fig)
    print("图4 -> T2_图4_逐会话回落.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    if "1" in which:
        fig1()
    if "2" in which:
        fig2()
    if "3" in which:
        fig3()
    if "4" in which:
        fig4()
