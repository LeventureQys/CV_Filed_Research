# -*- coding: utf-8 -*-
"""plan-v2.0 离线验证 · 折中方案 ②′（hybrid）：一致性优先的 A/B + 出图。

臂：
  raw      逐帧原样（现役交付）
  packavg  始终做包内去涨落（候选②，agg=median）
  hybrid   仅抖动时做包内去涨落（候选②′，agg=median）★
均用 `batch_runner_exp.exe`（不改产品代码）。

**主指标 = 反复同一荷载的一致性**（用户明确要求"不容损失"）：
  对每个加载沿 t：
    A      = 输入在 t 处的台阶（输入[5 s 后] − 输入[t−1.2 s]）
    inc(τ) = (显示[t+τ] − 输入[t+τ]) − (显示[t−1.2..−0.35] 中位 − 输入[同窗] 中位)
           = "相对于输入的显示增补量"，τ 固定是关键（同一荷载应在同一 τ 给出同一个值）
  一致性 = 各沿 inc(τ) 的 **极差**（越小越好 ⇒ 同一荷载给了同一个显示）
  另有 Δτ = inc(τ) − inc(τ=10 s)（最终锚定偏差）。

副指标：振荡数据上的抗抖、安静段偏移（是否被带偏），以及干净数据上的 Δ1s / G。
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.abspath(os.path.join(SCRIPTS, "build", "batch_runner_exp.exe"))
ODD = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                   "20260919_134056_single_device_f40a1b")
OUT = os.path.abspath(os.path.join(SCRIPTS, "..", "results"))
FIG = os.path.join(OUT, "figures")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def pick_font():
    for name in ("Microsoft YaHei", "SimHei", "DengXian"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK = pick_font()


def T(zh, en):
    return zh if CJK else en


def feed(el, V, mode, agg="median"):
    n = V.shape[1]
    lines = ["%d 100" % n]
    for i in range(len(el)):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    r = subprocess.run([RUNNER, mode, agg], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    rows = r.stdout.splitlines()
    body = rows[1:-1]
    si = np.array([float(x.split()[1]) for x in body])
    so = np.array([float(x.split()[2]) for x in body])
    c = [float(x) for x in rows[-1].split()[1:]]
    return si, so, c


def edges_of(el, tot):
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot, 3), np.percentile(tot, 97)
    if hi - lo <= 0:
        return []
    up, _ = L.edges_from_tot(np.convolve(tot, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    return tu


def consistency(el, tp, so, tu, taus=(1.0, 2.0, 5.0, 10.0)):
    """返回 {τ: (极差, 各沿值列表, 台阶列表)}"""
    res = {}
    for tau in taus:
        vals, amps = [], []
        for t in tu:
            m0 = (el >= t - 1.2) & (el < t - 0.35)
            m1 = (el >= t + tau - 0.15) & (el < t + tau + 0.15)
            m2 = (el >= t + 10.0 - 1.0) & (el < t + 10.0)
            if m0.sum() < 5 or m1.sum() < 1 or m2.sum() < 5:
                continue
            inc0 = float(np.median(so[m0] - tp[m0]))
            vals.append(float(np.median(so[m1] - tp[m1])) - inc0)
            amps.append(float(np.median(tp[m2])) - float(np.median(tp[m0])))
        if len(vals) >= 2:
            res[tau] = (float(np.max(vals) - np.min(vals)), vals, amps)
    return res


def main():
    os.makedirs(FIG, exist_ok=True)
    pre = L.load_stream(ODD, "device_001_pre_seg0.csv")
    elO, VO = pre["el"], pre["V"]
    tpO = VO.sum(1)

    pre2 = L.load_stream(L.DS_ZERO, "device_001_pre_seg0.csv")
    elN, VN = pre2["el"], pre2["V"]
    tpN = VN.sum(1)
    tuN = edges_of(elN, tpN)

    arms = {}
    for mode in ("raw", "packavg", "hybrid"):
        arms[mode] = {"odd": feed(elO, VO, mode), "new": feed(elN, VN, mode)}

    print("=== ① 折叠命中：抖动时是否有去涨落（振荡数据）===")
    print("%-9s %12s %10s %12s %12s" %
          ("臂", "agg组数", "总组数", "去涨落帧", "shape_hits"))
    for m in ("raw", "packavg", "hybrid"):
        _si, _so, c = arms[m]["odd"]
        print("%-9s %12.0f %10.0f %12.0f %12.0f"
              % (m, c[10], c[11], c[12], c[0]))

    print("\n=== ② 振荡数据：抗抖与安静段偏移 ===")
    print("%-9s %14s %16s %18s" % ("臂", "振荡段偏移std", "振荡段极差", "安静段偏移中位"))
    for m in ("raw", "packavg", "hybrid"):
        si, so, _c = arms[m]["odd"]
        off = so - si
        mo = (elO >= 69.7) & (elO < 71.7)
        mq = (elO >= 108) & (elO < 118)
        print("%-9s %14.0f %16.0f %18.0f"
              % (m, off[mo].std(), off[mo].max() - off[mo].min(), np.median(off[mq])))

    print("\n=== ③ ★反复同一荷载的一致性（新录制 %d 个加载沿）===" % len(tuN))
    print("   inc(τ) = 沿后 τ 的（显示−输入）− 沿前的（显示−输入）；同一荷载应在同一 τ 给同一值")
    print("   一致性判据 = 各沿 inc(τ) 的**极差**（越小越好）")
    cons = {}
    print("%-9s %14s %14s %14s %14s %16s" %
          ("臂", "极差@1s", "极差@2s", "极差@5s", "极差@10s", "台阶幅度中位"))
    for m in ("raw", "packavg", "hybrid"):
        si, so, _c = arms[m]["new"]
        r = consistency(elN, tpN, so, tuN)
        cons[m] = r
        row = [m]
        for tau in (1.0, 2.0, 5.0, 10.0):
            row.append("%14.0f" % r[tau][0] if tau in r else "%14s" % "-")
        amps = r.get(5.0, (0, [], []))[2]
        row.append("%16.0f" % np.median(amps) if amps else "%16s" % "-")
        print("%-9s %14s %14s %14s %14s %16s" % tuple(row))

    print("\n=== ④ 干净数据其它指标（新录制）===")
    print("%-9s %12s %12s %16s %12s" %
          ("臂", "Δ1s中位", "G中位", "受载段偏移中位", "全程极差"))
    for m in ("raw", "packavg", "hybrid"):
        si, so, _c = arms[m]["new"]
        d1s, gs = [], []
        for t in tuN:
            m0 = (elN >= t - 1.2) & (elN < t - 0.35)
            if m0.sum() < 5:
                continue
            bp = float(np.median(tpN[m0]))
            te = min(t + 30.0, elN[-1] - 0.2)
            m1 = (elN >= te - 1.5) & (elN <= te)
            A_ = float(np.median(tpN[m1])) - bp
            m2 = (elN >= t + 0.8) & (elN <= t + 1.2)
            d1s.append(float(np.median(so[m2] - tpN[m2])) if m2.any() else np.nan)
            m3 = (elN >= t + 4.0) & (elN <= t + 5.0)
            if m3.any() and abs(A_) > 500:
                gs.append((float(np.median(so[m3])) - float(np.median(so[m0]))) / A_)
        off = so - si
        print("%-9s %12.0f %12.3f %16.0f %12.0f"
              % (m, np.nanmedian(d1s), np.nanmedian(gs),
                 np.median(off[tpN > np.percentile(tpN, 60)]),
                 off.max() - off.min()))

    make_fig(elO, tpO, arms, elN, tpN, tuN, cons)
    return arms, cons


def make_fig(elO, tpO, arms, elN, tpN, tuN, cons):
    fig = plt.figure(figsize=(18, 11.5), dpi=110)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.15, 1.0], hspace=0.42, wspace=0.22)

    # (0,0) 振荡数据 66~84 s 显示输出
    ax = fig.add_subplot(gs[0, 0])
    a, b = 70.0, 76.0
    m = (elO >= a) & (elO < b)
    ax.plot(elO[m], tpO[m], color="#9aa0a6", lw=0.6, label=T("输入", "input"))
    for k, col in (("raw", "#d93025"), ("packavg", "#1a73e8"), ("hybrid", "#188038")):
        ax.plot(elO[m], arms[k]["odd"][1][m], color=col, lw=1.2, label=k)
    ax.set_title(T("振荡工况 70~76 s：三臂显示输出", "oscillating 70~76 s"), fontsize=10)
    ax.set_ylabel(T("显示总量 (ADC)", "display (ADC)"), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # (0,1) 振荡工况偏移（0.2 s 平滑）
    ax = fig.add_subplot(gs[0, 1])
    for k, col in (("raw", "#d93025"), ("packavg", "#1a73e8"), ("hybrid", "#188038")):
        si, so, _c = arms[k]["odd"]
        off = so - si
        kk = 20
        ax.plot(elO[m], np.convolve(off, np.ones(kk) / kk, mode="same")[m], color=col, lw=1.5, label=k)
    ax.axhline(0, color="#5f6368", lw=0.8, ls=":")
    ax.set_title(T("振荡工况 70~76 s：显示偏移（0.2 s 平滑）", "offset (0.2 s smoothed)"), fontsize=10)
    ax.set_ylabel(T("显示 − 输入 (ADC)", "display - input (ADC)"), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)

    # (1,0) ★一致性：各沿 inc(5s) 散点
    ax = fig.add_subplot(gs[1, 0])
    for k, col, mk in (("raw", "#d93025", "o"), ("packavg", "#1a73e8", "s"),
                       ("hybrid", "#188038", "^")):
        r = cons[k].get(5.0)
        if not r:
            continue
        vals, amps = r[1], r[2]
        ax.scatter(amps, vals, color=col, marker=mk, s=70, alpha=0.85,
                   label=T("%s（极差 %.0f ADC）" % (k, r[0]), "%s (range %.0f)" % (k, r[0])))
    ax.axhline(0, color="#5f6368", lw=0.8, ls=":")
    ax.set_xlabel(T("该次加载的输入台阶 A (ADC)", "input step A (ADC)"), fontsize=9)
    ax.set_ylabel(T("inc(5 s) = 显示相对输入的增补量 (ADC)", "inc(5 s) ADC"), fontsize=9)
    ax.set_title(T("★ 反复同一荷载一致性：同一台阶幅度应落在同一 inc 上\n"
                   "（点越分散 = 同一荷载给出的显示越不一致）",
                   "Repeat-load consistency (scatter = poor consistency)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)

    # (1,1) 一致性随 τ 的极差
    ax = fig.add_subplot(gs[1, 1])
    taus = [1.0, 2.0, 5.0, 10.0]
    w = 0.26
    xs = np.arange(len(taus))
    for i, (k, col) in enumerate((("raw", "#d93025"), ("packavg", "#1a73e8"),
                                  ("hybrid", "#188038"))):
        vals = [cons[k][t][0] if t in cons[k] else np.nan for t in taus]
        ax.bar(xs + (i - 1) * w, vals, width=w, color=col, label=k)
        for x, v in zip(xs + (i - 1) * w, vals):
            if v == v:
                ax.text(x, v, "%.0f" % v, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([T("沿后 %g s" % t, "t+%gs" % t) for t in taus])
    ax.set_ylabel(T("各沿 inc 的极差 (ADC)", "range across loads (ADC)"), fontsize=9)
    ax.set_title(T("一致性 vs 测量时刻 τ（越矮越好；这是用户要求「不容损失」的指标）",
                   "Consistency vs tau (lower = better)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=8)

    # (2,*) 干净数据上的 Δ1s 与 G（代价）
    ax = fig.add_subplot(gs[2, 0])
    for k, col in (("raw", "#d93025"), ("packavg", "#1a73e8"), ("hybrid", "#188038")):
        r = cons[k].get(1.0)
        if not r:
            continue
        ax.scatter(r[2], r[1], color=col, s=70, alpha=0.85, label=k)
    ax.axhline(0, color="#5f6368", lw=0.8, ls=":")
    ax.set_xlabel(T("输入台阶 A (ADC)", "input step A (ADC)"), fontsize=9)
    ax.set_ylabel(T("inc(1 s) (ADC)", "inc(1 s)"), fontsize=9)
    ax.set_title(T("沿后 1 s 的增补量（响应快慢的可视化）", "inc at t+1 s"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    labels, raws, pas, hys = [], [], [], []
    for m_ in (np.ones(len(tpN), bool),):
        pass
    labs = [T("受载段偏移中位", "loaded offset"), T("全程偏移极差", "overall range")]
    rw, pa, hy = [], [], []
    for k, arr in (("raw", rw), ("packavg", pa), ("hybrid", hy)):
        si, so, _c = arms[k]["new"]
        off = so - si
        arr.append(np.median(off[tpN > np.percentile(tpN, 60)]))
        arr.append(off.max() - off.min())
    xs = np.arange(2)
    ax.bar(xs - 0.26, rw, 0.26, color="#d93025", label="raw")
    ax.bar(xs, pa, 0.26, color="#1a73e8", label="packavg")
    ax.bar(xs + 0.26, hy, 0.26, color="#188038", label="hybrid")
    for i, v in enumerate(rw):
        ax.text(xs[i] - 0.26, v, "%.0f" % v, ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(pa):
        ax.text(xs[i], v, "%.0f" % v, ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(hy):
        ax.text(xs[i] + 0.26, v, "%.0f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_ylabel(T("ADC", "ADC"), fontsize=9)
    ax.set_title(T("干净数据（新录制 337 s）：代价检查", "clean-data cost check"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle(T("折中方案 ②′ 离线验证：raw / packavg(始终) / hybrid(仅抖动时) —— 一致性优先",
                   "Offline A/B: raw / packavg / hybrid"), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = os.path.join(FIG, "fig_hybrid_ab.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
