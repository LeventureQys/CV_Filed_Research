# -*- coding: utf-8 -*-
"""为零负载-中途切换负载-零负载-切换负载 下的两份录制出图。

figures/rec_<tag>_result.png  每份：① 全长时序 ② v4/v4r 相对 v3 的扣除量之差 ③ 自动挑选的放大窗
figures/rec_two_metrics.png   两份合并：事件台账（比值 vs 是否识别）、慢相窗时漂、最大偏差
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.facecolor"] = "white"

COL = {"raw": "#9e9e9e", "v3": "#1f77b4", "v4_fast5": "#d62728", "v4r_fast5": "#e08a00"}
LBL = {"raw": "原始（无补偿）", "v3": "GLM53 v3（现役）",
       "v4_fast5": "v4 免责5s（最新算法）", "v4r_fast5": "v4r（免责5s + 变载也免责）"}
LW = {"raw": 0.9, "v3": 1.15, "v4_fast5": 2.6, "v4r_fast5": 1.15}
LS = {"raw": "-", "v3": "-", "v4_fast5": "-", "v4r_fast5": (0, (6, 2))}
AL = {"raw": 0.80, "v3": 0.95, "v4_fast5": 0.45, "v4r_fast5": 0.85}
ZO = {"raw": 1, "v3": 3, "v4_fast5": 2, "v4r_fast5": 4}
ORDER = ["raw", "v3", "v4_fast5", "v4r_fast5"]
TAGS = ["1d9493", "13ffca"]
TITLE = {"1d9493": "20260916_212140_single_device_1d9493（64s）",
         "13ffca": "20260917_101646_single_device_13ffca（121s，最终测试目标）"}


def smooth(x, n):
    return pd.Series(x).rolling(max(3, int(n)), center=True, min_periods=1).median().to_numpy()


def audit(fig, name):
    if os.environ.get("FIG_CHECK") != "1":
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.axes:
        for t in ax.texts:
            bb = t.get_window_extent(r)
            if not (bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.x1 + 1 and bb.y1 <= fig.bbox.y1 + 1):
                bad.append(f"越出画布: {t.get_text()[:20]!r}")
            elif bb.x0 < ax.bbox.x0 - 2 or bb.x1 > ax.bbox.x1 + 2 or \
                    bb.y0 < ax.bbox.y0 - 2 or bb.y1 > ax.bbox.y1 + 2:
                bad.append(f"越出坐标区: {t.get_text()[:20]!r}")
    print(f"[audit {name}] " + ("OK" if not bad else f"{len(bad)} 处: " + "; ".join(bad)))


def mid_load(ev):
    if not len(ev):
        return ev
    thr = 0.30 * ev.pre.max()
    return ev[(ev.pre > thr) & (ev.post > thr)]


def pick_zooms(d, ev, k=2):
    tu, span = d["tu"], d["span"]
    wins = []
    for a, _ in d["periods"][:1]:
        lo, hi = max(0.0, float(tu[a]) - 3.0), min(span, float(tu[a]) + 12.0)
        wins.append((lo, hi, "载入沿（含 v4 免责期）"))
    if len(ev):
        cand = mid_load(ev)
        if not len(cand):
            cand = ev
        for _, r in cand.reindex(cand.jump.abs().sort_values(ascending=False).index).iterrows():
            lo, hi = max(0.0, r["t"] - 6.0), min(span, r["t"] + 14.0)
            if any(hi > w[0] and lo < w[1] for w in wins):
                continue
            kind = "变载" if pd.notna(r["restep"]) else "**漏检台阶**"
            wins.append((lo, hi, f"t={r['t']:.1f}s {kind} Δ={r['jump']:+.0f}（比值 {r['ratio']:.2f}）"))
            if len(wins) >= 1 + k:
                break
    return wins[:3]


all_ev, all_st = [], []
for tag in TAGS:
    f = os.path.join(RES, f"rec_{tag}.npz")
    if not os.path.exists(f):
        print(f"[skip] 缺 {f}")
        continue
    z = np.load(f)
    tu, Xu = z["tu"], z["Xu"]
    periods, events = z["periods"], z["events"]
    Ys = {k[2:]: z[k] for k in z.files if k.startswith("Y_")}
    dtm = tu[1] - tu[0]
    tot = Xu.sum(axis=1)
    tot_s = smooth(tot, 0.5 / dtm)
    Ysm = {k: (tot_s if k == "raw" else smooth(Ys[k].sum(axis=1), 0.5 / dtm)) for k in ORDER}
    DED = {k: smooth((Ys[k] - Xu).sum(axis=1), 0.5 / dtm) for k in ORDER[1:]}
    ev = pd.read_csv(os.path.join(RES, f"rec_{tag}_events.csv"))
    st = pd.read_csv(os.path.join(RES, f"rec_{tag}_metrics.csv"))
    ev["rec"], st["rec"] = tag, tag
    all_ev.append(ev)
    all_st.append(st)

    # 免责期 = v3 进入负载态后 5s（只有空载→负载才重挂）
    c = None
    onspans = []
    for a, b in periods:
        onspans.append((float(tu[a]), min(float(tu[a]) + 5.0, float(tu[b]))))

    fig = plt.figure(figsize=(19, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[2.4, 1.1, 1.7])

    ax = fig.add_subplot(gs[0, :])
    for a, b in periods:
        ax.axvspan(tu[a], tu[b], color="orange", alpha=0.07, zorder=0)
    for s0, s1 in onspans:
        ax.axvspan(s0, s1, color="#d62728", alpha=0.11, zorder=0)
    for e in events:
        ax.axvline(tu[e], color="k", ls=":", lw=0.7, alpha=0.45, zorder=1)
    for k in ORDER:
        ax.plot(tu, Ysm[k], color=COL[k], lw=LW[k], ls=LS[k], alpha=AL[k], label=LBL[k], zorder=ZO[k])
    ymax = float(Ysm["raw"].max())
    for _, r in ev.iterrows():
        if abs(r["jump"]) > 0.4 * ymax:
            ax.annotate(f"{r['jump']:+,.0f}", xy=(r["t"], tot_s[int(r["t"] / dtm)]),
                        xytext=(r["t"], tot_s[int(r["t"] / dtm)] + 0.10 * ymax), fontsize=9.5,
                        ha="center", color="#333333",
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#888888"))
    bad = mid_load(ev)
    bad = bad[bad.restep.isna()]
    for _, r in bad.iterrows():
        ax.annotate(f"漏检台阶 {r['jump']:+.0f}\n(比值 {r['ratio']:.2f})",
                    xy=(r["t"], tot_s[int(r["t"] / dtm)]),
                    xytext=(r["t"], max(0.03 * ymax, tot_s[int(r["t"] / dtm)] - 0.18 * ymax)),
                    fontsize=10, ha="center", color="#a01010",
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#d62728"))
    ax.set_xlim(0, tu[-1])
    ax.set_ylim(-0.05 * ymax, ymax * 1.16)
    ax.set_xlabel("时间 (s)", fontsize=11.5)
    ax.set_ylabel("整阵显示总量 (21ch ADC)", fontsize=11.5)
    ax.set_title(f"① 全长时序（整阵 21 通道求和，0.5s 中值平滑）  {TITLE.get(tag, tag)}  "
                 f"橙带=负载段  红带=v4 免责期  灰点线=变载事件（{len(events)} 个）", fontsize=13, loc="left")
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="orange", alpha=0.20, label="负载段"),
          Patch(facecolor="#d62728", alpha=0.20, label="v4 免责期（空载→负载后 0~5s）")]
    ax.legend(h, l, fontsize=11, loc="upper left", ncol=3, framealpha=0.93)
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[1, :])
    for a, b in periods:
        ax.axvspan(tu[a], tu[b], color="orange", alpha=0.05, zorder=0)
    ax.axhline(0, color="k", lw=0.9)
    for k in ORDER[2:]:
        ax.plot(tu, DED[k] - DED["v3"], color=COL[k], lw=LW[k] + 0.2, ls=LS[k],
                alpha=0.95, label=f"{LBL[k]} − v3", zorder=ZO[k])
    mx = float(max(np.abs(DED[k] - DED["v3"]).max() for k in ORDER[2:]))
    ax.set_ylim(-mx * 1.3, mx * 1.3)
    ax.set_xlim(0, tu[-1])
    ax.set_xlabel("时间 (s)", fontsize=11.5)
    ax.set_ylabel("扣除量之差 (ADC)", fontsize=11.5)
    ax.set_title("② v4 / v4r 相对现役 v3 的扣除量之差（整阵总量）：v4 的差别只出现在空载→负载之后",
                 fontsize=13, loc="left")
    ax.legend(fontsize=11, loc="upper left", ncol=2, framealpha=0.93)
    ax.grid(alpha=0.15)

    wins = pick_zooms({"tu": tu, "span": float(tu[-1]), "periods": periods}, ev, k=2)
    for j in range(3):
        ax = fig.add_subplot(gs[2, j])
        if j >= len(wins):
            ax.axis("off")
            continue
        lo, hi, title = wins[j]
        sl = slice(int(lo / dtm), min(len(tu), int(hi / dtm)))
        for a, b in periods:
            if tu[b] > lo and tu[a] < hi:
                ax.axvspan(max(tu[a], lo), min(tu[b], hi), color="orange", alpha=0.08, zorder=0)
        for s0, s1 in onspans:
            if lo <= s0 <= hi:
                ax.axvspan(s0, min(s1, hi), color="#d62728", alpha=0.13, zorder=0)
        ylo = min(Ys[k].sum(axis=1)[sl].min() for k in ORDER)
        yhi = max(Ys[k].sum(axis=1)[sl].max() for k in ORDER)
        pad = 0.10 * max(1.0, yhi - ylo)
        ax.set_ylim(ylo - pad, yhi + pad * 2.6)
        for e in events:
            if lo <= tu[e] <= hi:
                ax.axvline(tu[e], color="k", ls=":", lw=0.8, alpha=0.5, zorder=1)
        for k in ORDER:
            ax.plot(tu[sl], (Ys[k].sum(axis=1))[sl], color=COL[k], lw=LW[k] * 0.55 + 0.55,
                    ls=LS[k], alpha=AL[k], label=LBL[k], zorder=ZO[k])
        for s0, s1 in onspans:
            if lo <= s0 <= hi:
                ax.text((s0 + s1) / 2, yhi + pad * 2.1, "免责期", ha="center", va="center",
                        fontsize=9.5, color="#a01010")
        ax.set_xlim(lo, hi)
        ax.set_title(f"③{'abc'[j]} {title}", fontsize=11.5, loc="left")
        ax.set_xlabel("时间 (s)", fontsize=10)
        if j == 0:
            ax.set_ylabel("整阵显示总量 (ADC)", fontsize=10)
            ax.legend(fontsize=8.5, loc="upper left", ncol=2)
        ax.tick_params(labelsize=9.5)
        ax.grid(alpha=0.15)

    fig.suptitle(f"{TITLE.get(tag, tag)}    ·    v4 快相免责期算法离线复算（幅度窗 [3.5,5.0]s）",
                 fontsize=16)
    audit(fig, f"rec_{tag}")
    fig.savefig(os.path.join(FIG, f"rec_{tag}_result.png"), dpi=118)
    plt.close(fig)
    print(f"saved: figures/rec_{tag}_result.png   放大窗: {[w[2] for w in wins]}")

# ---------------- 合并指标图 ----------------
if all_ev:
    ev = pd.concat(all_ev, ignore_index=True)
    st = pd.concat(all_st, ignore_index=True)
    fig = plt.figure(figsize=(19, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    sub = mid_load(ev).copy()
    sub["detected"] = sub["restep"].notna()
    for tag, mk, c in (("1d9493", "o", "#1f77b4"), ("13ffca", "s", "#d62728")):
        s = sub[sub.rec == tag]
        if len(s) == 0:
            continue
        ax.scatter(s.ratio, s.gap_v3, s=110, marker=mk, color=c, alpha=0.85,
                   label=f"{tag}（n={len(s)}）", zorder=3)
        for _, r in s.iterrows():
            ax.annotate(f"{r['t']:.0f}s" + ("" if r["detected"] else " 漏检"),
                        (r["ratio"], r["gap_v3"]), fontsize=9,
                        color="#a01010" if not r["detected"] else "#333333",
                        xytext=(7, 6), textcoords="offset points")
    ax.axvline(0.29, color="#d62728", ls="--", lw=1.4)
    ax.text(0.295, ax.get_ylim()[1] * 0.92, "有效识别门限 ≈0.29", color="#a01010", fontsize=10.5)
    ax.set_xlabel("ADC 台阶 ÷ 变载前读数", fontsize=10.5)
    ax.set_ylabel("v3 最大 |显示−原始| (ADC)", fontsize=10.5)
    ax.set_title("④ 台阶相对大小  vs  v3 偏差（只含负载内变载）", fontsize=12, loc="left")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.15)

    ax = fig.add_subplot(gs[0, 1])
    sub = sub.sort_values(["rec", "t"]).reset_index(drop=True)
    x = np.arange(len(sub))
    w = 0.36
    ax.bar(x - w / 2, sub.ratio, w, color="#7f7f7f", alpha=0.9, label="台阶/变载前读数")
    ax.bar(x + w / 2, sub.gain_ratio_v3, w, color="#1f77b4", alpha=0.9,
           label="显示增益/原始增益（1.0=台阶完整透传）")
    ax.axhline(0.29, color="#d62728", ls="--", lw=1.4, label="有效识别门限 0.29")
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.annotate("漏检" if not r["detected"] else "识别",
                    (i, max(r["ratio"], r["gain_ratio_v3"])), fontsize=9,
                    color="#a01010" if not r["detected"] else "#1a7f1a",
                    ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.rec[:6]}\n{r.t:.0f}s" for _, r in sub.iterrows()], fontsize=8)
    ax.set_ylabel("比值", fontsize=10.5)
    ax.set_title("⑤ 逐事件：台阶比值 / 显示增益比 / 是否被识别", fontsize=12, loc="left")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.15, axis="y")

    ax = fig.add_subplot(gs[1, 0])
    wins = st[["rec", "window_s", "end_s", "dur_s"]].drop_duplicates()
    x = np.arange(len(wins))
    for i, k in enumerate(ORDER):
        vals = []
        for _, r in wins.iterrows():
            q = st[(st.rec == r["rec"]) & (st.window_s == r["window_s"]) & (st.algo == k)]
            vals.append(float(q.drift_pct.iloc[0]) if len(q) else 0.0)
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=COL[k], alpha=0.92, label=LBL[k])
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.rec[:6]}\n{r.window_s:.0f}~{r.end_s:.0f}s" for _, r in wins.iterrows()],
                       fontsize=8, rotation=0)
    ax.set_ylabel("慢相时漂残余（占本窗电平 %）", fontsize=10.5)
    ax.set_title("⑥ 慢相稳定窗时漂残余（事件后 8s 保护带）", fontsize=12, loc="left")
    ax.legend(fontsize=9.5, ncol=2)
    ax.grid(alpha=0.15, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    for i, k in enumerate(ORDER[1:]):
        vals = [st[(st.rec == tg) & (st.algo == k)].drift_pct.abs().mean() for tg in TAGS if (st.rec == tg).any()]
        ax.bar(np.arange(len(vals)) + (i - 1) * 0.27, vals, 0.27, color=COL[k], alpha=0.92,
               label=LBL[k])
    rawv = [st[(st.rec == tg) & (st.algo == "raw")].drift_pct.abs().mean() for tg in TAGS if (st.rec == tg).any()]
    ax.plot(np.arange(len(rawv)), rawv, "o--", color=COL["raw"], label="原始（参考）")
    ax.set_xticks(np.arange(len(rawv)))
    ax.set_xticklabels([t for t in TAGS if (st.rec == t).any()], fontsize=10)
    ax.set_ylabel("慢相时漂残余 |均值| (%)", fontsize=10.5)
    ax.set_title("⑦ 两份录制的慢相残余汇总", fontsize=12, loc="left")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.15, axis="y")

    fig.suptitle("零负载-中途切换负载-零负载-切换负载：两份录制 · v3/v4 指标对比", fontsize=15.5)
    audit(fig, "rec_two_metrics")
    fig.savefig(os.path.join(FIG, "rec_two_metrics.png"), dpi=118)
    plt.close(fig)
    print("saved: figures/rec_two_metrics.png")
