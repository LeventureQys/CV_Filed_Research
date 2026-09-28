# -*- coding: utf-8 -*-
"""新实采「切换负载」数据 · v4 快相免责期算法最终结果图。

图 1 figures/new_switch_load_result.png
   ① 整阵总量全长时序（raw / v3 / v4 / v4r）+ 负载段 + 变载事件 + v4 免责期
   ② v4 / v4r 相对 v3 的扣除量之差（差别只出现在 3 次「空载→负载」之后）
   ③ 三处放大：首次加载沿 / 卸载→再加载 / 末段漏检台阶导致的欠报
图 2 figures/new_switch_load_metrics.png
   ④ 变载事件最坏欠报  ⑤ 慢相窗时漂残余  ⑥ 主通道全长  ⑦ 漏检台阶处的扣除量失控
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
order = ["raw", "v3", "v4_fast5", "v4r_fast5"]

d = np.load(os.path.join(RES, "new_switch_load.npz"))
tu, Xu = d["tu"], d["Xu"]
periods, events = d["periods"], d["events"]
Ys = {k[2:]: d[k] for k in d.files if k.startswith("Y_")}
tot = Xu.sum(axis=1)
dtm = tu[1] - tu[0]
onsets_algo = [13.35, 136.48, 188.58]          # v3/v4 实际确认的 3 次空载→负载
ev_df = pd.read_csv(os.path.join(RES, "new_switch_load_events.csv"))
st_df = pd.read_csv(os.path.join(RES, "new_switch_load_metrics.csv"))


def smooth(x, s=0.5):
    return pd.Series(x).rolling(max(3, int(s / dtm)), center=True, min_periods=1).median().to_numpy()


tot_s = smooth(tot)
Ysm = {k: (tot_s if k == "raw" else smooth(Ys[k].sum(axis=1))) for k in order}
DED = {k: smooth((Ys[k] - Xu).sum(axis=1)) for k in order if k != "raw"}


def shade(ax, lo, hi, onset=True):
    for a, b in periods:
        if tu[b] > lo and tu[a] < hi:
            ax.axvspan(max(tu[a], lo), min(tu[b], hi), color="orange", alpha=0.08, zorder=0)
    if onset:
        for s in onsets_algo:
            if lo <= s <= hi:
                ax.axvspan(s, min(s + 5.0, hi), color="#d62728", alpha=0.13, zorder=0)


def audit(fig, name):
    """布局自检：注释框是否越出画布 / 越出所属坐标区（无法肉眼看图时的替代检查）。"""
    if os.environ.get("FIG_CHECK") != "1":
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.axes:
        ab = ax.bbox
        for t in ax.texts:
            bb = t.get_window_extent(r)
            if not (bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.x1 + 1 and bb.y1 <= fig.bbox.y1 + 1):
                bad.append(f"越出画布: {t.get_text()[:24]!r} bbox={bb.bounds} fig={fig.bbox.bounds}")
            elif bb.x0 < ab.x0 - 2 or bb.x1 > ab.x1 + 2 or bb.y0 < ab.y0 - 2 or bb.y1 > ab.y1 + 2:
                bad.append(f"越出坐标区: {t.get_text()[:24]!r} bbox={bb.bounds} ax={ab.bounds}")
        for t in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            bb = t.get_window_extent(r)
            if bb.x1 > fig.bbox.x1 + 2 or bb.x0 < -2 or bb.y1 > fig.bbox.y1 + 2 or bb.y0 < -2:
                bad.append(f"标题/轴标签越界: {t.get_text()[:24]!r}")
    print(f"[audit {name}] " + ("OK" if not bad else f"{len(bad)} 处问题"))
    for b in bad:
        print("   -", b)


# ============================================================ 图 1
fig = plt.figure(figsize=(20, 16), constrained_layout=True)
gs = fig.add_gridspec(3, 3, height_ratios=[2.45, 1.15, 1.7])

# ---------- ① 全长总览 ----------
ax = fig.add_subplot(gs[0, :])
shade(ax, 0, tu[-1])
for e in events:
    ax.axvline(tu[e], color="k", ls=":", lw=0.7, alpha=0.45, zorder=1)
for k in order:
    ax.plot(tu, Ysm[k], color=COL[k], lw=LW[k], ls=LS[k], alpha=AL[k],
            label=LBL[k], zorder=ZO[k])
ymax = float(Ysm["raw"].max())
for e in events:
    xp = np.median(tot_s[max(0, e - int(2 / dtm)):e])
    xq = np.median(tot_s[min(len(tu) - 1, e + int(4 / dtm)):min(len(tu), e + int(6 / dtm))])
    j = xq - xp
    if abs(j) > 6000:
        ax.annotate(f"{j:+,.0f}", xy=(tu[e], tot_s[e]), xytext=(tu[e], tot_s[e] + 3200),
                    fontsize=9.5, ha="center", color="#333333",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#888888"))
ax.annotate("t=240.9s 的 +5254 台阶（+26%）低于阶跃检测的有效门限（≈27% 电平）\n"
            "→ 被当成蠕变：扣除量冲到 9241 ADC，显示 16600 vs 原始 25842（欠报 36%）",
            xy=(246.0, 16800), xytext=(118.0, 2400), fontsize=10.5, color="#a01010",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0", ec="#d62728", lw=1.0),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#d62728"))
ax.set_xlim(0, tu[-1])
ax.set_ylim(-1500, ymax * 1.16)
ax.set_xlabel("时间 (s)", fontsize=11.5)
ax.set_ylabel("整阵显示总量 (21ch ADC)", fontsize=11.5)
ax.set_title("① 全长时序（整阵 21 通道求和，0.5s 中值平滑）  橙带=负载段  红带=v4 的 5s 快相免责期  "
             f"灰点线=检测到的变载事件（共 {len(events)} 个）", fontsize=13, loc="left")
h, l = ax.get_legend_handles_labels()
h += [Patch(facecolor="orange", alpha=0.20, label="负载段（原始总量 > 5% 峰值）"),
      Patch(facecolor="#d62728", alpha=0.20, label="v4 免责期（空载→负载后 0~5s，共 3 处）")]
ax.legend(h, l, fontsize=11, loc="upper left", ncol=3, framealpha=0.93)
ax.grid(alpha=0.15)

# ---------- ② v4 与 v3 的差别 ----------
ax = fig.add_subplot(gs[1, :])
shade(ax, 0, tu[-1])
ax.axhline(0, color="k", lw=0.9)
for k in ("v4_fast5", "v4r_fast5"):
    ax.plot(tu, DED[k] - DED["v3"], color=COL[k], lw=LW[k] + 0.2, ls=LS[k],
            alpha=0.95, label=f"{LBL[k]} − v3", zorder=ZO[k])
mx = float(max(np.abs(DED[k] - DED["v3"]).max() for k in ("v4_fast5", "v4r_fast5")))
ax.set_ylim(-mx * 1.30, mx * 1.30)
ax.set_xlim(0, tu[-1])
ax.set_xlabel("时间 (s)", fontsize=11.5)
ax.set_ylabel("扣除量之差 (ADC)", fontsize=11.5)
ax.set_title("② v4 / v4r 相对现役 v3 的扣除量之差（整阵总量）——v4 的差别只出现在 3 次「空载→负载」之后；"
             "16 次负载内变载处 v4 与 v3 完全一致（原型 v4 的免责期不由 restep 触发）",
             fontsize=13, loc="left")
ax.legend(fontsize=11, loc="upper left", ncol=2, framealpha=0.93)
ax.grid(alpha=0.15)

# ---------- ③ 三处放大 ----------
zooms = [(8.5, 23.5, "③a 首次加载沿（原始帧率）：红带内 v4 直通原始；v3 从 onset+3s 起开始扣除"),
         (66.0, 96.0, "③b 卸载→再加载：变载处 v3 与 v4 完全重合（免责期不介入 restep）"),
         (234.0, 252.0, "③c 漏检台阶：240.9s 的 +5254 台阶未触发变载，补偿把真实力当蠕变扣掉")]
for j, (lo, hi, title) in enumerate(zooms):
    ax = fig.add_subplot(gs[2, j])
    sl = slice(int(lo / dtm), min(len(tu), int(hi / dtm)))
    shade(ax, lo, hi)
    ylo = min(Ys[k].sum(axis=1)[sl].min() for k in order)
    yhi = max(Ys[k].sum(axis=1)[sl].max() for k in order)
    pad = 0.10 * max(1.0, yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad * 2.6)
    for e in events:
        if lo <= tu[e] <= hi:
            ax.axvline(tu[e], color="k", ls=":", lw=0.8, alpha=0.5, zorder=1)
    for k in order:
        ax.plot(tu[sl], (Ys[k].sum(axis=1))[sl], color=COL[k], lw=LW[k] * 0.55 + 0.55,
                ls=LS[k], alpha=AL[k], label=LBL[k], zorder=ZO[k])
    for s in onsets_algo:
        if lo <= s <= hi:
            ax.text(s + 2.5, yhi + pad * 2.1, "免责期", ha="center", va="center",
                    fontsize=9.5, color="#a01010")
    if j == 2:
        ax.annotate("240.9s 台阶未被识别", xy=(241.6, 25200), xytext=(235.0, 18000),
                    fontsize=10, color="#a01010",
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#d62728"))
    ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=11.5, loc="left")
    ax.set_xlabel("时间 (s)", fontsize=10)
    if j == 0:
        ax.set_ylabel("整阵显示总量 (ADC)", fontsize=10)
        ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    elif j == 2:
        ax.set_ylabel("整阵显示总量 (ADC)", fontsize=10)
        ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    ax.tick_params(labelsize=9.5)
    ax.grid(alpha=0.15)

fig.suptitle("20260917_133923_single_device_ee20bc   切换负载 255.7s / 100.5Hz / 21ch / ADC 域"
             "    ·    v4 快相免责期算法离线复算（幅度窗 [3.5,5.0]s）", fontsize=16)
audit(fig, "result")
fig.savefig(os.path.join(FIG, "new_switch_load_result.png"), dpi=118)
plt.close(fig)
print("saved: figures/new_switch_load_result.png")

# ============================================================ 图 2
fig = plt.figure(figsize=(20, 11.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.20)

# ④ 事件最坏欠报
ax = fig.add_subplot(gs[0, 0])
evs = sorted(ev_df.event_s.unique())
x = np.arange(len(evs))
w = 0.27
for i, k in enumerate(["v3", "v4_fast5", "v4r_fast5"]):
    vals = [float(ev_df[(ev_df.event_s == e) & (ev_df.algo == k)].worst_gap_pct.iloc[0]) for e in evs]
    ax.bar(x + (i - 1) * w, vals, w, color=COL[k], alpha=0.92, label=LBL[k])
ax.set_xticks(x)
ax.set_xticklabels([f"{e:.0f}" for e in evs], fontsize=9, rotation=45)
ax.set_xlabel("变载事件时刻 (s)", fontsize=10.5)
ax.set_ylabel("最坏欠报（占本次跳变 %）", fontsize=10.5)
ax.set_title("④ 变载事件最坏欠报（越小越好；202.6s 与 240.9s 两处台阶低于检测门限，"
             "v3/v4 欠报最大）", fontsize=12, loc="left")
ax.legend(fontsize=9.5)
ax.grid(alpha=0.15, axis="y")

# ⑤ 慢相窗时漂
ax = fig.add_subplot(gs[0, 1])
wins = st_df[["window_s", "end_s", "dur_s"]].drop_duplicates().values
x = np.arange(len(wins))
for i, k in enumerate(order):
    vals = [float(st_df[(st_df.window_s == w0) & (st_df.algo == k)].drift_pct.iloc[0])
            for w0, _, _ in wins]
    ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=COL[k], alpha=0.92, label=LBL[k])
ax.axhline(0, color="k", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f"{w0:.0f}~{w1:.0f}s\n({dd:.0f}s)" for w0, w1, dd in wins], fontsize=9)
ax.set_ylabel("慢相时漂残余（占本窗电平 %）", fontsize=10.5)
ax.set_title("⑤ 慢相稳定窗时漂残余（事件后 8s 保护带；越大说明显示还在漂）", fontsize=12, loc="left")
ax.legend(fontsize=9.5, ncol=2)
ax.grid(alpha=0.15, axis="y")

# ⑥ 主通道
ax = fig.add_subplot(gs[1, 0])
base = Xu[:int(9 / dtm)].mean(axis=0)
m = int(np.argmax(Xu[int(12 / dtm):int(129 / dtm)].mean(axis=0) - base))
for k in order:
    ax.plot(tu, smooth(Ys[k][:, m]), color=COL[k], lw=LW[k], ls=LS[k], alpha=AL[k],
            label=LBL[k], zorder=ZO[k])
for a, b in periods:
    ax.axvspan(tu[a], tu[b], color="orange", alpha=0.08)
ax.set_xlim(0, tu[-1])
ax.set_ylim(-base[m] * 1.4, float(smooth(Ys["raw"][:, m]).max()) * 1.15)
ax.set_xlabel("时间 (s)", fontsize=10.5)
ax.set_ylabel(f"主通道 ch{m} 显示值 (ADC)", fontsize=10.5)
ax.set_title(f"⑥ 主通道 ch{m}（负载段幅度最大）全长时序", fontsize=12, loc="left")
ax.legend(fontsize=9.5, ncol=2)
ax.grid(alpha=0.15)

# ⑦ 扣除量失控
ax = fig.add_subplot(gs[1, 1])
for k in ("v3", "v4_fast5", "v4r_fast5"):
    ax.plot(tu, DED[k], color=COL[k], lw=2.0 if k == "v4_fast5" else 1.4,
            ls=LS[k], alpha=0.95, label=f"{LBL[k]} 扣除量", zorder=ZO[k])
ax.axvline(240.86, color="#d62728", ls="--", lw=1.3)
ax.annotate("240.86s 真实台阶 +2575（随后续升到 +5300）\n此处未触发变载 → g 从 0.02 涨到 0.30",
            xy=(240.86, 6000), xytext=(225.0, 7200), fontsize=10, color="#a01010",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff0f0", ec="#d62728", lw=0.9),
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#d62728"))
ax.set_xlim(210, tu[-1])
ax.set_ylim(-500, 10500)
ax.set_xlabel("时间 (s)", fontsize=10.5)
ax.set_ylabel("蠕变扣除量（整阵总量, ADC）", fontsize=10.5)
ax.set_title("⑦ 被漏掉的台阶让补偿量失控：v3 与 v4 扣到 9241 ADC（≈显示真值的 36%），"
             "v4r 因变载重起免责期而未失控", fontsize=12, loc="left")
ax.legend(fontsize=9.5)
ax.grid(alpha=0.15)

fig.suptitle("20260917_133923_single_device_ee20bc    ·    v4 快相免责期算法指标分解", fontsize=15.5)
audit(fig, "metrics")
fig.savefig(os.path.join(FIG, "new_switch_load_metrics.png"), dpi=118)
plt.close(fig)
print("saved: figures/new_switch_load_metrics.png")

