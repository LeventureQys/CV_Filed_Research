# -*- coding: utf-8 -*-
"""差异最大的一组：20260917_133923（切换负载 255.7s）v3 / v4 / v5 图像对比。

figures/v5_diff_worst.png
 ① 全长时序（raw / v3 / v4 / v5）+ 问题事件标记
 ②③④ 三处漏检台阶放大：240.9s（最严重）/ 230.6s / 202.6s
 ⑤ 机理：v3 的 div=|fast−slow| 与 0.18·slow 门限（差一点点没够到）
 ⑥ 扣除量曲线：v3 的扣除量失控冲到 9241 ADC
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
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

TAG = "切换负载-快相无责"
z = np.load(os.path.join(RES, f"v5_{TAG}.npz"))
tu, Xu = z["tu"], z["Xu"]
periods, events = z["periods"], z["events"]
Ys = {k[2:]: z[k] for k in z.files if k.startswith("Y_")}
dtm = tu[1] - tu[0]
tot = Xu.sum(axis=1)
sm = lambda x, s=0.5: pd.Series(x).rolling(max(3, int(s / dtm)), center=True,
                                          min_periods=1).median().to_numpy()
tot_s = sm(tot)
order = ["raw", "v3", "v4", "v5", "v5b"]
COL = {"raw": "#9e9e9e", "v3": "#1f77b4", "v4": "#2ca02c", "v5": "#d62728", "v5b": "#e08a00"}
LBL = {"raw": "原始（无补偿）", "v3": "GLM53 v3（现役）", "v4": "v4 免责5s",
       "v5": "v5（当前：免责3s）", "v5b": "v5 @ 免责5s（上一版，供对比）"}

# v5 @ 免责 5s 变体（上一版参数），用于同图对比 3s/5s 的取舍
c5 = GLM53v5(Xu.shape[1])
c5.FAST_S, c5.EXEMPT_AWIN, c5.LEV_ARM_S = 5.0, 1.5, 5.0
Y5b = np.empty_like(Xu)
for i in range(len(tu)):
    Y5b[i] = c5.process(tu[i], Xu[i])
Ys["v5b"] = Y5b

Ysm = {k: (tot_s if k == "raw" else sm(Ys[k].sum(axis=1))) for k in order}
DED = {k: tot_s - Ysm[k] for k in order[1:]}

# ---- 机理量：逐帧取 v5 内部状态复算 div / 门限 / 电平差 ----
c = GLM53v5(Xu.shape[1])
div_a = np.zeros(len(tu)); thr_a = np.zeros(len(tu))
dlev_a = np.zeros(len(tu)); tlev_a = np.zeros(len(tu)); latch_a = np.zeros(len(tu), bool)
for i in range(len(tu)):
    c.process(tu[i], Xu[i])
    eps = c._eps()
    div_a[i] = abs(c.fast - c.slow)
    thr_a[i] = (max(c.STEP_REL * max(c.slow, eps), c.STEP_ABS * c.max_ts)
                if c.in_load else np.nan)
    lv_now = c._win_mean(tu[i] - c.LEV_FAST_S, tu[i])
    lv_ref = c._win_mean(tu[i] - c.LEV_FAST_S - c.LEV_LAG_S, tu[i] - c.LEV_FAST_S)
    if lv_now is not None and lv_ref is not None:
        dlev_a[i] = abs(lv_now - lv_ref)
        tlev_a[i] = max(c.LEV_REL * max(lv_ref, eps), c.LEV_ABS_FRAC * c.max_ts)
    else:
        dlev_a[i] = tlev_a[i] = np.nan
    latch_a[i] = c.lev_latch is not None

# v3 的同类量（用于对照：v3 只有 div 判据）
c3 = GLM53v3(Xu.shape[1])
div3 = np.zeros(len(tu)); thr3 = np.zeros(len(tu))
for i in range(len(tu)):
    c3.process(tu[i], Xu[i])
    eps = c3._eps()
    div3[i] = abs(c3.fast - c3.slow)
    thr3[i] = (max(c3.STEP_REL * max(c3.slow, eps), c3.STEP_ABS * c3.max_ts)
               if c3.in_load else np.nan)

# ---- 关键事件定位 ----
ev_df = pd.read_csv(os.path.join(RES, "v5_midload_events.csv"))
ev = ev_df[ev_df.rec == TAG].copy()
ev["d_gap"] = ev["gap_v3"] - ev["gap_v5"]
worst = ev.sort_values("d_gap", ascending=False).head(3)
print("差异最大的三个事件（v3 偏差 − v5 偏差）：")
print(worst[["t", "ratio", "jump", "gap_v3", "gap_v4", "gap_v5", "d_gap"]].to_string(index=False))

fig = plt.figure(figsize=(20, 15.5), constrained_layout=True)
gs = fig.add_gridspec(3, 6, height_ratios=[2.0, 1.7, 1.7])

# ---------- ① 全长 ----------
ax = fig.add_subplot(gs[0, :])
for a, b in periods:
    ax.axvspan(tu[a], tu[b], color="orange", alpha=0.07, zorder=0)
for e in events:
    ax.axvline(tu[e], color="k", ls=":", lw=0.7, alpha=0.4, zorder=1)
for k in order:
    ax.plot(tu, Ysm[k], color=COL[k],
            lw=2.8 if k == "v5" else (2.2 if k == "raw" else (1.6 if k == "v5b" else 1.5)),
            ls=":" if k == "raw" else ("--" if k == "v5b" else "-"),
            alpha=0.42 if k == "v5" else 0.95,
            zorder=4 if k == "v3" else (3 if k == "v4" else (6 if k == "v5b" else 2)),
            label=LBL[k])
for _, r in worst.iterrows():
    ax.axvspan(r["t"], r["t"] + 10, color="#d62728", alpha=0.13, zorder=0)
ax.annotate("240.9s：真实台阶 +5254（占电平 25.6%）被 v3/v4 当蠕变扣掉\n"
            "→ 扣除量冲到 9241 ADC，显示 16600 vs 原始 25842（欠报 36%）",
            xy=(247.0, 17000), xytext=(120.0, 2600), fontsize=11.5, color="#a01010",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff0f0", ec="#d62728", lw=1.1),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#d62728"))
h, l = ax.get_legend_handles_labels()
h += [Patch(facecolor="orange", alpha=0.20, label="负载段"),
      Patch(facecolor="#d62728", alpha=0.25, label="差异最大的三处漏检台阶")]
ax.legend(h, l, fontsize=11, loc="upper left", ncol=3, framealpha=0.93)
ax.set_xlim(0, tu[-1]); ax.set_ylim(-1500, tot_s.max() * 1.16)
ax.set_xlabel("时间 (s)", fontsize=11.5); ax.set_ylabel("整阵显示总量 (21ch ADC)", fontsize=11.5)
ax.set_title("① 全长时序（20260917_133923 切换负载 255.7s / 21ch / ADC 域）：v3 与 v4 几乎重合，"
             "v5 在三处漏检台阶上明显不同", fontsize=13.5, loc="left")
ax.grid(alpha=0.15)

# ---------- ②③④ 三处放大 ----------
for j, (_, r) in enumerate(worst.iterrows()):
    ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
    lo, hi = r["t"] - 6.0, r["t"] + 16.0
    sl = slice(int(lo / dtm), min(len(tu), int(hi / dtm)))
    for a, b in periods:
        if tu[b] > lo and tu[a] < hi:
            ax.axvspan(max(tu[a], lo), min(tu[b], hi), color="orange", alpha=0.07, zorder=0)
    for k in order:
        ax.plot(tu[sl], (Ys[k].sum(axis=1))[sl], color=COL[k],
                lw=2.0 if k in ("v3", "v5", "raw") else 1.5,
                ls=":" if k == "raw" else ("--" if k == "v5b" else "-"),
                alpha=0.95 if k != "v5b" else 0.85, label=LBL[k],
                zorder=5 if k == "v5b" else (4 if k == "v3" else 3))
    ylo = min(Ys[k].sum(axis=1)[sl].min() for k in order)
    yhi = max(Ys[k].sum(axis=1)[sl].max() for k in order)
    pad = 0.12 * max(1.0, yhi - ylo)
    ax.set_ylim(ylo - pad, yhi + pad * 2.2); ax.set_xlim(lo, hi)
    ax.axvline(r["t"], color="#d62728", ls="--", lw=1.3)
    ax.annotate(f"台阶 {r['jump']:+.0f}（比值 {r['ratio']:.2f}）\n"
                f"最大偏差 v3 {r['gap_v3']:.0f} / v4 {r['gap_v4']:.0f} / v5 {r['gap_v5']:.0f} ADC",
                xy=(r["t"], ylo + 0.5 * (yhi - ylo)),
                xytext=(lo + 1.0, ylo + pad * 0.4), fontsize=10.5, color="#a01010",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0", ec="#d62728", lw=1.0),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#d62728"))
    ax.set_xlabel("时间 (s)", fontsize=10.5)
    if j == 0:
        ax.set_ylabel("整阵显示总量 (ADC)", fontsize=10.5)
        ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_title(f"{'②③④'[j]} t={r['t']:.1f}s 处放大（{'最严重' if j == 0 else '第%d严重' % (j+1)}）",
                 fontsize=12, loc="left")
    ax.tick_params(labelsize=9.5); ax.grid(alpha=0.15)

# ---------- ⑤ 机理：div vs 门限 ----------
ax = fig.add_subplot(gs[2, 0:3])
lo, hi = 239.0, 245.0
sl = slice(int(lo / dtm), int(hi / dtm))
ax.plot(tu[sl], div_a[sl], color="#d62728", lw=2.2, label="v3/v5 都用到的 div = |fast−slow|")
ax.plot(tu[sl], thr_a[sl], color="#1f77b4", lw=2.0, ls="--",
        label="v3 的门限 0.18x slow（约 3690 ADC）")
ax.plot(tu[sl], dlev_a[sl], color="#e08a00", lw=2.2,
        label="v5 新增判据 |Δ电平(0.5s)|")
ax.plot(tu[sl], tlev_a[sl], color="#7f7f7f", lw=1.8, ls=":",
        label="v5 的门限 5%×参考电平（≈1025 ADC）")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlim(lo, hi)
ax.set_ylim(0, 6500)
ax.set_xlabel("时间 (s)", fontsize=10.5); ax.set_ylabel("判据量 (ADC)", fontsize=10.5)
ax.set_title("⑤ 为什么 v3 漏检：台阶发生后 div 峰值≈0.665×台阶=3494，"
             "而门限是 0.18×slow≈3690 —— 差 5% 没够到；v5 的电平差 5254 远超其门限 1025",
             fontsize=12, loc="left")
ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.15)

# ---------- ⑥ 扣除量 ----------
ax = fig.add_subplot(gs[2, 3:6])
lo, hi = 190.0, 252.0
sl = slice(int(lo / dtm), min(len(tu), int(hi / dtm)))
for a, b in periods:
    if tu[b] > lo and tu[a] < hi:
        ax.axvspan(max(tu[a], lo), min(tu[b], hi), color="orange", alpha=0.06, zorder=0)
for k in order[1:]:
    ax.plot(tu[sl], DED[k][sl], color=COL[k], lw=2.4 if k == "v5" else 1.6,
            ls="--" if k == "v5b" else "-", label=f"{LBL[k]} 扣除量")
ax.axhline(0, color="k", lw=0.9)
ax.annotate("v3/v4：台阶被当蠕变 → 扣除量失控冲到 9241 ADC（≈真值 36%）",
            xy=(246.5, 9241), xytext=(205.0, 8800), fontsize=11, color="#a01010",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0", ec="#d62728", lw=1.0),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#d62728"))
ax.annotate("v5：只扣真实蠕变（≈1.5~1.9k），且变载后立刻重建基线",
            xy=(243.0, 1700), xytext=(196.0, 3100), fontsize=11, color="#1a5a1a",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0fff0", ec="#2ca02c", lw=1.0),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#2ca02c"))
ax.set_xlim(lo, hi); ax.set_ylim(-500, 10100)
ax.set_xlabel("时间 (s)", fontsize=10.5); ax.set_ylabel("蠕变扣除量 (ADC)", fontsize=10.5)
ax.set_title("⑥ 扣除量：v3/v4 把真实加载吃成蠕变，v5 只扣真实蠕变", fontsize=12, loc="left")
ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.15)

fig.suptitle("差异最大的一组：20260917_133923 切换负载  ·  v3（现役）/ v4 / v5 对比",
             fontsize=16.5)
fig.savefig(os.path.join(FIG, "v5_diff_worst.png"), dpi=118)
plt.close(fig)
print("saved: figures/v5_diff_worst.png")

# 机理数据核对
k = int(240.86 / dtm)
seg = slice(int(240.8 / dtm), int(243.0 / dtm))
print(f"\n240.86s 处：div 峰值(v3 判据)={np.nanmax(div_a[seg]):.0f} ADC, "
      f"门限={np.nanmax(thr_a[seg]):.0f} ADC → {'够到' if np.nanmax(div_a[seg])>np.nanmax(thr_a[seg]) else '未够到'}")
print(f"           v5 电平差峰值={np.nanmax(dlev_a[seg]):.0f} ADC, "
      f"门限={np.nanmax(tlev_a[seg]):.0f} ADC → {'识别' if np.nanmax(dlev_a[seg])>np.nanmax(tlev_a[seg]) else '未识别'}")
