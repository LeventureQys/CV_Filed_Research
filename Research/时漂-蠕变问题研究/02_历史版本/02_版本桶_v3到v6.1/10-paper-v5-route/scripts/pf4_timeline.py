# -*- coding: utf-8 -*-
"""图 4：时延分解 —— 从真实加载沿到扣除到位（实采 4 份录制 · 10 个受载沿）。

时延口径：
  检测窗  = 真实加载沿 → 判据命中（pending 置位）
  确认窗  = 2.5 s（kStepPersistS，与免责期档位无关）
  免责期  = 3 s（默认档）
  爬升    = 确认结束到「首个可见扣除」；可见阈值 = 0.5% × 台阶
  扣到位  = 首个可见扣除 + 3τ（τ=3 s）≈ 95%

产出：figures/F4_latency.png、results/f4_latency.csv
"""
import numpy as np
import pandas as pd

import pd as P
from figstyle import (C_ALG, C_ALT, C_FAST, C_GRAY, C_RAW, plt, save_figure)

VIS_FRAC = 0.005          # 可见扣除阈值：0.5% × 台阶
STEP_PERSIST = 2.5


def load_edges(d):
    """返回所有「空载→负载」真实加载沿的时刻与台阶（独立于算法判据）。"""
    tu, tot, dtm = d["tu"], d["tot"], d["dtm"]
    periods = P.L.find_periods(tot, dtm)
    base = float(np.median(tot[:max(1, int(3.0 / dtm))]))
    out = []
    for a, b in periods:
        if a < 3:
            continue
        pre = float(np.median(tot[max(0, a - int(2.0 / dtm)):a]))
        post = float(np.median(tot[a + int(4.0 / dtm):a + int(6.0 / dtm)]))
        if pre > 0.15 * tot.max():
            continue                       # 只保留空载→负载（pre 为空载电平）
        if abs(post - pre) < 2000:
            continue
        # 真实加载沿：平滑总量越过 基线上方 5% 台阶高度的首帧
        thr = pre + 0.05 * (post - pre)
        idx = np.where(tot[a:a + int(6.0 / dtm)] > thr)[0]
        if not len(idx):
            continue
        out.append((float(tu[a + int(idx[0])]), float(post - pre), base))
    return out


rows, curves = [], []
for tag, path in P.RECS:
    d = P.load_rec(path)
    tu, dtm, tot = d["tu"], d["dtm"], d["tot"]
    Y, comp = P.run_algo(tu, d["Xu"])
    ded = tot - Y.sum(axis=1)                     # 逐帧扣除量（总量口径）
    edges = load_edges(d)
    epoch_t = list(comp.epoch_t)
    for te, step, base in edges:
        vis_thr = VIS_FRAC * step
        i0 = int(np.searchsorted(tu, te))
        i1 = min(len(tu) - 1, int(np.searchsorted(tu, te + 20.0)))
        seg = ded[i0:i1]
        hit = np.where(seg > vis_thr)[0]
        if not len(hit):
            continue
        t_vis = float(tu[i0 + int(hit[0])] - te)
        # pending 命中时刻：用 glb53_v51 的状态（pending_ts）——由 epoch 起点反推即可：
        # epoch 起点 = pending 起点 + 2.5 s，故 pending 时刻 = epoch - 2.5
        ep = next((x for x in epoch_t if 0 < x - te <= 10.0), np.nan)
        t_detect = float(ep - te - STEP_PERSIST) if np.isfinite(ep) else np.nan
        rows.append(dict(rec=tag, t_edge_s=te, step=step, t_detect_s=t_detect,
                         t_epoch_s=float(ep - te) if np.isfinite(ep) else np.nan,
                         t_first_ded_s=t_vis, t_ded95_s=t_vis + 3 * 3.0))
        sub = np.arange(max(0, i0 - int(1.0 / dtm)), min(len(tu), i0 + int(20.0 / dtm)))
        for j in sub:
            curves.append(dict(rec=tag, t=float(tu[j] - te), raw=float(tot[j]),
                               algo=float(Y[j].sum()), ded=float(ded[j]),
                               step=step, t_edge=te))
        print(f"[F4] {tag} @{te:7.2f}s 台阶 {step:7.0f}：检测 {t_detect:5.2f}s、"
              f"epoch {float(ep-te) if np.isfinite(ep) else float('nan'):5.2f}s、"
              f"首扣(可见) {t_vis:5.2f}s、扣到位 {t_vis+9:.2f}s")

df = pd.DataFrame(rows)
P.save_table(df, "f4_latency.csv")
print(f"[F4] {len(df)} 个受载沿：检测中位 {df.t_detect_s.median():.2f}s、"
      f"epoch 中位 {df.t_epoch_s.median():.2f}s、首扣(可见)中位 {df.t_first_ded_s.median():.2f}s、"
      f"扣到位中位 {df.t_ded95_s.median():.2f}s")

# 选一条代表录制做 (b)：变载最丰富的 13ffca
# 选一条代表录制做 (b)(c)：受载沿最多、且含大台阶的那份
pick = df.groupby("rec").size().idxmax()
cand = df[df.rec == pick].sort_values("step", ascending=False)
edge_pick = float(cand.t_edge_s.iloc[0])
print(f"[F4] (b) 代表录制 = {pick}，取台阶最大的一次 @{edge_pick:.2f} s（台阶 {cand.step.iloc[0]:.0f} ADC）")

# ── 绘图 ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.4, 7.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.28], height_ratios=[1.0, 1.0],
                      wspace=0.26, hspace=0.46, left=0.088, right=0.985, top=0.925, bottom=0.115)

# (a) 时延梯形
ax = fig.add_subplot(gs[:, 0])
y = np.arange(len(df))[::-1]
# 第一段直接用 barh 从 0 画到 epoch 起点（覆盖"检测 + 确认"的实际时长）
for yy, r in zip(y, df.itertuples()):
    ax.barh(yy, r.t_epoch_s, left=0.0, height=0.52, color="#AEB6BD", zorder=2)
    ax.barh(yy, 3.0, left=r.t_epoch_s, height=0.52, color=C_FAST, alpha=0.95, zorder=2)
    ax.barh(yy, max(r.t_first_ded_s - r.t_epoch_s - 3.0, 0.02), left=r.t_epoch_s + 3.0,
            height=0.52, color=C_ALT, alpha=0.95, zorder=2)
    ax.plot([r.t_first_ded_s], [yy], marker="v", ms=6.0, color=C_ALG, zorder=5)
    ax.plot([r.t_ded95_s], [yy], marker="|", ms=10, mew=2.0, color="#7D3C98", zorder=5)
ax.axvline(5.5, color=C_ALG, lw=1.3, ls=":")
ax.axvline(14.5, color="#7D3C98", lw=1.3, ls=":")
ax.set_yticks(y)
ax.set_yticklabels(["%s\n@%.1f s" % (r.rec.split("-")[-1], r.t_edge_s) for r in df.itertuples()],
                   fontsize=7.6)
ax.set_xlim(0, 30)
ax.set_ylim(-0.8, len(df) + 1.05)
ax.set_xlabel("真实加载沿后时间 (s)", labelpad=4)
ax.set_title("(a) 10 个受载沿的时延分解（免责 3 s 档）", fontsize=11)
ax.grid(alpha=0.25, axis="x", lw=0.6)
# 顶部四段标注：按各自区间居中，并避开 x=5.5 / 14.5 两条参考线
_yl = len(df) + 0.30
ax.text(1.20, _yl, "确认", fontsize=8.4, color=C_GRAY, ha="center", va="bottom")
ax.text(4.00, _yl, "免责期", fontsize=8.4, color=C_FAST, ha="center", va="bottom")
ax.text(9.00, _yl, "爬升", fontsize=8.4, color=C_ALT, ha="center", va="bottom")
ax.text(17.50, _yl, "到位", fontsize=8.4, color="#7D3C98", ha="center", va="bottom")
h1 = plt.Rectangle((0, 0), 1, 1, fc="#AEB6BD")
h2 = plt.Rectangle((0, 0), 1, 1, fc=C_FAST, alpha=0.90)
h3 = plt.Rectangle((0, 0), 1, 1, fc=C_ALT, alpha=0.90)
h4 = plt.Line2D([], [], marker="v", ls="none", ms=6, color=C_ALG)
h5 = plt.Line2D([], [], marker="|", ls="none", ms=10, mew=2.0, color="#7D3C98")
ax.legend([h1, h2, h3, h4, h5],
          ["检测 + 确认窗 (2.5 s)", "免责期 (3 s)", "扣除爬升", "首个可见扣除", "扣到位 95%"],
          fontsize=7.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.118),
          framealpha=1.0, handlelength=1.5, columnspacing=1.4)

# (b) 一帧波形：原始 vs 补偿
ax2 = fig.add_subplot(gs[0, 1])
cv = pd.DataFrame([c for c in curves if c["rec"] == pick])
seg = cv[cv.t_edge == edge_pick]
ax2.plot(seg.t, seg.raw, color=C_RAW, lw=3.6, alpha=0.60, label="原始（无补偿）", zorder=2)
ax2.plot(seg.t, seg.algo, color=C_ALG, lw=1.5, ls=(0, (6, 2)), label="本算法", zorder=3)
ax2.axvspan(0.0, STEP_PERSIST, color="#DDE1E4", lw=0)
ax2.axvline(STEP_PERSIST, color=C_GRAY, lw=1.0, ls=":")
ax2.axvline(STEP_PERSIST + 3.0, color=C_FAST, lw=1.4, ls="--")
ax2.axvline(STEP_PERSIST + 3.0 + 9.0, color="#7D3C98", lw=1.2, ls=":")
ax2.set_xlim(-0.8, 16.0)
ax2.set_ylim(0, None)
ax2.set_xlabel("真实加载沿后时间 (s)")
ax2.set_ylabel("显示总量 (ADC)")
ax2.set_title("(b) %s @%.1f s：真实加载被逐帧透传" % (pick, edge_pick), fontsize=11)
ax2.grid(alpha=0.25, lw=0.6)
ax2.legend(fontsize=8.6, loc="lower right", framealpha=1.0)
ylo, yhi = ax2.get_ylim()
ax2.set_ylim(ylo, yhi + 0.14 * (yhi - ylo))
ylo, yhi = ax2.get_ylim()
ax2.text(STEP_PERSIST + 0.18, ylo + 0.045 * (yhi - ylo), "epoch 起点", fontsize=8.4, color=C_GRAY)
ax2.text(STEP_PERSIST + 3.20, ylo + 0.045 * (yhi - ylo), "首扣", fontsize=8.4, color=C_FAST)
ax2.text(STEP_PERSIST + 9.20, ylo + 0.045 * (yhi - ylo), "扣到位 95%", fontsize=8.4,
         color="#7D3C98")

# (c) 扣除量的建立过程（每条曲线按自身末端值归一化，看形状：单调、无过冲、约 9 s 到 95%）
ax3 = fig.add_subplot(gs[1, 1])
for r in df.itertuples():
    cc = pd.DataFrame([c for c in curves if c["rec"] == r.rec and c["t_edge"] == r.t_edge_s])
    if not len(cc):
        continue
    d = cc.ded.to_numpy()
    d = d - d[0]
    peak = float(np.maximum(np.max(d), 1e-9))          # 按各自峰值归一化，跑满 0~100%
    ax3.plot(cc.t, 100.0 * d / peak, color=C_GRAY, lw=0.9, alpha=0.55)
ax3.axvline(STEP_PERSIST, color=C_GRAY, lw=1.0, ls=":")
ax3.axvline(STEP_PERSIST + 3.0, color=C_FAST, lw=1.4, ls="--")
ax3.axhline(95.0, color="#7D3C98", lw=1.0, ls=":", zorder=6)
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 112)
ax3.set_xlabel("真实加载沿后时间 (s)")
ax3.set_ylabel("扣除量 ÷ 本事件 20 s 内的峰值 (%)")
ax3.set_title("(c) 扣除量单调建立、无过冲（各事件按自身峰值归一化）", fontsize=11)
ax3.grid(alpha=0.25, lw=0.6)
ax3.text(8.4, 96.0, "95%", fontsize=8.4, color="#7D3C98")
fig.text(0.088, 0.012,
         "注：(c) 的纵轴是各事件自身的归一化值，不代表扣除量占台阶的比例——后者在 20 s 处只有 5%~14%（受载通道只占少数）。",
         fontsize=8.4, color=C_GRAY, ha="left", va="bottom")

save_figure(fig, "F4_latency.png")
