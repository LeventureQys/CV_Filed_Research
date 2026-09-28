# -*- coding: utf-8 -*-
"""论文图 F7：快相响应的取舍（v6 vs 现役无责期的量化对照）。

(a) 首次加载平稳时刻 T_stable：逐份对照（恒载 9 组 + 实采，n/a 表示 30 s 窗内未满足）
(b) 加载沿附近放大（右拇指/数据2 与 切换负载）：原始继续上漂 vs v6 停在平台
(c) 稳态精度代价：阶跃保真 与 加载沿 +5 s 误差（v6 的"多报"从哪来）
(d) 事件灵敏度代价：epoch 数（恒载 9 组合计 / 实采 4 份合计）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                            # noqa: E402
import pv_style as S                                             # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402

m = pd.read_csv(os.path.join(C.RES, "metrics_all.csv"))
st = pd.read_csv(os.path.join(C.RES, "metrics_settle.csv"))
ARMS = ["e1s", "e3s", "v6"]
LBL = {"e1s": "无责 1 s", "e3s": "无责 3 s（现役）", "v6": "v6"}
COL = {"e1s": S.C_E1, "e3s": S.C_E3, "v6": S.C_V6}

fig, axs = plt.subplots(2, 2, figsize=(14.6, 9.6))
fig.suptitle("图 F7  快相响应的取舍：把「等快相走完」换成「把快相算掉」之后", fontsize=14)

# ── (a) T_stable 逐份 ────────────────────────────────────────────
ax = axs[0, 0]
pv = st.pivot_table(index="dataset", columns="algo", values="t_stable").reindex(columns=ARMS)
order = [t for t, _ in C.ALL if t in pv.index]
pv = pv.reindex(order)
xs = np.arange(len(pv))
w = 0.27
for j, a in enumerate(ARMS):
    vals = pv[a].to_numpy(float)
    b = ax.bar(xs + (j - 1) * w, np.nan_to_num(vals), w, color=COL[a], label=LBL[a])
    for x, v in zip(xs + (j - 1) * w, vals):
        if np.isfinite(v):
            ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.4)
        else:
            ax.text(x, 0.6, "n/a", ha="center", fontsize=7.0, color="0.4")
ax.axhline(1.0, color="#c62828", ls="--", lw=1.4)
ax.text(len(pv) - 0.4, 1.7, "1 s 目标线", color="#c62828", fontsize=9, ha="right")
ax.set_xticks(xs)
ax.set_xticklabels([s.replace("/", "\n") for s in pv.index], fontsize=7.4)
ax.set_ylabel("T_stable：显示首次停下并保持 30 s (s)")
ax.set_ylim(0, float(np.nanmax(pv[ARMS].to_numpy(float))) * 1.20)
ax.set_title("(a) 首次加载的平稳时刻：v6 中位 0.55 s（恒载），无责档 2.98/3.82 s")
ax.legend(fontsize=8.6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.90))

# ── (b) 加载沿放大 ──────────────────────────────────────────────
ax = axs[0, 1]
for tag, xoff, col in (("右拇指指尖/数据2", 0.0, S.C_V6), ("切换负载-快相无责", 0.0, S.C_E1)):
    z = C.load_npz(tag)
    tu, tot = z["tu"], z["tot_s"]
    i0 = int(z["i0"][0]) // int(z["ds"][0]) if "ds" in z else int(z["i0"][0])
    dtm = float(z["dtm"][0]) * (int(z["ds"][0]) if "ds" in z else 1)
    win = np.arange(i0, min(len(tu), i0 + int(30 / dtm)))
    tt = tu[win] - tu[i0]
    tot_n = (tot[win] - tot[i0]) / (tot[i0 + int(5 / dtm)] - tot[i0])
    y6 = (z["Y_v6"][win] - tot[i0]) / (tot[i0 + int(5 / dtm)] - tot[i0])
    y3 = (z["Y_e3s"][win] - tot[i0]) / (tot[i0 + int(5 / dtm)] - tot[i0])
    n = 1 if tag.startswith("右") else 2
    ax.plot(tt, tot_n, color=S.C_RAW, lw=1.2, ls="-" if n == 1 else "--",
            label=("原始" if n == 1 else None))
    ax.plot(tt, y3, color=S.C_E3, lw=1.2, ls="-" if n == 1 else "--",
            label=("无责 3 s（现役）" if n == 1 else None))
    ax.plot(tt, y6, color=col, lw=2.0, ls="-" if n == 1 else "--",
            label=("v6" if n == 1 else None))
    ax.annotate(tag, xy=(18, float(y6[-1])), fontsize=8.8, color=col,
                xytext=(19.5, 1.02 + 0.06 * (n - 1) * 3))
ax.axhline(1.0, color=S.C_IDEAL, ls=":", lw=1.4)
ax.text(29.5, 1.008, "真值线", ha="right", fontsize=8.8, color=S.C_IDEAL)
ax.set_xlabel("加载沿后时间 (s)")
ax.set_ylabel("按「加载沿+5 s 电平」归一化")
ax.set_title("(b) 加载沿放大：v6 在 0.6~0.8 s 到平台；实线=右拇指/数据2，虚线=切换负载")
ax.legend(fontsize=8.6, loc="lower right")

# ── (c) 稳态精度的代价 ──────────────────────────────────────────
ax = axs[1, 0]
hold = m[m.kind == "恒载"]
q = hold.groupby("algo").step_ratio.mean().reindex(ARMS)
ax.bar(np.arange(len(ARMS)) - 0.19, q.values, 0.38, color=[COL[a] for a in ARMS],
       label="阶跃保真（左轴，柱）")
for xi, vi in zip(np.arange(len(ARMS)) - 0.19, q.values):
    ax.text(xi, vi - 0.010, f"{vi:.3f}", ha="center", fontsize=9, color="white")
ax.axhline(1.0, color="#c62828", ls="--", lw=1.2)
ax.text(-0.52, 1.002, "1.0 = 台阶完整透传", fontsize=8.6, color="#c62828", ha="left")
ax.set_xticks(range(len(ARMS)))
ax.set_xticklabels([LBL[a] for a in ARMS])
ax.set_ylim(0.985, 1.10)
ax.set_ylabel("阶跃保真（柱，左轴）")
ax2 = ax.twinx()
med = [st[(st.kind == "恒载") & (st.algo == a)].err5s.abs().median() for a in ARMS]
mx = [st[(st.kind == "恒载") & (st.algo == a)].err5s.abs().max() for a in ARMS]
# 把「|误差| %」线性映射到本坐标系的 1.02+err/100（0% → 1.02，5% → 1.07），
# 右轴刻度由同一条映射反算，避免双轴 artist 出界。
ax.plot(np.arange(len(ARMS)) + 0.19, [1.02 + m / 100 for m in med], "D",
        color="0.15", ms=8, label="|加载沿+5 s 误差| 中位（右轴）")
ax.vlines(np.arange(len(ARMS)) + 0.19, [1.02 + m / 100 for m in med],
          [1.02 + m / 100 for m in mx], color="0.15", lw=1.2, ls=":")
ax.plot(np.arange(len(ARMS)) + 0.19, [1.02 + m / 100 for m in mx], "v", color="0.15", ms=5,
        label="同口径最大值（▽）")
for xi, m_ in zip(np.arange(len(ARMS)) + 0.19, med):
    ax.text(xi, 1.02 + m_ / 100 + 0.0035, f"{m_:.2f}%", fontsize=8.4, color="0.2",
            va="bottom", ha="center")
ax2.set_ylim(7.0, 10.0)          # 与主轴同域：刻度值 t 对应主轴坐标 1.02 + t/100
ax2.set_yticks([7, 8, 9, 10])
ax2.set_ylabel("|加载沿 +5 s 相对误差| (%，右轴)", fontsize=9.5)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.4, loc="lower center", ncol=2)
ax.set_title("(c) v6 把台阶多报了 8.2%：形状先验误差直接变成静态错值")
S.note(ax, "v5 的 A 是「从数据里量出来的」；\n"
           "v6 的 Â 是「量 + 先验反演出来的」→\n先验不准就直接进显示", loc=(0.985, 0.03), fs=8.8)

# ── (d) 事件灵敏度代价 ──────────────────────────────────────────
ax = axs[1, 1]
ep_hold = {a: int(hold[hold.algo == a].epoch.sum()) for a in ARMS}
vary = m[m.kind == "实采"]
ep_vary = {a: int(vary[vary.algo == a].epoch.sum()) for a in ARMS}
w = 0.26
for j, a in enumerate(ARMS):
    x0 = (j - 1) * w
    b1 = ax.bar([-0.19 + x0], [ep_hold[a]], w, color=COL[a])
    b2 = ax.bar([0.19 + x0], [ep_vary[a]], w, color=COL[a], alpha=0.5, hatch="//")
    ax.text(-0.19 + x0, ep_hold[a] + 1.2, str(ep_hold[a]), ha="center", fontsize=9.5)
    ax.text(0.19 + x0, ep_vary[a] + 1.2, str(ep_vary[a]), ha="center", fontsize=9.5)
ax.set_xticks([-0.19, 0.19])
ax.set_xticklabels(["恒载 9 组合计", "实采 4 份合计"])
ax.set_xlim(-0.62, 0.62)
ax.set_ylabel("epoch 数（越少越稳）")
ax.set_title("(d) 取消 2.5 s 确认窗 + 6 s 抑制窗的代价：epoch 数 1.6~1.7 倍")
S.note(ax, "实心=无责档；斜纹=v6。\n"
           "v6 改用 3 帧持续 + 0.4 s 可撤销 + 尾部 10% 门\n"
           "→ 保压期的小台阶/手指调整更容易开新 epoch",
       loc=(0.985, 0.97), va="top", fs=8.8)

fig.subplots_adjust(left=0.058, right=0.985, top=0.925, bottom=0.075, wspace=0.20, hspace=0.31)
outp = os.path.join(C.FIG, "F7_fast_tradeoff.png")
fig.savefig(outp)
S.figcheck(fig, outp)
plt.close(fig)
print("saved", outp)

print("\n[F7 数字]")
for a in ARMS:
    s = st[st.algo == a]
    print(f"  {a:>4} T_stable 恒载中位 {s[s.kind=='恒载'].t_stable.median():.2f}s  "
          f"实采 {s[s.kind=='实采'].t_stable.median():.2f}s  "
          f"|err5s| 恒载 {s[s.kind=='恒载'].err5s.abs().median():.2f}%  "
          f"实采 {s[s.kind=='实采'].err5s.abs().median():.2f}%")
