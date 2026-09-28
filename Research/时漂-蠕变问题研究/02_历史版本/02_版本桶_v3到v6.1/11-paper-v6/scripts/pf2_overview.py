# -*- coding: utf-8 -*-
"""论文图 F2：v6 抗蠕变算法总览（逐帧数据流 + 状态机 + 三条输出链路）。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_common as C                                            # noqa: E402
import pv_style as S                                             # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch   # noqa: E402


def box(ax, x, y, w, h, title, body="", fc="#ffffff", ec="0.35", fs=10.2, bfs=8.6,
        tc="0.1", bc="0.3", lw=1.3, radius=0.022):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={radius}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if body:
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tc, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=bfs,
                color=bc, zorder=3, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=tc, zorder=3, linespacing=1.4)
    return (x, y, w, h)


def arrow(ax, p, q, color="0.3", lw=1.5, style="-|>", rad=0.0, ls="-", z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=lw,
                                 color=color, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


fig = plt.figure(figsize=(15.6, 9.4))
fig.suptitle("图 F2  v6 抗蠕变补偿算法总览：逐帧数据流、六类工况与慢相蠕变模块", fontsize=14.5)
ax = fig.add_axes([0.012, 0.035, 0.976, 0.888])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

# ══════════ ① 预处理 ══════════
box(ax, 1.2, 76, 20.5, 17, "① 预处理（每帧）",
    "total = Σv\n"
    "时间轴：样本序号 / fs\n"
    "（不用包时间戳做微分）\n"
    "3 帧中值总量 → 抗单帧掉点\n"
    "σ_d ← 1.4826·MAD(d)",
    fc="#eef4fb", ec="#5b8db8")

# ══════════ ② 检测器 ══════════
box(ax, 24.5, 76, 24.5, 17, "② 检测器 D（严格因果）",
    "d = mean(t−0.20, t] − mean[t−0.65, t−0.35]\n"
    "|d| > max( 5σ_d , 5%·lv_ref , 1%·max_tot )\n"
    "3 帧持续 + 真沿回溯（≤0.6 s 内找\n"
    "越过 pre+3%·跳变的首帧）",
    fc="#eef4fb", ec="#5b8db8")

# ══════════ ③ 分类器 ══════════
box(ax, 51.8, 76, 21.5, 17, "③ 工况分类（六类，锁定一次）",
    "C1 onset / C2 restep / C3 decrease\n"
    "C4 卸载 / C5 复合（形状失配）\n"
    "C6 瞬态（0.4 s 内可撤销）\n"
    "→ 选形状库 f、κ、T_glide",
    fc="#eef4fb", ec="#5b8db8")

# ══════════ ④ 逆模型 ══════════
box(ax, 76.2, 76, 22.6, 17, "④ 逆模型 F（形状约束反演）",
    "Â = Σ y(τ)·g(τ) / Σ g(τ)²\n"
    "  窗 [0.20, 0.80] s，电平域最小二乘\n"
    "限幅 0.5Δ_obs ≤ Â ≤ κ·Δ_obs\n"
    "失配自检 → 转 C5；停滞检测 → Â←Δ实测",
    fc="#eaf6ee", ec="#4c9a63")

# 顶部数据流箭头
for p, q in (((21.7, 84.5), (24.5, 84.5)), ((49.0, 84.5), (51.8, 84.5)),
             ((73.3, 84.5), (76.2, 84.5))):
    arrow(ax, p, q)
arrow(ax, (11.4, 93.0), (11.4, 96.0), lw=1.4)
ax.text(11.4, 97.2, "输入：时间戳 ts + n 通道显示值 v", ha="center", fontsize=11,
        fontweight="bold", color="0.12")

# ══════════ ⑤ 滑行器 ══════════
box(ax, 62.0, 47, 36.8, 21, "⑤ 滑行器 G（替代 v5 的「免责期」）",
    "y_target = Z + (Â − Z)·W ,  W = smoothstep((τ−τ_d)/T_glide)\n"
    "c_target = c0·(1−W) + (目标 − total)·W     ← 继承当前修正，不跳回原始\n"
    "限速 |Δc| ≤ r_max·Â·Δt（r_max = 0.8/s）,  T_glide ∈ [0.40, 0.80] s\n"
    "停滞检测：实测尾巴增长比 < 0.5×模型增长比 持续 0.45 s → 停止预判",
    fc="#f2edfa", ec="#7e57c2", fs=11.0, bfs=8.8)

# ══════════ 状态机 ══════════
box(ax, 1.2, 52, 21.5, 14, "状态机（每帧）",
    "idle → event（滑行）→ slow（慢相），卸载回 idle\n"
    "静默期：空载 0.50 s / 卸载后 0.80 s\n"
    "re-arm：抬升结束后才允许再建事件",
    fc="#fff8e8", ec="#c9a227", fs=10.4, bfs=8.5)

# ══════════ 交接 ══════════
box(ax, 25.5, 52, 33.5, 14, "交接（τ ≥ τ_ho = max(5 s, τ_d+T_glide)）",
    "A_i ← 各通道无蠕变总电平（pin 模式：钉住值 + Â·占比）\n"
    "g ← clip( median[ ded_old_i /(γ_iA_i) ], −0.5, 1.5 )   ← 扣除连续性锚定",
    fc="#eaf6ee", ec="#4c9a63", fs=10.4, bfs=8.6)

# ══════════ ⑥ 慢相模块 ══════════
box(ax, 1.2, 8, 57.8, 34, "⑥ 慢相蠕变模块（逐行沿用 v5，本文的「纯蠕变」主体）",
    "", fc="#e9f7f5", ec="#2e8b7f", fs=11.2)
ax.text(30.1, 37.2,
        "rel_i = (Z_i − A_i)/A_i ,  i ∈ L (受载通道：A_i > 0.10·max A)\n"
        "g_raw = median{ rel_i }             ← 鲁棒共识（器件差异是乘性、残差长尾）\n"
        "g ← g + (Δt/τ_g)(g_raw − g) ,  τ_g = 3.0 s\n"
        "γ_i ← Σ Δt·g·rel_i / Σ Δt·g²   （仅 g > 0.02 时更新）→ clip[0.3, 2.0]\n"
        "ded_i = clip( γ_i·A_i·g , −0.5A_i , +1.5A_i )\n"
        "v_i = Z_i − min( ded_i , max(Z_i, 0) )      ← 输出封顶，只改输出不改状态",
        ha="center", va="center", fontsize=9.0, color="0.15", zorder=3, linespacing=1.75)
ax.text(30.1, 11.2,
        "可选：A 慢修正（本文的取舍开关）—— 把 ΣA 以 0.2%/s 朝「交接时刻实测电平」拉，\n"
        "偏差在 ±2.5%·ΣA 死区内不动：换掉形状先验的平台静态偏置，代价是平台不再绝对平。",
        ha="center", va="center", fontsize=8.7, color=S.C_TRIM, zorder=3, linespacing=1.6)

# ══════════ ⑦ 输出与安全语义 ══════════
box(ax, 62.0, 8, 36.8, 34, "⑦ 输出链路（三条作用域互不重叠）", "", fc="#f7f7f7", ec="0.4", fs=11.2)
ax.text(80.4, 34.0,
        "空载段：v = Z（不估基线、不减基线、不扣蠕变）\n"
        "事件段（滑行/减重冻结）：v = Z − c_applied，扣除量连续\n"
        "慢相段：v = Z − ded（逐通道）\n"
        "输出封顶：ded ≤ max(Z, 0) → 显示不为负\n"
        "卸载：平滑电平落入空载带且 u > 0.3 s → 立即回直通\n"
        "非有限输入直接丢弃（不污染状态）",
        ha="center", va="center", fontsize=9.0, color="0.15", zorder=3, linespacing=1.75)
ax.text(80.4, 11.5,
        "安全语义（与 v5 一致，三条）：\n"
        "① 空载不归零　② 扣除量连续　③ 只改显示值，\n"
        "不改业务数据 / 录制 / 标定 / NG 判定",
        ha="center", va="center", fontsize=8.8, color="0.25", zorder=3, linespacing=1.6)

# ══════════ 连接箭头 ══════════
arrow(ax, (11.4, 76.0), (11.4, 66.2))                       # 预处理 → 状态机
arrow(ax, (11.4, 66.0), (11.4, 42.2))                       # 状态机 → 慢相
arrow(ax, (22.7, 59.0), (25.5, 59.0))                       # 状态机 → 交接
arrow(ax, (59.0, 59.0), (62.0, 59.0))                       # 交接 → 滑行
arrow(ax, (42.2, 52.0), (42.2, 42.2))                       # 交接 → 慢相
arrow(ax, (62.0, 50.0), (59.0, 34.0), color="#2e8b7f", rad=-0.12)   # 慢相 → 输出
arrow(ax, (80.4, 47.0), (80.4, 42.2), color="#7e57c2")      # 滑行 → 输出
arrow(ax, (30.1, 76.0), (30.1, 66.2))                       # 检测 → 分类（下行示意）
arrow(ax, (36.8, 83.5), (51.8, 83.5), color="#5b8db8", lw=1.2, ls=(0, (5, 3)))
arrow(ax, (5.0, 76.0), (5.0, 66.2), color="#c9a227", lw=1.2, ls=(0, (5, 3)))
ax.text(6.4, 70.6, "电平/缓冲", fontsize=8.2, color="#a9841f", ha="left")
ax.text(37.0, 88.8, "工况标签 → 选形状库", fontsize=8.2, color="#3f6f95", ha="left")

# ══════════ 底部时间轴 ══════════
y0 = 3.0
ax.plot([8, 96], [y0, y0], color="0.35", lw=2.2, solid_capstyle="butt", zorder=2)
segs = [(8, 20, "#d9d9d9", "0 → 0.2 s\n检测+回溯"),
        (20, 27, "#cdb4f0", "0.2~0.8 s\n滑行建修正"),
        (27, 36, "#9ad3c9", "1~5 s\n形状走完"),
        (36, 96, "#cfe8e4", "5 s → 段末\n慢相蠕变持续扣除")]
for a, b, col, lab in segs:
    ax.add_patch(plt.Rectangle((a, y0 - 1.5), b - a, 3.0, fc=col, ec="0.5", lw=0.8, zorder=3))
    ax.text((a + b) / 2, y0 + 3.2, lab, ha="center", va="bottom", fontsize=8.4, color="0.2")
ax.annotate("", xy=(8, y0 - 3.4), xytext=(30, y0 - 3.4),
            arrowprops=dict(arrowstyle="<|-|>", color=S.C_ANNO, lw=1.2))
ax.text(19, y0 - 6.6, "v5 首扣 5.5 s / 到位 14 s　→　v6 平台 ~0.6~0.8 s",
        ha="center", fontsize=9.4, color=S.C_ANNO, fontweight="bold")

fig.savefig(os.path.join(C.FIG, "F2_overview.png"))
S.figcheck(fig, os.path.join(C.FIG, "F2_overview.png"))
out = os.path.join(C.FIG, "F2_overview.png")
plt.close(fig)
print("saved", out)
