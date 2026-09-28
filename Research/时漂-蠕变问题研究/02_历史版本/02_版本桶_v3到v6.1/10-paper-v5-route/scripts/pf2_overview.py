# -*- coding: utf-8 -*-
"""图 2：逐帧数据流与状态机总览（示意图，坐标为排版量，不代表实测数值）。

产出：figures/F2_overview.png
"""
from figstyle import (C_ALG, C_FAST, C_GRAY, C_SLOW, plt, save_figure)

BOX = dict(fc="#FFFFFF", ec=C_GRAY, lw=1.15, zorder=2)
BOXF = dict(fc="#FDF2E3", ec=C_FAST, lw=1.35, zorder=2)
BOXA = dict(fc="#FDEDEC", ec=C_ALG, lw=1.35, zorder=2)
BOXS = dict(fc="#E8F6F3", ec=C_SLOW, lw=1.35, zorder=2)
BOXG = dict(fc="#F4F6F7", ec=C_GRAY, lw=1.15, zorder=2)


def box(ax, x, y, w, h, text, style=BOX, fs=8.9, tc="#1B2631"):
    ax.add_patch(plt.Rectangle((x, y), w, h, **style))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, linespacing=1.6)


def arrow(ax, p, q, color=C_GRAY, lw=1.4, style="-|>"):
    ax.annotate("", xy=q, xytext=p, zorder=3,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(13.4, 8.7))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 134)
ax.set_ylim(0, 87)
ax.axis("off")

ax.text(1.0, 82.0, "(a) 逐帧数据流：一条主干 + 三条互斥的输出链", fontsize=12.6, color="#1B2631")

# ── 主干第一排 ───────────────────────────────────────────────────
box(ax, 1.0, 65.6, 17.5, 8.0,
    "输入\n$t$ + 显示值向量 $\\mathbf{v}$（$n$ 通道）", BOX, fs=8.8)
box(ax, 20.5, 65.6, 20.0, 8.0,
    "$\\mathrm{total}=\\sum_i v_i$ → 环形缓冲 1024 帧\n三级 EMA：0.3 / 0.7 / 6.0 / 10.0 s", BOX, fs=8.5)
box(ax, 42.5, 65.6, 21.5, 8.0,
    "变载判据（取“或”）\n① $|\\mathrm{fast}-\\mathrm{slow}|>\\mathrm{thr}$\n② 短滞后电平差 $d_{lev}>5\\%$", BOX,
    fs=8.4)
box(ax, 66.0, 65.6, 18.0, 8.0,
    "$\\mathrm{pending}$ 计时\n连续保持 2.5 s 才确认\n（回落即取消）", BOX, fs=8.6)
box(ax, 86.0, 65.6, 18.0, 8.0,
    "状态机迁移\n空载→负载 / 负载内变载 /\n负载→空载", BOX, fs=8.6)
arrow(ax, (18.5, 69.6), (20.5, 69.6))
arrow(ax, (40.5, 69.6), (42.5, 69.6))
arrow(ax, (64.0, 69.6), (66.0, 69.6))
arrow(ax, (84.0, 69.6), (86.0, 69.6))

# ── 右列：epoch 与免责期 ─────────────────────────────────────────
box(ax, 106.5, 65.6, 26.5, 8.0,
    "进入/重启 epoch：\n$\\mathrm{onset}=t$，$\\mathrm{fast\\_done}=0$\n空载→负载时 $\\mathrm{carry}=0$",
    BOXF, fs=8.4)
arrow(ax, (104.0, 69.6), (106.5, 69.6), color=C_FAST)

box(ax, 86.0, 52.0, 18.0, 8.0,
    "空载态（$\\mathrm{in\\_load}=0$）\n输出 = 输入\n（不估基线、不扣蠕变）", BOX, fs=8.0)
arrow(ax, (95.0, 65.6), (95.0, 60.0))

box(ax, 106.5, 52.0, 26.5, 8.0,
    "快相免责期（$u<\\mathrm{fast\\_phase\\_s}$）\n不采 $A$、不积分 $g$；\n末 1/3 窗累积幅度样本",
    BOXF, fs=8.2)
arrow(ax, (119.7, 65.6), (119.7, 60.0), color=C_FAST)
ax.text(120.4, 62.3, "$u<\\mathrm{fast}$", fontsize=8.2, color=C_FAST)

box(ax, 106.5, 42.6, 26.5, 6.4,
    "期内输出 $\\mathbf{v}=\\mathbf{Z}-\\mathrm{carry}$", BOXA, fs=9.0)
arrow(ax, (119.7, 52.0), (119.7, 49.0), color=C_ALG)
ax.text(108.6, 29.6, "$u\\geq\\mathrm{fast}$", fontsize=8.0, color=C_FAST, ha="left",
        va="center")

box(ax, 106.5, 31.8, 26.5, 8.0,
    "免责期收尾当帧\n$A=\\overline{\\mathrm{Z}-\\mathrm{carry}}$\n$g\\leftarrow$ 由 $\\mathrm{carry}$ 锚定（含 $\\gamma$）",
    BOXF, fs=8.2)
arrow(ax, (119.7, 42.6), (119.7, 39.8), color=C_FAST)
arrow(ax, (110.0, 31.8), (110.0, 27.0), color=C_FAST)

# ── pending 冻结 ─────────────────────────────────────────────────
box(ax, 86.0, 42.0, 18.0, 6.8,
    "已受载且 $\\mathrm{pending}$：\n冻结扣除量，不积分 $g$、$\\gamma$", BOXG, fs=8.0)
arrow(ax, (95.0, 52.0), (95.0, 48.8))

box(ax, 86.0, 31.8, 18.0, 8.0,
    "输出封顶\n$\\mathrm{ded}_i\\leftarrow\\min(\\mathrm{ded}_i,\\max(Z_i,0))$\n只改输出、不改状态",
    BOXA, fs=8.0)
arrow(ax, (95.0, 42.0), (95.0, 39.8), color=C_ALG)

# ── 中间行：慢相估计 → 输出 ──────────────────────────────────────
box(ax, 44.0, 31.8, 40.0, 8.0,
    "慢相蠕变估计（仅受载通道 $A_i>0.1\\max A$）\n"
    "$g_{\\mathrm{raw}}=\\mathrm{median}\\{(Z_i-A_i)/A_i\\}$；$g\\leftarrow g+\\dfrac{\\Delta t}{3.0}(g_{\\mathrm{raw}}-g)$\n"
    "$\\gamma_i\\leftarrow\\mathrm{clip}\\left(\\sum\\Delta t\\,g\\,\\mathrm{rel}_i/\\sum\\Delta t\\,g^2,\\,0.3,\\,2.0\\right)$",
    BOXS, fs=8.1)

box(ax, 1.0, 31.8, 41.0, 8.0,
    "本帧输出\n$\\mathrm{ded}_i=\\mathrm{clip}(\\gamma_iA_ig,\\,-0.5A_i,\\,1.5A_i)$\n"
    "$\\mathbf{v}=\\mathbf{Z}-\\mathrm{ded}_i$（未受载通道直通）", BOXA, fs=8.4)
arrow(ax, (44.0, 35.8), (42.0, 35.8), color=C_ALG)

# ── 说明框 ───────────────────────────────────────────────────────
box(ax, 1.0, 15.4, 41.0, 8.8,
    "三条输出链，作用域互不重叠：\n"
    "① 非负载段 → 直通（不估基线、不减基线、不扣蠕变）\n"
    "② 受载且免责期内 → 只减 $\\mathrm{carry}$\n"
    "③ 受载且免责期结束 → 减 $\\gamma_iA_ig$，再封顶",
    BOX, fs=7.9)
arrow(ax, (21.0, 31.8), (21.0, 24.4), color=C_GRAY)

box(ax, 44.0, 14.9, 40.0, 9.8,
    "两条“冻结”分支不是一回事：\n"
    "pending 冻结：保护输入（新台阶不被旧模型吃掉）\n"
    "免责期：保护模型（快相不污染 $A$ 与 $g$）",
    BOXG, fs=7.9)
box(ax, 86.0, 15.4, 47.0, 8.8,
    "每帧只改显示值：\n"
    "业务 canonical 帧、录制原始数据、\n标定与 NG 判定均不受影响；\n"
    "无法建立可靠估计时（$A$ 全零）自然退化为直通。",
    BOX, fs=7.9)

ax.plot([3.0, 131.0], [13.6, 13.6], color="#D5D8DC", lw=1.1)

# ── (b) 时间轴 ───────────────────────────────────────────────────
ax.text(1.0, 11.6, "(b) 时间轴落点：四段时延都以“真实加载沿”为原点（色块宽度为示意，非等比例）",
        fontsize=12.0, color="#1B2631")

y_lane, h_lane = 4.6, 5.4
# 色块边界与下方刻度对齐：2.0=加载沿、39.0=epoch 起点（首扣 = 加载沿 + 5.5 s）、
# 116.0=扣到位 14.5 s。前两块为白底（仅灰边框），后两块带底色。
segs = [(2.0, 14.0, "① 检测窗\n0 ~ 0.5 s", "#FFFFFF"),
        (16.0, 23.0, "② 确认窗\n2.5 s（与档位无关）", "#FFFFFF"),
        (39.0, 22.0, "③ 免责期\n3 s / 5 s（可切）", "#FDF2E3"),
        (61.0, 55.0, "④ 此后持续跟随\n（蠕变仍在增长，扣除量永远略落后）", "#E8F6F3")]
for x, w, txt, fc in segs:
    ec = {"#FDF2E3": C_FAST, "#E8F6F3": C_SLOW, "#FDEDEC": C_ALG, "#FFFFFF": C_GRAY}[fc]
    ax.add_patch(plt.Rectangle((x, y_lane), w, h_lane, fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.text(x + w / 2, y_lane + h_lane / 2, txt, ha="center", va="center",
            fontsize=7.9, color="#1B2631", zorder=4, linespacing=1.6)

for xt, lab, ha in [(2.0, "加载沿", "left"), (39.0, "epoch 起点", "center"),
                    (61.0, "首扣 5.5 s", "center"), (116.0, "扣到位 14.5 s", "right")]:
    ax.plot([xt, xt], [y_lane - 0.85, y_lane - 0.25], color=C_GRAY, lw=1.0, zorder=3)
    ax.text(xt, y_lane - 1.05, lab, ha=ha, va="top", fontsize=8.2, color=C_GRAY)

save_figure(fig, "F2_overview.png")
