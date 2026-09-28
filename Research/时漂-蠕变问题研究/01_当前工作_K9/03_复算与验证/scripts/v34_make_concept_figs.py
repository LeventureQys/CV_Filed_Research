# -*- coding: utf-8 -*-
"""生成《在线双态蠕变观测器补偿算法》正文用的概念示意图。

全部图形为按正文公式合成的示意曲线与结构框图，不含任何代码、文件名、
类名、数据集标识或实测台账数值；坐标轴一律用归一化相对量。
输出：../figure/concept/fig01..fig14*.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.edgecolor"] = "#495057"
matplotlib.rcParams["axes.labelcolor"] = "#212529"
matplotlib.rcParams["text.color"] = "#212529"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure", "concept")
os.makedirs(OUT, exist_ok=True)

C_IN = "#4C6EF5"
C_DIS = "#E8590C"
C_FAST = "#0CA678"
C_SLOW = "#AE3EC9"
C_ZERO = "#868E96"
C_WARN = "#FA5252"
C_BAND = "#40C057"

TITLE_KW = dict(fontsize=13, fontweight="bold", color="#212529")
NOTE_KW = dict(fontsize=10, color="#495057")
SUB_KW = dict(fontsize=9.5, color="#495057")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def clean(ax, xlabel=None, ylabel=None, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=5)
    if title:
        ax.set_title(title, **TITLE_KW)


def panel_title(ax, name, note=None):
    if note:
        ax.set_title(name, fontsize=11.5, fontweight="bold", pad=24)
        ax.text(0.5, 1.012, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.5, color="#495057", linespacing=1.4)
    else:
        ax.set_title(name, fontsize=11.5, fontweight="bold")


def box(ax, x, y, w, h, text, fc="#F1F3F5", ec="#495057", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#212529", linespacing=1.6,
            zorder=3)


def arrow(ax, p, q, color="#495057", lw=1.4, style="-|>", rad=0.0, ls="-", zorder=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, linewidth=lw,
                                 color=color, linestyle=ls, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def fig01_problem():
    t = np.linspace(0, 265, 5300)
    dt = t[1] - t[0]
    tau = 150.0
    r_max = 0.30

    load = np.zeros_like(t)
    load[(t > 20) & (t < 150)] = 1.0
    load[t > 156] = 1.0
    creep = np.zeros_like(t)
    x = 0.0
    for i in range(1, len(t)):
        if load[i] > 0:
            x += dt * (r_max * load[i] - x) / tau
        else:
            x -= dt * x / tau
        creep[i] = x
    v = load + creep

    comp = np.zeros_like(t)
    c = 0.0
    for i in range(1, len(t)):
        if t[i] >= 150.0 and load[i] == 0:
            c = 0.0
        elif load[i] > 0:
            c += dt * (r_max * load[i] - c) / tau
        comp[i] = c
    disp = v - comp

    first_level = disp[(t > 100) & (t < 148)].mean()
    reload_level = disp[(t > 230) & (t < 262)].mean()

    fig, axs = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axs[0]
    ax.plot(t, disp, color=C_DIS, lw=2.6, zorder=3, label="显示（补偿后）")
    ax.plot(t, v, color=C_IN, lw=1.6, alpha=0.85, zorder=4, label="输入读数")
    ax.axhline(first_level, color="#ADB5BD", lw=1.2, ls=":", zorder=2)
    ax.text(2, first_level + 0.035, "首段显示电平\n（本应被保持）", fontsize=9.5, color="#495057",
            va="bottom", linespacing=1.5)
    ax.axvspan(150, 156, color=C_WARN, alpha=0.10, zorder=1)
    arrow_top = disp[(t > 158) & (t < 166)].mean()
    ax.annotate("", xy=(163, first_level), xytext=(163, arrow_top),
                arrowprops=dict(arrowstyle="<|-|>", color=C_WARN, lw=2.0, shrinkA=0, shrinkB=0))
    ax.plot([151, 163], [first_level, first_level], color=C_WARN, lw=1.0, ls=(0, (2, 2)),
            zorder=3)
    ax.plot([163, 190], [arrow_top, arrow_top], color=C_WARN, lw=1.0, ls=(0, (2, 2)), zorder=3)
    ax.text(198, 0.84, "同一负载\n前后显示不一致", color=C_WARN,
            fontsize=10.5, fontweight="bold", va="center", ha="center", linespacing=1.5)
    ax.text(153, 1.42, "整片卸载", color=C_WARN, fontsize=10, ha="center", va="top")
    ax.annotate("扣除量已清零 →\n该区间显示 = 输入读数", xy=(155.5, reload_level * 0.55),
                xytext=(166, 0.42), fontsize=9.5, color="#495057",
                arrowprops=dict(arrowstyle="-|>", color="#495057", lw=1.1))
    ax.set_ylim(-0.05, 1.55)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    clean(ax, ylabel="相对量")

    ax = axs[1]
    ax.plot(t, comp, color=C_SLOW, lw=1.8)
    ax.fill_between(t, 0, comp, color=C_SLOW, alpha=0.12)
    ax.annotate("卸载确认：状态清零", xy=(149, 0.172), xytext=(100, 0.275), color=C_WARN,
                fontsize=10.5, arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.4, shrinkB=3))
    ax.annotate("重载被当作全新事件\n补偿重新生长（追不上材料记忆）", xy=(215, 0.103),
                xytext=(262, 0.245), color=C_SLOW, fontsize=10.5, linespacing=1.5, ha="right",
                arrowprops=dict(arrowstyle="-|>", color=C_SLOW, lw=1.4))
    ax.set_ylim(-0.01, 0.32)
    clean(ax, xlabel="时间（示意）", ylabel="扣除量")
    fig.tight_layout()
    save(fig, "fig01_problem_bloodline.png")


def fig02_flow():
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.03, 0.76, 0.15, 0.14, "输入读数 $v$", fc="#E7F5FF", ec=C_IN, bold=True)
    box(ax, 0.23, 0.76, 0.19, 0.14, "扣零点\n$y=v-z_0$", fc="#F8F0FC", ec=C_ZERO)
    box(ax, 0.47, 0.76, 0.23, 0.14, "扣非弹性状态\n显示 $=v-x_1-x_2$", fc="#FFF4E6", ec=C_DIS)
    box(ax, 0.75, 0.76, 0.15, 0.14, "显示", fc="#FFF4E6", ec=C_DIS, bold=True)

    arrow(ax, (0.18, 0.83), (0.23, 0.83), color=C_IN, lw=1.6)
    arrow(ax, (0.42, 0.83), (0.47, 0.83), color=C_ZERO, lw=1.6)
    arrow(ax, (0.70, 0.83), (0.75, 0.83), color=C_DIS, lw=1.6)

    box(ax, 0.20, 0.42, 0.60, 0.16,
        "状态估计器（逐通道）\n"
        "弹性响应估计　$e=\\max(y-x_1-x_2,\\,0)$\n"
        "快态 $x_1$　·　慢态 $x_2$　·　零点 $z_0$ 在线演化",
        fc="#EBFBEE", ec=C_BAND, fs=10)

    arrow(ax, (0.105, 0.76), (0.325, 0.58), color="#0B7285", lw=1.5)
    ax.text(0.145, 0.655, "$v$", color="#0B7285", fontsize=11, fontweight="bold", va="center")

    arrow(ax, (0.20, 0.50), (0.012, 0.50), color=C_ZERO, lw=1.3, ls=(0, (4, 2)))
    arrow(ax, (0.012, 0.50), (0.012, 0.985), color=C_ZERO, lw=1.3, ls=(0, (4, 2)))
    arrow(ax, (0.012, 0.985), (0.325, 0.985), color=C_ZERO, lw=1.3, ls=(0, (4, 2)))
    arrow(ax, (0.325, 0.985), (0.325, 0.90), color=C_ZERO, lw=1.3, ls=(0, (4, 2)))
    ax.text(0.350, 0.915, "$z_0$", color=C_ZERO, fontsize=11, fontweight="bold", va="bottom")

    arrow(ax, (0.80, 0.50), (0.98, 0.50), color=C_SLOW, lw=1.3, ls=(0, (4, 2)))
    arrow(ax, (0.98, 0.50), (0.98, 0.985), color=C_SLOW, lw=1.3, ls=(0, (4, 2)))
    arrow(ax, (0.98, 0.985), (0.60, 0.985), color=C_SLOW, lw=1.3, ls=(0, (4, 2)))
    arrow(ax, (0.60, 0.985), (0.60, 0.90), color=C_SLOW, lw=1.3, ls=(0, (4, 2)))
    ax.text(0.625, 0.915, "$x_1+x_2$", color=C_SLOW, fontsize=11, fontweight="bold", va="bottom")

    ax.text(0.20, 0.24,
            "虚线为状态反馈：三个逐通道状态由同一帧的弹性估计驱动；\n"
            "整条链路里没有事件判定、没有跨帧账本，也没有需要清零的状态。",
            fontsize=10, color="#495057", va="top", linespacing=1.6)
    save(fig, "fig02_signal_flow.png")


def fig03_gate():
    t = np.linspace(0, 60, 3000)
    dt = t[1] - t[0]
    load = (t > 5) & (t < 30)
    e_lvl = np.where(load, 1.0, 0.0)
    y = e_lvl.copy()
    x1 = np.zeros_like(t)
    x2 = np.zeros_like(t)
    for i in range(1, len(t)):
        if e_lvl[i] > 0:
            x1[i] = x1[i - 1] + dt * (0.12 * e_lvl[i] - x1[i - 1]) / 8.0
            x2[i] = x2[i - 1] + dt * (0.30 * e_lvl[i] - x2[i - 1]) / 60.0
        else:
            x1[i] = x1[i - 1] - dt * x1[i - 1] / 6.0
            x2[i] = x2[i - 1] - dt * x2[i - 1] / 40.0
    ehat = np.maximum(y - x1 - x2, 0.0)

    fig, axs = plt.subplots(3, 1, figsize=(10.4, 7.2), sharex=True)
    axs[0].plot(t, y, color=C_IN, lw=1.9)
    axs[0].axvspan(5, 30, color=C_BAND, alpha=0.08)
    axs[0].text(17, 1.06, "受载", ha="center", fontsize=10, color="#2B8A3E")
    axs[0].text(45, 1.06, "空载", ha="center", fontsize=10, color="#868E96")
    clean(axs[0], ylabel="输入读数 $v$")
    axs[0].set_ylim(-0.05, 1.22)

    axs[1].plot(t, x1, color=C_FAST, lw=1.8, label="快态 $x_1$")
    axs[1].plot(t, x2, color=C_SLOW, lw=1.8, label="慢态 $x_2$")
    axs[1].axvspan(5, 30, color=C_BAND, alpha=0.08)
    axs[1].legend(loc="upper right", frameon=False, fontsize=10)
    axs[1].annotate("卸载后按各自时间常数衰减：快态迅速趋零，慢态仍明显残留",
                    xy=(40, x2[2000]), xytext=(0.6, 0.335), fontsize=9.5, color=C_SLOW,
                    arrowprops=dict(arrowstyle="-|>", color=C_SLOW, lw=1.2))
    axs[1].set_ylim(0, 0.40)
    clean(axs[1], ylabel="非弹性状态")

    axs[2].plot(t, ehat, color="#0B7285", lw=1.9)
    axs[2].fill_between(t, 0, ehat, color="#0B7285", alpha=0.12)
    axs[2].axhline(0, color=C_WARN, lw=1.0, ls=(0, (3, 2)), zorder=1)
    axs[2].axvspan(5, 30, color=C_BAND, alpha=0.08)
    axs[2].annotate("$e=\\max(y-x_1-x_2,\\,0)$ 触底\n→ 自动进入恢复分支", xy=(33, 0.0),
                    xytext=(35.5, 0.62), fontsize=10, color=C_WARN, linespacing=1.5,
                    arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.3))
    axs[2].text(0.6, 1.30, "无阈值判据、无受载状态机：门由弹性估计的自然下限形成",
                fontsize=10, color="#495057")
    axs[2].set_ylim(-0.05, 1.5)
    clean(axs[2], xlabel="时间（示意）", ylabel="弹性估计 $e$")
    fig.tight_layout()
    save(fig, "fig03_load_gate.png")


def fig04_model():
    fig = plt.figure(figsize=(11.8, 5.4))
    axm = fig.add_axes([0.02, 0.05, 0.44, 0.86])
    axm.set_xlim(0, 11)
    axm.set_ylim(0, 10)
    axm.axis("off")
    axm.set_title("模型：弹性 + 两个开尔文-沃伊特单元串联", **TITLE_KW)

    def spring(ax, x0, x1, y, n=8, amp=0.42):
        xs = np.linspace(x0, x1, 2 * n + 1)
        ys = np.full_like(xs, y)
        for k, idx in enumerate(range(1, len(xs) - 1)):
            ys[idx] = y + (amp if k % 2 == 0 else -amp)
        ax.plot(xs, ys, color="#212529", lw=1.6, zorder=3)

    def dashpot(ax, y, x0, x1):
        w = 1.1
        xc = (x0 + x1) / 2
        ax.plot([x0, xc - w / 2], [y, y], color="#212529", lw=1.5, zorder=3)
        ax.add_patch(Rectangle((xc - w / 2, y - 0.5), w, 1.0, fill=False, lw=1.5, ec="#212529",
                               zorder=3))
        ax.add_patch(Rectangle((xc - 0.42, y - 0.28), 0.2, 0.56, fill=True, fc="#DEE2E6",
                               ec="#212529", lw=1.2, zorder=3))
        ax.plot([xc + w / 2, x1], [y, y], color="#212529", lw=1.5, zorder=3)

    def voigt(ax, xa, xb, yc, tag, col, sub):
        ax.plot([xa, xa], [yc - 1.5, yc + 1.5], color="#212529", lw=1.8, zorder=2)
        ax.plot([xb, xb], [yc - 1.5, yc + 1.5], color="#212529", lw=1.8, zorder=2)
        spring(ax, xa, xb, yc + 0.95, n=6, amp=0.3)
        dashpot(ax, yc - 0.95, xa, xb)
        ax.text((xa + xb) / 2, yc + 2.0, f"{tag}（{sub}）", ha="center", fontsize=10.5,
                color=col, fontweight="bold")

    axm.plot([0.4, 0.9], [5.0, 5.0], color="#212529", lw=1.8)
    spring(axm, 0.9, 2.3, 5.0)
    axm.plot([2.3, 3.0], [5.0, 5.0], color="#212529", lw=1.8)
    axm.text(1.6, 6.1, "弹性（瞬时）", ha="center", fontsize=10.5, color=C_DIS, fontweight="bold")
    voigt(axm, 3.0, 5.6, 5.0, "快态", C_FAST, "秒级")
    voigt(axm, 6.6, 9.2, 5.0, "慢态", C_SLOW, "分钟级")
    axm.plot([5.6, 6.6], [6.5, 6.5], color="#212529", lw=1.8)
    axm.plot([5.6, 6.6], [3.5, 3.5], color="#212529", lw=1.8)
    axm.plot([9.2, 9.9], [6.5, 6.5], color="#212529", lw=1.8)
    axm.plot([9.2, 9.9], [3.5, 3.5], color="#212529", lw=1.8)
    axm.plot([9.9, 9.9], [3.5, 6.5], color="#212529", lw=1.8)
    axm.plot([9.9, 10.6], [5.0, 5.0], color="#212529", lw=1.8)
    axm.text(10.6, 5.3, "显示", fontsize=11, color=C_DIS, fontweight="bold")
    axm.text(0.4, 0.5, "串联组合：总变形 = 弹性 + 快相 + 慢相\n"
                       "显示只取弹性分量，故两个非弹性分量都要在线估计",
             fontsize=10, color="#495057", va="bottom", linespacing=1.6)

    ax = fig.add_axes([0.55, 0.13, 0.43, 0.74])
    t = np.linspace(0, 100, 4000)
    elastic = np.where(t > 5, 1.0, 0.0)
    fast = np.where(t > 5, 0.12 * (1 - np.exp(-(t - 5) / 8.0)), 0.0)
    slow = np.where(t > 5, 0.35 * (1 - np.exp(-(t - 5) / 150.0)), 0.0)
    ax.plot(t, elastic + fast + slow, color=C_IN, lw=1.9)
    ax.plot(t, elastic, color=C_DIS, lw=2.2)
    ax.plot(t, fast, color=C_FAST, lw=1.3, ls="--")
    ax.plot(t, slow, color=C_SLOW, lw=1.3, ls="--")
    ax.axvline(5, color="#ADB5BD", lw=1.0, ls=":")
    ax.text(101, 1.44, "输入读数", color=C_IN, fontsize=10, va="center")
    ax.text(101, 1.00, "显示 = 弹性分量", color=C_DIS, fontsize=10, va="center")
    ax.text(101, 0.135, "快态 $x_1$", color=C_FAST, fontsize=10, va="center")
    ax.text(101, 0.37, "慢态 $x_2$", color=C_SLOW, fontsize=10, va="center")
    ax.annotate("加载瞬间：弹性跳变", xy=(5.6, 1.0), xytext=(13, 0.52),
                arrowprops=dict(arrowstyle="-|>", color="#495057", lw=1.2), fontsize=9.5)
    ax.annotate("秒级快相", xy=(24, 1.083), xytext=(30, 1.24),
                arrowprops=dict(arrowstyle="-|>", color=C_FAST, lw=1.2), fontsize=9.5, color=C_FAST)
    ax.annotate("分钟级慢相", xy=(86, 1.245), xytext=(44, 1.38),
                arrowprops=dict(arrowstyle="-|>", color=C_SLOW, lw=1.2), fontsize=9.5,
                color=C_SLOW)
    ax.set_ylim(-0.04, 1.58)
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    clean(ax, xlabel="时间（示意）", ylabel="相对量",
          title="同一阶跃在三个分量上的分解")
    save(fig, "fig04_dual_state_model.png")


def fig05_rules():
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.6))

    ax = axs[0]
    t = np.linspace(0, 40, 2000)
    dt = t[1] - t[0]
    load = (t > 4) & (t < 22)
    e = np.where(load, 1.0, 0.0)
    x1 = np.zeros_like(t)
    for i in range(1, len(t)):
        if e[i] > 0:
            x1[i] = x1[i - 1] + dt * (0.12 * e[i] - x1[i - 1]) / 8.0
        else:
            x1[i] = x1[i - 1] - dt * x1[i - 1] / 6.0
    ax.plot(t, x1 / 0.12, color=C_FAST, lw=2.0)
    ax.axvspan(4, 22, color=C_BAND, alpha=0.08)
    ax.axhline(1.0, color="#ADB5BD", lw=1.0, ls="--")
    ax.text(23.5, 1.045, "饱和值 $r_1e$", fontsize=10, color="#495057", va="bottom")
    ax.annotate("受载：朝 $r_1e$ 收敛\n$\\tau_{c1}$ 量级 8 s", xy=(9.6, 0.505), xytext=(11.0, 0.20),
                arrowprops=dict(arrowstyle="-|>", color=C_FAST, lw=1.3), fontsize=10,
                color=C_FAST, linespacing=1.5)
    ax.annotate("空载：按 $\\tau_{r1}$ 衰减\n（不清零）", xy=(25.0, 0.60), xytext=(27.0, 0.80),
                arrowprops=dict(arrowstyle="-|>", color=C_FAST, lw=1.3), fontsize=10,
                color=C_FAST, linespacing=1.5)
    ax.set_ylim(0, 1.20)
    ax.set_xlim(0, 40)
    clean(ax, xlabel="时间（示意）", ylabel="归一化状态", title="快态：两条分支，同一状态量")
    ax.text(4.6, 1.13, "受载", fontsize=10, color="#2B8A3E")
    ax.text(22.6, 1.13, "空载", fontsize=10, color="#2B8A3E")

    ax = axs[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.24, 0.86, 0.52, 0.12,
        "读数低通导数　$\\mathrm{slope}=(v-v_{lp})/\\tau_{slope}$\n"
        "（斜率中扣除快态已解释部分）", fc="#F1F3F5", ec=C_ZERO, fs=9.5)
    box(ax, 0.28, 0.70, 0.44, 0.12, "沿门　$|\\mathrm{slope}|<2\\%\\cdot e\\,/\\mathrm{s}$",
        fc="#EBFBEE", ec=C_BAND, fs=9.5)
    box(ax, 0.28, 0.54, 0.44, 0.12, "速率上限　$\\pm 1\\%\\cdot e\\,/\\mathrm{s}$",
        fc="#FFF9DB", ec="#F59F00", fs=9.5)
    box(ax, 0.24, 0.38, 0.52, 0.12,
        "积分进慢态　$x_2\\leftarrow x_2+\\Delta t\\cdot\\operatorname{clip}(\\cdot)$",
        fc="#F8F0FC", ec=C_SLOW, fs=9.5)
    box(ax, 0.24, 0.22, 0.52, 0.12, "幅度钳位　$0\\leq x_2\\leq r_{2\\max}e$",
        fc="#F8F0FC", ec=C_SLOW, fs=9.5)
    arrow(ax, (0.50, 0.86), (0.50, 0.82), color=C_ZERO, lw=1.4)
    arrow(ax, (0.50, 0.70), (0.50, 0.66), color=C_BAND, lw=1.4)
    arrow(ax, (0.50, 0.54), (0.50, 0.50), color="#F59F00", lw=1.4)
    arrow(ax, (0.50, 0.38), (0.50, 0.34), color=C_SLOW, lw=1.4)
    ax.text(0.79, 0.76, "台阶沿与快速变化\n不进慢态", fontsize=9.5, color="#2B8A3E", va="center",
            linespacing=1.5)
    ax.text(0.79, 0.60, "即使误判，\n累积速率有限", fontsize=9.5, color="#B08A00", va="center",
            linespacing=1.5)
    ax.text(0.79, 0.285, "慢态不会超过\n载荷比例上限", fontsize=9.5, color=C_SLOW, va="center",
            linespacing=1.5)
    ax.set_title("慢态：两层限制各管一件事", **TITLE_KW)
    fig.tight_layout()
    save(fig, "fig05_state_evolution.png")


def fig08_zero():
    t = np.linspace(0, 120, 6000)
    dt = t[1] - t[0]
    load = (t > 25) & (t < 70)
    true_zero = np.where(t < 45, 0.55,
                         np.where(t < 88, 0.55 - 0.05 * (t - 45) / 43.0, 0.44))
    v = np.where(load, 1.0 + true_zero, true_zero)

    z0 = np.zeros_like(t)
    z0[0] = v[0]
    ymax = np.zeros_like(t)
    for i in range(1, len(t)):
        y = v[i] - z0[i - 1]
        ymax[i] = max(y, ymax[i - 1] - dt * ymax[i - 1] / 600.0)
        if y < 0.05 * ymax[i]:
            z0[i] = z0[i - 1] + dt * (v[i] - z0[i - 1]) / 8.0
        else:
            z0[i] = z0[i - 1]
    y = v - z0

    fig, axs = plt.subplots(2, 1, figsize=(10.8, 6.6), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1]))
    ax = axs[0]
    ax.plot(t, v, color=C_IN, lw=1.7, label="输入读数 $v$")
    ax.plot(t, z0, color=C_ZERO, lw=2.2, label="零点 $z_0$（逐通道演化）")
    ax.plot(t, ymax, color="#ADB5BD", lw=1.2, ls="--", label="峰值包络 $y_{\\max}$（慢衰减）")
    ax.axvspan(25, 70, color=C_WARN, alpha=0.07)
    ax.text(47.5, 1.70, "受载：$z_0$ 冻结", ha="center", fontsize=10.5, color=C_WARN,
            fontweight="bold")
    ax.text(12, 1.70, "空载：$z_0$ 向读数靠拢", ha="center", fontsize=10.5, color="#2B8A3E",
            fontweight="bold")
    ax.text(96, 0.62, "空载：直通 + 慢速再校准", ha="center", fontsize=10.5, color="#2B8A3E",
            fontweight="bold")
    ax.set_ylim(0.30, 1.95)
    ax.set_xlim(0, 120)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    clean(ax, ylabel="相对量")

    ax = axs[1]
    ax.plot(t, y, color="#0B7285", lw=1.8, label="负载响应 $y=v-z_0$")
    ax.fill_between(t, 0, 0.05 * ymax, where=(ymax > 1e-9), color=C_BAND, alpha=0.35,
                    label="近零带（$5\\%\\cdot y_{\\max}$）")
    ax.axhline(0, color="#ADB5BD", lw=1.0)
    ax.axvspan(25, 70, color=C_WARN, alpha=0.07)
    ax.annotate("$y$ 落在近零带内即触发再校准；\n受载时 $y$ 远高于带宽，$z_0$ 冻结",
                xy=(104, -0.012), xytext=(70, 0.60), fontsize=9.5, linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", color="#2B8A3E", lw=1.2), color="#2B8A3E")
    ax.set_ylim(-0.06, 1.30)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    clean(ax, xlabel="时间（示意）", ylabel="相对量")
    fig.tight_layout()
    save(fig, "fig08_zero_tracking.png")


def fig09_startup():
    t = np.linspace(0, 40, 4000)
    dt = t[1] - t[0]
    tau_c = 35.0
    creep = np.where(t > 2, 0.25 * (1 - np.exp(-(t - 2) / tau_c)), 0.0)

    load_step = np.where(t > 4, 1.0, 0.0)
    vA = load_step + creep
    dispA = load_step.copy()

    vB = np.where(t < 12, 0.55 + load_step + creep, 0.0)
    post = t >= 12
    vB = np.where(post, 0.25 * (1 - np.exp(-(t - 12) / tau_c)), vB)
    dispB = np.where(t < 12, vB, 0.0)

    vC = 0.55 + load_step + creep
    dispC = vC.copy()

    cases = [
        ("零载启动", "零点 = 真空点：补偿正常，显示钉在弹性电平", vA, dispA, None),
        ("带载启动 + 调零（推荐）", "调零后读数塌入近零带 → 零点重锚 → 此后正常补偿", vB, dispB, 12.0),
        ("带载启动不调零", "弹性估计约为零：安全但不补偿，显示跟随输入并随蠕变上爬",
         vC, dispC, None),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    for ax, (name, note, v, disp, tare) in zip(axs, cases):
        ax.plot(t, disp, color=C_DIS, lw=2.0, label="显示", zorder=3)
        ax.plot(t, v, color=C_IN, lw=1.5, ls=(0, (5, 3)), label="输入读数", zorder=4)
        if tare is not None:
            ax.plot([tare, tare], [-0.1, 1.55], color=C_WARN, lw=1.3, ls=(0, (3, 2)), zorder=5)
            ax.text(tare + 0.6, 1.50, "调零", fontsize=9.5, color=C_WARN, va="top")
        panel_title(ax, name, note)
        ax.set_ylim(-0.1, 2.0)
        ax.set_xlim(0, 40)
        ax.set_xticks([0, 10, 20, 30, 40])
        clean(ax, xlabel="时间（示意）")
    axs[0].set_ylabel("相对量")
    axs[0].annotate("显示被钉在弹性电平", xy=(20, 0.998), xytext=(10, 0.45),
                    fontsize=9.5, color=C_DIS,
                    arrowprops=dict(arrowstyle="-|>", color=C_DIS, lw=1.2, shrinkB=0))
    axs[0].annotate("输入读数含蠕变、缓慢上爬", xy=(36, 1.155), xytext=(38, 1.58),
                    fontsize=9.5, color=C_IN, ha="right",
                    arrowprops=dict(arrowstyle="-|>", color=C_IN, lw=1.2, shrinkB=0))
    axs[0].legend(loc="lower right", frameon=False, fontsize=9)
    axs[1].annotate("调零后 $z_0$ 重锚，显示保持平直", xy=(30, 0.0), xytext=(16.0, 0.45),
                    fontsize=9.5, color=C_DIS,
                    arrowprops=dict(arrowstyle="-|>", color=C_DIS, lw=1.2, shrinkB=0))
    axs[2].annotate("显示与输入重合、随蠕变上爬", xy=(36, 1.712), xytext=(9, 0.60),
                    fontsize=9.5, color=C_WARN,
                    arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.2, shrinkB=0))
    fig.tight_layout()
    save(fig, "fig09_startup_semantics.png")


def fig10_timescales():
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    items = [
        (3.0, 3.0, "慢漂移速率低通", "$\\tau_{slope}$", C_ZERO, 1),
        (6.0, 6.0, "快态恢复", "$\\tau_{r1}$", C_FAST, -1),
        (8.0, 8.0, "快态收敛", "$\\tau_{c1}$", C_FAST, 1),
        (8.0, 20.0, "零点再校准", "$\\tau_0$", C_BAND, -1),
        (150.0, 150.0, "慢态恢复", "$\\tau_{r2}$", C_SLOW, 1),
        (600.0, 600.0, "峰值包络衰减", "$\\tau_{y\\max}$", C_ZERO, -1),
    ]
    for x, tx, label, sym, col, side in items:
        dy = 0.06 if (label == "快态收敛") else (-0.06 if (label == "零点再校准") else 0.0)
        ax.plot([x, x], [dy, side * 0.52], color=col, lw=1.6)
        ax.plot([x], [dy], marker="o", ms=8, color=col)
        if abs(tx - x) > 1e-9:
            ax.plot([x, tx], [side * 0.52, side * 0.52], color=col, lw=1.0, ls=(0, (3, 2)))
        ax.text(tx, side * 0.58, f"{label}\n{sym}", ha="center",
                va="bottom" if side > 0 else "top", fontsize=10, color=col, linespacing=1.4)
    ax.set_xscale("log")
    ax.set_xlim(1.6, 2600)
    ax.set_ylim(-1.6, 1.45)
    ax.set_yticks([])
    ax.plot([40, 40], [-1.0, 1.25], color="#DEE2E6", lw=1.2, ls="--")
    ax.text(2.1, -1.42, "秒级：沿响应、逐帧平滑与再校准", fontsize=10, color="#495057")
    ax.text(55, -1.42, "分钟～十分钟级：慢蠕变与包络", fontsize=10, color="#495057")
    ax.set_xlabel("时间常数（s，对数轴）")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig10_timescales.png")


def fig11_slow():
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8))
    t = np.linspace(0, 400, 4000)
    dt = t[1] - t[0]
    load = t > 10

    ax = axs[0]
    for amp, col in [(0.07, "#1098AD"), (0.34, "#D6336C")]:
        creep = np.where(load, amp * (1 - np.exp(-(t - 10) / 90.0)), 0.0)
        v = np.where(load, 1.0, 0.0) + creep
        fixed = np.where(load, 0.20 * (1 - np.exp(-(t - 10) / 90.0)), 0.0)
        disp = v - fixed
        ax.plot(t, v, color=col, lw=1.5, alpha=0.9, zorder=3)
        ax.plot(t, disp, color=col, lw=2.0, ls="--", zorder=4)
    ax.axhline(1.0, color="#ADB5BD", lw=1.1, ls=":", zorder=2)
    ax.annotate("小幅度工况：固定幅度偏大\n→ 过扣（显示低于弹性电平）", xy=(300, 0.888),
                xytext=(150, 0.62), fontsize=9.5, color="#1098AD", linespacing=1.5, va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#1098AD", lw=1.2, shrinkB=3))
    ax.annotate("大幅度工况：欠扣（显示仍随时间上爬）", xy=(100, 1.085),
                xytext=(190, 1.17), fontsize=9.5, color="#D6336C", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#D6336C", lw=1.2, shrinkB=3))
    ax.set_ylim(0, 1.45)
    ax.set_xlim(0, 400)
    panel_title(ax, "固定幅度慢态：一个常数落在两种工况之间",
                "实线 = 输入读数，虚线 = 显示，灰色点线 = 弹性电平（应钉住的目标）")
    clean(ax, xlabel="时间（示意）", ylabel="相对量")

    ax = axs[1]
    creep = np.where(load, 0.20 * (1 - np.exp(-(t - 10) / 90.0)), 0.0)
    v = np.where(load, 1.0, 0.0) + creep
    comp = np.zeros_like(t)
    vlp = np.zeros_like(t)
    for i in range(1, len(t)):
        vlp[i] = vlp[i - 1] + dt * (v[i] - vlp[i - 1]) / 3.0
        slope = (v[i] - vlp[i]) / 3.0
        if load[i] and abs(slope) < 0.02:
            comp[i] = comp[i - 1] + dt * np.clip(slope, -0.01, 0.01)
        else:
            comp[i] = comp[i - 1]
    disp = v - comp
    ax.plot(t, comp, color=C_SLOW, lw=1.8, ls="--", label="慢态扣除量（跟踪实测斜率）")
    ax.plot(t, v, color=C_IN, lw=1.5, label="输入读数")
    ax.plot(t, disp, color=C_DIS, lw=2.2, label="显示（钉平）")
    ax.axhline(1.0, color="#ADB5BD", lw=1.1, ls=":")
    ax.text(392, 1.005, "弹性电平", fontsize=9.5, color="#495057", va="bottom", ha="right")
    ax.annotate("扣除量与输入同步上涨，\n二者之差即显示", xy=(250, 0.205), xytext=(96, 0.33),
                fontsize=9.5, color=C_SLOW, linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", color=C_SLOW, lw=1.2, shrinkB=3))
    ax.set_ylim(0, 1.45)
    ax.set_xlim(0, 400)
    ax.legend(loc="center right", frameon=False, fontsize=9.5)
    panel_title(ax, "速率跟踪慢态：显示按构造钉平")
    clean(ax, xlabel="时间（示意）", ylabel="相对量")
    fig.tight_layout()
    save(fig, "fig11_slow_state.png")


def fig12_response():
    cases = [
        ("首次大台阶", 0.0, 0.90, "状态从零建立，路径最长"),
        ("整片卸载后重载", 0.55, 0.55, "状态未清零 → 重平衡路径短"),
        ("受载态小台阶", 0.0, 4.5, "无重锚快捷路径 → 只能等指数重平衡"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    t = np.linspace(0, 60, 3000)
    for ax, (name, start, tau, note) in zip(axs, cases):
        s = np.clip(t - 4, 0, None)
        norm = start + (1.0 - start) * (1 - np.exp(-s / tau))
        ax.axhspan(0.95, 1.05, color=C_BAND, alpha=0.16, label="终值 ±5% 带")
        ax.axhline(1.0, color="#ADB5BD", lw=1.1, ls=":")
        ax.plot(t, norm, color=C_DIS, lw=2.2, label="显示（归一化到终值）")
        panel_title(ax, name, note)
        ax.set_xlim(0, 60)
        ax.set_ylim(-0.12, 1.25)
        clean(ax, xlabel="时间（示意）")
    axs[0].annotate("沿", xy=(4.0, 0.0), xytext=(7.5, -0.09), fontsize=9.5,
                    arrowprops=dict(arrowstyle="-|>", color="#495057", lw=1.1))
    axs[1].annotate("起点已接近终值", xy=(4.2, 0.57), xytext=(12, 0.30), fontsize=9.5,
                    color="#2B8A3E", arrowprops=dict(arrowstyle="-|>", color="#2B8A3E", lw=1.2))
    axs[2].annotate("同样的相对误差\n要等更久", xy=(26, 0.99), xytext=(26, 0.42), fontsize=9.5,
                    color=C_WARN, linespacing=1.5,
                    arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.2))
    axs[0].set_ylabel("显示 / 终值")
    axs[0].legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig12_response_paths.png")


def fig13_consistency():
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    seg_x = np.arange(9)
    band = 0.10
    xticks = list(range(9))

    ax = axs[0]
    no_state = np.array([1.00, 1.02, 0.99, 1.01, 1.00, 0.62, 0.64, 0.63, 0.65])
    ax.axhspan(1.0 - band, 1.0 + band, color=C_BAND, alpha=0.12)
    ax.text(-0.5, 1.0 + band + 0.02, "判据带 ±10%（相对该类首段电平）",
            fontsize=9.5, color="#2B8A3E", va="bottom")
    ax.plot(seg_x[:5], no_state[:5], "o", ms=9, color=C_WARN)
    ax.plot(seg_x[5:], no_state[5:], "o", ms=9, color=C_WARN)
    ax.annotate("卸载-重载剪断连续性：\n同一负载分裂成两族", xy=(5.0, 0.63), xytext=(0.4, 0.43),
                fontsize=10, color=C_WARN, linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.3, shrinkB=9))
    ax.annotate("", xy=(4.4, 0.645), xytext=(4.4, 0.995),
                arrowprops=dict(arrowstyle="<|-|>", color=C_WARN, lw=1.6))
    ax.text(4.55, 0.80, "族极差", fontsize=10, color=C_WARN, fontweight="bold")
    ax.set_title("状态不跨卸载连续（示意）", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0.35, 1.24)
    ax.set_xlim(-0.6, 8.6)
    ax.set_xticks(xticks)
    clean(ax, xlabel="同一负载的逐次受载段", ylabel="显示电平（归一化）")

    ax = axs[1]
    with_state = np.array([1.00, 1.03, 0.97, 1.02, 0.99, 1.01, 0.96, 1.04, 1.02])
    ax.axhspan(1.0 - band, 1.0 + band, color=C_BAND, alpha=0.12)
    ax.plot(seg_x, with_state, "o", ms=9, color=C_FAST)
    ax.annotate("状态随材料连续：\n全部受载段落在判据带内", xy=(6, 0.96), xytext=(1.2, 0.52),
                fontsize=10, color="#2B8A3E", linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", color=C_FAST, lw=1.3))
    ax.set_title("状态跨卸载连续（本文路线，示意）", fontsize=11.5, fontweight="bold")
    ax.set_xlim(-0.6, 8.6)
    ax.set_xticks(xticks)
    clean(ax, xlabel="同一负载的逐次受载段")
    fig.tight_layout()
    save(fig, "fig13_consistency_criterion.png")


def fig14_cost():
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8))

    ax = axs[0]
    cats = ["算术\n（乘加 / 加减）", "比较与限幅\n（含 min / max）", "分支\n（受载/空载、沿门、包络）",
            "除法 / 超越函数"]
    vals = [31, 8, 3, 0]
    cols = ["#4C6EF5", "#F59F00", "#AE3EC9", C_WARN]
    bars = ax.barh(cats[::-1], vals[::-1], color=cols[::-1], height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2,
                f"{v}" if v > 0 else "0（常数除法化为乘倒数）",
                va="center", fontsize=10, color="#495057")
    ax.set_xlim(0, 46)
    ax.set_xlabel("每通道每帧的标量运算次数（解析计数，最重的受载分支）")
    ax.set_title("单次更新的运算构成", fontsize=11.5, fontweight="bold")
    clean(ax)

    ax = axs[1]
    n = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    flops = 42
    ops_per_frame = n * flops
    us_per_frame = ops_per_frame / 1e9 * 1e6
    ax.loglog(n, us_per_frame, "o-", color="#4C6EF5", lw=2.0, ms=6,
              label="每帧耗时估算（$10^9$ 次运算/s 单核）")
    ax.loglog(n, us_per_frame * 4.0, "s--", color="#F59F00", lw=1.6, ms=5,
              label="悲观 4 倍（$2.5\\times10^8$ 次运算/s）")
    ax.axhline(10000, color=C_WARN, lw=1.4, ls=":")
    ax.text(1.3, 11400, "10 ms 帧预算（100 Hz）", fontsize=9.5, color=C_WARN, va="bottom")
    ax.set_xlabel("通道总数")
    ax.set_ylabel("每帧占用时间（µs，估算）")
    ax.set_title("随通道数线性增长，常用通道数下仍留数个量级余量", fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    clean(ax)
    fig.tight_layout()
    save(fig, "fig14_compute_cost.png")


if __name__ == "__main__":
    fig01_problem()
    fig02_flow()
    fig03_gate()
    fig04_model()
    fig05_rules()
    fig08_zero()
    fig09_startup()
    fig10_timescales()
    fig11_slow()
    fig12_response()
    fig13_consistency()
    fig14_cost()
