# -*- coding: utf-8 -*-
"""v3.6 图 1：目标录制总览（输入/显示/补偿量）+ 后段放大 + 逐次加载对齐形状。"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402

FIG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ds = K.load(K.DS_TARGET)
    el = ds["pre"]["el"]
    tin, tout, ded = ds["tot_in"], ds["tot_out"], ds["ded"]

    fig, ax = plt.subplots(3, 1, figsize=(16, 12))
    ax[0].plot(el, tin, color="0.65", lw=1.0, label="算法输入（pre 读数）")
    ax[0].plot(el, tout, color="tab:blue", lw=1.4, label="现场显示（seg 结果）")
    ax[0].set_title("目标录制 20260919_160854 · 全程总波形（Σ21 通道，ADC）")
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)
    for t in (3.34, 15.18, 41.75, 63.41, 232.23, 245.31):
        ax[0].axvline(t, color="tab:red", ls=":", lw=0.8, alpha=0.6)

    ax[1].plot(el, ded, color="tab:green", lw=1.2)
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_title("补偿量 ded = Σ输入 − Σ显示")
    ax[1].grid(alpha=0.3)

    m = (el >= 230) & (el <= 312)
    ax[2].plot(el[m], tin[m], color="0.6", lw=1.6, label="输入")
    ax[2].plot(el[m], tout[m], color="tab:blue", lw=1.4, label="显示")
    ax[2].set_title("后段放大 230~312 s：完全卸载(240 s) → 重载(245.3 s) 之后补偿消失")
    ax[2].legend(loc="lower right")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(FIG, "F0_overview.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # 逐沿对齐形状
    from v36_probe_edges import find_edges, wmed
    edges = find_edges(el, tin)
    ups = [e[0] for e in edges if e[1] > 0]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for t in ups:
        b = wmed(el, tin, t - 2.5, t - 0.5)
        g = np.arange(0.0, 30.0, 0.25)
        yi = np.array([wmed(el, tin, t + u, t + u + 0.25) - b for u in g])
        yo = np.array([wmed(el, tout, t + u, t + u + 0.25) - b for u in g])
        ax[0].plot(g, yi, lw=1.2, label="t=%.1fs" % t)
        ax[1].plot(g, yo, lw=1.2, label="t=%.1fs" % t)
    ax[0].set_title("输入增量（沿后 0~30 s，相对沿前基线）")
    ax[1].set_title("显示增量（同上）—— 各次加载差异巨大")
    for a in ax:
        a.grid(alpha=0.3)
        a.set_xlabel("沿后时间 (s)")
        a.set_ylabel("增量 (ADC)")
        a.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(FIG, "F1_edge_align.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print("-> %s\n-> %s" % (p1, p2))


if __name__ == "__main__":
    main()
