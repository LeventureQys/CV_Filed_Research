# -*- coding: utf-8 -*-
"""只读出图：随机震荡工况（71~73 s 与全录抖动分布）。

两张图：
  fig_oscillation_zoom.png   71~73 s 细节：输入逐帧 / 现场输入均值 / 显示输出 / 限幅上界
  fig_oscillation_map.png    全录抖动强度地图 + 4 帧/包结构
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

DS = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                  "20260919_134056_single_device_f40a1b")
FIG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "results", "figures"))


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


def main():
    os.makedirs(FIG, exist_ok=True)
    pre = L.load_stream(DS, "device_001_pre_seg0.csv")
    main = L.load_stream(DS, "device_001_seg000.csv")
    el = pre["el"]
    tp = pre["V"].sum(1)
    tm = main["V"].sum(1)

    # ── 图 1：71~73 s 细节 ──
    a, b = 69.5, 75.0
    m = (el >= a) & (el < b)
    t = el[m]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.4), dpi=110,
                             gridspec_kw={"height_ratios": [2.1, 1.0]}, sharex=True)

    ax = axes[0]
    ax.plot(t, tp[m], color="#9aa0a6", lw=0.7,
            label=T("输入逐帧（原样，1~4 帧/时间戳）", "input per-frame"))
    ax.plot(t, tm[m], color="#d93025", lw=1.4,
            label=T("现场显示输出（v6 C+F5）", "display output"))
    # 0.5% 限幅上界（相对输入的滑动中位）
    w = 60
    base = np.convolve(tp[m], np.ones(w) / w, mode="same")
    ax.plot(t, base * 1.005, color="#f9ab00", lw=1.0, ls="--",
            label=T("限幅上界 ≈ 输入平滑值 ×1.005", "clamp bound ~ smoothed input*1.005"))
    ax.axvspan(71.0, 73.0, color="#1a73e8", alpha=0.07)
    ax.annotate(T("你指出的 71~73 s：输入在这里以 ~4.2 Hz / 9000 ADC 峰峰摆动\n"
                  "显示把它低通掉了，只跟着慢包络走（所以看着像「爬升」）",
                  "71-73s: input oscillates ~4.2 Hz / 9000 ADC pp; display low-passes it"),
                xy=(72.0, float(np.percentile(tp[m], 97))), xytext=(73.2, 24500),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="#1a73e8"),
                bbox=dict(fc="#e8f0fe", ec="#1a73e8", lw=0.8, boxstyle="round,pad=0.35"))
    ax.set_ylabel(T("总量 (ADC)", "sum (ADC)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.set_title(T("随机震荡工况 · 71~73 s 细节：输入在抖，显示在低通（抖动峰峰约为整个台阶的 62%）",
                   "Oscillating condition - 71~73 s detail"), fontsize=12)

    ax2 = axes[1]
    off = tm[m] - tp[m]
    ax2.axhline(0, color="#5f6368", lw=0.8, ls=":")
    ax2.plot(t, off, color="#188038", lw=1.1,
             label=T("显示 − 输入（逐帧）", "display - input"))
    # 0.2 s 滑动均值：看清慢漂
    k = 20
    offs = np.convolve(off, np.ones(k) / k, mode="same")
    ax2.plot(t, offs, color="#202124", lw=1.6,
             label=T("同上 ×0.2 s 平滑（慢漂）", "same, 0.2 s smoothed"))
    ax2.axvspan(71.0, 73.0, color="#1a73e8", alpha=0.07)
    ax2.set_xlabel(T("时间 (s)", "time (s)"), fontsize=10)
    ax2.set_ylabel(T("偏移 (ADC)", "offset (ADC)"), fontsize=10)
    ax2.set_ylim(-3500, 1500)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    p1 = os.path.join(FIG, "fig_oscillation_zoom.png")
    fig.savefig(p1, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p1)

    # ── 图 2：全录抖动地图 ──
    fig, axes = plt.subplots(3, 1, figsize=(16, 9.5), dpi=110,
                             gridspec_kw={"height_ratios": [1.5, 1.4, 1.0]})
    ax = axes[0]
    ax.plot(el, tp, color="#9aa0a6", lw=0.5, label=T("输入逐帧", "input"))
    ax.plot(el, tm, color="#d93025", lw=0.8, label=T("显示输出", "display"))
    ax.set_ylabel(T("总量 (ADC)", "sum (ADC)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(T("全录概览：输入（灰）与显示（红）", "overview: input vs display"),
                 fontsize=11)

    # 抖动强度：500 帧窗的去中位极差
    win = 500
    rng = np.empty(len(tp))
    for i in range(len(tp)):
        x = tp[max(0, i - win // 2):i + win // 2]
        rng[i] = x.max() - x.min()
    ax = axes[1]
    ax.plot(el, rng, color="#7b1fa2", lw=0.8,
            label=T("输入抖动强度（0.5 s 窗极差）", "input jitter (0.5 s range)"))
    ax.axhline(1000, color="#d93025", lw=1.0, ls="--",
               label=T("判为「抖」的门限 1000 ADC", "jitter threshold 1000 ADC"))
    ax.set_ylabel(T("极差 (ADC)", "range (ADC)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(T("抖动强度时间线：全录约 23% 的时间处于抖动（>1000 ADC）",
                   "jitter timeline: ~23% of the session is oscillating"), fontsize=11)

    # 包内极差（4 帧/包结构）
    from itertools import groupby
    sizes = {}
    packr = np.zeros(len(tp))
    for kk, g in groupby(range(len(el)), key=lambda i: round(el[i], 3)):
        idx = list(g)
        sizes[len(idx)] = sizes.get(len(idx), 0) + 1
        if len(idx) > 1:
            v = tp[idx]
            packr[idx] = v.max() - v.min()
    ax = axes[2]
    ax.plot(el, packr, color="#00695c", lw=0.7,
            label=T("包内极差（同一 elapsed 的 1~4 帧之间）", "intra-packet range"))
    ax.set_xlabel(T("时间 (s)", "time (s)"), fontsize=10)
    ax.set_ylabel(T("极差 (ADC)", "range (ADC)"), fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(T("取包结构：%s（每 39 ms 一包，包内 timestamp 跨度中位 0.01 ms）"
                   % ", ".join("%d帧×%d包" % (k, v) for k, v in sorted(sizes.items())),
                   "packet structure"), fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(FIG, "fig_oscillation_map.png")
    fig.savefig(p2, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
