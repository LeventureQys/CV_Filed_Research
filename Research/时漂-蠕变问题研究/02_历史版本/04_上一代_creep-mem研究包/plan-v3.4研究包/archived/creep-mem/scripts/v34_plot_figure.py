# -*- coding: utf-8 -*-
"""v3.4 出图：目标会话全量回放（mem OFF vs ON）的输入/显示总量对比图。

输出: ../figure/v34_零基线-反复增减同一负载_对比.png
用法: python v34_plot_figure.py   （在 scripts/ 目录下）
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_regression import run
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")
OUT = os.path.join(HERE, "..", "figure",
                   "v34_零基线-反复增减同一负载_对比.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    pre, s0, o0 = run(DS, ["--mem", "0"])     # plan-v3.1 行为
    el = pre["el"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.6]})

    # ── 顶部: 修复前(=plan-v3.1) ──
    ax = axes[0]
    ax.plot(el, s0, color="#999", lw=0.6, label="输入（算法前读数 pre）")
    ax.plot(el, o0, color="#d62728", lw=0.8,
            label="显示 · mem OFF（plan-v3.1，修复前）")
    ax.set_ylabel("21 通道总量 (ADC)")
    ax.set_title("修复前（plan-v3.1）：整片卸载后重载丢失基线 —— "
                 "245 s 后显示≈输入，与 15~239 s 补偿族相差 ~2000 ADC", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    # ── 中部: 修复后(plan-v3.4) ──
    _, s1, o1 = run(DS, ["--mem", "120"])
    ax = axes[1]
    ax.plot(el, s1, color="#999", lw=0.6, label="输入（算法前读数 pre）")
    ax.plot(el, o1, color="#1f77b4", lw=0.8,
            label="显示 · mem ON（plan-v3.4 creep-mem，修复后）")
    for t, txt in ((15, "15s 加载\n（补偿族基准）"), (245, "245s 重载\n（修复目标位）")):
        ax.axvline(t, color="#2ca02c", ls="--", lw=1.0, alpha=0.8)
        ax.text(t + 1.5, ax.get_ylim()[0] + 300, txt, fontsize=8,
                color="#2ca02c", va="bottom")
    ax.set_ylabel("21 通道总量 (ADC)")
    ax.set_title("修复后（plan-v3.4 creep-mem）：重载段恢复蠕变补偿，"
                 "满载族显示极差 2516 → 1018 ADC，半载族 1911 → 641 ADC", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    # ── 底部: 修复前后显示差 ──
    ax = axes[2]
    ax.plot(el, o1 - o0, color="#9467bd", lw=0.7,
            label="显示(ON) − 显示(OFF)")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("Δ 显示 (ADC)")
    ax.set_title("修复带来的变化：只发生在整片卸载-重载之后的段落（15~239 s 逐位不变）",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("零基线-反复增减同一负载 · 20260919_160854_single_device_7b3977 · "
                 "真实 C++ 本体回放（参数集 plan-v3.4 creep-mem, mem_tau_s=120s）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=150)
    print("saved:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
