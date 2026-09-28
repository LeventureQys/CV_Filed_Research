# -*- coding: utf-8 -*-
"""t8_figs.py -- T8 的三张图（读 results/t8_*.csv，不重跑任何实验）。

  T8_01_matrix_radar.png  评估矩阵（五路线 × 十维度标准化热力图 + 关键维度雷达）
  T8_02_verdict.png       可替代性裁决（分工况 × 三态 + 判据数字）
  T8_03_discount.png      乐观 vs 悲观折扣（对数轴）

用法：python scripts/t8_figs.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from t8_common import RESULTS, DIM_ORDER, COL_ORDER, setup_cjk_font, save_fig  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def main():
    cjk = setup_cjk_font()
    L = (lambda zh, en: zh) if cjk else (lambda zh, en: en)

    # ------------------------------------------------ 图 1：评估矩阵
    m = pd.read_csv(os.path.join(RESULTS, "t8_matrix.csv"))
    m["key"] = m["dimension"] + "||" + m["route"]

    # 标准化方向：已人工判定「越大越好 / 越小越好」，映射到 0~1（1 = 最好）
    # (维度, 路线) -> (原始数值, 方向)  方向 +1 = 越大越好, -1 = 越小越好
    SPEC = {
        ("稳定时间", "v5.1"): (2.89, -1), ("稳定时间", "v6"): (1.80, -1),
        ("稳定时间", "v6.1"): (1.16, -1), ("稳定时间", "T3-B第三条路"): (0.52, -1),
        ("稳态时漂", "v6"): (34.14, -1), ("稳态时漂", "T3-B第三条路"): (9.15, -1),
        ("台阶保真", "v5.1"): (0.98, +1), ("台阶保真", "v6"): (1.033, +1),
        ("台阶保真", "v6.1"): (0.920, +1), ("台阶保真", "T3-B第三条路"): (0.982, +1),
        ("过充", "v5.1"): (54.39, -1), ("过充", "v6"): (120.81, -1),
        ("过充", "v6.1"): (74.83, -1), ("过充", "T3-B第三条路"): (3.02, -1),
        ("下冲", "v6"): (18.59, -1), ("下冲", "T3-B第三条路"): (18.59, -1),
        ("重复性", "v5.1"): (3.07, -1), ("重复性", "v6"): (5.09, -1),
        ("重复性", "v6.1"): (3.37, -1), ("重复性", "T3-B第三条路"): (2.57, -1),
        ("强扰动鲁棒性", "v6"): (100.0, -1), ("强扰动鲁棒性", "v5.1"): (100.0, -1),
        ("强扰动鲁棒性", "T3-B第三条路"): (2713.3, -1),
        ("卸载行为", "v5.1"): (0.8889, +1), ("卸载行为", "v6"): (1.0, +1),
        ("计算量", "v5.1"): (1.05, -1), ("计算量", "v6"): (1.71, -1),
        ("计算量", "v6.1"): (1.71, -1), ("计算量", "T3-B第三条路"): (1.71, -1),
        ("可标定性", "v5.1"): (0.0, -1), ("可标定性", "v6"): (5.52, -1),
        ("可标定性", "v6.1"): (1.06, -1), ("可标定性", "T3-B第三条路"): (0.5, -1),
    }
    dims = [d for d in DIM_ORDER]
    routes = [c for c in COL_ORDER]
    grid = np.full((len(dims), len(routes)), np.nan)
    for i, d in enumerate(dims):
        vals = [SPEC.get((d, r), (np.nan, -1))[0] for r in routes]
        dirs = [SPEC.get((d, r), (np.nan, -1))[1] for r in routes]
        arr = np.array(vals, dtype=float)
        ok = np.isfinite(arr)
        if not ok.any():
            continue
        lo, hi = np.nanmin(arr[ok]), np.nanmax(arr[ok])
        if hi - lo < 1e-12:
            norm = np.where(ok, 0.5, np.nan)
        else:
            norm = (arr - lo) / (hi - lo)
            for j in range(len(routes)):
                if ok[j] and dirs[j] < 0:
                    norm[j] = 1.0 - norm[j]
        grid[i, :] = norm

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(routes)))
    ax.set_xticklabels(routes, fontsize=10)
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims, fontsize=10)
    for i in range(len(dims)):
        for j in range(len(routes)):
            v = grid[i, j]
            txt = "缺数据" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if (not np.isfinite(v) or 0.25 < v < 0.8) else "white")
    ax.set_title(L("T8-Q1 评估矩阵（标准化分值：1=该维度最好，灰=缺数据）\n"
                   "原始数值见 results/t8_matrix.csv（每格带 value+unit+source_task+source_file+column）",
                   "T8-Q1 Assessment matrix (normalized; 1=best in dimension, grey=no data)"),
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    # 雷达：只画可标定 vs 稳定 vs 过充 等有值的维度
    ax2 = axes[1]
    rdim = [d for d in dims if np.isfinite(grid[dims.index(d), :]).sum() >= 3]
    ang = np.linspace(0, 2 * np.pi, len(rdim), endpoint=False).tolist()
    ang += ang[:1]
    colors = {"v5.1": "#1f77b4", "v6": "#d62728", "v6.1": "#ff7f0e", "T3-B第三条路": "#2ca02c"}
    for j, r in enumerate(routes):
        if r == "v3":
            continue
        vals = [grid[dims.index(d), j] for d in rdim]
        if not np.isfinite(vals).all():
            continue
        v = vals + vals[:1]
        ax2.plot(ang, v, "o-", lw=2, ms=4, color=colors.get(r, "k"), label=r)
    ax2.set_xticks(ang[:-1])
    ax2.set_xticklabels(rdim, fontsize=9)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_ylim(0, 1.05)
    ax2.set_title(L("关键维度雷达（仅画 4 条有完整数据的路线）",
                    "Key-dimension radar (4 routes with complete data)"), fontsize=11)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)
    save_fig(fig, "T8_01_matrix_radar.png")

    # ------------------------------------------------ 图 2：可替代性裁决
    s = pd.read_csv(os.path.join(RESULTS, "t8_substitutability.csv"))
    verdict_score = {"v6 可替代（有条件）": 2.0, "有条件可替代": 1.0, "**不可替代**": 0.0}
    conds, score = [], []
    for _, row in s.iterrows():
        c = str(row["condition"])
        conds.append(c[:2] + "\n" + c[2:14].replace("（", "\n（") if len(c) > 14 else c)
        v = str(row["verdict"])
        sc = 2.0 if "可替代" in v and "不可替代" not in v and "有条件" not in v else (
            0.0 if "不可替代" in v else 1.0)
        score.append(sc)
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    cols = ["#2ca02c" if x == 2 else ("#ffbf00" if x == 1 else "#d62728") for x in score]
    bars = ax.bar(range(len(conds)), score, color=cols, width=0.55, edgecolor="black")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([L("不可替代", "Not substitutable"),
                        L("有条件可替代", "Conditionally"),
                        L("可替代（有条件）", "Substitutable")], fontsize=10)
    ax.set_ylim(0, 2.9)
    for b, sc, (_, row) in zip(bars, score, s.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, sc + 0.08, str(row["verdict"]).replace("**", ""),
                ha="center", fontsize=9, fontweight="bold")
    # 关键判据注释
    notes = [
        L("中位 1.80 s / 达标 5/9=55.6%\nworst 13.16 s ⇒ 只可承诺中位",
          "median 1.80 s / pass 5/9\nworst 13.16 s"),
        L("restep 19 个仅 2 个可测\nOS5 中位 −18.59% / G=0.473",
          "restep 2/19 measurable\nOS5 -18.59% / G=0.473"),
        L("全卸载 9/9 达标、残余 max 2.81%\n部分卸载 n=1（仅定性）",
          "unload 9/9 pass, resid<=2.81%\npartial unload n=1"),
        L("±1000 ADC：失败率 100%\nA50 仅 75.7 ADC，锚点错 94% 电平",
          "±1000 ADC: 100% fail\nA50 75.7 ADC"),
    ]
    for i, nt in enumerate(notes):
        ax.text(i, -0.42, nt, ha="center", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="grey", lw=0.6))
    ax.set_title(L("T8-Q3 可替代性裁决（分工况；判据与出处见 results/t8_substitutability.csv）",
                   "T8-Q3 Substitutability verdict by operating case"), fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(bottom=0.32)
    save_fig(fig, "T8_02_verdict.png")

    # ------------------------------------------------ 图 3：乐观 vs 悲观
    o = pd.read_csv(os.path.join(RESULTS, "t8_optimism_discount.csv"))
    labels = ["O2 形状泛化\nonset→跨形态", "O5 强扰动\nA50 vs 用户量级",              "O6 重复性\nRstep vs R5", "O7 路径 RMS\nv5.1 vs v6",
              "O8 卸载残余\n中位 vs max", "O9 对称性\n保守 vs 中位",
              "O4 κ 尾部\n1.15 vs 1.10", "O1 达标率\n1 s vs 2 s"]
    opt = [3.10, 75.7, 3.52, 0.26, 0.19, 13.5, 78.50, 33.3]
    pes = [56.39, 1000.0, 5.09, 7.90, 2.81, 24.0, 5.36, 55.6]
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2, opt, w, label=L("乐观（自身标定 / 有利口径）", "Optimistic"),
                color="#7fb3d5", edgecolor="black")
    b2 = ax.bar(x + w / 2, pes, w, label=L("悲观（留一·跨录制 / 用户量级）", "Pessimistic"),
                color="#c0392b", edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel(L("数值（对数轴；单位各异，见图例与 csv）", "value (log scale)"), fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12,
                    f"{b.get_height():g}", ha="center", fontsize=8)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_title(L("T8-Q2 乐观性折扣：每项都必须给两个数，决策列见 results/t8_optimism_discount.csv:decide_with",
                   "T8-Q2 Optimism discount: two numbers per item"), fontsize=11.5)
    fig.subplots_adjust(bottom=0.22)
    save_fig(fig, "T8_03_discount.png")
    print("figures done")


if __name__ == "__main__":
    main()
