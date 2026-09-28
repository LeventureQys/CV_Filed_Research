# -*- coding: utf-8 -*-
"""论文配图统一样式（中文标签、网格、字号、配色）。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 115,
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.framealpha": 0.92,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# 论文统一配色
C_RAW = "0.60"          # 原始（无补偿）
C_V6 = "#2ca02c"        # v6
C_E1 = "#1f77b4"        # 无责 1 s
C_E3 = "#ff7f0e"        # 无责 3 s（现役）
C_TRIM = "#d62728"      # A 慢修正
C_IDEAL = "#17becf"     # 理想值
C_EXEMPT = "#ffb74d"    # 免责/瞬态区
C_GLIDE = "#7e57c2"     # 滑行区
C_SLOW = "#4db6ac"      # 慢相区
C_ANNO = "#8d6e63"


def tag(ax, s, loc="upper left", fs=9.5, color="0.15", **kw):
    """面板序号标签 (a)(b)(c)…"""
    ax.text(0.015 if "left" in loc else 0.985, 0.975, s, transform=ax.transAxes,
            ha="left" if "left" in loc else "right", va="top", fontsize=fs,
            color=color, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5), **kw)


def note(ax, s, loc=(0.985, 0.03), ha="right", va="bottom", fs=8.8, color="0.25"):
    ax.text(loc[0], loc[1], s, transform=ax.transAxes, ha=ha, va=va, fontsize=fs,
            color=color, bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.5))


def figcheck(fig, path, quiet=False):
    """机器自检（非视觉）：空白面板 / 越界 / 文本重叠。

    越界判定只针对**文本**（文字被画布截断才是排版事故）；线条与矩形在坐标轴裁剪下
    报"出界"会产生大量假阳性（bar/line 的 window_extent 反映数据范围，不反映可见范围），
    因此不纳入。视觉验收由 `pv_vision_check.py`（DeepSeek vision）另做。
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    empty, oob_tot, ov_tot, details = [], 0, 0, []
    for i, a in enumerate(fig.axes):
        arts = list(a.lines) + list(a.patches) + list(a.texts)
        if a.title.get_text():
            arts.append(a.title)
        has_data = any(x.get_visible() for x in arts)
        if not has_data:
            empty.append(i)
        oob = 0
        for art in list(a.texts) + ([a.title] if a.title.get_text() else []):
            if not art.get_visible():
                continue
            try:
                bb = art.get_window_extent(r)
            except Exception:
                continue
            area = bb.width * bb.height
            if area <= 0:
                continue
            x0, x1 = max(bb.x0, fb.x0), min(bb.x1, fb.x1)
            y0, y1 = max(bb.y0, fb.y0), min(bb.y1, fb.y1)
            if max(0.0, x1 - x0) * max(0.0, y1 - y0) / area < 0.85:
                oob += 1
        items = [(t.get_text().strip()[:24], t.get_window_extent(r))
                 for t in a.texts if t.get_text().strip()]
        if a.title.get_text().strip():
            items.append(("<标题>", a.title.get_window_extent(r)))
        ov = 0
        for j in range(len(items)):
            for k in range(j + 1, len(items)):
                b1, b2 = items[j][1], items[k][1]
                x0, x1 = max(b1.x0, b2.x0), min(b1.x1, b2.x1)
                y0, y1 = max(b1.y0, b2.y0), min(b1.y1, b2.y1)
                inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                if inter > 0.12 * min(b1.width * b1.height, b2.width * b2.height):
                    ov += 1
        oob_tot += oob
        ov_tot += ov
        details.append(f"ax{i} artists={len(arts)} oob={oob} overlap={ov}")
    res = dict(axes=len(fig.axes), empty=empty, oob=oob_tot, overlap=ov_tot)
    if not quiet:
        print(f"[figcheck] {path.split(chr(92))[-1]}: axes={res['axes']} "
              f"empty={empty or 'none'} oob={oob_tot} text_overlap={ov_tot}")
        if oob_tot or ov_tot:
            print("   " + " | ".join(details))
    return res
