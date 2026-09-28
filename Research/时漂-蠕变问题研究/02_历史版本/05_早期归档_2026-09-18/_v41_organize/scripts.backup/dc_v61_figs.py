# -*- coding: utf-8 -*-
"""v6.1 出图：尖峰的修复验证（L1 时序放大 + L2 指标）。

L1_v61_spike_fix.png  四格时序放大（切换负载 @185.98 / @133.84 / 再切换负载 @57.28 / 恒载 数据1 首加载）
L2_v61_metrics.png    尖峰指标柱状对比 + 恒载指标对比（读 results/v61_ab_*.csv）

含 figcheck（非视觉结构自检：空面板 / artist 越界 / 文本重叠）。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402
from glm53_v61 import GLM53v61                              # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
COL = {"raw": "0.62", "v6": "#2ca02c", "v61": "#d62728"}
LBL = {"raw": "原始", "v6": "v6", "v61": "v6.1"}


def _inter(b1, b2):
    return (max(0.0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
            * max(0.0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0)))


def figcheck(fig, path):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    print(f"\n[figcheck] {os.path.basename(path)}  {fb.width:.0f}x{fb.height:.0f}px  axes={len(fig.axes)}")
    empty, oob_tot, ov_tot = [], 0, 0
    for i, a in enumerate(fig.axes):
        has = bool(a.lines or a.patches or a.collections or a.images)
        if not has:
            empty.append(i)
        oob = 0
        for art in list(a.lines) + list(a.patches) + list(a.texts):
            if not art.get_visible():
                continue
            try:
                bb = art.get_window_extent(r)
            except Exception:
                continue
            if bb.width * bb.height > 0 and _inter(bb, fb) / (bb.width * bb.height) < 0.85:
                oob += 1
        items = [(t.get_text().strip()[:24], t.get_window_extent(r))
                 for t in a.texts if t.get_text().strip()]
        if a.title.get_text().strip():
            items.append(("<标题>", a.title.get_window_extent(r)))
        ov = 0
        for j in range(len(items)):
            for k in range(j + 1, len(items)):
                b1, b2 = items[j][1], items[k][1]
                if _inter(b1, b2) > 0.12 * min(b1.width * b1.height, b2.width * b2.height):
                    ov += 1
        oob_tot += oob
        ov_tot += ov
        print(f"  ax{i} lines={len(a.lines):2d} patches={len(a.patches):3d} texts={len(a.texts):2d} "
              f"有内容={has} 越界={oob} 重叠={ov}")
    print(f"  -> 空白 {empty or '无'}；越界合计 {oob_tot}；重叠合计 {ov_tot}")


def run(cls, d):
    tu, Xu = d["tu"], d["Xu"]
    c = cls(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return c, L.med_smooth(Y.sum(axis=1), 0.5 / d["dtm"])


PANELS = [(9, 183.0, 191.0, "① 切换负载 @185.98 s：v6 +10.9% 尖峰 → v6.1 +2.7%"),
          (9, 131.5, 138.0, "② 切换负载 @133.84 s：v6 +4.9% 过充平台 → v6.1 -2.9%"),
          (10, 55.0, 62.0, "③ 再切换负载 @57.28 s：v6 +6.6% → v6.1 -1.3%"),
          (0, 7.0, 17.0, "④ 恒载 右拇指/数据1：同一机制，绝对量只有 ~1 个显示单位")]

curves = {}
for i, _, _, _ in PANELS:
    tag, path = C.ALL[i]
    if tag in curves:
        continue
    d = L.prep(path)
    curves[tag] = dict(d=d, raw=L.med_smooth(d["Xu"].sum(axis=1), 0.5 / d["dtm"]))
    for name, cls in (("v6", GLM53v6), ("v61", GLM53v61)):
        _, ys = run(cls, d)
        curves[tag][name] = ys

fig, axes = plt.subplots(2, 2, figsize=(17.5, 9.0))
fig.suptitle("v6.1 · 阶跃尖峰的定位与修复（灰=原始，绿=v6，红=v6.1）", fontsize=14)
for ax, (i, ta, tb, title) in zip(axes.ravel(), PANELS):
    tag, _ = C.ALL[i]
    cu = curves[tag]
    tu = cu["d"]["tu"]
    m = (tu >= ta) & (tu <= tb)
    ax.plot(tu[m], cu["raw"][m], color=COL["raw"], lw=1.6, label=LBL["raw"])
    ax.plot(tu[m], cu["v6"][m], color=COL["v6"], lw=1.6, label=LBL["v6"])
    ax.plot(tu[m], cu["v61"][m], color=COL["v61"], lw=1.6, label=LBL["v61"])
    ax.set_title(f"{title}\n{tag}", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    if tag.endswith("数据1"):
        ax.set_ylabel("总量 (显示单位)", fontsize=9)
    else:
        ax.set_ylabel("总量 (ADC)", fontsize=9)
axes.ravel()[0].legend(fontsize=9, loc="lower right")
fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.06, hspace=0.34, wspace=0.16)
p1 = os.path.join(FIG, "L1_v61_spike_fix.png")
fig.savefig(p1, dpi=115)
figcheck(fig, p1)
plt.close(fig)
print(f"图已保存：{p1}")

# ── L2 指标 ──
srcs = [os.path.join(RES, f) for f in ("v61_ab_final.csv", "v61_ab_final_settle.csv")]
if not all(os.path.exists(p) for p in srcs):
    print("[skip] L2：缺 v61_ab_final.csv（先跑 cw_v61_ab.py --arms both --rom 1.06 --onsetonly 1 --tag final）")
    sys.exit(0)
df = pd.read_csv(srcs[0])
st = pd.read_csv(srcs[1])
algos = [a for a in ("v6", "v61_s1.06_o1") if a in set(df.algo)]
ALBL = {"v6": "v6", "v61_s1.06_o1": "v6.1"}
v = df[df.kind == "实采"]
h = df[df.kind == "恒载"]

fig2, axs = plt.subplots(1, 3, figsize=(18.0, 5.4))
fig2.suptitle("v6 / v6.1 指标对比（13 份同口径）", fontsize=13.5)

ax = axs[0]
g = v.groupby("algo").agg(超调max=("sp_over_max", "max"), 回落max=("sp_tr_max", "max"),
                          超调中位=("sp_over_max", "median")).reindex(algos)
xs = np.arange(2)
for j, a in enumerate(algos):
    b = ax.bar(xs + (j - (len(algos) - 1) / 2) * 0.26,
               [g.loc[a, "超调max"], g.loc[a, "回落max"]], 0.25,
               color=COL["v6"] if a == "v6" else COL["v61"], alpha=1.0 - 0.25 * j,
               label=ALBL.get(a, a))
    ax.bar_label(b, fmt="%.1f", fontsize=9)
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xticks(xs); ax.set_xticklabels(["阶跃后最大超调\n(占阶跃 %)", "峰后最大回落\n(占阶跃 %)"])
ax.set_title("(1) 实采 27 个真阶跃：尖峰口径（越小越好）", fontsize=10.5)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = axs[1]
g2 = v.groupby("algo").agg(全程偏差=("max_gap", "median"), 变载窗偏差=("gap_med", "median")).reindex(algos)
for j, a in enumerate(algos):
    b = ax.bar(xs + (j - (len(algos) - 1) / 2) * 0.26,
               [g2.loc[a, "全程偏差"], g2.loc[a, "变载窗偏差"]], 0.25,
               color=COL["v6"] if a == "v6" else COL["v61"], alpha=1.0 - 0.25 * j,
               label=ALBL.get(a, a))
    ax.bar_label(b, fmt="%.0f", fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels(["全程最大偏差\n中位 (ADC)", "变载窗偏差\n中位 (ADC)"])
ax.set_title("(2) 实采 4 份：偏差（越小越好）", fontsize=10.5)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = axs[2]
g3 = h.groupby("algo").agg(全段=("drift_main", lambda s: s.abs().mean()),
                           慢相段=("drift_slow", lambda s: s.abs().mean()),
                           保真=("step_ratio", "mean")).reindex(algos)
xs3 = np.arange(3)
for j, a in enumerate(algos):
    b = ax.bar(xs3 + (j - (len(algos) - 1) / 2) * 0.26,
               [g3.loc[a, "全段"], g3.loc[a, "慢相段"], g3.loc[a, "保真"]], 0.25,
               color=COL["v6"] if a == "v6" else COL["v61"], alpha=1.0 - 0.25 * j,
               label=ALBL.get(a, a))
    ax.bar_label(b, fmt="%.2f", fontsize=8.5)
ax.axhline(1.0, color="0.4", ls=":", lw=1.2)
ax.set_xticks(xs3); ax.set_xticklabels(["恒载 全段时漂\n(%)", "恒载 慢相段时漂\n(%)", "恒载 阶跃保真\n(1.0=不偏)"])
ax.set_title("(3) 恒载 9 组（时漂越小越好；保真越接近 1 越好）", fontsize=10.5)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

fig2.subplots_adjust(left=0.055, right=0.99, top=0.88, bottom=0.13, wspace=0.24)
p2 = os.path.join(FIG, "L2_v61_metrics.png")
fig2.savefig(p2, dpi=115)
figcheck(fig2, p2)
plt.close(fig2)
print(f"图已保存：{p2}")
