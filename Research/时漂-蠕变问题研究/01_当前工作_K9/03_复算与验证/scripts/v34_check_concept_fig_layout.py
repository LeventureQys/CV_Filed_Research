# -*- coding: utf-8 -*-
"""概念示意图版面自检：文本互相压盖、文本压住线条/边框、缺字形与被裁切。

用法：python v34_check_concept_fig_layout.py [图名关键词]
不带参数时检查全部 12 张图，只打印有问题的项。

判据（与 vision 复核互补，只抓可程序化的那一类）：
  1. T-T：两个可见文本的渲染包围盒重叠 > 3px；
  2. T-G：某个文本包围盒内被线条/框图边框覆盖的墨迹占比 > 1.5% 且 > 25px
     （渲染一版只含图形、不含文字的掩膜来统计）；
  3. CLIP：文本超出画布；EDGE-INK：成品图四边有着墨（疑似被裁切）；
  4. 缺字形警告（matplotlib 找不到该字形时会打印，这里转成检查项）。
"""

import importlib.util
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "v34_make_concept_figs.py")
FIGURES = ["fig01_problem", "fig02_flow", "fig03_gate", "fig04_model", "fig05_rules",
           "fig08_zero", "fig09_startup", "fig10_timescales", "fig11_slow",
           "fig12_response", "fig13_consistency", "fig14_cost"]

spec = importlib.util.spec_from_file_location("mk", GEN)
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)

orig_save = mk.save
report = {}


def all_texts(fig):
    items = []
    for ai, ax in enumerate(fig.axes):
        if ax.axison:
            if ax.get_title():
                items.append(("title", ai, ax.title))
            if ax.xaxis.label.get_text():
                items.append(("xlabel", ai, ax.xaxis.label))
            if ax.yaxis.label.get_text():
                items.append(("ylabel", ai, ax.yaxis.label))
            for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                items.append(("tick", ai, t))
        for t in ax.texts:
            items.append(("text", ai, t))
    for t in fig.texts:
        items.append(("figtext", -1, t))
    return items


def in_view(ax, axis, loc):
    lo, hi = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    if lo > hi:
        lo, hi = hi, lo
    tol = abs(hi - lo) * 1e-9 + 1e-12
    return (lo - tol) <= loc <= (hi + tol)


def render_ink(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.float32) / 255.0
    return (buf < 0.985).any(axis=2)


def check_save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figbb = fig.bbox

    legend_texts, legend_frames = [], []
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            legend_texts.extend(lg.get_texts())
            legend_frames.append(lg)

    probs, texts = [], []
    for kind, ai, t in all_texts(fig):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        if kind == "tick":
            ax = fig.axes[ai]
            loc = t.get_position()[1] if t in ax.get_yticklabels() else t.get_position()[0]
            axis = "y" if t in ax.get_yticklabels() else "x"
            if not in_view(ax, axis, loc):
                continue
        try:
            if getattr(t, "arrow_patch", None) is not None:
                bb = matplotlib.text.Text.get_window_extent(t, renderer=renderer)
            else:
                bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > figbb.x1 + 1 or bb.y1 > figbb.y1 + 1:
            probs.append("CLIP   %r at %.0f,%.0f" % (s.replace("\n", " / ")[:40], bb.x0, bb.y0))
        texts.append((s.replace("\n", " / "), t, bb))

    for t in legend_texts:
        if t.get_text().strip():
            texts.append((t.get_text(), t, t.get_window_extent(renderer=renderer)))

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][2], texts[j][2]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 3 and oy > 3:
                probs.append("T-T    %.0fx%.0f  %r <> %r"
                             % (ox, oy, texts[i][0][:32], texts[j][0][:32]))

    patch_faces, patch_hide, coll_alpha, coll_hide = [], [], [], []
    for ax in fig.axes:
        for pa in ax.patches:
            if not pa.get_visible():
                continue
            if pa.get_alpha() is not None and pa.get_alpha() < 0.9:
                patch_hide.append(pa)
            else:
                patch_faces.append((pa, pa.get_facecolor()))
        for co in ax.collections:
            if not co.get_visible():
                continue
            if co.get_alpha() is None or co.get_alpha() >= 0.9:
                coll_alpha.append((co, co.get_alpha()))
            else:
                coll_hide.append(co)

    objs, seen = [], set()
    for item in texts + [(None, t, None) for t in legend_texts]:
        t = item[1]
        if id(t) not in seen:
            seen.add(id(t))
            objs.append(t)
    for ax in fig.axes:
        for t in [ax.title, ax.xaxis.label, ax.yaxis.label] + \
                list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if id(t) not in seen:
                seen.add(id(t))
                objs.append(t)
    for t in fig.texts:
        if id(t) not in seen:
            seen.add(id(t))
            objs.append(t)

    state = [(t, t.get_visible()) for t in objs]
    for t in objs:
        t.set_visible(False)
    for lg in legend_frames:
        lg.set_visible(False)
    for pa in patch_hide:
        pa.set_visible(False)
    for co in coll_hide:
        co.set_visible(False)
    for pa, fc in patch_faces:
        pa.set_facecolor("none")
    for co, a in coll_alpha:
        co.set_alpha(0.0)
    ink = render_ink(fig)
    for pa, fc in patch_faces:
        pa.set_facecolor(fc)
    for co, a in coll_alpha:
        co.set_alpha(a)
    for pa in patch_hide:
        pa.set_visible(True)
    for co in coll_hide:
        co.set_visible(True)
    for lg in legend_frames:
        lg.set_visible(True)
    for t, vis in state:
        t.set_visible(vis)

    H = ink.shape[0]
    for s, t, bb in texts:
        r0, r1 = max(0, int(round(H - bb.y1))), min(H, int(round(H - bb.y0)))
        c0, c1 = max(0, int(round(bb.x0))), min(ink.shape[1], int(round(bb.x1)))
        if r1 <= r0 or c1 <= c0:
            continue
        sub = ink[r0:r1, c0:c1]
        if sub.mean() > 0.015 and sub.sum() > 25:
            probs.append("T-G    %.1f%% (%d px) of %r covered by lines"
                         % (sub.mean() * 100, int(sub.sum()), s[:38]))

    report[name] = probs
    path = os.path.join(mk.OUT, name)
    orig_save(fig, name)
    img = mpimg.imread(path)
    edge = (img[:, :, :3] < 0.985).any(axis=2)
    for tag, line in (("top", edge[0, :]), ("bottom", edge[-1, :]),
                      ("left", edge[:, 0]), ("right", edge[:, -1])):
        if int(line.sum()) > 2:
            probs.append("EDGE-INK %s: %d px" % (tag, int(line.sum())))


mk.save = check_save
wanted = sys.argv[1] if len(sys.argv) > 1 else ""
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for fn in FIGURES:
        if wanted and wanted not in fn:
            continue
        getattr(mk, fn)()

glyph = sorted({str(w.message).split("\n")[0][:140] for w in caught
                if "missing from font" in str(w.message) or "Glyph" in str(w.message)})

bad = 0
for name in sorted(report):
    probs = report[name]
    if probs:
        print("ISSUE " + name + "  (%d)" % len(probs))
        for p in probs:
            print("        " + p)
        bad += len(probs)
    else:
        print("OK    " + name)
print("total findings:", bad)
print("missing glyphs:", "none" if not glyph else glyph)
