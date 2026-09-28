# -*- coding: utf-8 -*-
"""中途切换-13ffca：v6 与 v6.1 的**全程**对比（时序 + 偏差 + 逐段台账）。

产出：
  figures/M1_v61_13ffca.png   4 格：全程时序 / 全程偏差 / 最大差异处放大 / 卸载沿放大
  results/v61_13ffca_curves.csv   逐帧（降采样）raw / v6 / v6.1
  results/v61_13ffca_events.csv   逐事件台账（超调/残留/回落/该段显示稳态）
  results/_v61_13ffca.log
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
COL = {"raw": "0.60", "v6": "#2ca02c", "v61": "#d62728"}
TAG = "中途切换-13ffca"
IDX = [t for t, _ in C.ALL].index(TAG)
path = C.ALL[IDX][1]
d = L.prep(path)
tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
raw_i = Xu.sum(axis=1)
rs = L.med_smooth(raw_i, 0.5 / dtm)
span = float(tu[-1])

ARMS = [("v6", GLM53v6, {}), ("v6.1", GLM53v61, {})]
res = {}
for name, cls, kw in ARMS:
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for j in range(len(tu)):
        Y[j] = c.process(tu[j], Xu[j])
    res[name] = dict(c=c, Y=Y, ys=L.med_smooth(Y.sum(axis=1), 0.5 / dtm))
    print(f"{name:<5} epoch={len(c.epoch_t):3d}  A_peak={c.A_peak:9,.0f}  "
          f"显示峰值={Y.sum(axis=1).max():9,.0f}  原始峰值={raw_i.max():9,.0f}")

err = {n: res[n]["ys"] - rs for n in res}
print(f"\n录制：{TAG}  {span:.1f}s  {Xu.shape[1]}ch  采样 {1/dtm:.1f}Hz")
print(f"全程 max|显示−原始|：v6 {np.abs(err['v6']).max():,.0f} @t={tu[int(np.argmax(np.abs(err['v6'])))]:.2f}s"
      f"   v6.1 {np.abs(err['v6.1']).max():,.0f} @t={tu[int(np.argmax(np.abs(err['v6.1'])))]:.2f}s")
dd = res["v6"]["ys"] - res["v6.1"]["ys"]
k = int(np.argmax(np.abs(dd)))
print(f"两臂差异最大处：t={tu[k]:.2f}s  v6={res['v6']['ys'][k]:,.0f}  v6.1={res['v6.1']['ys'][k]:,.0f}"
      f"（差 {dd[k]:+,.0f}）")
print(f"两臂逐帧差：max {np.abs(dd).max():,.0f}  中位 {np.median(np.abs(dd)):,.0f}  "
      f"RMS {np.sqrt((dd**2).mean()):,.0f}")

# ── 逐事件台账 ──
rows = []
peak_all = float(rs.max())
for t0, kd in res["v6"]["c"].kind_log:
    i0 = int(np.searchsorted(tu, t0))
    a, b = min(len(tu) - 1, i0 + int(4.6 / dtm)), min(len(tu), i0 + int(5.4 / dtm))
    if b - a < 3:
        continue
    pre = float(np.median(rs[max(0, i0 - int(0.3 / dtm)):max(1, i0)]))
    P = float(np.median(rs[a:b]))
    step = P - pre
    row = dict(t0=round(t0, 2), ev=kd, step=round(step))
    for n in res:
        ys = res[n]["ys"]
        j5 = min(len(tu) - 1, i0 + int(5.0 / dtm))
        seg = ys[i0:j5 + 1]
        kk = int(np.argmax(seg))
        e = min(len(tu), i0 + kk + int(6.0 / dtm))
        win = raw_i[i0 + kk:e]
        if len(win):
            cut = np.where(win < 0.85 * win.max())[0]
            if len(cut):
                e = max(i0 + kk + int(cut[0]), i0 + kk + 1)
        i20 = min(len(tu) - 1, i0 + int(20.0 / dtm))
        row[f"{n}_超调%"] = round(100 * (seg[kk] - P) / step, 1) if step > 1 else np.nan
        row[f"{n}_5s残留%"] = round(100 * (ys[j5] - P) / step, 1) if step > 1 else np.nan
        row[f"{n}_回落%"] = round(100 * (seg[kk] - min(ys[i0 + kk:e])) / step, 1) if step > 1 else np.nan
        row[f"{n}_20s电平"] = round(float(ys[i20]), 0)
    row["段内原始20s"] = round(float(rs[min(len(tu) - 1, i0 + int(20.0 / dtm))]), 0)
    rows.append(row)
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(RES, "v61_13ffca_events.csv"), index=False, encoding="utf-8-sig")
print("\n逐事件台账（13ffca 全部事件）：")
print(ev.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

# ── 分段偏差统计（按 10 s 窗，看两臂在哪些段差异大）──
nw = int(10.0 / dtm)
print("\n每 10 s 的 |显示−原始| 均值（只列两臂差异 > 150 ADC 的窗）：")
print(f"{'t窗(s)':>14} {'原始均值':>10} {'v6偏差':>10} {'v6.1偏差':>10} {'两臂差':>10}")
for s in range(0, len(tu) - nw, nw):
    a6 = float(np.mean(np.abs(err["v6"][s:s + nw])))
    a1 = float(np.mean(np.abs(err["v6.1"][s:s + nw])))
    if abs(a6 - a1) > 150:
        print(f"{tu[s]:6.0f}~{tu[s+nw]:6.0f} {np.mean(rs[s:s+nw]):10,.0f} "
              f"{a6:10,.0f} {a1:10,.0f} {a6-a1:+10,.0f}")

cur = pd.DataFrame(dict(t=tu[::4], raw=rs[::4], v6=res["v6"]["ys"][::4], v61=res["v6.1"]["ys"][::4],
                        e6=err["v6"][::4], e61=err["v6.1"][::4]))
cur.to_csv(os.path.join(RES, "v61_13ffca_curves.csv"), index=False, encoding="utf-8-sig")

# ── 图 ──
tm = tu[k]
fig, axes = plt.subplots(5, 1, figsize=(16.5, 15.2))
fig.suptitle(f"{TAG} · v6 与 v6.1 全程对比（灰=原始，绿=v6，红=v6.1）", fontsize=14)

ax = axes[0]
ax.plot(tu, rs, color=COL["raw"], lw=1.5, label="原始")
ax.plot(tu, res["v6"]["ys"], color=COL["v6"], lw=1.4, label="v6")
ax.plot(tu, res["v6.1"]["ys"], color=COL["v61"], lw=1.4, label="v6.1")
ax.set_title(f"(a) 全程 {span:.0f}s · 21ch · {1/dtm:.1f}Hz", fontsize=10.5)
ax.set_ylabel("总量 (ADC)", fontsize=9)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
ax.tick_params(labelsize=8.5)

ax = axes[1]
ax.axhline(0, color="0.35", lw=1.0)
ax.plot(tu, err["v6"], color=COL["v6"], lw=1.4, label="v6 − 原始")
ax.plot(tu, err["v6.1"], color=COL["v61"], lw=1.4, label="v6.1 − 原始")
ax.set_title("(b) 全程偏差：显示 − 原始（0 以上=显示偏高；恒载段为负=蠕变被扣掉，属正常）", fontsize=10.5)
ax.set_ylabel("偏差 (ADC)", fontsize=9)
ax.legend(fontsize=9, loc="lower left")
ax.grid(alpha=0.25)
ax.tick_params(labelsize=8.5)

for ax, (ta, tb, title) in zip(axes[2:], [
        (84.0, 95.0, "(c) onset @87.81s 放大：v6 冲到 29,810（+6.9%），v6.1 停在 28,127（+0.9%）"),
        (96.0, 106.0, "(d) 小台阶 @98.88s 放大：显示冲击（v6 +284.7% / 回落 11.2% → v6.1 +165.4% / 回落 2.1%）"),
        (58.0, 72.0, "(e) restep @61.02s + 卸载 @68.9s 放大：两臂差 ~500 ADC 的静态电平差")]):
    m = (tu >= ta) & (tu <= tb)
    ax.plot(tu[m], rs[m], color=COL["raw"], lw=1.5, label="原始")
    ax.plot(tu[m], res["v6"]["ys"][m], color=COL["v6"], lw=1.4, label="v6")
    ax.plot(tu[m], res["v6.1"]["ys"][m], color=COL["v61"], lw=1.4, label="v6.1")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("总量 (ADC)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8.5)
fig.subplots_adjust(left=0.055, right=0.99, top=0.945, bottom=0.038, hspace=0.46)
p = os.path.join(FIG, "M1_v61_13ffca.png")
fig.savefig(p, dpi=112)


def _inter(b1, b2):
    return (max(0.0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
            * max(0.0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0)))


fig.canvas.draw()
r = fig.canvas.get_renderer()
fb = fig.bbox
oob = ov = 0
for i, a in enumerate(fig.axes):
    o = 0
    for art in list(a.lines) + list(a.texts):
        try:
            bb = art.get_window_extent(r)
        except Exception:
            continue
        if bb.width * bb.height > 0 and _inter(bb, fb) / (bb.width * bb.height) < 0.85:
            o += 1
    items = [(t.get_text().strip()[:22], t.get_window_extent(r)) for t in a.texts
             if t.get_text().strip()]
    items.append(("<标题>", a.title.get_window_extent(r)))
    v = 0
    for j in range(len(items)):
        for kk in range(j + 1, len(items)):
            b1, b2 = items[j][1], items[k][1]
            if _inter(b1, b2) > 0.12 * min(b1.width * b1.height, b2.width * b2.height):
                v += 1
    oob += o
    ov += v
    print(f"[figcheck] ax{i} lines={len(a.lines)} 越界={o} 重叠={v}")
print(f"[figcheck] 越界合计 {oob} 重叠合计 {ov}")
plt.close(fig)
print(f"\n图已保存：{p}")
