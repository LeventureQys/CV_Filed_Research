# -*- coding: utf-8 -*-
"""v6.1 的 F1 依据：13 份 onset 事件的**实测形状包络**与**逆模型高估倍数**。

输出两张表（Document/08-v6.1算法说明.md §3 引用）：
  ① 逐 τ 的实测形状包络 g(τ)=[Z(τ)−Z(0)]/[Z(5s)−Z(0)]：中位 / p75 / p90 / max，
     与 v6 ROM 逐点对照 —— 用来看 v6 ROM 落在包络的哪个位置；
  ② 每个 onset 事件的"逆模型高估倍数" Â₁/A_true（用 v6 ROM 与电平域最小二乘算），
     给出分布，并算出"要让最大高估 ≤ 0"所需的 ROM_SCALE。

事件原点口径：raw 总量 3 帧中值 → 前置电平 + 3% 跳变的首帧前一帧 = t0（与 v6 的 `_backdate` 同规则）。
产物：results/v61_shape_cal.csv、results/v61_overest.csv、results/_v61_shape_cal.log
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import ROM_TAU, ROM_G, g_shape                 # noqa: E402

TAUS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]


def backdate(tu, tot, hit_i):
    """与 v6 `_backdate` 同规则：命中前 0.6 s 内找越过 前置+3%跳变 的首帧，取其前一帧。"""
    dtm = float(tu[1] - tu[0])
    i = int(hit_i)
    a = max(0, i - int(0.6 / dtm))
    seg = tot[a:i + 1]
    if len(seg) < 8:
        return None
    q = max(3, len(seg) // 4)
    pre = float(np.median(seg[:q]))
    jump = float(seg[-1] - pre)
    if jump <= 0:
        return None
    tgt = pre + 0.03 * jump
    w = np.where(seg >= tgt)[0]
    if not len(w):
        return None
    idx = max(0, int(w[0]) - 1)
    return a + idx, pre


rows, over = [], []
for tag, path in C.ALL:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    kind = C.KIND[tag]
    evs = L.detect_events(tot, dtm)
    for hit, dl in evs:
        if dl <= 0:
            continue
        bd = backdate(tu, tot, hit)
        if bd is None:
            continue
        i0, pre = bd
        j5 = i0 + int(5.0 / dtm)
        if j5 >= len(tu) - 5:
            continue
        A_true = float(np.median(tot[j5 - int(0.4 / dtm):j5 + int(0.4 / dtm)])) - pre
        if A_true <= 0:
            continue
        # 事件后 5 s 内不允许有其它加载（否则该"阶跃"不是单阶跃）
        if any(hit < h2 <= j5 + int(1.0 / dtm) and d2 > 0 for h2, d2 in evs):
            continue
        shp = {}
        for t in TAUS:
            j = i0 + int(round(t / dtm))
            if j >= len(tu):
                break
            shp[t] = (float(np.mean(tot[max(0, j - 2):j + 3])) - pre) / A_true
        rows.append(dict(dataset=tag, kind=kind, t0=float(tu[i0]), A_true=A_true,
                         **{f"f{t:.2f}": v for t, v in shp.items()}))
        # v6 逆模型（电平域 LS，窗 [0.2, 0.8]）在各决策时刻的高估倍数
        tt = np.array([(j - i0) * dtm for j in range(i0, i0 + int(1.0 / dtm))])
        yy = np.array([tot[j] - pre for j in range(i0, i0 + int(1.0 / dtm))])
        for td in (0.25, 0.30, 0.50):
            m = (tt >= 0.20) & (tt <= min(td, 0.80))
            if m.sum() < 4:
                continue
            g = g_shape(tt[m])
            A1 = float((yy[m] * g).sum() / (g * g).sum())
            over.append(dict(dataset=tag, kind=kind, t0=float(tu[i0]), td=td,
                             A1=A1, A_true=A_true, ratio=A1 / A_true))

sh = pd.DataFrame(rows)
ov = pd.DataFrame(over)
sh.to_csv(os.path.join(RES, "v61_shape_cal.csv"), index=False, encoding="utf-8-sig")
ov.to_csv(os.path.join(RES, "v61_overest.csv"), index=False, encoding="utf-8-sig")

print("=" * 118)
print(f"① 实测形状包络（{len(sh)} 个单阶跃 onset 事件：恒载 {int((sh.kind=='恒载').sum())} / "
      f"实采 {int((sh.kind=='实采').sum())}）")
print("=" * 118)
tab = []
for t in TAUS:
    ck = f"f{t:.2f}"
    if ck not in sh.columns:
        continue
    col = sh[ck].dropna()
    tab.append(dict(tau=t, v6_ROM=float(g_shape(t)), 中位=float(col.median()),
                    p75=float(col.quantile(0.75)), p90=float(col.quantile(0.90)),
                    最大=float(col.max()), 最小=float(col.min()), n=len(col)))
tb = pd.DataFrame(tab)
print(tb.round(3).to_string(index=False))
print("\n分族中位：")
tab2 = []
for t in TAUS:
    ck = f"f{t:.2f}"
    if ck not in sh.columns:
        continue
    h = sh[sh.kind == "恒载"][ck].dropna()
    v = sh[sh.kind == "实采"][ck].dropna()
    tab2.append(dict(tau=t, v6_ROM=float(g_shape(t)), 恒载中位=float(h.median()) if len(h) else np.nan,
                     实采中位=float(v.median()) if len(v) else np.nan,
                     恒载max=float(h.max()) if len(h) else np.nan,
                     实采max=float(v.max()) if len(v) else np.nan))
print(pd.DataFrame(tab2).round(3).to_string(index=False))

print("\n" + "=" * 118)
print("② 逆模型高估倍数 Â₁/A_true（v6 ROM + 电平域 LS）")
print("=" * 118)
for td in (0.25, 0.30, 0.50):
    s = ov[ov.td == td]
    if not len(s):
        continue
    print(f"\nτ_d={td:.2f}s  n={len(s)}")
    print(f"  全样本：中位 {s.ratio.median():.3f}  p75 {s.ratio.quantile(0.75):.3f}  "
          f"p90 {s.ratio.quantile(0.90):.3f}  max {s.ratio.max():.3f}  min {s.ratio.min():.3f}")
    for k in ("恒载", "实采"):
        sk = s[s.kind == k]
        if len(sk):
            print(f"  {k}（n={len(sk)}）：中位 {sk.ratio.median():.3f}  "
                  f"p90 {sk.ratio.quantile(0.90):.3f}  max {sk.ratio.max():.3f}  "
                  f"min {sk.ratio.min():.3f}")
    top = s.reindex(s.ratio.sort_values(ascending=False).index).head(6)
    print("  最大的 6 个：")
    for _, r in top.iterrows():
        print(f"    {r.dataset:>18} t0={r.t0:8.2f}  Â₁={r.A1:10,.1f} 真值={r.A_true:10,.1f}  ×{r.ratio:.3f}")

print("\n④ 单阶跃 onset 事件台账（用于核对事件筛选是否漏掉大事件）")
led = sh[["dataset", "kind", "t0", "A_true"]].copy()
ov25 = ov[ov.td == 0.25][["dataset", "t0", "A1", "ratio"]]
led = led.merge(ov25, on=["dataset", "t0"], how="left")
led["ratio"] = led["ratio"].round(3)
print(led.sort_values(["kind", "dataset", "t0"]).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
print(f"\n恒载事件数 {int((led.kind=='恒载').sum())}（应为 9，缺的是'录制开始时已加载/5 s 内有二次加载'的）")

print("\n③ 让最大高估不超过 0 所需的 ROM_SCALE（= max ratio，含 2% 余量）")
s = ov[ov.td >= 0.25]
if len(s):
    print(f"  全样本（τ_d≥0.25 s）max ratio = {s.ratio.max():.3f} ⇒ ROM_SCALE ≥ {s.ratio.max():.3f}"
          f"（含 2% 余量 {s.ratio.max()*1.02:.3f}）")
    print(f"  中位比 {s.ratio.median():.3f} ⇒ 该取值对中位事件的偏差 "
          f"{100*(s.ratio.median()/s.ratio.max()-1):+.1f}%")
    p90 = s.ratio.quantile(0.90)
    print(f"  p90 比 {p90:.3f} ⇒ ROM_SCALE = p90 时：中位偏差 "
          f"{100*(s.ratio.median()/p90-1):+.1f}%、最大超调 {100*(s.ratio.max()/p90-1):+.1f}%")
