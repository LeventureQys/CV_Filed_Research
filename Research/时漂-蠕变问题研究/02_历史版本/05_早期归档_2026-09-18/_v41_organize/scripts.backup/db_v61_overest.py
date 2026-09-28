# -*- coding: utf-8 -*-
"""F1 的直接依据：以 **v6 自己的事件台账** 为准，量逆模型的"高估倍数" Â/A_true。

对 v6 检出的每个事件，取 τ=1.0 s 时 `ev["hist"]` 里的 (τ, inc)，
用 v6 的 ROM + 电平域最小二乘（窗口 [0.2, min(τ,0.8)]，与 `glm53_v6._inv_est` 同式）算 Â，
与真值 A_true = raw(t0+4.6~5.4 s 中位) − base 相比。

产物：results/v61_overest_v6.csv、results/_v61_overest_v6.log
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import GLM53v6, g_shape                        # noqa: E402


SHAPE_TAUS = (0.20, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00)


class Trace(GLM53v6):
    def __init__(self, n):
        super().__init__(n)
        self.snap = []
        self._shape = {}

    def process(self, ts, v):
        out = super().process(ts, v)
        ev = self.ev
        if ev is None:
            if self._shape:
                self._shape = {}
            return out
        t0 = ev["t0"]
        tau = ts - t0
        for t in SHAPE_TAUS:
            if t not in self._shape and tau >= t:
                self._shape[t] = float(np.sum(v) - ev["base"])
        if not ev.get("_snap") and tau >= 1.0:
            ev["_snap"] = True
            self.snap.append(dict(t0=float(t0), kind=ev["kind"], base=float(ev["base"]),
                                  hist=[(float(a), float(b)) for a, b in ev["hist"]],
                                  shape=dict(self._shape)))
            self._shape = {}
        return out


def ls_A(hist, td, scale=1.0):
    tt = np.array([h[0] for h in hist])
    yy = np.array([h[1] for h in hist])
    m = (tt >= 0.20) & (tt <= min(td, 0.80))
    if m.sum() < 5:
        return np.nan
    g = np.minimum(1.0, scale * g_shape(tt[m]))
    return float((yy[m] * g).sum() / (g * g).sum())


rows = []
for tag, path in C.ALL:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    raw = Xu.sum(axis=1)
    c = Trace(Xu.shape[1])
    for j in range(len(tu)):
        c.process(tu[j], Xu[j])
    for s in c.snap:
        i0 = int(np.searchsorted(tu, s["t0"]))
        a = min(len(tu) - 1, i0 + int(4.6 / dtm))
        b = min(len(tu), i0 + int(5.4 / dtm))
        if b - a < 3:
            continue
        A_true = float(np.median(raw[a:b])) - s["base"]
        if A_true <= 1e-9:
            continue
        A1 = ls_A(s["hist"], 1.0)
        if not np.isfinite(A1):
            continue
        row = dict(dataset=tag, kind=C.KIND[tag], t0=s["t0"], ev=s["kind"],
                   A1=A1, A_true=A_true, ratio=A1 / A_true, lam_needed=A1 / A_true)
        shp = s.get("shape", {})
        for t in (0.20, 0.30, 0.50, 0.80, 0.00):
            if t and t in shp:
                row[f"f{t:.2f}"] = shp[t] / A_true
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "v61_overest_v6.csv"), index=False, encoding="utf-8-sig")

print("=" * 112)
print(f"逆模型高估倍数（v6 自己的事件台账，τ=1.0 s 时的 Â 与真值比） n={len(df)}")
print("=" * 112)
print(df.groupby(["kind", "ev"]).agg(n=("ratio", "size"), 中位=("ratio", "median"),
                                     p90=("ratio", lambda s: s.quantile(0.9)),
                                     最大=("ratio", "max"), 最小=("ratio", "min")).round(3).to_string())
print("\n逐事件（按高估倍数降序，前 20）：")
top = df.sort_values("ratio", ascending=False).head(20)
for _, r in top.iterrows():
    print(f"  {r.dataset:>18} [{r.kind}] t0={r.t0:8.2f} {r.ev:<14s} "
          f"Â={r.A1:10,.1f} 真值={r.A_true:10,.1f}  ×{r.ratio:.3f}")

on = df[df.ev == "onset"]
print(f"\nonset 事件（n={len(on)}）：中位 ×{on.ratio.median():.3f}  "
      f"p90 ×{on.ratio.quantile(0.9):.3f}  max ×{on.ratio.max():.3f}")
print("  ⇒ 若要\"任何 onset 都不高估\"：ROM_SCALE ≥ %.3f（含 2%% 余量 %.3f）"
      % (on.ratio.max(), on.ratio.max() * 1.02))
for s in (1.00, 1.04, 1.06, 1.08, 1.10, 1.12):
    e = 100 * (on.ratio / s - 1)
    print(f"  ROM_SCALE={s:.2f}：onset 偏差 中位 {e.median():+5.1f}%  "
          f"p90 {e.quantile(0.9):+5.1f}%  max {e.max():+5.1f}%  min {e.min():+5.1f}%")

print("\n" + "=" * 112)
print("onset 实测形状 g(τ)=inc(τ)/A_true（以 v6 事件原点为准）")
print("=" * 112)
cols = [c for c in ("f0.20", "f0.30", "f0.50", "f0.80") if c in on.columns]
tab = []
for c in cols:
    t = float(c[1:])
    x = on[c].dropna()
    tab.append(dict(tau=t, v6_ROM=float(g_shape(t)), 中位=float(x.median()),
                    p90=float(x.quantile(0.9)), 最大=float(x.max()), 最小=float(x.min()), n=len(x)))
tb = pd.DataFrame(tab)
print(tb.round(3).to_string(index=False))
print("\nv6 ROM 在 0.20~0.80 s 段比实测中位低 "
      f"{100*(tb['中位']/tb['v6_ROM'] - 1).mean():+.1f}%（逐点均值）"
      f" ⇒ Â 高估同量级（LS 加权后实测结果见上表）")
