# -*- coding: utf-8 -*-
"""分析 fall_events.json：大回落事件（base fall>1000）的归因与各变体表现。"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "..", "results", "fall_events.json"), encoding="utf-8"))

by = {}
for r in rows:
    by.setdefault(r["variant"], {})[(r["cond"], round(r["t"], 2))] = r

base = by["base"]
big = sorted([k for k, r in base.items() if r["fall"] > 1000.0],
             key=lambda k: -base[k]["fall"])
print(f"base 大回落事件（fall>1000）共 {len(big)} 个 / {len(base)}")
print(f"{'工况/时刻':58s} {'base':>8s} {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s} {'E':>8s} {'输入':>8s} | {'Δx1':>7s} {'Δx2':>7s} (base峰后)")
for k in big[:20]:
    r = base[k]
    vals = []
    for v in ("base", "A_tsl1", "B_soft10", "C_gate05", "D_M2+B", "E_combo", "INPUT"):
        rr = by.get(v, {}).get(k)
        vals.append(f"{rr['fall']:8.0f}" if rr else "       -")
    print(f"{k[0][-40:] + '@' + str(k[1]):58s} {' '.join(vals)} | {r['dx1']:7.0f} {r['dx2']:7.0f}")

print("\n[大回落事件上各变体 fall 中位/均值]")
for v in ("base", "A_tsl1", "B_soft10", "C_gate05", "D_M2+B", "E_combo", "INPUT"):
    sub = [by[v][k]["fall"] for k in big if k in by.get(v, {})]
    if sub:
        print(f"{v:10s} n={len(sub):2d} median={np.median(sub):8.0f} mean={np.mean(sub):8.0f}")

print("\n[大回落事件 t_peak 分布]")
for v in ("base", "E_combo", "INPUT"):
    sub = [by[v][k]["t_peak"] for k in big if k in by.get(v, {})]
    if sub:
        print(f"{v:10s} median={np.median(sub):5.2f}s  p10={np.percentile(sub,10):5.2f}  p90={np.percentile(sub,90):5.2f}")

print("\n[base 大回落事件的峰后分解（Δx1/Δx2/残差=fall-Δin+Δx1+Δx2）]")
for k in big[:12]:
    r = base[k]
    ri = by.get("INPUT", {}).get(k)
    if ri:
        res = r["fall"] - ri["fall"] + r["dx1"] + r["dx2"]
        print(f"{k[0][-36:]+'@'+str(k[1]):52s} fall={r['fall']:7.0f} in_fall={ri['fall']:7.0f} "
              f"dx1={r['dx1']:6.0f} dx2={r['dx2']:6.0f} 其余={res:6.0f}")
