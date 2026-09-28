# -*- coding: utf-8 -*-
"""v2.0 A1 · Q3 双通路分解 + Q4 轨迹（只读 results/ 与数据集）。

输出口径（显式）：
  off(t)      = main(t) − pre(t)（总量 ADC，正=显示被抬高）
  通路 a(t)   = −Σ share·c_target，仅当 state==event（滑行器/事件扣除）
  通路 b(t)   = −Σ clamp(γA g, ...)，仅当 state==slow（慢相扣除）
  off ≈ a + b 在全部帧成立（由 frames CSV 的 off_glide/off_slow 列给出，本脚本校验残差）
"""
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import v20_lib as L          # noqa: E402

RES = os.path.abspath(os.path.join(_HERE, "..", "results"))
frs = list(csv.DictReader(open(os.path.join(RES, "a1_replay_frames.csv"), encoding="utf-8")))
evs = list(csv.DictReader(open(os.path.join(RES, "a1_replay_events.csv"), encoding="utf-8")))

ds = L.load_dataset(L.DS_ZERO)
pre, main = ds["pre"], ds["main"]
el = pre["el"]
tp, tm = pre["V"].sum(1), main["V"].sum(1)
proto = np.array([float(r["out_total"]) for r in frs])
st = np.array([r["state"] for r in frs])
og = np.array([float(r["off_glide"]) for r in frs])
osl = np.array([float(r["off_slow"]) for r in frs])
Asum = np.array([float(r["A_sum"]) for r in frs])
g = np.array([float(r["g"]) for r in frs])
mnt = np.array([float(r["min_ts"]) for r in frs])
lvr = np.array([float(r["level_ref"]) for r in frs])
idn = np.array([int(r["idle_now"]) for r in frs], bool)
off_f = tm - tp
off_p = proto - tp

print("=== 通路分解残差自检（off 是否 == a+b）===")
print(f"  |off_p − (a+b)| 最大={np.max(np.abs(off_p-(og+osl))):.3f} ADC  "
      f"中位={np.median(np.abs(off_p-(og+osl))):.3f} ADC   (口径自洽)")

segs = L.plateau_segments(tp, el, hyst_frac=0.25, min_dur=0.25)
merged = []
for s in segs:
    if merged and (s[2] - s[1]) < 100:
        merged[-1] = (merged[-1][0], merged[-1][1], s[2])
    else:
        merged.append(s)

print("\n=== Q3 逐平台两条通路（中位；单位 ADC 总量）===")
print(f"{'#':>3}{'kind':>8}{'t0':>8}{'t1':>8}{'off_field':>10}{'off_proto':>10}"
      f"{'a_glide':>9}{'b_slow':>9}{'a+b':>8}{'frac_a':>8}{'frac_b':>8}"
      f"{'A_sum_med':>10}{'g_med':>10}")
rows = []
for i, (k, a, b) in enumerate(merged):
    sl = slice(a, b + 1)
    A = np.median(og[sl])
    B = np.median(osl[sl])
    S = A + B
    rows.append((i, k, el[a], el[b], np.median(off_f[sl]), np.median(off_p[sl]), A, B))
    print(f"{i:>3}{k:>8}{el[a]:8.2f}{el[b]:8.2f}{np.median(off_f[sl]):10.0f}"
          f"{np.median(off_p[sl]):10.0f}{A:9.0f}{B:9.0f}{S:8.0f}"
          f"{(A/S if abs(S)>1e-9 else 0):8.2f}{(B/S if abs(S)>1e-9 else 0):8.2f}"
          f"{np.median(Asum[sl]):10.0f}{np.median(g[sl]):10.5f}")

print("\n=== Q3 全段能量口径：两条通路各自的「扣除·帧」积分 ===")
tot_a = float(np.sum(np.abs(og)))
tot_b = float(np.sum(np.abs(osl)))
print(f"  Σ|a| = {tot_a:.3e} ADC·帧  ({100*tot_a/(tot_a+tot_b):.1f}%)")
print(f"  Σ|b| = {tot_b:.3e} ADC·帧  ({100*tot_b/(tot_a+tot_b):.1f}%)")
print(f"  两种口径的极值：事件期 a 最大 |a|={np.max(np.abs(og)):.0f} ADC；"
      f"慢相期 b 最大 |b|={np.max(np.abs(osl)):.0f} ADC")

print("\n=== Q4 关键转折（每 10 s 一点）===")
print(f"{'t':>7}{'pre':>8}{'off_f':>8}{'off_p':>8}{'min_ts':>9}{'1.5min':>9}"
      f"{'level_ref':>10}{'0.1levref':>10}{'g':>10}{'A_sum':>9}{'idle':>5}{'state':>7}")
t = 0.0
while t < el[-1]:
    i = int(np.searchsorted(el, t))
    if i < len(frs):
        print(f"{el[i]:7.1f}{tp[i]:8.0f}{off_f[i]:8.0f}{off_p[i]:8.0f}{mnt[i]:9.0f}"
              f"{1.5*mnt[i]:9.0f}{lvr[i]:10.0f}{0.1*lvr[i]:10.0f}{g[i]:10.5f}"
              f"{Asum[i]:9.0f}{int(idn[i]):5d}{st[i]:>7}")
    t += 10

print(f"\nmin_ts 全程最小={mnt.min():.0f} 最大={mnt.max():.0f} 末值={mnt[-1]:.0f}")
print(f"1.5·min_ts 全程范围={1.5*mnt.min():.0f}~{1.5*mnt.max():.0f}")
print(f"level_ref 全程 min={lvr.min():.0f} max={lvr.max():.0f}；0.1·level_ref 最大={0.1*lvr.max():.0f}")
print(f"g: min={g.min():.5f} max={g.max():.5f}；g<0 占比={100*np.mean(g<0):.1f}% "
      f"（其中 slow 态 g<0 占比={100*np.mean(g[st=='slow']<0):.1f}%）")
print(f"A_sum: min={Asum.min():.0f} max={Asum.max():.0f} 末值={Asum[-1]:.0f}；"
      f"A_sum>0 帧占比={100*np.mean(Asum>1e-9):.1f}%")
print(f"idle_now 真帧={int(idn.sum())}（{100*idn.mean():.2f}%）")
for s in ("idle", "event", "slow"):
    m = st == s
    print(f"  状态 {s:<6} 帧={int(m.sum()):6d}  idle_now 真={int(idn[m].sum()):6d}"
          f"（{100*idn[m].mean():5.1f}%）  该态 off_p 中位={np.median(off_p[m]):8.0f}")

print("\n=== Q4 最终状态锁定检查：末 30 s 的 off / A_sum / g ===")
m = el > el[-1] - 30
print(f"  off_field 中位={np.median(off_f[m]):.0f} 范围={off_f[m].min():.0f}~{off_f[m].max():.0f}")
print(f"  off_proto 中位={np.median(off_p[m]):.0f} 范围={off_p[m].min():.0f}~{off_p[m].max():.0f}")
print(f"  A_sum 中位={np.median(Asum[m]):.0f}  g 中位={np.median(g[m]):.5f}")
print(f"  末 30 s 真实空载段占比={100*np.mean(tp[m]<4000):.1f}%")

print("\n=== 「越飘越远」判据：各平台 idle 段偏移是否复位 ===")
print(f"{'#':>3}{'kind':>8}{'t0':>8}{'t1':>8}{'pre_med':>9}{'off_med':>9}{'off_end':>9}")
for i, (k, a, b) in enumerate(merged):
    sl = slice(a, b + 1)
    print(f"{i:>3}{k:>8}{el[a]:8.2f}{el[b]:8.2f}{np.median(tp[sl]):9.0f}"
          f"{np.median(off_f[sl]):9.0f}{off_f[b]:9.0f}")
sys.exit(0)
