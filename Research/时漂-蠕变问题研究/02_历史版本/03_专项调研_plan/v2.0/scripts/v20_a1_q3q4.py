# -*- coding: utf-8 -*-
"""v2.0 A1 · Q3 修正版通路分解（按 state 互斥归属）+ 44 s 事件表（只读）。

修正说明：`off_glide`（滑行器/事件扣除 = −Σshare·c_target）在**慢相态也会被
非零记录**（它是 `_share_vector` 的旁路量，不区分 state），因此「两条通路」必须
按 `state` 互斥归属：state==event → 通路(a)；state==slow → 通路(b)。
本脚本按此口径重算，并给出 off ≈ a+b 的逐帧残差自检。
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
off_f = tm - tp
off_p = proto - tp

ev_m = st == "event"
sl_m = st == "slow"
# 符号约定：frames CSV 的 off_glide/off_slow 是「扣除量」（off = −Σ扣除），
# 本脚本统一转到「显示偏移」口径：a = −off_glide, b = −off_slow ⇒ off_p = a + b。
a_all = -og
b_all = -osl
a = np.where(ev_m, a_all, 0.0)
b = np.where(sl_m, b_all, 0.0)
resid = off_p - (a + b)
print(f"[自检1] 逐帧 off_p == a_all+b_all ？"
      f"|残差| 中位={np.median(np.abs(a_all+b_all-off_p)):.4f} "
      f"最大={np.max(np.abs(a_all+b_all-off_p)):.4f} ADC")
print(f"[自检2] 按 state 互斥归属后 off_p ≈ a+b ？"
      f"|残差| 中位={np.median(np.abs(resid)):.4f} 最大={np.max(np.abs(resid)):.4f} ADC")
n_off = int(np.sum(np.abs(resid) > 1.0))
print(f"        残差 >1 ADC 的帧数={n_off}（占 {100*n_off/len(frs):.3f}%，"
      f"仅状态切换的单帧内两条通路同时贡献）")

segs = L.plateau_segments(tp, el, hyst_frac=0.25, min_dur=0.25)
merged = []
for s in segs:
    if merged and (s[2] - s[1]) < 100:
        merged[-1] = (merged[-1][0], merged[-1][1], s[2])
    else:
        merged.append(s)

print("\n=== Q3 逐平台两条通路（中位，ADC 总量）===")
print(f"{'#':>3}{'kind':>8}{'t0':>8}{'t1':>8}{'off_field':>10}{'off_proto':>10}"
      f"{'a_glide':>9}{'b_slow':>9}{'a+b':>8}{'a_share':>8}{'b_share':>8}"
      f"{'evt_s':>7}{'slow_s':>7}{'idle_s':>7}")
for i, (k, A_, B_) in enumerate(merged):
    sl = slice(A_, B_ + 1)
    av = np.median(a[sl]) if np.any(ev_m[sl]) else 0.0
    bv = np.median(b[sl]) if np.any(sl_m[sl]) else 0.0
    S = av + bv
    ne = int(ev_m[sl].sum())
    ns = int(sl_m[sl].sum())
    ni = int((~ev_m[sl] & ~sl_m[sl]).sum())
    print(f"{i:>3}{k:>8}{el[A_]:8.2f}{el[B_]:8.2f}{np.median(off_f[sl]):10.0f}"
          f"{np.median(off_p[sl]):10.0f}{av:9.0f}{bv:9.0f}{S:8.0f}"
          f"{(av/S if abs(S)>1e-9 else 0):8.2f}{(bv/S if abs(S)>1e-9 else 0):8.2f}"
          f"{ne/100:7.2f}{ns/100:7.2f}{ni/100:7.2f}")

print("\n=== 全段时权口径 ===")
print(f"  事件态帧={int(ev_m.sum())}（{ev_m.mean()*100:.1f}%）  Σ|a|={np.sum(np.abs(a)):.3e} ADC·帧"
      f"  Σ|a|·dt={np.sum(np.abs(a))/100/1000:.1f} kADC·s")
print(f"  慢相态帧={int(sl_m.sum())}（{sl_m.mean()*100:.1f}%）  Σ|b|={np.sum(np.abs(b)):.3e} ADC·帧"
      f"  Σ|b|·dt={np.sum(np.abs(b))/100/1000:.1f} kADC·s")
print(f"  占比：通路a={100*np.sum(np.abs(a))/(np.sum(np.abs(a))+np.sum(np.abs(b))):.1f}% "
      f"通路b={100*np.sum(np.abs(b))/(np.sum(np.abs(a))+np.sum(np.abs(b))):.1f}%")
print(f"  事件态 |a| 中位={np.median(np.abs(a[ev_m])):.0f}  慢相态 |b| 中位={np.median(np.abs(b[sl_m])):.0f} ADC")

print("\n=== 恒载平台上的偏移归属（判断「谁在主导漂移」）===")
for i, (k, A_, B_) in enumerate(merged):
    if k != "loaded" or (B_ - A_) < 500:
        continue
    sl = slice(A_, B_ + 1)
    print(f"  plat{i:<3} t={el[A_]:6.1f}~{el[B_]:6.1f} ({el[B_]-el[A_]:5.1f}s) "
          f"off_field 中位={np.median(off_f[sl]):6.0f} off_proto 中位={np.median(off_p[sl]):6.0f} "
          f"事件态帧占比={100*ev_m[sl].mean():5.1f}% 慢相态帧占比={100*sl_m[sl].mean():5.1f}% "
          f"事件态 off_p 中位={np.median(off_p[sl][ev_m[sl]]) if ev_m[sl].any() else 0:6.0f} "
          f"慢相态 off_p 中位={np.median(off_p[sl][sl_m[sl]]) if sl_m[sl].any() else 0:6.0f}")

print("\n=== 44 s 窗口（41.5~47.5 s）逐 0.05 s 关键量 ===")
print(f"{'t':>7}{'pre':>8}{'main':>8}{'proto':>8}{'off_f':>8}{'off_p':>8}{'state':>7}"
      f"{'kind':>14}{'A_hat':>9}{'g':>10}{'A_sum':>9}{'a':>8}{'b':>8}{'idle':>5}")
t = 41.5
while t <= 47.5:
    i = int(np.searchsorted(el, t))
    if i >= len(frs):
        break
    r = frs[i]
    print(f"{el[i]:7.2f}{tp[i]:8.0f}{tm[i]:8.0f}{proto[i]:8.0f}{off_f[i]:8.0f}{off_p[i]:8.0f}"
          f"{r['state']:>7}{r['kind']:>14}{r['A_hat']:>9}{g[i]:10.5f}{Asum[i]:9.0f}"
          f"{a[i]:8.0f}{b[i]:8.0f}{int(r['idle_now']):5d}")
    t += 0.05

print("\n=== 该窗口内 A_sum 的阶跃（|ΔA_sum|>200 的帧）===")
prev = Asum[0]
for i in range(1, len(frs)):
    if 41.0 <= el[i] <= 48.0 and abs(Asum[i] - prev) > 200:
        print(f"  t={el[i]:6.2f}  A_sum {prev:8.0f} → {Asum[i]:8.0f}  (Δ={Asum[i]-prev:+8.0f}) "
              f"g={g[i]:.5f} state={st[i]}")
    prev = Asum[i]
sys.exit(0)
