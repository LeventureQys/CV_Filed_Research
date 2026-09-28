# -*- coding: utf-8 -*-
"""v2.0 A1 · 事件日志 + 44 s 窗口逐帧明细 + 原型/现场分岔定位（只读）。"""
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import v20_lib as L          # noqa: E402

RES = os.path.abspath(os.path.join(_HERE, "..", "results"))
EV = os.path.join(RES, "a1_replay_events.csv")
FR = os.path.join(RES, "a1_replay_frames.csv")


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


rows = list(csv.DictReader(open(EV, encoding="utf-8")))
frs = list(csv.DictReader(open(FR, encoding="utf-8")))
tref = fl(frs[0]["t"])
print(f"events total={len(rows)}  frames={len(frs)}  tref={tref:.3f}")

print("\n[39~49 s 内全部事件]")
sel = [r for r in rows if 38.5 <= (fl(r["ts"]) - tref) <= 49.5]
print(f"count={len(sel)}")
for r in sel:
    t = fl(r["ts"]) - tref
    t0 = fl(r["t0"])
    t0s = f"{t0-tref:.2f}" if t0 is not None else "-"
    print(f"  t={t:8.2f} {r['ev']:<12} {r['kind']:<14} t0={t0s:>7} "
          f"A_hat={r['A_hat']:>8} A_before={r['A_sum_before']:>9} A_after={r['A_sum_after']:>9} "
          f"g_bef={r['g_before']:>10} g_aft={r['g_after']:>10} "
          f"{r['state_before']}->{r['state_after']} C={r['C']}")

print("\n[38~50 s 逐帧 0.1 s 采样：原型 vs 现场]")
ds = L.load_dataset(L.DS_ZERO)
pre, main = ds["pre"], ds["main"]
el = pre["el"]
tp = pre["V"].sum(1)
tm = main["V"].sum(1)
proto = np.array([fl(r["out_total"]) for r in frs])
off_f = tm - tp
off_p = proto - tp
d = proto - tm
print(f"{'t':>7}{'pre':>8}{'main':>8}{'proto':>8}{'off_f':>8}{'off_p':>8}{'d':>8}"
      f"{'state':>7}{'kind':>14}{'A_hat':>9}{'g':>9}{'A_sum':>9}{'off_g':>8}{'off_s':>8}")
t = 38.0
while t < 50.0:
    idx = int(np.searchsorted(el, t))
    if idx >= len(frs):
        break
    r = frs[idx]
    print(f"{el[idx]:7.2f}{tp[idx]:8.0f}{tm[idx]:8.0f}{proto[idx]:8.0f}"
          f"{off_f[idx]:8.0f}{off_p[idx]:8.0f}{d[idx]:8.0f}{r['state']:>7}"
          f"{r['kind']:>14}{r['A_hat']:>9}{r['g']:>9}{r['A_sum']:>9}"
          f"{r['off_glide']:>8}{r['off_slow']:>8}")
    t += 0.1

print("\n[|d|>150 ADC 的分岔区段（逐帧，30 帧合并）]")
big = np.where(np.abs(d) > 150)[0]
groups = []
for i in big:
    if groups and i - groups[-1][-1] <= 30:
        groups[-1].append(i)
    else:
        groups.append([i])
for g in groups:
    a, b = g[0], g[-1]
    print(f"  t={el[a]:7.2f}~{el[b]:7.2f} ({el[b]-el[a]:6.1f}s)  d 中位={np.median(d[a:b+1]):8.0f} "
          f"RMS={np.sqrt(np.mean(d[a:b+1]**2)):8.0f}  max|d|={np.max(np.abs(d[a:b+1])):8.0f} "
          f"off_f 中位={np.median(off_f[a:b+1]):8.0f} off_p 中位={np.median(off_p[a:b+1]):8.0f}")

print("\n[41.5~47.5 s 逐 0.1 s：现场 vs 原型 + 内部状态]")
print(f"{'t':>7}{'pre':>8}{'main':>8}{'proto':>8}{'off_f':>8}{'off_p':>8}{'d':>7} "
      f"{'state':<7}{'kind':<14}{'A_hat':>9}{'g':>9}{'A_sum':>8}{'idle':>5}")
t = 41.5
while t <= 47.6:
    i = int(np.searchsorted(el, t))
    if i >= len(frs):
        break
    r = frs[i]
    print(f"{el[i]:7.2f}{tp[i]:8.0f}{tm[i]:8.0f}{proto[i]:8.0f}{off_f[i]:8.0f}"
          f"{off_p[i]:8.0f}{d[i]:7.0f} {r['state']:<7}{r['kind']:<14}"
          f"{r['A_hat']:>9}{r['g']:>9}{float(r['A_sum']):8.0f}{r['idle_now']:>5}")
    t += 0.1

print("\n[全录制 |d| 分位]")
for q in (50, 75, 90, 95, 99, 100):
    print(f"  p{q:<3} |d| = {np.percentile(np.abs(d), q):8.1f} ADC")
print(f"  d>150 帧占比 = {100*np.mean(np.abs(d)>150):.1f}%  "
      f"d<50 帧占比 = {100*np.mean(np.abs(d)<50):.1f}%")

print("\n[首次 |d|>100 的帧]")
j = int(np.argmax(np.abs(d) > 100))
print(f"  i={j} t={el[j]:.3f}s d={d[j]:.1f}  state={frs[j]['state']} kind={frs[j]['kind']}")

print("\n[9~12 s 逐帧：分歧起点]")
print(f"{'t':>7}{'pre':>8}{'main':>8}{'proto':>8}{'off_f':>8}{'off_p':>8}{'d':>8}  "
      f"{'state':<7}{'kind':<14}{'A_hat':>9}{'g':>9}{'A_sum':>9}  {'off_g':>8}{'off_s':>8}")
for i in range(len(el)):
    if 9.0 <= el[i] <= 12.0:
        r = frs[i]
        print(f"{el[i]:7.2f}{tp[i]:8.0f}{tm[i]:8.0f}{proto[i]:8.0f}{off_f[i]:8.0f}"
              f"{off_p[i]:8.0f}{d[i]:8.0f}  {r['state']:<7}{r['kind']:<14}"
              f"{r['A_hat']:>9}{r['g']:>9}{r['A_sum']:>9}  {r['off_glide']:>8}{r['off_slow']:>8}")

print("\n[0~12 s 每 0.2 s 粗采样]")
t = 0.0
print(f"{'t':>7}{'pre':>8}{'main':>8}{'proto':>8}{'off_f':>8}{'off_p':>8}{'d':>8}  "
      f"{'state':<7}{'kind':<14}{'A_hat':>9}{'g':>10}{'A_sum':>9}")
while t < 12.0:
    i = int(np.searchsorted(el, t))
    if i >= len(frs):
        break
    r = frs[i]
    print(f"{el[i]:7.2f}{tp[i]:8.0f}{tm[i]:8.0f}{proto[i]:8.0f}{off_f[i]:8.0f}"
          f"{off_p[i]:8.0f}{d[i]:8.0f}  {r['state']:<7}{r['kind']:<14}"
          f"{r['A_hat']:>9}{r['g']:>10}{r['A_sum']:>9}")
    t += 0.2

print("\n[全部 0~20 s 事件]")
for r in rows:
    ts = fl(r["ts"]) - tref
    if ts <= 20.0:
        t0 = fl(r["t0"])
        t0s = f"{t0-tref:.2f}" if t0 is not None else "-"
        print(f"  t={ts:8.2f} {r['ev']:<12} {r['kind']:<14} t0={t0s:>7} "
              f"A_before={r['A_sum_before']:>9} A_after={r['A_sum_after']:>9} "
              f"g_bef={r['g_before']:>10} g_aft={r['g_after']:>10} "
              f"{r['state_before']}->{r['state_after']} C={r['C']}")
sys.exit(0)
