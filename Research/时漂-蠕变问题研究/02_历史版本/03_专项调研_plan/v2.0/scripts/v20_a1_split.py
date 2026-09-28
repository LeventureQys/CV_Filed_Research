# -*- coding: utf-8 -*-
"""v2.0 A1 · 分歧点定位：10.3~11.2 s 逐帧 + 0~22 s 事件序列（只读）。"""
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
tref = float(frs[0]["t"])

ds = L.load_dataset(L.DS_ZERO)
pre, main = ds["pre"], ds["main"]
el = pre["el"]
tp = pre["V"].sum(1)
tm = main["V"].sum(1)
proto = np.array([float(r["out_total"]) for r in frs])

print("=== 10.30~11.20 s 逐帧 ===")
hdr = (f"{'t':>7}{'pre':>8}{'main':>8}{'proto':>8}{'off_f':>8}{'off_p':>8}{'d':>7} "
       f"{'state':<6}{'kind':<14}{'A_hat':>9}{'g':>9}{'A_sum':>9}{'off_g':>8}{'off_s':>8}")
print(hdr)
for i, r in enumerate(frs):
    if 10.30 <= el[i] <= 11.20:
        print(f"{el[i]:7.2f}{tp[i]:8.0f}{tm[i]:8.0f}{proto[i]:8.0f}"
              f"{tm[i]-tp[i]:8.0f}{proto[i]-tp[i]:8.0f}{proto[i]-tm[i]:7.0f} "
              f"{r['state']:<6}{r['kind']:<14}{r['A_hat']:>9}{r['g']:>9}{r['A_sum']:>9}"
              f"{r['off_glide']:>8}{r['off_slow']:>8}")

print("\n=== 0~22 s 全部事件（时间相对首帧）===")
for r in evs:
    ts = float(r["ts"]) - tref
    if ts <= 22.0:
        t0 = float(r["t0"]) - tref if r["t0"] else None
        t0s = f"{t0:.2f}" if t0 is not None else "-"
        print(f"  t={ts:7.2f} {r['ev']:<12} {r['kind']:<14} t0={t0s:>7} "
              f"Ab={r['A_sum_before']:>9} Aa={r['A_sum_after']:>9} "
              f"gb={r['g_before']:>10} ga={r['g_after']:>10} "
              f"{r['state_before']}->{r['state_after']} C={r['C']}")

print("\n=== 慢相段内 d 的稳定性（每 10 s 中位/RMS）===")
d = proto - tm
t = 0.0
while t < el[-1]:
    m = (el >= t) & (el < t + 10)
    if m.any():
        print(f"  t={t:6.0f}~{t+10:6.0f}  d 中位={np.median(d[m]):8.0f}  "
              f"RMS={np.sqrt(np.mean(d[m]**2)):8.0f}  |d| 中位={np.median(np.abs(d[m])):8.0f}")
    t += 10
sys.exit(0)
