# -*- coding: utf-8 -*-
"""v2.0 A1 · 44 s 关键窗口的现场/原型数值对照（只读）。"""
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import v20_lib as L          # noqa: E402

RES = os.path.abspath(os.path.join(_HERE, "..", "results"))
frs = list(csv.DictReader(open(os.path.join(RES, "a1_replay_frames.csv"), encoding="utf-8")))
ds = L.load_dataset(L.DS_ZERO)
pre, main = ds["pre"], ds["main"]
el = pre["el"]
tp, tm = pre["V"].sum(1), main["V"].sum(1)
proto = np.array([float(r["out_total"]) for r in frs])
Asum = np.array([float(r["A_sum"]) for r in frs])
g = np.array([float(r["g"]) for r in frs])
st = np.array([r["state"] for r in frs])


def stat(t0, t1, name):
    m = (el >= t0) & (el <= t1)
    print(f"{name:<34} n={int(m.sum()):5d} pre中位={np.median(tp[m]):7.0f} "
          f"main中位={np.median(tm[m]):7.0f} proto中位={np.median(proto[m]):7.0f} "
          f"off_f={np.median(tm[m]-tp[m]):7.0f} off_p={np.median(proto[m]-tp[m]):7.0f} "
          f"d(proto-main)={np.median(proto[m]-tm[m]):7.0f}")


print("=== 现场 vs 原型 分窗口对照（总量 ADC）===")
stat(0.0, 2.2, "0.0~2.2  (初始空载)")
stat(2.2, 5.3, "2.2~5.3  (首次加载+滑行)")
stat(5.3, 40.0, "5.3~40.0 (长保压前段)")
stat(40.0, 42.0, "40.0~42.0(42 s 加载前)")
stat(42.0, 43.8, "42.0~43.8(加载沿后)")
stat(43.8, 45.4, "43.8~45.4(卸载后/handoff 前)")
stat(45.4, 46.4, "45.4~46.4(handoff 后)")
stat(46.4, 49.0, "46.4~49.0(重锚后)")
stat(49.0, 55.0, "49.0~55.0")
stat(55.0, 111.0, "55.0~111.0(长保压后段)")
stat(111.0, 120.0, "111~120 (回到空载)")

print("\n=== 关键瞬时值 ===")
for t in (41.5, 42.0, 43.5, 44.0, 45.3, 45.4, 46.0, 46.5, 47.0, 48.0, 50.0, 55.0, 60.0):
    i = int(np.searchsorted(el, t))
    print(f"  t={el[i]:7.2f} pre={tp[i]:7.0f} main={tm[i]:7.0f} proto={proto[i]:7.0f} "
          f"off_f={tm[i]-tp[i]:6.0f} off_p={proto[i]-tp[i]:6.0f} "
          f"d={proto[i]-tm[i]:6.0f} A_sum={Asum[i]:7.0f} g={g[i]:9.5f} {st[i]}")

print("\n=== 该录制 κ 触发检查：重建 Â 的 κ 截断（解析）===")
k_lo, k_hi = 1.05, 1.30
print("  Â = min(max(a_ls, inc), κ·inc) ⇒ κ 越小 Â 越受抑；本录制事件 A_hat 最大值="
      f"{max(float(r['A_hat']) for r in frs if r['A_hat'] != ''):.0f} ADC")

print("\n=== 现场 main 流在事件期的最大偏移与结束时偏移 ===")
for name, t0, t1 in (("42.0~45.4", 42.0, 45.4), ("46.4~49.0", 46.4, 49.0),
                     ("49.0~55.0", 49.0, 55.0), ("55.0~111.0", 55.0, 111.0)):
    m = (el >= t0) & (el <= t1)
    print(f"  {name}: off_f min={ (tm-tp)[m].min():7.0f} max={ (tm-tp)[m].max():7.0f} "
          f"末值={ (tm-tp)[m][-1]:7.0f}")
sys.exit(0)
