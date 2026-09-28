# -*- coding: utf-8 -*-
"""右拇指/数据2：全程形状 + 卸载逐帧，用于定位用户报告的"卸载处奇怪下降"。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                       # noqa: E402
from glm53_v51 import GLM53v51           # noqa: E402
from glm53_v6 import GLM53v6             # noqa: E402

d = L.prep(os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"))
tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
tot = Xu.sum(axis=1)


class T(GLM53v51):
    def __init__(s, n):
        super().__init__(n)
        s.epoch_t = []

    def _begin(s, ts):
        super()._begin(ts)
        s.epoch_t.append(float(ts))

    def _restep(s, ts, z):
        super()._restep(ts, z)
        s.epoch_t.append(float(ts))


c5 = T(Xu.shape[1]); c5.FAST_S, c5.EXEMPT_AWIN, c5.LEV_ARM_S = 3.0, 1.0, 3.0
Y5 = np.empty_like(Xu)
for i in range(len(tu)):
    Y5[i] = c5.process(tu[i], Xu[i])
c6 = GLM53v6(Xu.shape[1])
Y6 = np.empty_like(Xu)
for i in range(len(tu)):
    Y6[i] = c6.process(tu[i], Xu[i])

print("=== 右拇指/数据2 全程（每 5 s）===")
print("      t      raw    v5-3s       v6   v5-raw   v6-raw")
for tt in np.arange(12, 162, 5.0):
    i = int(np.searchsorted(tu, tt))
    if i >= len(tu):
        break
    print(f"{tu[i]:7.1f}{tot[i]:9.2f}{Y5[i].sum():9.2f}{Y6[i].sum():9.2f}"
          f"{Y5[i].sum()-tot[i]:9.2f}{Y6[i].sum()-tot[i]:9.2f}")

print("\n=== 卸载逐帧 159.25~160.05（每帧）===")
i0 = int(np.searchsorted(tu, 159.25)); i1 = int(np.searchsorted(tu, 160.05))
print("        t      raw    v5-3s       v6")
for i in range(i0, i1):
    print(f"{tu[i]:9.3f}{tot[i]:9.3f}{Y5[i].sum():9.3f}{Y6[i].sum():9.3f}")

print("\n=== v6 相对原始的单帧变化最大处（全程 top10，|Δv6−Δraw|）===")
d6 = np.diff(Y6.sum(axis=1)); dr = np.diff(tot)
ex = np.abs(d6 - dr)
for k in np.argsort(ex)[-10:][::-1]:
    print(f"  t={tu[k]:7.2f}  raw {tot[k]:9.2f}->{tot[k+1]:9.2f}  "
          f"v6 {Y6[k].sum():9.2f}->{Y6[k+1].sum():9.2f}  超额 {ex[k]:8.2f}")
