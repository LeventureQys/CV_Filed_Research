# -*- coding: utf-8 -*-
"""定位 v6c 与 v5.1 在实录上的分叉点（临时脚本）。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, os.pardir, "v4.1flash", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402
from glm53_v6c import GLM53v6c                         # noqa: E402

B = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS)), "变化负载")
P = os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc",
                 "device_001_seg000.csv")

d = L.prep(P)
tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
tot_s = L.med_smooth(tot, 0.5 / dtm)
evt = L.event_table(d, [e for e, _ in L.detect_events(tot, dtm)], {}, [], gain_s=6.0, algos=[])
print("真实事件：")
for _, e in evt.iterrows():
    print(f"  t={e.t:8.2f}s 台阶={e.jump:+9.0f} 前={e.pre:8.0f} 后={e.post:8.0f}")


class T51(GLM53v51):
    def __init__(self, n):
        super().__init__(n)
        self.ev = []

    def _begin(self, ts):
        super()._begin(ts)
        self.ev.append((float(ts), "begin"))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.ev.append((float(ts), "restep"))


class T6c(GLM53v6c):
    def __init__(self, n):
        super().__init__(n)
        self.ev = []

    def _begin(self, ts):
        super()._begin(ts)
        self.ev.append((float(ts), "begin"))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.ev.append((float(ts), "restep"))


res = {}
for nm, cls in [("v5.1", T51), ("v6c", T6c)]:
    c = cls(Xu.shape[1])
    D = np.empty(len(tu))
    G = np.empty(len(tu))
    A = np.empty(len(tu))
    FD = np.empty(len(tu))
    for i in range(len(tu)):
        y = c.process(tu[i], Xu[i])
        D[i] = (Xu[i] - y).sum()
        G[i] = c.g
        A[i] = c.A.max()
        FD[i] = 1.0 if c.fast_done else 0.0
    res[nm] = (c, D, G, A, FD)
    print(f"\n[{nm}] epoch ({len(c.ev)}):", [(round(t, 2), k) for t, k in c.ev])
    print(f"  末帧 g={c.g:+.4f} A={c.A.max():.0f} 后期扣除中位={np.median(D[tu>tu[-1]-10]):.0f}")

c5, D5, G5, A5, FD5 = res["v5.1"]
c6, D6, G6, A6, FD6 = res["v6c"]
dd = np.abs(D6 - D5)
print(f"\n扣除量差异：max={dd.max():.1f} @t={tu[int(np.argmax(dd))]:.2f}s  "
      f"中位={np.median(dd):.1f}")
# 分叉点 = 差异首次超过 1% 峰值
thr = 0.01 * tot_s.max()
idx = np.where(dd > thr)[0]
print(f"分叉点（>1%峰值={thr:.0f}）：t={tu[idx[0]]:.2f}s" if len(idx) else "无分叉")
for tt in (t for t in (16.3, 17, 18, 20, 25, 30, 38, 40, 45, 55, 60, 62, 70, 76, 78, 80, 85, 90, 95, 100)):
    j = int(np.searchsorted(tu, tt))
    if j >= len(tu):
        continue
    print(f"  t={tt:6.1f}s  raw={tot[j]:8.0f}  v5.1: ded={D5[j]:8.1f} g={G5[j]:+.4f} A={A5[j]:7.0f} fd={FD5[j]:.0f}"
          f"  |  v6c: ded={D6[j]:8.1f} g={G6[j]:+.4f} A={A6[j]:7.0f} fd={FD6[j]:.0f}  Δ={D6[j]-D5[j]:+7.1f}")
