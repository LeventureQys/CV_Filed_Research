# -*- coding: utf-8 -*-
"""单份实录的 v6 事件台账 + 最大偏差定位（找离群原因）。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                       # noqa: E402
from glm53_v51 import GLM53v51           # noqa: E402
from glm53_v6 import GLM53v6             # noqa: E402

CASES = [("切换负载-快相无责", os.path.join(
    TEMP, "变化负载", "切换负载-快相无责的测试",
    "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
    ("中途切换-1d9493", os.path.join(
        TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv"))]

for tag, path in CASES:
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    ts_s = L.med_smooth(tot, 0.5 / dtm)

    class _T(GLM53v51):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    c5 = _T(Xu.shape[1]); c5.FAST_S, c5.EXEMPT_AWIN, c5.LEV_ARM_S = 3.0, 1.0, 3.0
    Y5 = np.empty_like(Xu)
    for i in range(len(tu)):
        Y5[i] = c5.process(tu[i], Xu[i])
    c6 = GLM53v6(Xu.shape[1]); c6.debug = True
    Y6 = np.empty_like(Xu)
    for i in range(len(tu)):
        Y6[i] = c6.process(tu[i], Xu[i])
    y5s = L.med_smooth(Y5.sum(axis=1), 0.5 / dtm)
    y6s = L.med_smooth(Y6.sum(axis=1), 0.5 / dtm)
    g5 = np.abs(y5s - ts_s); g6 = np.abs(y6s - ts_s)
    print("=" * 112)
    print(f"{tag}  峰值 {tot.max():.0f}  v6 事件 {len(c6.epoch_t)}  A_peak {c6.A_peak:.0f}")
    for t, k in c6.kind_log:
        print(f"      @{t:7.2f}s {k}")
    for name, arr in (("v5-3s", g5), ("v6", g6)):
        i = int(np.argmax(arr))
        print(f"   {name} 最大偏差 {arr[i]:.0f} @ t={tu[i]:.2f}s  (raw {tot[i]:.0f} / 显示 "
              f"{(y5s if name == 'v5-3s' else y6s)[i]:.0f})")
    print("   t(s)      raw     v5-3s       v6   |  v6事件")
    for tt in np.arange(8, 135, 4.0):
        i = int(np.searchsorted(tu, tt))
        if i >= len(tu):
            break
        ev = [f"{t:.0f}:{k[:6]}" for t, k in c6.kind_log if abs(t - tu[i]) < 2.0]
        print(f"   {tu[i]:7.1f}{tot[i]:10.0f}{y5s[i]:10.0f}{y6s[i]:10.0f}   | {' '.join(ev)}")
