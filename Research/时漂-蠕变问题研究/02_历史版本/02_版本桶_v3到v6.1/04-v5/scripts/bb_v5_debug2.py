# -*- coding: utf-8 -*-
"""v5 场景 6（两档间隔 8s）失败根因：逐帧打印 gamma / hold_comp / carry。"""
import os
import sys
import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

REC0 = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
        r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
        r"\device_001_seg000.csv")
d0 = L.prep(REC0)
a, b = int(133.84 / d0["dtm"]), int(182.0 / d0["dtm"])
uu = d0["tu"][a:b] - d0["tu"][a]
yy = d0["tot"][a:b] / d0["tot"][a + int(5.0 / d0["dtm"])] - 1.0
fit = lambda x, a1, t1, a2, t2: a1 * (1 - np.exp(-x / t1)) + a2 * (1 - np.exp(-x / t2))
P, _ = optimize.curve_fit(fit, uu, yy, p0=[0.03, 3.0, 0.05, 60.0],
                          bounds=([0, 0.2, 0, 5], [1, 60, 1, 3000]))
cr = lambda x: np.where(x > 0, fit(np.maximum(x, 0), *P), 0.0)
FS, dt, DUR = 100.0, 0.01, 300.0
tt = np.arange(0.0, DUR, dt)
sig = np.zeros_like(tt)
m = tt > 20.0
sig[m] += 10000.0 * (1 + cr(tt[m] - 20.0))
for t0, k in ((220.0, 0.5), (228.0, 0.5)):
    mm = tt > t0
    sig[mm] += k * 10000.0 * (1 + cr(tt[mm] - t0))

c = GLM53v5(1)
print(f"{'t':>8}{'原始':>8}{'显示':>8}{'A':>8}{'g':>9}{'gamma':>8}{'g2':>10}{'grel':>10}"
      f"{'carry':>7}{'hc':>7}{'hld':>4}{'pnd':>4}{'fd':>3}")
for i in range(len(tt)):
    Y = c.process(tt[i], np.array([sig[i]]))[0]
    if 227.0 <= tt[i] <= 232.0 and (round(tt[i] * 100) % 5 == 0):
        print(f"{tt[i]:8.2f}{sig[i]:8.0f}{Y:8.0f}{c.A.max():8.0f}{c.g:+9.5f}"
              f"{c.gamma[0]:8.3f}{c.g2:10.6f}{c.g_rel[0]:10.6f}{c.carry.max():7.0f}"
              f"{(0 if c.hold_comp is None else float(c.hold_comp.max())):7.0f}"
              f"{int(c.hold):4d}{int(c.pending):4d}{int(c.fast_done):3d}")
