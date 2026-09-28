# -*- coding: utf-8 -*-
"""v5 失败场景（20N+5N、两档间隔 8s）的内部量追踪。"""
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
FS, dt, DUR = 100.0, 0.01, 400.0
tt = np.arange(0.0, DUR, dt)


def build(base_n, steps):
    s = np.zeros_like(tt)
    m = tt > 20.0
    s[m] += base_n * 10000.0 * (1 + cr(tt[m] - 20.0))
    for t0, k in steps:
        mm = tt > t0
        s[mm] += k * 10000.0 * (1 + cr(tt[mm] - t0))
    return s


def trace(title, base_n, steps, lo, hi):
    sig = build(base_n, steps)
    ideal = (base_n + sum(k for _, k in steps)) * 10000.0
    c = GLM53v5(1)
    rows = []
    for i in range(len(tt)):
        Y = c.process(tt[i], np.array([sig[i]]))[0]
        if lo <= tt[i] <= hi:
            ded = sig[i] - Y
            rows.append((tt[i], sig[i], Y, ded, c.in_load, c.a_captured, c.A.max(), c.g,
                         c.carry.max(), c.hold, c.pending, c.fast_done,
                         c.a_new_frames, 0 if c.hold_comp is None else float(np.max(c.hold_comp))))
    print(f"\n=== {title}（理想 {ideal:.0f}）===")
    print(f"{'t':>7}{'原始':>8}{'显示':>8}{'扣除':>8}{'in':>4}{'aC':>4}{'A':>8}{'g':>8}"
          f"{'carry':>8}{'hld':>4}{'pnd':>5}{'fastdone':>9}{'anf':>5}{'hc':>8}")
    step = max(1, len(rows) // 60)
    for r in rows[::step]:
        print(f"{r[0]:7.1f}{r[1]:8.0f}{r[2]:8.0f}{r[3]:8.0f}{int(r[4]):4d}{int(r[5]):4d}"
              f"{r[6]:8.0f}{r[7]:+8.4f}{r[8]:8.0f}{int(r[9]):4d}{int(r[10]):5d}{int(r[11]):9d}"
              f"{r[12]:5d}{r[13]:8.0f}")


trace("20N 保压 280s → +5N（台阶 12.5%）", 2.0, [(300.0, 0.25)], 295.0, 340.0)
trace("10N → 200s → +5N → 8s 后 +5N", 1.0, [(220.0, 0.50), (228.0, 0.50)], 215.0, 260.0)
