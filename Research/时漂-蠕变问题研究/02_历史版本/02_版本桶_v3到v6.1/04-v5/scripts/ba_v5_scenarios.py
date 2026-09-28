# -*- coding: utf-8 -*-
"""v5 验证（合成）：砝码场景 —— 10N 保压后加码，看显示稳到哪。

对比 raw / v3(现役) / v4(免责5s) / v5(本次修复)。
蠕变律由实测保压段拟合得到。
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
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
FS = 100.0
dt = 1.0 / FS
DUR = 600.0
tt = np.arange(0.0, DUR, dt)

ALGOS = [("raw", None, {}),
         ("v3", GLM53v3, {}),
         ("v4", L.GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
         ("v5", GLM53v5, {})]


def build(steps, base_n=1.0, base10=True):
    """基准 = 10N(10000)；steps=[(时刻, 相对基准10N 的倍数)]"""
    s = np.zeros_like(tt)
    m = tt > 20.0
    s[m] += base_n * 10000.0 * (1 + cr(tt[m] - 20.0))
    for t0, k in steps:
        mm = tt > t0
        s[mm] += k * 10000.0 * (1 + cr(tt[mm] - t0))
    return s


def run(cls, sig, **kw):
    c = cls(1)
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty(len(tt))
    for i in range(len(tt)):
        Y[i] = c.process(tt[i], np.array([sig[i]]))[0]
    return Y, c


CASES = [
    ("1. 10N → 保压 280s → +2.5N（台阶 25%）", [(300.0, 0.25)], 1.0),
    ("2. 10N → 保压 280s → +5N（台阶 50%）", [(300.0, 0.50)], 1.0),
    ("3. 10N → 保压 280s → +10N（台阶 100%）", [(300.0, 1.00)], 1.0),
    ("4. 20N → 保压 280s → +5N（台阶 25%）", [(300.0, 0.25)], 2.0),
    ("5. 10N → 保压 200s → +5N → 80s 后 +5N", [(220.0, 0.50), (300.0, 0.50)], 1.0),
    ("6. 10N → 保压 200s → +5N → 8s 后 +5N", [(220.0, 0.50), (228.0, 0.50)], 1.0),
    ("7. 10N → 保压 200s → +5N → 3s 后 +5N", [(220.0, 0.50), (223.0, 0.50)], 1.0),
]

print("=" * 112)
print("v5 合成砝码场景验证（蠕变律取自实测：%.2f%%·τ=%.1fs + %.2f%%·τ=%.0fs）"
      % (100 * P[0], P[1], 100 * P[2], P[3]))
print("=" * 112)
print(f"{'场景':>44}{'理想':>9}{'原始':>9}" + "".join(f"{k:>18}" for k, _, _ in ALGOS[1:]))
for title, steps, base_n in CASES:
    sig = build(steps, base_n)
    ideal = (base_n + sum(k for _, k in steps)) * 10000.0
    tail = slice(int((DUR - 40) / dt), None)
    line = f"{title:>44}{ideal:9.0f}{sig[tail].mean():9.0f}"
    for k, cls, kw in ALGOS[1:]:
        Y, c = run(cls, sig, **kw)
        end = Y[tail].mean()
        line += f"{end:8.0f}({100*(end-ideal)/ideal:+5.1f}%)"
    print(line)

print("\n（括号内 = 相对理想值的偏差；理想值 = 各档砝码之和对应的 ADC）")

# 逐帧轨迹：最关键的两例
for title, steps, base_n in (CASES[0], CASES[1]):
    sig = build(steps, base_n)
    ideal = (base_n + sum(k for _, k in steps)) * 10000.0
    print(f"\n--- {title} ---")
    print(f"  {'t(s)':>7}{'原始':>9}{'理想':>9}" + "".join(f"{k:>10}" for k, _, _ in ALGOS[1:]))
    Ys = {k: run(cls, sig, **kw)[0] for k, cls, kw in ALGOS[1:]}
    for w in (295, 300.5, 302, 305, 308, 312, 320, 340, 400, 500, 595):
        i = int(w / dt)
        print(f"  {w:7.1f}{sig[i]:9.0f}{ideal:9.0f}" +
              "".join(f"{Ys[k][i]:10.0f}" for k, _, _ in ALGOS[1:]))
