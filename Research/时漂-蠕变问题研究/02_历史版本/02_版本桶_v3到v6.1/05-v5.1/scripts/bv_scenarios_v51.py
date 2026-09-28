# -*- coding: utf-8 -*-
"""合成砝码 7 场景 · **仅当前实现**（v5.1，`glm53_v51.py`）复算，供文档 §10.1。

与 `ba_v5_scenarios.py` 同一套场景与蠕变律（由实采保压段拟合），只跑 raw 与当前实现，
不引入任何历史版本对照。
产出：results/scenarios_v51.csv、控制台表
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

REC0 = os.path.join(TEMP, "变化负载", "切换负载-快相无责的测试",
                    "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")
d0 = L.prep(REC0)
a, b = int(133.84 / d0["dtm"]), int(182.0 / d0["dtm"])
uu = d0["tu"][a:b] - d0["tu"][a]
yy = d0["tot"][a:b] / d0["tot"][a + int(5.0 / d0["dtm"])] - 1.0
ft = lambda x, a1, t1, a2, t2: a1 * (1 - np.exp(-x / t1)) + a2 * (1 - np.exp(-x / t2))  # noqa: E731
P, _ = optimize.curve_fit(ft, uu, yy, p0=[0.03, 3.0, 0.05, 60.0],
                          bounds=([0, 0.2, 0, 5], [1, 60, 1, 3000]))
cr = lambda x: np.where(x > 0, ft(np.maximum(x, 0), *P), 0.0)          # noqa: E731
FS, dt, DUR = 100.0, 0.01, 600.0
tt = np.arange(0.0, DUR, dt)
print(f"蠕变律（实测拟合）：{100 * P[0]:.2f}%·τ={P[1]:.1f}s + {100 * P[2]:.2f}%·τ={P[3]:.0f}s")


def build(steps, base_n=1.0):
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


CASES = [("1. 10N → 保压 280s → +2.5N（台阶 25%）", [(300.0, 0.25)], 1.0),
         ("2. 10N → 保压 280s → +5N（台阶 50%）", [(300.0, 0.50)], 1.0),
         ("3. 10N → 保压 280s → +10N（台阶 100%）", [(300.0, 1.00)], 1.0),
         ("4. 20N → 保压 280s → +5N（台阶 12.5%）", [(300.0, 0.25)], 2.0),
         ("5. 10N → 保压 200s → +5N → 80s 后 +5N", [(220.0, 0.50), (300.0, 0.50)], 1.0),
         ("6. 10N → 保压 200s → +5N → 8s 后 +5N", [(220.0, 0.50), (228.0, 0.50)], 1.0),
         ("7. 10N → 保压 200s → +5N → 3s 后 +5N", [(220.0, 0.50), (223.0, 0.50)], 1.0)]

tail = slice(int((DUR - 40) / dt), None)
rows = []
print("=" * 96)
print("合成砝码场景 · 当前实现（免责 3 s 默认档）")
print(f"{'场景':>40}{'理想':>9}{'原始(含蠕变)':>13}{'本算法':>16}")
print("-" * 96)
for title, steps, base_n in CASES:
    sig = build(steps, base_n)
    ideal = (base_n + sum(k for _, k in steps)) * 10000.0
    Y, c = run(GLM53v51, sig)
    end = Y[tail].mean()
    rows.append(dict(case=title, ideal=ideal, raw=sig[tail].mean(), algo=end,
                     dev_pct=100 * (end - ideal) / ideal))
    print(f"{title:>40}{ideal:9.0f}{sig[tail].mean():13.0f}{end:8.0f}({100*(end-ideal)/ideal:+5.1f}%)")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "scenarios_v51.csv"), index=False, encoding="utf-8-sig")
print(f"\n偏差范围：{df.dev_pct.min():+.2f}% ~ {df.dev_pct.max():+.2f}%（|偏差| 中位 {df.dev_pct.abs().median():.2f}%）")

# 关键两例的逐帧轨迹（加载沿附近）
for title, steps, base_n in (CASES[0], CASES[1]):
    sig = build(steps, base_n)
    ideal = (base_n + sum(k for _, k in steps)) * 10000.0
    Y, c = run(GLM53v51, sig)
    print(f"\n--- {title} ---")
    print(f"  {'t(s)':>7}{'原始':>9}{'理想':>9}{'本算法':>9}{'偏差':>8}")
    for w in (295, 300.5, 302, 305, 308, 312, 320, 340, 400, 500, 595):
        i = int(w / dt)
        print(f"  {w:7.1f}{sig[i]:9.0f}{ideal:9.0f}{Y[i]:9.0f}{Y[i] - ideal:8.0f}")
