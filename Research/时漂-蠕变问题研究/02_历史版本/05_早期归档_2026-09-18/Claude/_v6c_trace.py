# -*- coding: utf-8 -*-
"""v6c 快相窗时序追踪（临时脚本）。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, os.pardir, "v4.1flash", "scripts"))
sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)
sys.stdout.reconfigure(encoding="utf-8")
from glm53_v51 import GLM53v51          # noqa: E402
from glm53_v6c import GLM53v6c, g_shape  # noqa: E402

DT, N = 0.01, 5
rng = np.random.default_rng(7)
T = 40.0
t = np.arange(0.0, T, DT)
X = np.full((len(t), N), 40.0)
for i in range(len(t)):
    if t[i] >= 5.0:
        uu = t[i] - 5.0
        fast = float(g_shape(uu)) if uu < 5.0 else 1.0
        slow = 1.0 + 0.35 * (1.0 - np.exp(-uu / 25.0))
        X[i] += 3000.0 * fast * slow / N
X *= (1.0 + 0.05 * np.arange(N) / N)
X += rng.normal(0.0, 3.0, X.shape)

c = GLM53v6c(N)
rows = []
for i in range(len(t)):
    y = c.process(t[i], X[i])
    rows.append((t[i], int(c.in_load), int(c.fast_done), int(c.hold),
                 c.epoch_start, c._shape_tau(t[i]) if c.real_step_ts else -1.0,
                 c.shape_ratio, float(c.A_obs.max()), float(c.ramp_target_est.max()),
                 float(c._ded_now.max()), c.epoch_ramp, c._ramp_dur, c._ramp_t0,
                 float((X[i] - y).sum())))
r = np.array(rows, dtype=float)
print("t      inL fd hold ep_start tau_real ratio  A_obs   tgt     ded    ramp dur   t0     totDed")
for k in range(len(r)):
    if 6.8 <= r[k, 0] <= 13.0 and (k % 5 == 0):
        print(f"{r[k,0]:6.2f} {r[k,1]:3.0f} {r[k,2]:2.0f} {r[k,3]:4.0f} {r[k,4]:8.2f} "
              f"{r[k,5]:8.2f} {r[k,6]:5.3f} {r[k,7]:7.1f} {r[k,8]:7.1f} {r[k,9]:7.2f} "
              f"{r[k,10]:5.2f} {r[k,11]:5.2f} {r[k,12]:6.2f} {r[k,13]:8.1f}")

# 与 v5.1 对比：显示轨迹
c5 = GLM53v51(N)
Y5 = np.empty_like(X)
Y6 = np.empty_like(X)
c6 = GLM53v6c(N)
for i in range(len(t)):
    Y5[i] = c5.process(t[i], X[i])
    Y6[i] = c6.process(t[i], X[i])
D5, D6 = (X - Y5).sum(axis=1), (X - Y6).sum(axis=1)
print()
print("显示总电平（原始和 / v5.1 / v6c）：")
for tt in (7.0, 8.0, 9.0, 9.5, 10.0, 10.1, 10.5, 11.0, 12.0, 14.0, 20.0, 30.0):
    j = int(tt / DT)
    print(f"  t={tt:5.1f}s  raw={X[j].sum():8.1f}  v5.1={Y5[j].sum():8.1f} (扣{D5[j]:7.1f})  "
          f"v6c={Y6[j].sum():8.1f} (扣{D6[j]:7.1f})")
print()
print("v6c epoch:", [round(x, 2) for x in c6.epoch_t], "real:", [round(x, 2) for x in c6.real_step_t])
print("v6c handoff_ts =", round(c6.handoff_ts, 2), " ratio@ho =", round(c6.shape_ratio, 4),
      " 限速命中 =", c6._ramp_limited)
