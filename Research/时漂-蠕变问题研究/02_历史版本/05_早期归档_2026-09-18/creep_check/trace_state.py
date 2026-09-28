# -*- coding: utf-8 -*-
"""状态机追踪：定位改动后负载段/基准锚定为何不生效。"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
sys.stdout.reconfigure(encoding="utf-8")
from check_estimator import Compensator

fs = 100.0
w = np.array([0.5, 0.3, 0.2])
rng = np.random.default_rng(3)
comp = Compensator(3)
prev = None
for k in range(int(60 * fs)):
    ts = k / fs
    lv = 0.0 if ts < 15.0 else 100000.0
    raw = lv * w + rng.normal(0.0, 30.0, 3)
    v = raw.copy()
    comp.process(ts, v)
    state = (comp.in_load, comp.bl_ready, comp.established, round(comp.a_est, 1),
             round(comp.y0, 1), round(comp.carry_total, 1))
    if state != prev and (k % 25 == 0 or True):
        print("t=%6.2f in_load=%-5s bl=%-5s est=%-5s a=%9.1f y0=%9.1f carry=%8.1f "
              "sm=%9.1f idle=%9.1f peak=%9.1f dev=%7.2f noise=%8.2f enter_hold=%.2f"
              % (ts, comp.in_load, comp.bl_ready, comp.established, comp.a_est, comp.y0,
                 comp.carry_total, comp.sm, comp.idle_ref, comp.peak_ref, comp.idle_dev,
                 comp.noise, comp.enter_hold))
        prev = state
