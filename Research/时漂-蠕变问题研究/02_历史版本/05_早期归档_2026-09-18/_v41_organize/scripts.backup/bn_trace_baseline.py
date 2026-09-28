# -*- coding: utf-8 -*-
"""直接量测：加载沿期间空载基线 b_ 被"偷吃"了多少（这是零点负偏置的来源之一）。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
PATH = os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")
T0, T1 = 38.95, 39.60

d = L.prep(PATH)
tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
c = GLM53v3(Xu.shape[1])
print("  t      raw总量   ts_smooth  门控上限0.2*max_ts  in_load armed   b_总量  显示总量  说明")
prev_armed = False
for i in range(len(tu)):
    ts = tu[i]
    Y = c.process(ts, Xu[i])
    if not (T0 <= ts <= T1):
        continue
    gate = c.BASE_GATE_FRAC * c.max_ts
    note = ""
    if c.armed and not prev_armed:
        note = "★ armed 变 True（首次卸载后开启基线跟踪）"
    if c.in_load and abs(c.b.sum()) > 1.0 and not note:
        note = "（b_ 正在跟随加载沿）"
    prev_armed = c.armed
    if i % 5 == 0 or note:
        print(f"{ts:7.2f} {Xu[i].sum():9.0f} {c.ts_smooth:11.0f} {gate:16.0f}  "
              f"{str(c.in_load):7s} {str(c.armed):5s} {c.b.sum():8.1f} {Y.sum():9.1f}  {note}")
