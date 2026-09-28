# -*- coding: utf-8 -*-
"""诊断：v5 在恒载 9 组上的 epoch 重启次数（检查电平判据是否被快相/蠕变误触发）。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402


class V5T(GLM53v5):
    def __init__(self, n):
        super().__init__(n)
        self.epoch_t = []
        self.lev_hits = []

    def _begin(self, ts):
        super()._begin(ts)
        self.epoch_t.append(round(float(ts), 2))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.epoch_t.append(round(float(ts), 2))


for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"]:
    for i in (1, 2, 3):
        p = os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv")
        if not os.path.exists(p):
            continue
        d = L.prep(p)
        c = V5T(d["Xu"].shape[1])
        for k in range(len(d["tu"])):
            c.process(d["tu"][k], d["Xu"][k])
        print(f"{loc}/数据{i}: epoch={len(c.epoch_t)} @ {c.epoch_t}  "
              f"g_end={c.g:+.3f} γ∈[{c.gamma.min():.2f},{c.gamma.max():.2f}]")
