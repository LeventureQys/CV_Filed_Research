# -*- coding: utf-8 -*-
"""v6c 额外 epoch 的来源排查（临时脚本）：打印每个 epoch 的时间/类型/触发时的电平。"""
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
RECS = [("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv"))]


class T51(GLM53v51):
    def __init__(self, n):
        super().__init__(n)
        self.log = []

    def _begin(self, ts):
        super()._begin(ts)
        self.log.append((float(ts), "begin"))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.log.append((float(ts), "restep"))


class T6c(GLM53v6c):
    def __init__(self, n):
        super().__init__(n)
        self.log = []

    def _begin(self, ts):
        super()._begin(ts)
        self.log.append((float(ts), "begin", self.epoch_kind))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.log.append((float(ts), "restep", self.epoch_kind))


for tag, path in RECS:
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    ev = [e for e, _ in L.detect_events(tot, dtm)]
    evt = L.event_table(d, ev, {}, [], gain_s=6.0, algos=[])
    print("=" * 110)
    print(f"[{tag}]  真实变载事件（detect_events）：")
    for _, e in evt.iterrows():
        print(f"    t={e.t:8.2f}s  台阶={e.jump:+8.0f}  前={e.pre:8.0f}  后={e.post:8.0f}  "
              f"ratio={e.ratio:5.2f}")
    for nm, cls in [("v5.1", T51), ("v6c", T6c)]:
        c = cls(Xu.shape[1])
        for i in range(len(tu)):
            c.process(tu[i], Xu[i])
        print(f"  {nm} epoch 日志（{len(c.log)} 个）：")
        for row in c.log:
            ts = row[0]
            j = int(np.searchsorted(tu, ts))
            lv = float(tot_s[max(0, j - 50):j + 1].mean())
            print(f"    t={ts:8.2f}s  {row[1]:>7s}  触发帧电平={lv:9.0f}")
