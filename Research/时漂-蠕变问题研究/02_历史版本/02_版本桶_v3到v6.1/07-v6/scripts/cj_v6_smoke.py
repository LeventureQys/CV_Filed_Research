# -*- coding: utf-8 -*-
"""v6 原型冒烟自检：单份数据上跑 raw / 免责1s / 免责3s / v6，打印关键状态与轨迹。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                       # noqa: E402
from glm53_v51 import GLM53v51           # noqa: E402
from glm53_v6 import GLM53v6             # noqa: E402

CASES = [
    ("右拇指_1", os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv")),
    ("中途13ffca", os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                                "最终测试目标", "device_001_seg000.csv")),
]


def run_v5(tu, Xu, fast):
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

    c = _T(Xu.shape[1])
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = fast, fast / 3.0, fast
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def run_v6(tu, Xu):
    c = GLM53v6(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


for tag, path in CASES:
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    print("=" * 112)
    print(f"{tag}  时长 {d['span']:.1f}s  ch={Xu.shape[1]}  dt={1000*dtm:.2f}ms  峰值 {tot.max():.0f}")
    Yr = Xu
    Y1, c1 = run_v5(tu, Xu, 1.0)
    Y3, c3 = run_v5(tu, Xu, 3.0)
    Y6, c6 = run_v6(tu, Xu)
    print(f"  v6: sig_d={c6.sig_d:.3f}  事件={len(c6.epoch_t)}  撤销={c6.n_revoke}  "
          f"C5={c6.n_c5}  A.max={c6.A.max():.3f}  g={c6.g:+.4f}  state={c6.state}")
    for t, k in c6.kind_log:
        print(f"       @{t:7.2f}s  {k}")
    print(f"  v5-1s: 事件={len(c1.epoch_t)} A.max={c1.A.max():.3f} g={c1.g:+.4f}")
    print(f"  v5-3s: 事件={len(c3.epoch_t)} A.max={c3.A.max():.3f} g={c3.g:+.4f}")
    # 第一事件后 6 s 的轨迹
    if c6.epoch_t:
        t0 = c6.epoch_t[0]
        i0 = int(np.searchsorted(tu, t0))
        print(f"  首事件 t={t0:.3f}s 起（每 0.25 s：t, raw, v5-1s, v5-3s, v6）")
        for k in range(0, int(6.0 / dtm) + 1, max(1, int(0.25 / dtm))):
            i = min(i0 + k, len(tu) - 1)
            print(f"    +{tu[i]-t0:5.2f}s  {tot[i]:10.2f}  {Y1[i].sum():10.2f}  "
                  f"{Y3[i].sum():10.2f}  {Y6[i].sum():10.2f}")
