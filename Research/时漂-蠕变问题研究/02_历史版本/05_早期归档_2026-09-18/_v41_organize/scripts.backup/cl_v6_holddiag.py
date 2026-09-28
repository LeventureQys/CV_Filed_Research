# -*- coding: utf-8 -*-
"""v6 慢相段诊断：加载后整段保压内 raw / v5 / v6 + v6 内部状态（A.sum / g / loaded 数 / 事件）。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                       # noqa: E402
from glm53_v51 import GLM53v51           # noqa: E402
from glm53_v6 import GLM53v6             # noqa: E402

CASES = [
    ("右拇指_1", os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv")),
    ("左拇指_1", os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv")),
    ("四指_1", os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv")),
]

for tag, path in CASES:
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)


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


    c5 = _T(Xu.shape[1]); c5.FAST_S, c5.EXEMPT_AWIN, c5.LEV_ARM_S = 3.0, 1.0, 3.0
    Y5 = np.empty_like(Xu)
    for i in range(len(tu)):
        Y5[i] = c5.process(tu[i], Xu[i])

    c6 = GLM53v6(Xu.shape[1])
    Y6 = np.empty_like(Xu)
    for i in range(len(tu)):
        Y6[i] = c6.process(tu[i], Xu[i])

    # 负载段
    ld = tot > 0.15 * tot.max()
    dd = np.diff(ld.astype(int))
    s = list(np.where(dd == 1)[0] + 1); e = list(np.where(dd == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    i0, i1 = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]
    print("=" * 118)
    print(f"{tag}  负载段 {tu[i0]:.1f}~{tu[i1]:.1f}s  A_peak(v6)={c6.A_peak:.3f}  "
          f"事件={len(c6.epoch_t)}")
    for t, k in c6.kind_log:
        print(f"      v6 事件 @{t:7.2f}s {k}")
    print(f"{'t(s)':>7}{'raw':>10}{'v5-3s':>10}{'v6':>10}{'v6显示-raw':>12}"
          f"{'A.sum':>10}{'g':>10}{'loaded':>8}{'state':>8}")
    for tt in [0, 0.5, 1, 2, 3, 5, 8, 12, 20, 40, 60, 90, 120]:
        i = min(i0 + int(tt / dtm), i1 - 1, len(tu) - 1)
        print(f"{tu[i]-tu[i0]:7.1f}{tot[i]:10.3f}{Y5[i].sum():10.3f}{Y6[i].sum():10.3f}"
              f"{Y6[i].sum()-tot[i]:12.3f}{c6.A.sum():10.3f}{c6.g:10.4f}"
              f"{int(c6.loaded.sum()):8d}{c6.state:>8}")
