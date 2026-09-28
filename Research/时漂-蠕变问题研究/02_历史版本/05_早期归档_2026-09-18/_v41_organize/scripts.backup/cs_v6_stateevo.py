# -*- coding: utf-8 -*-
"""A/g/扣除量随时间演化：定位 v6 平台偏高(切换负载)与长保压缓慢下滑(右拇指/数据2)的来源。"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                       # noqa: E402
from glm53_v6 import GLM53v6             # noqa: E402


def dump(path, label, a, b, step):
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    c = GLM53v6(Xu.shape[1])
    n = len(tu)
    Y = np.empty(n); Asum = np.zeros(n); gs = np.zeros(n); ld = np.zeros(n, int)
    for i in range(n):
        out = c.process(tu[i], Xu[i])
        Y[i] = out.sum(); Asum[i] = c.A.sum(); gs[i] = c.g; ld[i] = int(c.loaded.sum())
    print("=" * 118)
    print(f"{label}")
    for t, k in c.kind_log:
        print(f"      @{t:8.2f}s {k}")
    print(f"{'t':>8}{'raw':>11}{'v6':>11}{'扣除':>9}{'A.sum':>10}{'g':>9}{'ld':>4}"
          f"{'A加载和':>10}{'隐含A·g':>10}")
    i0 = int(np.searchsorted(tu, a)); i1 = int(np.searchsorted(tu, b))
    for i in range(i0, min(i1, n), max(1, int(step / dtm))):
        ded = tot[i] - Y[i]
        Ald = float(c.A[c.loaded].sum()) if ld[i] else 0.0
        print(f"{tu[i]:8.2f}{tot[i]:11.2f}{Y[i]:11.2f}{ded:9.2f}{Asum[i]:10.2f}"
              f"{gs[i]:9.4f}{ld[i]:4d}{Ald:10.2f}{Ald*gs[i]:10.2f}")


dump(os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"),
     "右拇指/数据2 —— 长保压（每 5 s）", 20, 159.6, 5.0)
dump(os.path.join(TEMP, "变化负载", "切换负载-快相无责的测试",
                  "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"),
     "切换负载 —— 133~186 s（每 2 s）", 133, 186, 2.0)
