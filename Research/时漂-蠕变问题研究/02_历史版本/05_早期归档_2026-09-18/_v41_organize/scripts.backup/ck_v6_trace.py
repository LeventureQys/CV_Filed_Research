# -*- coding: utf-8 -*-
"""v6 原型单事件追踪：打印事件生命周期 + 逐 0.1 s 的 Â / 修正量 / 目标。"""
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

CASES = [
    ("右拇指_1", os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"), 8.2),
    ("中途13ffca", os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                                "最终测试目标", "device_001_seg000.csv"), 8.3),
    ("中途13ffca-变载", os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv"), 18.7),
]

for tag, path, t_zoom in CASES:
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    c = GLM53v6(Xu.shape[1])
    c.debug = True
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    print("=" * 110)
    print(f"{tag}   事件 {len(c.epoch_t)}  撤销 {c.n_revoke}  A_peak {c.A_peak:.3f}")
    for r in c.dbg:
        tag = r[1]
        nums = "  ".join(f"{x:12.4g}" for x in r[3:])
        kind = r[2] if isinstance(r[2], str) else f"{r[2]:.4g}"
        print(f"   {r[0]:8.3f}  {tag:8s} {kind:16s} {nums}")
    # 放大窗
    i0 = int(np.searchsorted(tu, t_zoom))
    p5 = float(np.median(tot[i0 + int(4.6 / dtm):i0 + int(5.4 / dtm)]))
    print(f"   真实 5s 电平 = {p5:.3f}（事件前级 {np.median(tot[i0-int(1.5/dtm):i0-int(0.3/dtm)]):.3f}）")
    print("   +t(s)     原始      v6")
    for k in range(0, int(3.0 / dtm) + 1, max(1, int(0.10 / dtm))):
        i = min(i0 + k, len(tu) - 1)
        print(f"   {tu[i]-tu[i0]:6.2f}  {tot[i]:9.3f}  {Y[i].sum():9.3f}")
