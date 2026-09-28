# -*- coding: utf-8 -*-
"""v6c 与 v5.1 逐段扣除对照（临时脚本）：看交接点、交接前后扣除跳变、末端 g。"""
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
P = os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc",
                 "device_001_seg000.csv")

d = L.prep(P)
tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
tot_s = L.med_smooth(tot, 0.5 / dtm)

for nm, cls in [("v5.1", GLM53v51), ("v6c", GLM53v6c)]:
    c = cls(Xu.shape[1])
    D = np.empty(len(tu))
    H = []
    for i in range(len(tu)):
        y = c.process(tu[i], Xu[i])
        D[i] = (Xu[i] - y).sum()
        H.append((float(tu[i]), float(c.g), float(c.A.max()), float(c.fast_done),
                  float(c.in_load), float(getattr(c, "epoch_ramp", 0.0)),
                  float(c.shape_ratio) if hasattr(c, "shape_ratio") else 1.0))
    H = np.array(H)
    print("=" * 100)
    print(f"[{nm}] epoch={len(getattr(c, 'epoch_t', []))}  g_end={c.g:+.4f}  A_max={c.A.max():.0f}")
    print(f"  末端扣除中位 = {np.median(D[tu > tu[-1] - 20]):.0f}  "
          f"末端显示中位 = {np.median((Xu - 0).sum(axis=1)[tu > tu[-1] - 20]) - np.median(D[tu > tu[-1] - 20]):.0f}")
    for tt in (30, 40, 50, 60, 61, 62, 63, 64, 65, 70, 80, 90, 100, 110, 120):
        if tt > tu[-1]:
            continue
        j = int(np.searchsorted(tu, tt))
        print(f"    t={tt:6.1f}s  g={H[j,1]:+.4f}  A={H[j,2]:7.0f}  ded={D[j]:8.1f}  "
              f"fd={H[j,3]:.0f} inL={H[j,4]:.0f} ramp={H[j,5]:.2f} ratio={H[j,6]:.3f}")

# 交接帧附近的扣除跳变（v6c）
c = GLM53v6c(Xu.shape[1])
D6 = np.empty(len(tu))
hs = None
for i in range(len(tu)):
    y = c.process(tu[i], Xu[i])
    D6[i] = (Xu[i] - y).sum()
    if c.handoff_ts > 0 and hs is None:
        hs = c.handoff_ts
print("=" * 100)
print(f"v6c 首个交接 @{hs}")
if hs:
    j = int(np.searchsorted(tu, hs))
    print("  交接前后逐帧总扣除：")
    for k in range(j - 6, j + 6):
        if 0 <= k < len(tu):
            print(f"    t={tu[k]:7.2f}s  扣除={D6[k]:9.3f}  Δ={D6[k]-D6[k-1]:+8.3f}")
