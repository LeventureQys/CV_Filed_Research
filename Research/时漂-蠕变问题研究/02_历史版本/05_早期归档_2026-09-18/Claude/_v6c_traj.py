# -*- coding: utf-8 -*-
"""显示轨迹对照（临时脚本）：v5.1 vs v6c 在 1d9493 首个加载沿后的显示电平与扣除量。"""
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
P = os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")
d = L.prep(P)
tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
tot_s = L.med_smooth(tot, 0.5 / dtm)

out = {}
for nm, cls in [("v5.1", GLM53v51), ("v6c", GLM53v6c)]:
    c = cls(Xu.shape[1])
    Y = np.empty_like(Xu)
    G = np.empty(len(tu))
    A = np.empty(len(tu))
    FD = np.empty(len(tu))
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
        G[i] = c.g
        A[i] = c.A.max()
        FD[i] = 1.0 if c.fast_done else 0.0
    out[nm] = (c, Y, G, A, FD)
    print(f"[{nm}] epoch_t={[round(x,2) for x in getattr(c,'epoch_t',[])]} "
          f"g_end={c.g:+.4f} A={c.A.max():.0f}")

c5, Y5, G5, A5, FD5 = out["v5.1"]
c6, Y6, G6, A6, FD6 = out["v6c"]
totY5, totY6 = Y5.sum(axis=1), Y6.sum(axis=1)
print()
print("首个加载沿在 t≈10.91s；下面看 10~30s")
print(f"{'t':>7} {'raw':>8} {'v5.1显示':>9} {'v5.1扣':>8} {'g':>7} {'fd':>3} | "
      f"{'v6c显示':>9} {'v6c扣':>8} {'g':>7} {'fd':>3} {'ramp':>5}")
for tt in np.arange(10.0, 30.01, 0.5):
    j = int(np.searchsorted(tu, tt))
    if j >= len(tu):
        break
    r = getattr(c6, "_dbg_ramp", None)
    print(f"{tt:7.2f} {tot[j]:8.0f} {totY5[j]:9.0f} {tot[j]-totY5[j]:8.1f} {G5[j]:+7.4f} {FD5[j]:3.0f} | "
          f"{totY6[j]:9.0f} {tot[j]-totY6[j]:8.1f} {G6[j]:+7.4f} {FD6[j]:3.0f}")

# 终值对照
print()
for nm, ty in [("v5.1", totY5), ("v6c", totY6)]:
    print(f"  {nm}: t=25s显示={ty[int(np.searchsorted(tu,25))]:.0f}  "
          f"t=29s={ty[int(np.searchsorted(tu,29))]:.0f}  "
          f"段末(20~30s)中位={np.median(ty[int(np.searchsorted(tu,20)):int(np.searchsorted(tu,30))]):.0f}")
