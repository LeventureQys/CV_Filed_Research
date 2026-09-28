# -*- coding: utf-8 -*-
"""追轨迹：卸载沿之后 v5 为什么比 v3 更久地把显示压在 ≤0（零点塌陷）。

输出：u / pending / hold / in_load / A_max / g / 扣除量 / 显示总量 的逐帧（抽样）轨迹。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
PATH = os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")
T0, T1 = 58.5, 63.9

d = L.prep(PATH)
tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
tot_s = L.med_smooth(tot, 0.5 / dtm)
peak = float(tot_s.max())
print(f"峰值 {peak:.0f}；窗口 {T0}~{T1}s")


def trace(name, c):
    ded_prev = np.zeros(c.n)
    print(f"\n===== {name} =====")
    print("   t     raw_tot  显示总量  扣除总量  in_load pending hold  u(s)  A_max    g     事件")
    ev_last = None
    for i in range(len(tu)):
        ts = tu[i]
        before = dict(in_load=c.in_load, onset=c.onset_ts, pending=c.pending)
        Y = c.process(ts, Xu[i])
        if not (T0 <= ts <= T1):
            continue
        s = float(Y.sum())
        raw = float(tot[i])
        ded = raw - s
        ev = ""
        if not before["in_load"] and c.in_load:
            ev = "★BEGIN(空载→负载)"
        elif before["onset"] != c.onset_ts and c.in_load:
            ev = "◆RESTEP(变载重捕获)"
        elif before["in_load"] and not c.in_load:
            ev = "○UNLOAD(判定卸载)"
        if i % 10 == 0 or ev:
            print(f"{ts:7.2f} {raw:9.0f} {s:9.0f} {ded:9.0f}   {str(c.in_load):5s} "
                  f"{str(c.pending):5s} {str(c.hold):5s} "
                  f"{(ts-c.onset_ts if c.in_load else float('nan')):6.2f} "
                  f"{c.A.max():7.0f} {c.g:7.3f}  {ev}")


trace("v3", GLM53v3(Xu.shape[1]))
trace("v5", GLM53v5(Xu.shape[1]))
