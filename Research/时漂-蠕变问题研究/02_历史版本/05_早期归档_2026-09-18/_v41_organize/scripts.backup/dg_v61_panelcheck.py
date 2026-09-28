# -*- coding: utf-8 -*-
"""L1 四格的数值验收（本会话模型无图像输入能力，用数字代替目视）：每格给出
原始/v6/v6.1 在窗内的峰值与"显示−原始"的极值，确认图上确实能看到差异。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402
from glm53_v61 import GLM53v61                              # noqa: E402

PANELS = [(9, 183.0, 191.0, "① 切换负载 @185.98"),
          (9, 131.5, 138.0, "② 切换负载 @133.84"),
          (10, 55.0, 62.0, "③ 再切换负载 @57.28"),
          (0, 7.0, 17.0, "④ 恒载 右拇指/数据1")]

for i, ta, tb, title in PANELS:
    tag, path = C.ALL[i]
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    rs = L.med_smooth(Xu.sum(axis=1), 0.5 / dtm)
    Ys = {}
    for nm, cls in (("v6", GLM53v6), ("v61", GLM53v61)):
        c = cls(Xu.shape[1])
        Y = np.empty_like(Xu)
        for j in range(len(tu)):
            Y[j] = c.process(tu[j], Xu[j])
        Ys[nm] = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    m = (tu >= ta) & (tu <= tb)
    e6, e1 = Ys["v6"][m] - rs[m], Ys["v61"][m] - rs[m]
    print(f"{title}  [{tag}]  {ta}~{tb}s  窗内 {m.sum()} 帧")
    print(f"    原始   max={rs[m].max():9,.1f}  min={rs[m].min():9,.1f}")
    print(f"    v6     max={Ys['v6'][m].max():9,.1f}   显示−原始 ∈ [{e6.min():+8,.1f}, {e6.max():+8,.1f}]")
    print(f"    v6.1   max={Ys['v61'][m].max():9,.1f}   显示−原始 ∈ [{e1.min():+8,.1f}, {e1.max():+8,.1f}]")
    k = int(np.argmax(np.abs(Ys["v6"][m] - Ys["v61"][m])))
    print(f"    两臂差异最大处 t={tu[m][k]:.2f}s：v6={Ys['v6'][m][k]:,.1f} vs v6.1={Ys['v61'][m][k]:,.1f}"
          f"（差 {Ys['v6'][m][k]-Ys['v61'][m][k]:+,.1f}）")
