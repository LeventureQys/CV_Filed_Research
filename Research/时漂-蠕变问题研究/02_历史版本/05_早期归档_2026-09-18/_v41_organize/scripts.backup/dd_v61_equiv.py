# -*- coding: utf-8 -*-
"""等价性自检：把 v6.1 的四处修复关掉后，必须与 v6 逐帧一致（保证 A/B 是同一套机器）。

用法：python dd_v61_equiv.py [数据索引...]
"""
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

idx = [int(x) for x in (sys.argv[1:] or [9, 0, 12])]
worst = 0.0
for i in idx:
    tag, path = C.ALL[i]
    d = L.prep(path)
    tu, Xu = d["tu"], d["Xu"]
    Ys = []
    for cls, kw in ((GLM53v6, {}),
                    (GLM53v61, dict(ROM_SCALE=1.0, RATE_DOWN=0.0,
                                    STALL_HOLD_S=0.45, STALL_TAIL_KEEP=0.0))):
        c = cls(Xu.shape[1])
        for k, val in kw.items():
            setattr(c, k, val)
        Y = np.empty_like(Xu)
        for j in range(len(tu)):
            Y[j] = c.process(tu[j], Xu[j])
        Ys.append((Y, c))
    dd = float(np.abs(Ys[0][0] - Ys[1][0]).max())
    worst = max(worst, dd)
    print(f"{tag:>18}  逐帧最大差 = {dd:.3e}   epoch {len(Ys[0][1].epoch_t)} vs {len(Ys[1][1].epoch_t)}"
          f"   kindlog 一致 = {Ys[0][1].kind_log == Ys[1][1].kind_log}")
print(f"\n最大差 = {worst:.3e}  ⇒ {'一致（A/B 可比）' if worst < 1e-9 else '不一致，需要核对覆写代码'}")
