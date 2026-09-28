# -*- coding: utf-8 -*-
"""中途切换-13ffca 的分项归因（含**小台阶**事件，不做 step 过滤）：v6 / v6.1 / 关F1 / 关F2。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402
from glm53_v61 import GLM53v61                              # noqa: E402

BASE = dict(ROM_SCALE=1.06, RATE_DOWN=0.25, CONSERVATIVE_ONSET_ONLY=True)
ARMS = [("v6", GLM53v6, {}),
        ("v6.1", GLM53v61, dict(BASE)),
        ("关F1", GLM53v61, dict(BASE, ROM_SCALE=1.0)),
        ("关F2", GLM53v61, dict(BASE, RATE_DOWN=0.0))]

tag, path = [x for x in C.ALL if x[0] == "中途切换-13ffca"][0]
d = L.prep(path)
tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
raw_i = Xu.sum(axis=1)
rs = L.med_smooth(raw_i, 0.5 / dtm)

rows = []
for name, cls, kw in ARMS:
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for j in range(len(tu)):
        Y[j] = c.process(tu[j], Xu[j])
    ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    for t0, kd in c.kind_log:
        i0 = int(np.searchsorted(tu, t0))
        pre = float(np.median(rs[max(0, i0 - int(0.3 / dtm)):max(1, i0)]))
        a, b = min(len(tu) - 1, i0 + int(4.6 / dtm)), min(len(tu), i0 + int(5.4 / dtm))
        if b - a < 3:
            continue
        P = float(np.median(rs[a:b]))
        step = P - pre
        j5 = min(len(tu) - 1, i0 + int(5.0 / dtm))
        seg = ys[i0:j5 + 1]
        k = int(np.argmax(seg))
        e = min(len(tu), i0 + k + int(6.0 / dtm))
        win = raw_i[i0 + k:e]
        if len(win):
            cut = np.where(win < 0.85 * win.max())[0]
            if len(cut):
                e = max(i0 + k + int(cut[0]), i0 + k + 1)
        rows.append(dict(t0=round(t0, 2), ev=kd, step=round(step), arm=name,
                         over=100 * (seg[k] - P) / step if abs(step) > 1 else np.nan,
                         over_abs=seg[k] - P,
                         trans=100 * (seg[k] - min(ys[i0 + k:e])) / step if abs(step) > 1 else np.nan,
                         peak_raw=seg[k]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "v61_13ffca_ablation.csv"), index=False, encoding="utf-8-sig")
pv = df.pivot_table(index=["t0", "ev", "step"], columns="arm",
                    values=["over", "peak_raw", "trans"])
print("中途切换-13ffca · 每事件（含小台阶，不做 step 过滤）")
print("  超调/回落 单位 = 占该次阶跃 %；peak_raw = 显示峰值 ADC")
hdr = f"{'t0':>7} {'ev':<14} {'step':>8} | " + " | ".join(
    f"{a:>22}" for a in [x[0] for x in ARMS])
print(hdr)
for (t0, ev, st), r in pv.iterrows():
    cells = []
    for a in [x[0] for x in ARMS]:
        o, pk, tr = r[("over", a)], r[("peak_raw", a)], r[("trans", a)]
        cells.append(f"超{o:+7.1f}% 峰{pk:9,.0f}")
    print(f"{t0:7.2f} {ev:<14} {st:8,.0f} | " + " | ".join(cells))
print("\n回落（占阶跃 %）：")
for (t0, ev, st), r in pv.iterrows():
    print(f"  {t0:7.2f} {ev:<14} " + "  ".join(
        f"{a}={r[('trans', a)]:+7.1f}" for a in [x[0] for x in ARMS]))
