# -*- coding: utf-8 -*-
"""逐事件台账：v6 / v6.1(λ=1.06) / v6.1(λ=1.08) 在每个真阶跃上的"超调 / 5 s 残留 / 瞬态回落"。
用于 `Document/08-v6.1算法说明.md` §4.1。只列 step ≥ 5% × 录制峰值的真阶跃。
"""
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

ARMS = [("v6", GLM53v6, {}),
        ("v61_1.06", GLM53v61, dict(ROM_SCALE=1.06, RATE_DOWN=0.25, CONSERVATIVE_ONSET_ONLY=True)),
        ("v61_1.08", GLM53v61, dict(ROM_SCALE=1.08, RATE_DOWN=0.25, CONSERVATIVE_ONSET_ONLY=True)),
        ("v61_1.10", GLM53v61, dict(ROM_SCALE=1.10, RATE_DOWN=0.25, CONSERVATIVE_ONSET_ONLY=True))]

rows = []
for tag, path in C.ALL:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    raw_inst = Xu.sum(axis=1)
    rs = L.med_smooth(raw_inst, 0.5 / dtm)
    peak = float(rs.max())
    for name, cls, kw in ARMS:
        c = cls(Xu.shape[1])
        for k, v in kw.items():
            setattr(c, k, v)
        Y = np.empty_like(Xu)
        for j in range(len(tu)):
            Y[j] = c.process(tu[j], Xu[j])
        ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        n = len(tu)
        for t0, kd in c.kind_log:
            i0 = int(np.searchsorted(tu, t0))
            pre = float(np.median(rs[max(0, i0 - int(0.3 / dtm)):max(1, i0)]))
            a = min(n - 1, i0 + int(4.6 / dtm))
            b = min(n, i0 + int(5.4 / dtm))
            if b - a < 3:
                continue
            P = float(np.median(rs[a:b]))
            step = P - pre
            if step < 0.05 * peak:
                continue
            j5 = min(n - 1, i0 + int(5.0 / dtm))
            seg = ys[i0:j5 + 1]
            k = int(np.argmax(seg))
            e = min(n, i0 + k + int(6.0 / dtm))
            # 回落窗遇卸载沿即截止：用**原始瞬时值**判"载荷在卸"（掉到窗内峰值 85% 以下），
            # 因为 0.5 s 中值平滑会让卸载沿滞后，晚一帧就会把"显示跟着掉到 0"算进回落
            win = raw_inst[i0 + k:e]
            if len(win):
                cut = np.where(win < 0.85 * win.max())[0]
                if len(cut):
                    e = max(i0 + k + int(cut[0]), i0 + k + 1)
            rows.append(dict(dataset=tag, kind=C.KIND[tag], t0=round(t0, 2), ev=kd,
                             step=round(step), arm=name,
                             over_peak=100 * (seg[k] - P) / step,
                             over_5s=100 * (ys[j5] - P) / step,
                             transient=100 * (seg[k] - min(ys[i0 + k:e])) / step))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "v61_event_ledger.csv"), index=False, encoding="utf-8-sig")

for kind in ("实采", "恒载"):
    s = df[df.kind == kind]
    if not len(s):
        continue
    print("=" * 112)
    print(f"{kind} · 每事件 超调 / 5 s 残留 / 瞬态回落（占阶跃 %）")
    print("=" * 112)
    pv = s.pivot_table(index=["dataset", "t0", "ev", "step"], columns="arm",
                       values=["over_peak", "over_5s", "transient"])
    arms = [a for a, _, _ in ARMS]
    for (ds, t0, ev, st), r in pv.iterrows():
        print(f"  {ds:>18} t0={t0:7.2f} {ev:<14s} step={st:8,.0f} | 超调 " +
              " → ".join(f"{r[('over_peak', a)]:+7.1f}" for a in arms) +
              " | 残留5s " + " → ".join(f"{r[('over_5s', a)]:+6.1f}" for a in arms) +
              " | 回落 " + " → ".join(f"{r[('transient', a)]:+7.1f}" for a in arms))
    print()
print("汇总（中位 / 最大，占阶跃 %）：")
print(df.groupby(["kind", "arm"]).agg(超调中位=("over_peak", "median"), 超调max=("over_peak", "max"),
                                      回落max=("transient", "max"),
                                      残留中位=("over_5s", "median")).round(2).to_string())
print(f"\n产物：results/v61_event_ledger.csv")
