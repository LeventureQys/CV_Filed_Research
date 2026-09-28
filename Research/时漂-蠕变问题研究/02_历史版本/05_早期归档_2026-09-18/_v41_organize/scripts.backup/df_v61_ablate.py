# -*- coding: utf-8 -*-
"""F1/F2/F3 的分项归因（ablation）：每档只关掉一处修复，看指标怎么动。

档位：
  v6               基线（= v6.1 四处修复全关，逐帧 0 差已验证）
  v61              **最终配置**：F1(λ=1.06, onset) + F2(RATE_DOWN=0.25) + F3(STALL_HOLD=0.35)
  v61_noF1         关 F1（λ=1.0）—— 其余同最终
  v61_noF2         关 F2（RATE_DOWN=0）—— 其余同最终
  v61_noF3         关 F3（STALL_HOLD=0.45）—— 其余同最终
输出：尖峰口径（超调/回落）按族与关键事件汇总，用于 `Document/08-v6.1算法说明.md` §3.1 的归因表。
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

BASE = dict(ROM_SCALE=1.06, RATE_DOWN=0.25, STALL_HOLD_S=0.35,
            STALL_TAIL_KEEP=0.0, CONSERVATIVE_ONSET_ONLY=True)
ARMS = [("v6", GLM53v6, {}),
        ("v61", GLM53v61, dict(BASE)),
        ("v61_noF1", GLM53v61, dict(BASE, ROM_SCALE=1.0)),
        ("v61_noF2", GLM53v61, dict(BASE, RATE_DOWN=0.0)),
        ("v61_noF3", GLM53v61, dict(BASE, STALL_HOLD_S=0.45)),
        ("v61_noF2F3", GLM53v61, dict(BASE, RATE_DOWN=0.0, STALL_HOLD_S=0.45))]
KEY = {("中途切换-1d9493", 20.65), ("切换负载-快相无责", 185.98),
       ("切换负载-快相无责", 133.84), ("中途切换-13ffca", 61.02)}


def ev_rows(tag, kind, tu, dtm, c, rs, ys, peak):
    n = len(tu)
    out = []
    for t0, kd in c.kind_log:
        i0 = int(np.searchsorted(tu, t0))
        pre = float(np.median(rs[max(0, i0 - int(0.3 / dtm)):max(1, i0)]))
        a, b = min(n - 1, i0 + int(4.6 / dtm)), min(n, i0 + int(5.4 / dtm))
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
        cut = np.where(rs[i0 + k:e] < 0.7 * P)[0]
        if len(cut):
            e = i0 + k + int(cut[0])
        out.append(dict(dataset=tag, kind=kind, t0=round(t0, 2), ev=kd, step=step,
                        over=100 * (seg[k] - P) / step,
                        trans=100 * (seg[k] - min(ys[i0 + k:e])) / step))
    return out


rows = []
for tag, path in C.ALL:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    raw = Xu.sum(axis=1)
    rs = L.med_smooth(raw, 0.5 / dtm)
    peak = float(rs.max())
    for name, cls, kw in ARMS:
        c = cls(Xu.shape[1])
        for k, v in kw.items():
            setattr(c, k, v)
        Y = np.empty_like(Xu)
        for j in range(len(tu)):
            Y[j] = c.process(tu[j], Xu[j])
        ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        for r in ev_rows(tag, C.KIND[tag], tu, dtm, c, rs, ys, peak):
            r["arm"] = name
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "v61_ablation.csv"), index=False, encoding="utf-8-sig")
order = [a for a, _, _ in ARMS]
print("=" * 118)
print("分项归因：尖峰口径（占阶跃 %）")
print("=" * 118)
print(df.groupby(["kind", "arm"]).agg(超调中位=("over", "median"), 超调max=("over", "max"),
                                      超调min=("over", "min"), 回落max=("trans", "max")
                                      ).reindex(pd.MultiIndex.from_product(
                                          [["实采", "恒载"], order]), ).round(2).to_string())
print("\n关键事件（超调 / 回落，占阶跃 %；v6 → 各档）：")
for ds, t0 in sorted(KEY):
    s = df[(df.dataset == ds) & (np.abs(df.t0 - t0) < 0.05)]
    if not len(s):
        continue
    s = s.set_index("arm").reindex(order)
    print(f"  {ds:>18} t0={t0:7.2f} {s.ev.dropna().iloc[0] if s.ev.notna().any() else '':<14s} "
          f"step={s.step.dropna().iloc[0]:8,.0f} | 超调 " +
          " ".join(f"{a}={s.loc[a, 'over']:+6.1f}" if pd.notna(s.loc[a, "over"]) else f"{a}=  n/a"
                   for a in order) +
          " | 回落 " + " ".join(f"{s.loc[a, 'trans']:+7.1f}" if pd.notna(s.loc[a, "trans"]) else "   n/a"
                                for a in order))
print(f"\n产物：results/v61_ablation.csv")
