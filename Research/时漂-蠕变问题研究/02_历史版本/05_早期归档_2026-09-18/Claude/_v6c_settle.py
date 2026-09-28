# -*- coding: utf-8 -*-
"""稳定时间口径（临时脚本）：真实加载沿 → 显示进入并停留在终值 ±tol 的时刻。

终值 = 该负载段末 10s 的显示中位；tol 取该段幅度的一定比例。
同时给出"过充量"（显示超出终值的最大值），验证 v6 的过充病是否被治好。
"""
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
from glm53_v6 import GLM53v6                           # noqa: E402
from glm53_v6c import GLM53v6c                         # noqa: E402

B = os.path.join(os.path.dirname(os.path.dirname(SCRIPTS)), "变化负载")
RECS = [("快相无责-ee20bc", os.path.join(B, "切换负载-快相无责的测试",
                                         "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]


def settle_time(t, y, tot_s, e, seg_end, tol_frac=0.02):
    """进入并保持在终值 ±tol 的时刻（相对加载沿 e）。"""
    j0 = int(np.searchsorted(t, e))
    j1 = int(np.searchsorted(t, seg_end))
    if j1 - j0 < int(5.0 / (t[1] - t[0])):
        return float("nan"), float("nan"), float("nan")
    amp = float(np.median(tot_s[j1 - 100:j1]) - np.median(tot_s[max(0, j0 - 200):j0]))
    final = float(np.median(y[j1 - 100:j1]))
    tol = tol_frac * abs(amp) if abs(amp) > 1e-9 else 1.0
    inside = np.abs(y[j0:j1] - final) <= tol
    # 从后往前找最后一个"越界"点，其后即稳定
    out = np.where(~inside)[0]
    if len(out) == 0:
        return 0.0, abs(float(y[j0:j1].max() - final)), abs(amp)
    k = int(out[-1]) + 1
    if k >= len(inside):
        return float("nan"), abs(float(y[j0:j1].max() - final)), abs(amp)
    return float(t[j0 + k] - e), max(0.0, float(y[j0:j1].max() - final)), abs(amp)


print("=" * 118)
print(f"{'录制':>18} {'算法':>6} {'加载沿s':>9} {'台阶':>8} {'稳定s':>7} {'过充ADC':>9} {'过充%':>7} "
      f"{'稳定s/台阶':>10}")
print("-" * 118)
agg = {}
for tag, path in RECS:
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    ev = L.detect_events(tot, dtm)
    periods = L.find_periods(tot, dtm)
    for nm, cls in [("v5.1", GLM53v51), ("v6c", GLM53v6c), ("v6", GLM53v6)]:
        c = cls(Xu.shape[1])
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        rows = []
        for e, _ in ev:
            seg_end = next((b for a, b in periods if a <= e < b), len(tu) - 1)
            if seg_end - e < int(8.0 / dtm):
                continue
            st, over, amp = settle_time(tu, y, tot_s, tu[e], tu[seg_end])
            if not np.isfinite(st):
                continue
            rows.append((tu[e], amp, st, over))
        agg.setdefault(nm, []).extend(rows)
        if rows:
            med = float(np.median([r[2] for r in rows]))
            print(f"  {tag:>16} {nm:>6} " + f"{rows[0][0]:9.2f} {rows[0][1]:8.0f} "
                  f"{rows[0][2]:7.2f} {rows[0][3]:9.0f} {100*rows[0][3]/max(rows[0][1],1):7.2f} "
                  f"{rows[0][2]:10.2f}")
            for r in rows[1:]:
                print(f"  {'':>16} {'':>6} " + f"{r[0]:9.2f} {r[1]:8.0f} "
                      f"{r[2]:7.2f} {r[3]:9.0f} {100*r[3]/max(r[1],1):7.2f} {r[2]:10.2f}")

print("=" * 118)
print("汇总（所有 >=8s 的负载段）：")
for nm in ("v5.1", "v6c", "v6"):
    rs = agg.get(nm, [])
    if not rs:
        continue
    st = np.array([r[2] for r in rs])
    ov = np.array([100 * r[3] / max(r[1], 1.0) for r in rs])
    print(f"  {nm:5s} n={len(rs):2d}  稳定时间 中位={np.median(st):6.2f}s 均值={st.mean():6.2f}s "
          f"最大={st.max():6.2f}s  |  过充% 中位={np.median(ov):5.2f} 最大={ov.max():6.2f}")
