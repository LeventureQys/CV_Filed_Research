# -*- coding: utf-8 -*-
"""GLM53 探针 3：结论汇总——切换瞬间的"纯算法跳变"量化

对每个事件，把显示序列与原始序列做差（算法偏移 off = out_total - total），
off 的逐帧跃变即"与真实负载无关的纯算法跳变"。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")

CASES = {"A": [("onset", 14.30), ("unload", 31.91), ("onset", 34.98), ("restep", 40.98),
               ("unload", 51.25), ("onset", 54.32), ("restep", 60.33)],
         "B": [("onset", 13.21), ("restep", 23.83), ("unload", 31.35), ("onset", 34.42),
               ("unload", 37.45), ("onset", 41.59), ("unload", 61.5)]}

print(f"{'数据':<4}{'事件':<8}{'t':>7}{'显示跳变':>11}{'原始同刻':>10}{'纯算法跳变':>12}{'前漂移(2s)':>12}{'后漂移(2s)':>12}")
print("-" * 78)
rows = []
for tag, evs in CASES.items():
    tr = pd.read_csv(os.path.join(RES, f"g_trace_{tag}.csv"))
    keep = np.r_[True, np.diff(tr.ts.values) > 1e-6]
    u = tr[keep].reset_index(drop=True)
    off = (u.out_total - u.total).values
    doff = np.diff(off)
    for name, t0 in evs:
        i = int(np.argmin(np.abs(u.ts.values - t0)))
        i = max(i, 1)
        d_disp = u.out_total.values[i] - u.out_total.values[i - 1]
        d_raw = u.total.values[i] - u.total.values[i - 1]
        pre = (u.out_total[(u.ts > t0 - 2) & (u.ts < t0)].iloc[-1] -
               u.out_total[(u.ts > t0 - 2) & (u.ts < t0)].iloc[0]) if len(u[(u.ts > t0 - 2) & (u.ts < t0)]) > 1 else 0
        post = (u.out_total[(u.ts > t0) & (u.ts < t0 + 2)].iloc[-1] -
                u.out_total[(u.ts > t0) & (u.ts < t0 + 2)].iloc[0]) if len(u[(u.ts > t0) & (u.ts < t0 + 2)]) > 1 else 0
        print(f"{tag:<4}{name:<8}{u.ts.values[i]:>7.2f}{d_disp:>11,.0f}{d_raw:>10,.0f}{d_disp - d_raw:>12,.0f}{pre:>12,.0f}{post:>12,.0f}")
        rows.append(dict(dataset=tag, event=name, t=round(float(u.ts.values[i]), 2),
                         d_disp=d_disp, d_raw=d_raw, algo_jump=d_disp - d_raw,
                         pre2s=pre, post2s=post))
    # 纯算法跳变的全局最大
    j = int(np.argmax(np.abs(doff)))
    print(f"     -> {tag} 全局最大纯算法跳变 {doff[j]:+,.0f} ADC @ t={u.ts.values[j+1]:.2f}s")
pd.DataFrame(rows).to_csv(os.path.join(RES, "g_jump_summary.csv"), index=False, encoding="utf-8-sig")
print("\n(纯算法跳变 = 显示变化 - 原始同刻变化, 正值=显示凭空上跳, 负值=显示凭空下跳)")
