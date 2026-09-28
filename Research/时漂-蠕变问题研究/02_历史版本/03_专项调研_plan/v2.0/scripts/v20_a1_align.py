# -*- coding: utf-8 -*-
"""v2.0 A1 · main 流与 pre 流的帧对齐自检（只读）。

检查：同 packet（同 timestamp）内 pre 与 main 是否逐帧对应；零偏移帧是不是
「算法输出未被应用」的帧；按分组内重排（倒序）后 main 是否与「某一种帧序」自洽。
"""
import sys
import os
import collections

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import v20_lib as L          # noqa: E402

ds = L.load_dataset(L.DS_ZERO)
pre, main = ds["pre"], ds["main"]
tp, tm = pre["V"].sum(1), main["V"].sum(1)
ts = pre["ts"]
off = tm - tp
el = pre["el"]
print(f"frames={len(ts)}")
print(f"elapsed 唯一值数={len(np.unique(el))} timestamp 唯一值数={len(np.unique(np.round(ts,6)))}")

groups = []
cur = [0]
for i in range(1, len(ts)):
    if abs(ts[i] - ts[i - 1]) < 1e-9:
        cur.append(i)
    else:
        groups.append(cur)
        cur = [i]
groups.append(cur)

sizes = collections.Counter(len(g) for g in groups)
print(f"同 timestamp 分组: {len(groups)} 个，大小分布={dict(sorted(sizes.items()))}")

zero = np.abs(off) < 1e-9
print(f"off==0 帧数={int(zero.sum())}（{100*zero.mean():.2f}%）")

# 零偏移帧的同分组邻居是否也零偏移
both = sum(1 for g in groups if len(g) > 1 and all(zero[i] for i in g))
mixed = sum(1 for g in groups if len(g) > 1 and any(zero[i] for i in g) and not all(zero[i] for i in g))
print(f"多帧分组中「整组零偏移」={both}，「部分零偏移」={mixed}")

# 逐帧检查：main[i] 是否等于 pre[i+1] 之类（错位）
shifts = {}
for s in (-3, -2, -1, 1, 2, 3):
    if s > 0:
        a, b = tm[:-s], tp[s:]
    else:
        a, b = tm[-s:], tp[:s]
    shifts[s] = float(np.median(np.abs(a - b)))
print("main 与 pre 在不同错位下的 |差| 中位（应只有 s=0 最小）:", 
      {k: round(v, 1) for k, v in shifts.items()})
print(f"  s=0: {np.median(np.abs(tm-tp)):.1f}")

# 每个分组内，pre 值集合与 main 值集合的重叠度
same_set = 0
for g in groups:
    if len(g) < 2:
        continue
    if np.allclose(np.sort(tp[g]), np.sort(tm[g]), atol=1e-6):
        same_set += 1
print(f"多帧分组中 pre 集合==main 集合 的个数={same_set}/{sum(1 for g in groups if len(g)>1)}")

# 零偏移帧所在时刻分布
zidx = np.where(zero)[0]
print(f"零偏移帧的时间分布：前 10%={int((el[zidx] < el[-1]*0.1).sum())}，"
      f"后 10%={int((el[zidx] > el[-1]*0.9).sum())}")
print(f"零偏移帧中位于多帧分组内的比例="
      f"{100*np.mean([len(g) > 1 for g in groups for _ in g if False] or [0]):.0f}%")
cnt_multi_zero = sum(1 for g in groups if len(g) > 1 for i in g if zero[i])
print(f"多帧分组内的零偏移帧={cnt_multi_zero}，单帧分组的零偏移帧={int(zero.sum())-cnt_multi_zero}")

# 主指标在不同子集上的稳定性（用回放输出对比）
import csv
fr = list(csv.DictReader(open(os.path.join(os.path.dirname(_HERE), "results",
                                          "a1_replay_frames.csv"), encoding="utf-8")))
proto = np.array([float(r["out_total"]) for r in fr])
d = proto - tm
m = ~zero
print(f"d 中位={np.median(d):.1f} RMS={np.sqrt(np.mean(d**2)):.1f}（全部帧）")
print(f"d 中位={np.median(d[m]):.1f} RMS={np.sqrt(np.mean(d[m]**2)):.1f}（排除 off==0 帧）")
print(f"d 中位={np.median(d[zero]):.1f} RMS={np.sqrt(np.mean(d[zero]**2)):.1f}（仅 off==0 帧）")
sys.exit(0)
