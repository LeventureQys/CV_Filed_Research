# -*- coding: utf-8 -*-
"""检查总量 Σ 的形态：确认加载/卸载沿位置与尾部是否真正回到 0。"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, DATASETS

for name in DATASETS:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1 / k
    X = D["X"]
    tot = X.sum(axis=1)
    n = len(tot)
    seg = detect_segments(tot, t, fs=fs)
    rl = seg["raw_load"]
    w = lambda x: int(round(x * fs))
    print("=" * 90)
    print(f"### {name}  Σ 单位=N   前0.5s中位={np.median(tot[:w(0.5)]):.4f}N  "
          f"峰值={tot.max():.4f}N")
    print(f"    上升沿 @{t[rl[0]]:.2f}s   下降沿 @{t[rl[1]]:.2f}s   "
          f"fall_found={seg['fall_found']}")
    print(f"    分段: 前 0~{t[seg['pre'][1]]:.2f}s | "
          f"负载 {t[seg['load'][0]]:.2f}~{t[seg['load'][1]]:.2f}s | "
          f"后 {t[seg['post'][0]]:.2f}~{t[-1]:.2f}s")
    print("    关键点 Σ 值 (N):")
    for lab, i in [("前空载末", seg['pre'][1] - 1),
                   ("上升沿", rl[0]),
                   ("+1s", min(rl[0] + w(1), n - 1)),
                   ("+3s", min(rl[0] + w(3), n - 1)),
                   ("+10s", min(rl[0] + w(10), n - 1)),
                   ("下降沿-1s", max(0, rl[1] - w(1))),
                   ("下降沿", rl[1] if rl[1] < n else n - 1),
                   ("下降沿+1s", min(rl[1] + w(1), n - 1)),
                   ("末尾", n - 1)]:
        print(f"      {lab:>10s} t={t[i]:8.2f}s  Σ={tot[i]:9.4f}N")
    # 尾部 10 秒分段均值
    print("    尾部 10s 每段均值:")
    for a_ in range(10, 0, -2):
        i0 = n - w(a_); i1 = n - w(a_ - 2)
        print(f"      最后 {a_:2d}~{a_-2:2d}s : {tot[i0:i1].mean():9.5f}N")
    # 下降沿附近细看
    print("    下降沿附近 (每 0.1s):")
    for d_ in range(-5, 11):
        i = rl[1] + w(d_ * 0.1)
        if 0 <= i < n:
            print(f"      t-t_fall={d_*0.1:+.1f}s  Σ={tot[i]:9.5f}N")
