# -*- coding: utf-8 -*-
"""步骤6：诊断评测脚本中的异常（阶跃定义、t_ref 对齐、模型拟合病态）。"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, DATASETS

np.set_printoptions(precision=5, suppress=True)

for name in DATASETS:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), D["t"], fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    base = X[a:bb].mean(axis=0)
    Xn = X - base
    resp = Xn[c0:d0].max(axis=0)
    act = np.where(resp > 0.05)[0]
    main = int(act[np.argmax(resp[act])])
    y = Xn[:, main]
    w = lambda s: int(round(s * fs))
    print("=" * 90)
    ch = D["ch_cols"]
    print(f"### {name}  主通道 {ch[main]}  负载起点帧={c0} (t={t[c0]:.3f}s)")
    print(f"    前空载均值(原始X)={base[main]:.5f} N   = {base[main]*1000:.3f} mN")
    print("    阶跃附近逐段均值 (相对前空载基线, mN):")
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.15),
                   (0.15, 0.30), (0.30, 0.50), (0.5, 1.0), (1.0, 2.0),
                   (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, 60.0)):
        s0 = c0 + w(lo); s1 = c0 + w(hi)
        if s1 > d0:
            break
        print(f"      [{lo:5.2f},{hi:6.2f})s : {y[s0:s1].mean()*1000:9.3f} mN"
              f"   (std={y[s0:s1].std()*1000:6.3f})")
    print(f"    负载末5s={y[d0-w(5):d0].mean()*1000:.3f} mN   负载最大={resp[main]*1000:.3f} mN")
    # 总通道
    tot = Xn.sum(axis=1)
    print(f"    Σ通道: 前空载均值={tot[a:bb].mean()*1000:.3f} mN  "
          f"负载首0.5s={tot[c0:c0+w(0.5)].mean()*1000:.3f}  "
          f"负载末5s={tot[d0-w(5):d0].mean()*1000:.3f} mN")
    # 负载起点的定位精度：看 Σ 在 c0 附近
    print("    Σ 在负载起点附近 (±0.3s 每 0.05s):")
    lo = c0 - w(0.3)
    for i in range(12):
        s = lo + i * w(0.05)
        print(f"      t={t[s]-t[c0]:+.3f}s  Σ={tot[s]*1000:9.3f} mN")
    # 关键：0~0.15s 的上升 vs 1~2s 的值
    r015 = y[c0:c0 + w(0.15)].mean()
    v12 = y[c0 + w(1.0):c0 + w(2.0)].mean()
    print(f"    >>> 0~0.15s={r015*1000:.3f} mN, 1~2s={v12*1000:.3f} mN, 比值={r015/v12:.3f}")
    print(f"    >>> 末5s={y[d0-w(5):d0].mean()*1000:.3f} mN, 相对1~2s={100*(y[d0-w(5):d0].mean()/v12-1):.2f}%")
