# -*- coding: utf-8 -*-
"""快速诊断：恒载段各通道的形态（是否单调上升？主通道是什么样子？）"""
import json
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
ROOT = os.path.dirname(TEMP)
RES = os.path.join(OUT, "results")


def load(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def find_segment(total, frac=0.15):
    thr = frac * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if loaded[0]:
        s = np.r_[0, s]
    if loaded[-1]:
        e = np.r_[e, len(loaded)]
    segs = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)
    return segs[0]


lines = []
for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"]:
    for name in ["数据1", "数据2", "数据3"]:
        p = os.path.join(TEMP, loc, name, "device_001_seg000.csv")
        t, X = load(p)
        s0, s1 = find_segment(X.sum(axis=1))
        u = t[s0:s1] - t[s0]
        amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
        main = int(np.argmax(amp))
        loaded = amp > 0.10 * amp.max()
        nL = s1 - s0
        # 每个受载通道的漂移
        dr = X[s0:s1][-nL // 10:].mean(axis=0) - X[s0:s1][:nL // 10].mean(axis=0)
        pct = 100 * dr[loaded] / amp[loaded]
        lines.append(f"\n=== {loc}/{name}  时长 {t[-1]:.1f}s 负载段 {u[-1]:.1f}s "
                     f"通道 {X.shape[1]} 受载 {loaded.sum()} 主通道 ch{main} amp={amp[main]:.4f}")
        lines.append(f"  受载通道漂移%: 中位 {np.median(pct):+.1f}  均值 {pct.mean():+.1f} "
                     f"正漂移通道数 {(pct > 2).sum()}/{len(pct)}  负漂移 {(pct < -2).sum()}")
        lines.append(f"  主通道 ch{main} 负载段: 首 {X[s0:s0+nL//10, main].mean():.4f} "
                     f"→ 末 {X[s1-nL//10:s1, main].mean():.4f}  (漂移 {100*dr[main]/amp[main]:+.1f}%)")
        # 候选蠕变通道：amp 大且漂移为正
        cand = [i for i in np.where(loaded)[0] if dr[i] > 0]
        if cand:
            prof = X[s0:s1][:, cand].mean(axis=1) - X[s0:s1][:, cand].mean(axis=1)[0]
            dtm = np.median(np.diff(t)[np.diff(t) > 0])
            k10 = min(len(prof) - 1, int(10.0 / dtm))
            lines.append(f"  仅取正漂移通道({len(cand)}个)平均曲线: 首 {prof[0]:+.4f} "
                         f"10s {prof[k10]:+.4f} 末 {prof[-1]:+.4f}")

with open(os.path.join(RES, "_shape_diag.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("saved")
