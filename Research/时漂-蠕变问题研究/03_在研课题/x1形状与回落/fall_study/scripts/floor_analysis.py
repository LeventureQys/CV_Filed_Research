# -*- coding: utf-8 -*-
"""量化"快相 <1s 落位"的物理地板：
(1) 机械加载斜坡宽度 ramp_s（输入从离基线到 90% 台阶的时间，任何算法的显示都到不了这之前）
(2) 斜坡顶之后 10s 内的继续蠕变占台阶的比例 creep_frac（要 1s 落位就必须"瞬间"吸收掉的部分）
    —— 若 creep_frac 跨事件稳定，则可用置位型前馈瞬间吃掉；若散布大，则置位必过/欠冲。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, *[".."] * 9))
sys.path.insert(0, os.path.join(REPO, "toolbox", "数据解析工具"))
sys.path.insert(0, HERE)

from dptool.session_csv import load_table  # noqa: E402
from fall_study_run import DATA_ROOT, FPS, medfilt, find_events, find_sessions  # noqa: E402

rows = []
for cond, root in find_sessions():
    t = load_table(os.path.join(root, "device_001_pre_seg0.csv"), use_cache=False)
    ch = np.asarray(t.channels(), dtype=np.float64)
    ch = ch[:, np.isfinite(ch).all(axis=0)]
    ts = t.time_axis("frame_index") / FPS
    tot = medfilt(ch.sum(axis=1))
    events, _ = find_events(ts, tot)
    for i in events:
        pre = np.median(tot[max(0, i - int(1.5 * FPS)):i])
        w_step = slice(i + int(2.0 * FPS), i + int(4.0 * FPS))
        step = np.median(tot[w_step]) - pre
        if step < 2000:
            continue
        # 斜坡终点：首次到达 pre + 0.9*step
        tgt = pre + 0.9 * step
        w_up = (ts >= ts[i] - 1.0) & (ts <= ts[i] + 8.0)
        idx = np.where(w_up)[0]
        hit = idx[tot[idx] >= tgt]
        if len(hit) == 0:
            continue
        ramp_end = ts[hit[0]] - ts[i]
        top = tot[hit[0]]
        w_pl = (ts >= ts[i] + 10.0) & (ts <= ts[i] + 18.0)
        plat = np.median(tot[w_pl])
        rows.append((cond, ramp_end, (plat - top) / step, step))

r = np.array([(x[1], x[2]) for x in rows])
print(f"n={len(r)}")
print(f"斜坡宽 ramp_end: p10={np.percentile(r[:,0],10):.2f} 中位={np.median(r[:,0]):.2f} "
      f"p90={np.percentile(r[:,0],90):.2f} s")
print(f"斜坡顶后蠕变占台阶比 creep_frac: p10={np.percentile(r[:,1],10)*100:.1f}% "
      f"中位={np.median(r[:,1])*100:.1f}% p90={np.percentile(r[:,1],90)*100:.1f}%  极差={r[:,1].min()*100:.1f}%~{r[:,1].max()*100:.1f}%")
print(f"ramp_end<1s 的事件占比: {np.mean(r[:,0]<1.0)*100:.0f}%；<1.5s: {np.mean(r[:,0]<1.5)*100:.0f}%")
with open(os.path.join(HERE, "..", "results", "floor_analysis.txt"), "w", encoding="utf-8") as fh:
    fh.write(repr(rows))
