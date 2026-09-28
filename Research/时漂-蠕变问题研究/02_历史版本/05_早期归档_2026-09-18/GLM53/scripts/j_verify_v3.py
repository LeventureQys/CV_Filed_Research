# -*- coding: utf-8 -*-
"""GLM53 v3 验收：切换事件处的显示跳变量（v2 vs v3 逐事件对比）。

验收指标（用户关注点）: 切换负载时显示不得出现大于真实信号变化的跳变。
对每个 onset/restep/unload 事件计算:
  jump_disp  = 事件帧附近单帧最大 |Δ显示|（唯一时间戳帧，±1.5s 窗内）
  jump_raw   = 同帧 |Δ原始|
  excess     = jump_disp - jump_raw （>0 表示显示跳得比原始多）
  hold_dev   = 事件前 pending 窗口(检测到确认)内 |显示-原始| 的最大偏离
输出: results/j_v3_jump_metrics.csv; figures/j1_v3_zoom_{A,B}.png
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))  # temp
OUT = os.path.dirname(HERE)
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")

spec = importlib.util.spec_from_file_location("fv", os.path.join(HERE, "f_varying_load.py"))
fv = importlib.util.module_from_spec(spec)
sys.modules["fv"] = fv
spec.loader.exec_module(fv)

VARY = {"A": "零负载-切换负载-零负载-再切换负载",
        "B": "零负载-中途切换负载-零负载-切换负载"}

rows = []
for tag, loc in VARY.items():
    t, X, tc = fv.load_csv(os.path.join(BASE, "变化负载", loc, "device_001_seg000.csv"))
    raw = X.sum(1)
    t = t.to_numpy() if hasattr(t, "to_numpy") else t
    uniq = np.r_[True, np.diff(t) > 1e-6]

    for name, cls in [("v2", fv.CompV2), ("v3", fv.CompV3)]:
        comp = cls()
        Y = np.empty_like(X, dtype=float)
        for i in range(len(t)):
            Y[i] = comp.process(t[i], X[i].astype(float))
        disp = Y.sum(1)
        # 事件窗口内的单帧跳变（唯一时间戳帧）
        for ets, etag in comp.events:
            ets = float(ets)
            m = (t >= ets - 1.5) & (t <= ets + 1.5) & uniq
            ti = np.where(m)[0]
            if len(ti) < 3:
                continue
            d_disp = np.abs(np.diff(disp[ti]))
            d_raw = np.abs(np.diff(raw[ti]))
            j = int(np.argmax(d_disp))
            # pending 窗口(事件前 step_persist+检测余量)内显示偏离原始的最大值
            mh = (t >= ets - 4.0) & (t < ets - 0.2)
            hold_dev = float(np.max(np.abs(disp[mh] - raw[mh]))) if mh.any() else 0.0
            # 事件后窗口(0.5~4.5s, 避开卸载边缘): 显示-原始的均值偏移与漂移率
            mp = (t >= ets + 0.5) & (t <= ets + 4.5) & uniq
            off = float(np.mean(disp[mp] - raw[mp])) if mp.any() else np.nan
            k = 0.0
            if mp.sum() >= 10:
                k = float(np.polyfit(t[mp] - ets, disp[mp] - raw[mp], 1)[0])
            rows.append(dict(dataset=tag, algo=name, event=etag, t=round(ets, 2),
                             jump_disp=round(float(d_disp[j])),
                             jump_raw=round(float(d_raw[j])),
                             excess=round(float(d_disp[j] - d_raw[j])),
                             hold_dev=round(hold_dev),
                             post_off=round(off) if not np.isnan(off) else np.nan,
                             post_off_rate=round(k)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "j_v3_jump_metrics.csv"), index=False, encoding="utf-8-sig")
print(df.to_string(index=False))

# 对照图: 每个事件窗口 raw/v2/v3
for tag, loc in VARY.items():
    t, X, tc = fv.load_csv(os.path.join(BASE, "变化负载", loc, "device_001_seg000.csv"))
    raw = X.sum(1)
    Y2, c2 = fv.run_case(fv.CompV2, X, t)
    Y3, c3 = fv.run_case(fv.CompV3, X, t)
    evs = [(float(a), b) for a, b in c3.events if b in ("onset", "restep", "unload")]
    fig, axes = plt.subplots(len(evs), 1, figsize=(12, 2.6 * max(len(evs), 1)),
                             constrained_layout=True, squeeze=False)
    for ax, (t0, etag) in zip(axes[:, 0], evs):
        m = (t >= t0 - 4) & (t <= t0 + 8)
        ax.plot(t[m], raw[m], color="k", lw=0.7, alpha=0.5, label="raw")
        ax.plot(t[m], Y2.sum(1)[m], color="tab:blue", lw=0.9, label="v2")
        ax.plot(t[m], Y3.sum(1)[m], color="tab:green", lw=0.9, label="v3")
        ax.axvline(t0, color="gray", ls=":")
        ax.set_title(f"{loc[:14]}… {etag}@{t0:.1f}s")
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, f"j1_v3_zoom_{tag}.png"), dpi=140)
    plt.close(fig)

print("saved: figures/j1_v3_zoom_A/B.png; results/j_v3_jump_metrics.csv")
