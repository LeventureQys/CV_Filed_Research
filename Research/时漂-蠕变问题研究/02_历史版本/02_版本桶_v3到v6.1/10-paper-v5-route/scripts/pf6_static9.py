# -*- coding: utf-8 -*-
"""图 6：恒载 9 组的长时间保压漂移（实采，显示域力值）。

主通道逐帧曲线：原始 vs 本算法；每格标注该组「全段时漂残余」与「慢相段时漂残余」
（口径：负载段末 10% − 首 10%，慢相段从 onset+5 s 起算）。

产出：figures/F6_static9.png、results/f6_static9.csv
"""
import numpy as np
import pandas as pd

import pd as P
from figstyle import (C_ALG, C_FAST, C_GRAY, C_RAW, plt, save_figure)

ROWS = [("右拇指指尖", ["数据1", "数据2", "数据3"]),
        ("左拇指指尖", ["数据1", "数据2", "数据3"]),
        ("四指指尖", ["数据1", "数据2", "数据3"])]

# 免责期档位：默认 3 s；命令行给一个正数即切换（用于两档同窗口对比）
import sys
FAST_S = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
SUFFIX = "" if abs(FAST_S - 3.0) < 1e-9 else f"_fast{FAST_S:g}s"
print(f"[F6] 免责期 = {FAST_S} s")

rows = []
fig, axes = plt.subplots(3, 3, figsize=(12.6, 7.4))
fig.subplots_adjust(left=0.055, right=0.988, top=0.925, bottom=0.075,
                    wspace=0.20, hspace=0.38)

for i, (loc, names) in enumerate(ROWS):
    for j, name in enumerate(names):
        ax = axes[i][j]
        p = P.static_path(loc, name)
        d = P.load_rec(p, main_ch=P.STATIC_MAIN[loc])
        tu, dtm, x, tot = d["tu"], d["dtm"], d["x"], d["tot"]
        s0, s1 = P.L.find_periods(tot, dtm)[0]
        # 退掉负载段末端的卸载沿，避免末尾出现一条与算法无关的掉零竖线
        s1 = min(s1 - int(round(0.29 / dtm)), len(tu) - 1)
        tq = tu[s0:s1] - tu[s0]
        # 实际算法跑在 21 通道上；这里只画主通道，但补偿量取该通道自身的输出
        Y, comp = P.run_algo(tu, d["Xu"], FAST_S=FAST_S)
        yc = Y[:, P.STATIC_MAIN[loc]]
        amp = float(np.mean(x[s0:s1]) - np.mean(x[:max(1, s0)]))
        nL = s1 - s0
        dr = float((yc[s0:s1][-nL // 10:].mean() - yc[s0:s1][:nL // 10].mean()) / amp * 100)
        dr_raw = float((x[s0:s1][-nL // 10:].mean() - x[s0:s1][:nL // 10].mean()) / amp * 100)
        a5 = s0 + int(5.0 / dtm)
        n5 = max(1, s1 - a5)
        dr5 = float((yc[a5:s1][-n5 // 10:].mean() - yc[a5:s1][:n5 // 10].mean()) / amp * 100)
        dr5_raw = float((x[a5:s1][-n5 // 10:].mean() - x[a5:s1][:n5 // 10].mean()) / amp * 100)
        # 受载通道中位（口径与 bw_metrics_current.py 一致：逐通道"末 10% − 首 10%"÷ 该通道幅度取中位）
        amp_v = np.mean(Y[s0:s1], axis=0) - np.mean(Y[:max(1, s0)], axis=0)
        loaded = amp_v > 0.10 * amp_v.max()
        segY = Y[s0:s1]
        per_ch = (segY[-nL // 10:].mean(axis=0) - segY[:nL // 10].mean(axis=0)) / amp
        drm = float(np.median(per_ch[loaded])) if loaded.any() else float("nan")

        # 画面画"相对本段均值的增量"：y 轴落在 0 附近，两条曲线的分离一眼可见；
        # 时漂残余仍按原始口径（末 10% − 首 10%，除以幅度）计算，标注在右下角。
        xd = x[s0:s1] - float(np.mean(x[s0:s1]))
        yd = yc[s0:s1] - float(np.mean(yc[s0:s1]))
        ax.plot(tq, xd, color=C_RAW, lw=1.7, label="原始（无补偿）")
        ax.plot(tq, yd, color=C_ALG, lw=1.4, label="本算法")
        ax.set_xlim(0, tq[-1])
        lo = min(float(np.percentile(xd, 0.5)), float(np.percentile(yd, 0.5)))
        hi = max(float(np.percentile(xd, 99.5)), float(np.percentile(yd, 99.5)))
        span = max(hi - lo, 1e-9)
        ax.set_ylim(lo - 1.0 * span, hi + 0.35 * span)
        ax.set_title(f"{loc} / {name}", fontsize=9.6)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.set_ylabel("相对本段均值的增量 (N)", fontsize=9)
        if i == 2:
            ax.set_xlabel("负载段内时间 (s)", fontsize=9)
        # 注记放右下角（该区域无数据）；不透明底，避免网格线透出
        ax.text(0.96, 0.06, "全段 %.2f%% → %.2f%%\n慢相段 %.2f%% → %.2f%%"
                % (dr_raw, dr, dr5_raw, dr5),
                transform=ax.transAxes, fontsize=7.9, va="bottom", ha="right", linespacing=1.6,
                bbox=dict(fc="white", ec="#D5D8DC", lw=0.7, alpha=1.0, pad=2.6))
        if i == 0 and j == 0:
            ax.legend(fontsize=8.0, loc="upper left", framealpha=1.0, borderaxespad=0.7)
        rows.append(dict(loc=loc, dataset=name, amp=amp, dur_s=float(tq[-1]),
                         drift_full_raw=dr_raw, drift_full_algo=dr,
                         drift_slow_raw=dr5_raw, drift_slow_algo=dr5,
                         drift_loaded_med=drm,
                         A_max=float(comp.A.max()), loaded_n=int(comp.loaded.sum()),
                         gamma_lo=float(comp.gamma.min()), gamma_hi=float(comp.gamma.max())))
        print(f"[F6] {loc}/{name}: 全段 {dr_raw:6.2f}% → {dr:5.2f}%；慢相段 {dr5_raw:6.2f}% → {dr5:5.2f}%")

df = pd.DataFrame(rows)
P.save_table(df, f"f6_static9{SUFFIX}.csv")
print(f"[F6] 9 组均值(|·|)：全段 原始 {df.drift_full_raw.abs().mean():.2f}% → "
      f"本算法 {df.drift_full_algo.abs().mean():.2f}%；慢相段 "
      f"{df.drift_slow_raw.abs().mean():.2f}% → {df.drift_slow_algo.abs().mean():.2f}%；"
      f"受载中位 {df.drift_loaded_med.abs().mean():.2f}%")

fig.suptitle("恒载 9 组：长时间保压下的显示值（主通道；标注为该组的时漂残余）",
             fontsize=12.0, y=0.985)
save_figure(fig, f"F6_static9{SUFFIX}.png")
