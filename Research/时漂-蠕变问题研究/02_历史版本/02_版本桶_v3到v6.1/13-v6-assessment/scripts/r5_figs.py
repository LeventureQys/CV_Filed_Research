# -*- coding: utf-8 -*-
"""r5：出图 —— 快相形状的稳定性、on/restep 差异、ROM 失配带来的过充分布。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

import matplotlib                                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TG = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80])
ROG = np.array([0.680, 0.740, 0.790, 0.824, 0.854, 0.873, 0.886])


def main():
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    sh = [c for c in ev.columns if c.startswith("sh_")][:7]
    ev["sensor"] = np.where(ev.rec.str.startswith("右"), "右拇指",
                            np.where(ev.rec.str.startswith("左"), "左拇指",
                                     np.where(ev.rec.str.startswith("四"), "四指", "实录")))
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.4))

    # (a) 全体 onset 形状 + 现役 ROM
    o = ev[(ev.kind == "onset") & ev.A5.gt(0)]
    Y = o[sh].to_numpy(float)
    med = np.nanmedian(Y, 0)
    p10, p90 = np.nanpercentile(Y, 10, axis=0), np.nanpercentile(Y, 90, axis=0)
    for r in Y:
        ax[0, 0].plot(TG, r, color="0.75", lw=.7)
    ax[0, 0].fill_between(TG, p10, p90, color="#a5d6a7", alpha=.45, label="p10~p90")
    ax[0, 0].plot(TG, med, "o-", color="#2e7d32", lw=2.2, label="实测中位")
    ax[0, 0].plot(TG, ROG, "s--", color="#c62828", lw=2, label="现役形状 ROM")
    ax[0, 0].set_title("(a) 19 个 onset 的归一化快相形状（灰=各次）vs 现役 ROM")
    ax[0, 0].set_xlabel("τ（s，自加载沿）"); ax[0, 0].set_ylabel("y(τ)/A(5 s)")
    ax[0, 0].legend(fontsize=9); ax[0, 0].grid(alpha=.3)

    # (b) 分传感器
    for s, c in (("右拇指", "#1565c0"), ("左拇指", "#ef6c00"), ("四指", "#6a1b9a"), ("实录", "#2e7d32")):
        sub = o[o.sensor == s]
        if len(sub) < 2:
            continue
        ax[0, 1].plot(TG, np.nanmedian(sub[sh].to_numpy(float), 0), "o-", color=c,
                      label="%s（n=%d）" % (s, len(sub)))
    ax[0, 1].plot(TG, ROG, "s--", color="#c62828", label="现役 ROM")
    ax[0, 1].set_title("(b) 同一 ROM 面对不同传感器（0.2 s 处差 15 个百分点）")
    ax[0, 1].set_xlabel("τ（s）"); ax[0, 1].set_ylabel("y(τ)/A(5 s)")
    ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=.3)

    # (c) onset vs restep
    for k, c in (("onset", "#2e7d32"), ("restep", "#c62828")):
        sub = ev[(ev.kind == k) & ev.A5.gt(0)]
        if len(sub) < 2:
            continue
        ax[1, 0].plot(TG, np.nanmedian(sub[sh].to_numpy(float), 0), "o-", color=c,
                      label="%s（n=%d）" % (k, len(sub)))
        ax[1, 0].fill_between(TG, np.nanpercentile(sub[sh].to_numpy(float), 25, axis=0),
                              np.nanpercentile(sub[sh].to_numpy(float), 75, axis=0),
                              color=c, alpha=.15)
    ax[1, 0].plot(TG, ROG, "s--", color="0.4", label="现役 ROM（按 onset 标定）")
    ax[1, 0].set_title("(c) 零基线起（onset）vs 带载叠加（restep）：起步差 10~20 pt")
    ax[1, 0].set_xlabel("τ（s）"); ax[1, 0].set_ylabel("y(τ)/A(5 s)")
    ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=.3)

    # (d) 过充分布
    data, labs = [], []
    for k, c in (("onset", "#2e7d32"), ("restep", "#c62828")):
        v = ev[(ev.kind == k) & ev.over_vs_A5.notna()].over_vs_A5.to_numpy(float)
        if len(v):
            data.append(v); labs.append("%s\n(n=%d)" % (k, len(v)))
    bp = ax[1, 1].boxplot(data, labels=labs, widths=.5, patch_artist=True, showfliers=True)
    for b, c in zip(bp["boxes"], ("#a5d6a7", "#ef9a9a")):
        b.set_facecolor(c)
    ax[1, 1].axhline(0, color="k", lw=.8)
    ax[1, 1].axhspan(-3, 3, color="#c8e6c9", alpha=.5, label="±3% 可接受带")
    ax[1, 1].set_ylabel("Â 相对 A(5 s) 的偏差（%）")
    ax[1, 1].set_title("(d) 用同一个 onset 形状库算 Â：onset 偏上、restep 偏下")
    ax[1, 1].legend(fontsize=9); ax[1, 1].grid(alpha=.3)

    fig.suptitle("v6 快相处理的地基：形状是否稳定、两种形态差多少", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIG, "r5_shape_stability.png")
    fig.savefig(p, dpi=125)
    print("-> figures/r5_shape_stability.png")
    print("  (a) 中位形状 @0.2/0.5/1 s = %.3f / %.3f / %.3f；现役 ROM = %.3f / %.3f / %.3f"
          % (med[2], med[4], 6 < len(med) and med[6] or np.nan, ROG[2], ROG[4], ROG[6]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
