# -*- coding: utf-8 -*-
"""r7：三阶段分解图（回答 Q2）—— 典型事件的三段标注 + 9 组构成 + 完成度 + 慢相 τ/幅度。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FLASH = os.path.dirname(os.path.dirname(OUT))
TEMP = os.path.dirname(FLASH)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(FLASH, "progress", "04-v5", "scripts"))
import ad_lib as L  # noqa: E402

import matplotlib                                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    ph = pd.read_csv(os.path.join(RES, "phases.csv"))
    hold = ph[ph.rec.str.startswith(("右", "左", "四"))].copy()
    ev = pd.read_csv(os.path.join(RES, "events.csv"))
    evh = ev[(ev.dom == "显示域") & (ev.kind == "onset")]

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 8.6))

    # (a) 典型事件：右拇指/数据1 的加载段，标注三阶段
    name, path, ch = "右拇指指尖", os.path.join(TEMP, "右拇指指尖", "数据1",
                                                "device_001_seg000.csv"), 17
    d = L.prep(path)
    tu, y = d["tu"], d["Xu"][:, ch]
    k = int(np.argmax(np.diff(L.med_smooth(y, max(1, int(0.1 / d["dtm"]))),
                              prepend=y[0])[:len(y)])) if False else None
    # 用与 r1/r2 相同的方式找真沿
    w = max(1, int(0.10 / d["dtm"]))
    ys = L.med_smooth(y, w)
    dd = np.zeros_like(ys)
    dd[w:-w] = ys[2 * w:] - ys[:-2 * w]
    k = int(np.argmax(dd))
    pre = float(np.median(y[max(0, k - int(2.0 / d["dtm"])):k]))
    t0, t1 = tu[k] - 1.0, tu[k] + 9.0
    m = (tu >= t0) & (tu <= t1)
    ax[0, 0].plot(tu[m] - tu[k], y[m], color="0.35", lw=1.4, label="原始读数（右拇指/数据1）")
    A02 = float(np.median(y[k + int(0.15 / d["dtm"]):k + int(0.25 / d["dtm"])]))
    ax[0, 0].axvspan(0, 0.2, color="#ffcdd2", alpha=.8, label="① 机械阶跃（<0.2 s）")
    ax[0, 0].axvspan(0.2, 5.0, color="#fff9c4", alpha=.8, label="② 快相爬升（0.2→5 s）")
    ax[0, 0].axvspan(5.0, 9.0, color="#c8e6c9", alpha=.8, label="③ 慢相爬升（5 s→）")
    ax[0, 0].axhline(A02, color="#c62828", ls="--", lw=1)
    ax[0, 0].annotate("0.2 s：已完成 88.4%\n（9 组中位）", xy=(0.2, A02), xytext=(2.0, pre + 0.55 * (A02 - pre)),
                      fontsize=9, arrowprops=dict(arrowstyle="->", color="#c62828"))
    ax[0, 0].set_xlabel("自加载沿（s）"); ax[0, 0].set_ylabel("显示读数（N）")
    ax[0, 0].set_title("(a) 一次真实加载的三段结构")
    ax[0, 0].legend(fontsize=8, loc="lower right"); ax[0, 0].grid(alpha=.3)

    # (b) 9 组构成（段级 phases.csv）
    lab = [r.replace("/", "\n") for r in hold.rec]
    step = hold.A_step / hold.A_end * 100
    fast = (hold.A_5s - hold.A_step) / hold.A_end * 100
    slow = (hold.A_end - hold.A_5s) / hold.A_end * 100
    x = np.arange(len(hold))
    ax[0, 1].bar(x, step, color="#ef9a9a", label="① 阶跃")
    ax[0, 1].bar(x, fast, bottom=step, color="#ffe082", label="② 快相")
    ax[0, 1].bar(x, slow, bottom=step + fast, color="#a5d6a7", label="③ 慢相（5 s→段末）")
    for i, v in enumerate(slow):
        ax[0, 1].text(i, 100.5, "%.0f%%" % v, ha="center", fontsize=8, color="#2e7d32")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(lab, fontsize=7.5)
    ax[0, 1].set_ylabel("占该段总涨幅（%）"); ax[0, 1].set_ylim(0, 112)
    ax[0, 1].set_title("(b) 恒载 9 组的三段幅度构成（绿字=慢相占比）")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=.3, axis="y")

    # (c) 1 s / 2 s 完成度分布
    data = [evh.sh_100 * 100, evh.sh_200 * 100]
    bp = ax[1, 0].boxplot(data, labels=["1 s 时", "2 s 时"], widths=.45,
                          patch_artist=True, showfliers=False)
    for b, c in zip(bp["boxes"], ("#90caf9", "#a5d6a7")):
        b.set_facecolor(c)
    for i, dd2 in enumerate(data, start=1):
        ax[1, 0].scatter(np.full(len(dd2), i) + np.random.uniform(-.08, .08, len(dd2)),
                         dd2, s=18, color="#37474f", zorder=3)
    ax[1, 0].axhline(100, color="k", lw=.8)
    ax[1, 0].set_ylabel("占 5 s 增量（%）")
    ax[1, 0].set_title("(c) 快相在 1 s / 2 s 的完成度（19 个 onset）")
    ax[1, 0].grid(alpha=.3, axis="y")

    # (d) 慢相 τ vs 幅度
    ax[1, 1].scatter(hold.slow_frac, hold.slow_tau, s=48, color="#2e7d32")
    for _, r in hold.iterrows():
        ax[1, 1].annotate(r.rec.split("/")[0][:2] + r.rec[-1], (r.slow_frac, r.slow_tau), fontsize=7.5)
    ax[1, 1].set_xlabel("慢相幅度（占 5 s 增量 %）"); ax[1, 1].set_ylabel("慢相时间常数 τ（s）")
    ax[1, 1].set_title("(d) 慢相：幅度 5~45%、τ 19.7~71 s（不收敛）")
    ax[1, 1].grid(alpha=.3)

    fig.suptitle("v6 的地基：加载的三段结构（阶跃 / 快相 / 慢相）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(FIG, "r7_three_phases.png")
    fig.savefig(p, dpi=125)
    print("-> figures/r7_three_phases.png")
    print("  恒载 9 组：阶跃中位 %.1f%%、快相 %.1f%%、慢相 %.1f%%（占段内总涨幅）"
          % (step.median(), fast.median(), slow.median()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
