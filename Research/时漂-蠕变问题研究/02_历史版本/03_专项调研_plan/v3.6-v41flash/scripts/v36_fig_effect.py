# -*- coding: utf-8 -*-
"""v3.6 效果图：目标录制上的 v3.1 vs v3.6 对照（总波形 / 后段放大 / 逐段一致性柱状）。"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402
import v36_replay as R  # noqa: E402
from v36_probe_internal import R_arm  # noqa: E402
from v36_ab import EPISODES, episode_table  # noqa: E402

FIG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ARM_CAND = "v36:--seed-gain:1.0:--v36-d:400"
ARM_V34 = "v34:120"


def main():
    ds = K.load(K.DS_TARGET)
    el, tin, tlive = ds["pre"]["el"], ds["tot_in"], ds["tot_out"]
    s1 = R.run_arm(ds, *R_arm("v31"))
    s3 = R.run_arm(ds, *R_arm(ARM_V34))
    s2 = R.run_arm(ds, *R_arm(ARM_CAND))

    fig, ax = plt.subplots(3, 1, figsize=(17, 13))
    ax[0].plot(el, tin, color="0.7", lw=1.0, label="算法输入（读数）")
    ax[0].plot(el, tlive, color="tab:blue", lw=1.1, label="现役基线（plan v3.1）")
    ax[0].plot(el, s3["sum_out"], color="tab:green", lw=1.1, alpha=0.85, label="plan v3.4（已落地）")
    ax[0].plot(el, s2["sum_out"], color="tab:red", lw=1.1, alpha=0.85, label="候选 v3.6")
    ax[0].set_title("目标录制 20260919_160854 · 全程总波形（Σ21 通道，ADC）")
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)

    m = (el >= 236) & (el <= 312)
    ax[1].plot(el[m], tin[m], color="0.65", lw=1.5, label="输入")
    ax[1].plot(el[m], tlive[m], color="tab:blue", lw=1.3, label="现役基线（补偿丢失）")
    ax[1].plot(el[m], s3["sum_out"][m], color="tab:green", lw=1.3, label="plan v3.4（已落地）")
    ax[1].plot(el[m], s2["sum_out"][m], color="tab:red", lw=1.3, label="候选 v3.6")
    ax[1].set_title("后段放大 236~312 s：完全卸载(240 s) → 重载(245.3 s) 之后")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)

    r1, r3, r2 = episode_table(s1), episode_table(s3), episode_table(s2)
    lab = [r["lab"].split()[0] for r in r1]
    x = np.arange(len(lab))
    ax[2].bar(x - 0.27, [r["out_lvl"] for r in r1], width=0.27,
              color="tab:blue", label="v3.1 显示电平")
    ax[2].bar(x, [r["out_lvl"] for r in r3], width=0.27,
              color="tab:green", label="v3.4 显示电平")
    ax[2].bar(x + 0.27, [r["out_lvl"] for r in r2], width=0.27,
              color="tab:red", label="v3.6 显示电平")
    ax[2].plot(x, [r["in_lvl"] for r in r1], "k.--", lw=1.0, label="输入电平")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(lab, fontsize=8)
    ax[2].set_title("各平台显示电平：同一负载应当同一高度（E1~E9）")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p1 = os.path.join(FIG, "F1_v36_vs_v31.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(17, 8))
    ax[0].plot(el, s1["ded"], color="tab:blue", lw=1.0, label="v3.1 补偿量 ded")
    ax[0].plot(el, s3["ded"], color="tab:green", lw=1.0, alpha=0.8, label="v3.4 补偿量 ded")
    ax[0].plot(el, s2["ded"], color="tab:red", lw=1.0, alpha=0.8, label="v3.6 补偿量 ded")
    ax[0].axhline(0, color="0.4", lw=0.8)
    ax[0].set_title("补偿量 ded = Σ输入 − Σ显示（全程）")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    m = (el >= 236) & (el <= 312)
    ax[1].plot(el[m], s1["ded"][m], color="tab:blue", lw=1.3, label="v3.1")
    ax[1].plot(el[m], s3["ded"][m], color="tab:green", lw=1.3, label="v3.4")
    ax[1].plot(el[m], s2["ded"][m], color="tab:red", lw=1.3, label="v3.6")
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].set_title("后段放大：v3.1 的 ded 掉到 0（丢基线），两条修复臂都接回来了")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(FIG, "F2_ded.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    print("-> %s\n-> %s" % (p1, p2))


if __name__ == "__main__":
    main()
