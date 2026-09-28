# -*- coding: utf-8 -*-
"""130_palm8_plot：161747 长保压 419 s——输入/现参数显示/推荐档显示/激进档显示 对照图。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")

OUT = TEMP / "palm8" / "out"
FIG = TEMP / "figures"
FIG.mkdir(exist_ok=True)
E1 = 16502.0
T0 = 1.2

CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
       "tau_r_slow_idle_s": 0.5}
REC = {**CUR, "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0,
       "slope_cap_frac": 0.005, "r_slow_max": 0.15}
RF75 = {**REC, "r_fast": 0.075}

z = np.load(OUT / "streams.npz")
t, V = z["t"], z["V"]
fps = (len(t) - 1) / (t[-1] - t[0])
tin = V.sum(axis=1) / 1000.0
m = t >= T0

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
ax = axes[0]
ax.plot(t[m], tin[m], color="#888", lw=1.0, label="输入（算法关，录制流）")
ax.axhline(E1 / 1000.0, color="k", ls="--", lw=0.8)
ax.text(300, E1 / 1000.0 + 0.05, "弹性电平 E=16.50 N", fontsize=8)
ax.set_ylabel("总值 (N)")
ax.set_title("手掌 20260927_161747 · 单台阶 419 s 长保压（蠕变 +15.6%）· 输入与候选参数显示")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for name, p, c in [("现参数 rf.04 τc1=1 conf1.5 soft1.5 cap.025 rsm.03", CUR, "#d62728"),
                   ("推荐 rf.06 τc1=2 conf2 soft1.5 cap.005 rsm.15", REC, "#1a9850"),
                   ("激进 rf.075 其余同推荐", RF75, "#1f77b4")]:
    r = obs.run(t, V, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"], fps) / 1000.0
    i10 = int(np.searchsorted(t, T0 + 10))
    ax.plot(t[m], y[m], color=c, lw=1.2,
            label=f"{name}｜落点{y[-1]-E1/1000:+.2f}N 10s→末{y[-1]-y[i10]:+.2f}N")
ax.axhline(E1 / 1000.0, color="k", ls="--", lw=0.8)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("显示总值 (N)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "手掌_161747_长保压对照.png", dpi=140)
print("saved:", FIG / "手掌_161747_长保压对照.png")
