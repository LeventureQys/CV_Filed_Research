# -*- coding: utf-8 -*-
"""134_palm8_minplot：161747——现参数 vs 最小改动档 vs 推荐档 尾漂清理对照。"""
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
E1 = 16502.0
T0 = 1.2

CUR = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
       "slope_cap_frac": 0.025, "r_slow_max": 0.03, "tau_r_fast_s": 6.0,
       "tau_r_slow_idle_s": 0.5}
MINSET = {**CUR, "slope_cap_frac": 0.005, "r_slow_max": 0.15}
MINRF2 = {**MINSET, "r_fast": 0.02}
REC = {**CUR, "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0}

z = np.load(OUT / "streams.npz")
t, V = z["t"], z["V"]
fps = (len(t) - 1) / (t[-1] - t[0])
m = t >= T0
tin = V.sum(axis=1) / 1000.0

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
ax = axes[0]
ax.plot(t[m], tin[m], color="#888", lw=1.0, label="输入（算法关，录制流）")
ax.axhline(E1 / 1000.0, color="k", ls="--", lw=0.8)
ax.text(300, E1 / 1000.0 + 0.1, "弹性电平 E=16.50 N", fontsize=8)
ax.set_ylabel("总值 (N)")
ax.set_title("手掌 161747 · 419 s 长保压 · 尾漂清理：只改 2 个寄存器（207/208）即可冻结显示")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for name, p, c in [("现参数 rf.04 cap.025 rsm.03", CUR, "#d62728"),
                   ("最小档 只改 cap.005 + rsm.15（207=5, 208=15）", MINSET, "#1a9850"),
                   ("最小档+rf.02（再改 201=2，恒压代价最低）", MINRF2, "#ff7f0e"),
                   ("推荐档 rf.06 τc1=2 conf2 cap.005 rsm.15（参照）", REC, "#1f77b4")]:
    r = obs.run(t, V, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"], fps) / 1000.0
    i10 = int(np.searchsorted(t, T0 + 10))
    ax.plot(t[m], y[m], color=c, lw=1.2,
            label=f"{name}｜落点{y[-1]-E1/1000:+.2f}N 10s→末{y[-1]-y[i10]:+.2f}N")
ax.axhline(E1 / 1000.0, color="k", ls="--", lw=0.8)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("显示总值 (N)")
ax.legend(loc="lower right", fontsize=7.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "手掌_161747_最小参数尾漂清理.png", dpi=140)
print("saved")
