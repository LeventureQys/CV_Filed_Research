# -*- coding: utf-8 -*-
"""v5 对比图：合成砝码场景 + 四份实采录制的台阶捕获与偏差。"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.facecolor"] = "white"
sys.path.insert(0, HERE)
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

COL = {"raw": "#9e9e9e", "v3": "#1f77b4", "v4": "#2ca02c", "v5": "#d62728"}
LBL = {"raw": "原始（无补偿）", "v3": "GLM53 v3（现役）", "v4": "v4 免责5s", "v5": "v5（本次修复）"}

REC0 = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
        r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc"
        r"\device_001_seg000.csv")
d0 = L.prep(REC0)
a, b = int(133.84 / d0["dtm"]), int(182.0 / d0["dtm"])
uu = d0["tu"][a:b] - d0["tu"][a]
yy = d0["tot"][a:b] / d0["tot"][a + int(5.0 / d0["dtm"])] - 1.0
fit = lambda x, a1, t1, a2, t2: a1 * (1 - np.exp(-x / t1)) + a2 * (1 - np.exp(-x / t2))
P, _ = optimize.curve_fit(fit, uu, yy, p0=[0.03, 3.0, 0.05, 60.0],
                          bounds=([0, 0.2, 0, 5], [1, 60, 1, 3000]))
cr = lambda x: np.where(x > 0, fit(np.maximum(x, 0), *P), 0.0)
FS, dt, DUR = 100.0, 0.01, 560.0
tt = np.arange(0.0, DUR, dt)


def build(steps, base_n=1.0):
    s = np.zeros_like(tt)
    m = tt > 20.0
    s[m] += base_n * 10000.0 * (1 + cr(tt[m] - 20.0))
    for t0, k in steps:
        mm = tt > t0
        s[mm] += k * 10000.0 * (1 + cr(tt[mm] - t0))
    return s


def run(cls, sig, **kw):
    c = cls(1)
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty(len(tt))
    for i in range(len(tt)):
        Y[i] = c.process(tt[i], np.array([sig[i]]))[0]
    return Y


ALGOS = [("raw", None, {}), ("v3", GLM53v3, {}),
         ("v4", L.GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
         ("v5", GLM53v5, {})]

fig = plt.figure(figsize=(19, 11.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

# ① 10N → +5N 轨迹
ax = fig.add_subplot(gs[0, 0])
steps, base_n = [(300.0, 0.5)], 1.0
sig = build(steps, base_n)
ideal = (base_n + sum(k for _, k in steps)) * 10000.0
Ys = {k: (sig if cls is None else run(cls, sig, **kw)) for k, cls, kw in ALGOS}
sl = slice(int(290 / dt), int(430 / dt))
for k, _, _ in ALGOS:
    ax.plot(tt[sl] - 300, Ys[k][sl], color=COL[k], lw=2.4 if k in ("v5", "raw") else 1.5,
            ls=":" if k == "raw" else "-", alpha=0.95 if k != "raw" else 0.9, label=LBL[k])
ax.axhline(ideal, color="k", ls="--", lw=1.3)
ax.text(128, ideal + 250, f"真值 {ideal:.0f}（10N+5N）", fontsize=10.5, ha="right")
ax.axhspan(9750, 10250, color="#1f77b4", alpha=0.10)
ax.text(127, 9650, "10N 电平", fontsize=10, color="#1f77b4", ha="right")
ax.set_xlim(-10, 130)
ax.set_ylim(9200, 17600)
ax.set_xlabel("加码后时间 (s)", fontsize=11)
ax.set_ylabel("显示值 (ADC，10N = 10000)", fontsize=11)
ax.set_title("① 10N 保压 280s → +5N：v3/v4 偏高 +7.9%，v5 稳在真值", fontsize=12.5, loc="left")
ax.legend(fontsize=9.5, loc="center right")
ax.grid(alpha=0.15)

# ② 各场景偏差柱状
ax = fig.add_subplot(gs[0, 1])
CASES = [("+2.5N\n(台阶25%)", [(300.0, 0.25)], 1.0),
         ("+5N\n(台阶50%)", [(300.0, 0.50)], 1.0),
         ("+10N\n(台阶100%)", [(300.0, 1.00)], 1.0),
         ("20N+5N\n(台阶12.5%)", [(300.0, 0.25)], 2.0),
         ("两档\n间隔80s", [(220.0, 0.50), (300.0, 0.50)], 1.0),
         ("两档\n间隔8s", [(220.0, 0.50), (228.0, 0.50)], 1.0)]
x = np.arange(len(CASES))
w = 0.26
for i, (k, cls, kw) in enumerate(ALGOS[1:]):
    vals = []
    for _, steps, base_n in CASES:
        sig = build(steps, base_n)
        ideal = (base_n + sum(kk for _, kk in steps)) * 10000.0
        Y = run(cls, sig, **kw)
        vals.append(100 * (Y[-int(30 / dt):].mean() - ideal) / ideal)
    ax.bar(x + (i - 1) * w, vals, w, color=COL[k], alpha=0.92, label=LBL[k])
ax.axhline(0, color="k", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CASES], fontsize=9)
ax.set_ylabel("稳态相对真值的偏差 (%)", fontsize=11)
ax.set_title("② 合成砝码场景：各算法稳态偏差（0 = 准）", fontsize=12.5, loc="left")
ax.legend(fontsize=9.5)
ax.grid(alpha=0.15, axis="y")

# ③ 实采租：台阶捕获比
ax = fig.add_subplot(gs[1, 0])
ml = pd.read_csv(os.path.join(RES, "v5_midload_events.csv"))
mm = ml[ml.valid_gain.astype(bool)]
x = np.arange(len(mm))
for i, k in enumerate(["v3", "v4", "v5"]):
    vals = (mm[f"gain_{k}"] / mm["raw_gain"]).values
    ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=COL[k], alpha=0.92, label=LBL[k])
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ax.text(len(mm) - 0.5, 1.03, "1.0 = 台阶被完整透传", fontsize=9.5, ha="right")
ax.set_xticks(x)
ax.set_xticklabels([f"{r.rec[:5]}\n{r.t:.0f}s" for _, r in mm.iterrows()], fontsize=7.5)
ax.set_ylabel("台阶捕获比（显示增量 ÷ 原始增量）", fontsize=11)
ax.set_title(f"③ 四份实采录制的负载内变载（{len(mm)} 个有效事件）", fontsize=12.5, loc="left")
ax.legend(fontsize=9.5)
ax.grid(alpha=0.15, axis="y")

# ④ 全程最大偏差
ax = fig.add_subplot(gs[1, 1])
sv = pd.read_csv(os.path.join(RES, "v5_maxgap.csv"))
piv = sv.pivot_table(index="rec", columns="algo", values="pct_of_peak")
x = np.arange(len(piv))
for i, k in enumerate(["v3", "v4", "v5"]):
    ax.bar(x + (i - 1) * 0.27, piv[k].values, 0.27, color=COL[k], alpha=0.92, label=LBL[k])
for i, rec in enumerate(piv.index):
    for j, k in enumerate(["v3", "v4", "v5"]):
        ax.text(i + (j - 1) * 0.27, piv[k].values[i] + 0.6, f"{piv[k].values[i]:.0f}",
                ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([r[:12] for r in piv.index], fontsize=9)
ax.set_ylabel("全程 max|显示−原始| ÷ 峰值 (%)", fontsize=11)
ax.set_title("④ 全程最大偏差（越小越好）", fontsize=12.5, loc="left")
ax.legend(fontsize=9.5)
ax.grid(alpha=0.15, axis="y")

fig.suptitle("v5（变载识别与基线修复版）· 与现役 v3 / v4 的对比", fontsize=16)
fig.savefig(os.path.join(FIG, "v5_compare.png"), dpi=118)
plt.close(fig)
print("saved: figures/v5_compare.png")
