# -*- coding: utf-8 -*-
"""变化负载图：中途变载处 v3 vs v4_fast5 的逐帧对比 + 最坏欠报汇总。"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
FIG = os.path.join(OUT, "figures")
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3  # noqa: E402
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)

# 左：数据B 中途变载
t, X = load_rec(os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                             "device_001_seg000.csv"))
span = t[-1] - t[0]
dt = span / (len(t) - 1)
tu = np.arange(0.0, span, dt)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
tot = Xu.sum(axis=1)
Y3 = run(tu, Xu, GLM53v3).sum(axis=1)
Y4 = run(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0).sum(axis=1)
ax = axes[0]
ax.plot(tu, tot, color="0.55", lw=1.0, label="原始总量")
ax.plot(tu, Y3, color="tab:blue", lw=1.2, label="v3（现有）")
ax.plot(tu, Y4, color="tab:red", lw=1.2, label="v4（快相免责 5s）")
ax.axvline(20.95, color="k", ls="--", lw=1.0)
ax.annotate("中途 5N→10N 变载", xy=(20.95, 26000), xytext=(24, 27000), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_xlim(10, 40)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("显示总量 (ADC)")
ax.set_title("数据B：中途变载处 v3 欠报 35%、v4 欠报 21.6%\n"
             "v4 免责期不补快相 ⇒ 变载看得更清", fontsize=10)
ax.legend(fontsize=8, loc="lower right")

# 右：最坏欠报对比
dfm = pd.read_csv(os.path.join(RES, "varying_identifiability.csv"))
p = dfm.pivot_table(index=["dataset", "event_s"], columns="algo", values="worst_gap_pct")
p = p.dropna(subset=["v3"])
p = p[p["v3"] < 100]          # 去掉极小跳变（卸载到零）导致的比例失真点
xs = np.arange(len(p))
w = 0.36
axes[1].bar(xs - w / 2, p["v3"], width=w, color="tab:blue", label="v3（现有）")
axes[1].bar(xs + w / 2, p["v4_fast5"], width=w, color="tab:red", label="v4（快相免责 5s）")
axes[1].set_xticks(xs)
axes[1].set_xticklabels([f"{a}/{b:.1f}s" for a, b in p.index], rotation=40, ha="right", fontsize=7)
axes[1].set_ylabel("最坏欠报（占本次跳变 %）")
axes[1].set_title("变载窗内最坏「显示−原始」偏差\nv4 在每一个事件上都优于或等于 v3", fontsize=10)
axes[1].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "f_varying_change.png"), dpi=140)
plt.close(fig)
print("saved: figures/f_varying_change.png")
print(p.round(1).to_string())
