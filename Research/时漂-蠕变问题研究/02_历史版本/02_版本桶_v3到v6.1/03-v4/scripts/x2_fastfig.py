# -*- coding: utf-8 -*-
"""快相分析图：归一化快相轮廓（9 组叠加）+ 快/慢相分解 + v3/v4 时序对比。"""
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
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
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


def smooth(x, dt, tau=2.0):
    a = dt / tau
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def prep(loc, name):
    t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    ld = tot > 0.15 * tot.max()
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    s0, s1 = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    base = tot[:s0].mean()
    n_on = next(i for i in range(s0, s0 + 300)
                if tot[i] > base + 0.05 * (tot[s0:s1].max() - base))
    return dict(tu=tu, Xu=Xu, dt=dt, s0=s0, s1=s1, m=m, n_on=n_on, amp=amp[m])


# ---------- 图 1：归一化快相轮廓 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
COL = {"右拇指指尖": "tab:blue", "左拇指指尖": "tab:green", "四指指尖": "tab:red"}
for loc in COL:
    for name in ["数据1", "数据2", "数据3"]:
        p = prep(loc, name)
        Xs = smooth(p["Xu"][:, p["m"]], p["dt"], 2.0)
        b = Xs[:p["s0"]].mean()
        a5 = Xs[p["n_on"] + int(round(5.0 / p["dt"]))] - b
        tt = np.arange(0, 8, 1 / p["dt"])
        kk = p["n_on"] + np.round(tt / p["dt"]).astype(int)
        v = (Xs[kk] - b) / a5
        axes[0].plot(tt, v, color=COL[loc], lw=1.1, alpha=0.75,
                     label=loc if name == "数据1" else None)
axes[0].axvline(1.0, color="k", ls=":", lw=1.0)
axes[0].axvline(4.0, color="k", ls="--", lw=1.0)
axes[0].text(1.05, 0.05, "机械加载段结束 ≈1s", fontsize=8)
axes[0].text(4.05, 0.05, "快相基本结束 ≈4s", fontsize=8)
axes[0].set_xlim(0, 8)
axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel("加载后时间 (s)")
axes[0].set_ylabel("归一化读数（除以 onset+5s 处的增量）")
axes[0].set_title("9 组录制的归一化「快相」轮廓\n三条位置的曲线几乎重合 ⇒ 形状高度可复现", fontsize=10)
axes[0].legend(fontsize=8, loc="lower right")

# 右：快慢相占比堆叠柱
labs, fast, slow = [], [], []
for loc in COL:
    for name in ["数据1", "数据2", "数据3"]:
        p = prep(loc, name)
        Xs = smooth(p["Xu"][:, p["m"]], p["dt"], 2.0)
        b = Xs[:p["s0"]].mean()
        f = Xs[p["n_on"] + int(round(4.0 / p["dt"]))] - Xs[p["n_on"]]
        t_ = Xs[p["s1"] - 1] - Xs[p["n_on"] + int(round(4.0 / p["dt"]))]
        labs.append(f"{loc[:2]}/{name[-1]}")
        fast.append(f)
        slow.append(t_)
xs = np.arange(len(labs))
axes[1].bar(xs, np.array(fast) / (np.array(fast) + np.array(slow)) * 100,
            color="tab:orange", label="快相 0~4s")
axes[1].bar(xs, np.array(slow) / (np.array(fast) + np.array(slow)) * 100,
            bottom=np.array(fast) / (np.array(fast) + np.array(slow)) * 100,
            color="tab:blue", label="慢相 4s~结束")
axes[1].set_xticks(xs)
axes[1].set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
axes[1].set_ylabel("占总增量 (%)")
axes[1].set_title("快相（0~4s）与慢相（4s~结束）的增量占比", fontsize=10)
axes[1].legend(fontsize=8)

# 右二：v3 vs v4 时序（右拇指数据1）
p = prep("右拇指指尖", "数据1")
Yv3 = run(p["tu"], p["Xu"], GLM53v3)
Yv4 = run(p["tu"], p["Xu"], GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)
tt = p["tu"] - p["tu"][p["n_on"]]
sel = slice(p["n_on"], min(p["s1"], p["n_on"] + int(40 / p["dt"])))
axes[2].plot(tt[sel], p["Xu"][sel, p["m"]], color="0.6", lw=0.8, label="原始")
axes[2].plot(tt[sel], Yv3[sel, p["m"]], color="tab:blue", lw=1.1, label="v3（现有）")
axes[2].plot(tt[sel], Yv4[sel, p["m"]], color="tab:red", lw=1.1, label="v4（快相免责 5s）")
axes[2].axvspan(0, 5, color="orange", alpha=0.12)
axes[2].set_xlim(0, 40)
axes[2].set_xlabel("加载后时间 (s)")
axes[2].set_ylabel("读数")
axes[2].set_title("右拇指/数据1 ch17：免责期内显示直通\n（橙色区），之后只对慢相建蠕变模型", fontsize=10)
axes[2].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "f_fastphase_1.png"), dpi=140)
plt.close(fig)
print("saved: figures/f_fastphase_1.png")
