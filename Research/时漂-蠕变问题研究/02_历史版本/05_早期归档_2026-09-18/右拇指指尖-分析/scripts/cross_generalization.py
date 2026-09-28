# -*- coding: utf-8 -*-
"""蠕变模型跨数据泛化测试：用其它数据集标定的蠕变参数补偿当前数据集（模拟在线模型补偿）
模型: r(t) = 1 + a*(1-exp(-t/tau))  —— 单位阶跃下响应增长比
补偿: corrected = measured / r(t-t_load)，再乘回首段均值保持量纲
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "figures")
OUT = os.path.join(BASE, "results")
DATASETS = ["数据1", "数据2", "数据3"]


def load_csv(path):
    return pd.read_csv(path, skiprows=24)


def segment(t, total, thr_ratio=0.15):
    thr = thr_ratio * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if loaded[0]:
        starts = np.r_[0, starts]
    if loaded[-1]:
        ends = np.r_[ends, len(loaded)]
    segs = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)
    return segs[0]


def exp_model(t, a, tau, c):
    return a * (1 - np.exp(-t / tau)) + c


data = {}
for name in DATASETS:
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    total = X.sum(axis=1)
    s0, s1 = segment(t, total)
    load_mean = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    main_ch = int(np.argmax(load_mean))
    loaded_mask = load_mean > 0.3 * load_mean.max()
    data[name] = dict(t=t, X=X, s0=s0, s1=s1, main_ch=main_ch,
                      loaded_mask=loaded_mask, ch_cols=ch_cols)

# 1) 每数据集拟合主通道蠕变参数 a, tau（相对阶跃归一）
params = {}
for name in DATASETS:
    d = data[name]
    t, s0, s1, mc = d["t"], d["s0"], d["s1"], d["main_ch"]
    sig = d["X"][s0:s1, mc] - d["X"][:s0, mc].mean()
    tl = t[s0:s1] - t[s0]
    popt, _ = curve_fit(exp_model, tl, sig,
                        p0=[sig[-1] - sig[0], 50, sig[0]], maxfev=20000)
    a, tau, c = popt
    params[name] = dict(a_ratio=a / c, tau=tau, c=c)
    print(f"{name}: 蠕变幅度比 a/c={a/c:.3f}  τ={tau:.1f}s")

# 2) 留一法：用其它两数据集的平均参数补偿本数据集全阵列
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
rows = []
for ax, name in zip(axes, DATASETS):
    others = [n for n in DATASETS if n != name]
    a_cal = np.mean([params[n]["a_ratio"] for n in others])
    tau_cal = np.mean([params[n]["tau"] for n in others])
    d = data[name]
    t, s0, s1 = d["t"], d["s0"], d["s1"]
    tl = t[s0:s1] - t[s0]
    r = 1 + a_cal * (1 - np.exp(-tl / tau_cal))

    Xb = d["X"] - d["X"][:s0].mean(axis=0)   # 固定基线
    Y = Xb.copy()
    # 只补偿受载通道；以负载首5%均值作为阶跃估计，除蠕变比后再归一
    n5 = max((s1 - s0) // 20, 5)
    loaded = d["loaded_mask"]
    Y[s0:s1][:, loaded] = Xb[s0:s1][:, loaded] / r[:, None] * 1.0
    # 说明: r(0)=1 故首段不变，末段被压回

    mc = d["main_ch"]
    n10 = max((s1 - s0) // 10, 10)
    raw_step = Xb[s0:s0 + n10, mc].mean()
    resid = abs(Y[s1 - n10:s1, mc].mean() - Y[s0:s0 + n10, mc].mean()) / raw_step * 100
    rows.append([name, a_cal, tau_cal, resid])
    print(f"{name}: 用 {others} 参数(a={a_cal:.3f},τ={tau_cal:.0f}s)补偿 → 主通道残余漂移={resid:.1f}%")

    ax.plot(tl, Xb[s0:s1, mc], lw=0.5, color="gray", label="原始(扣固定基线)")
    ax.plot(tl, Y[s0:s1, mc], lw=1.2, color="tab:red",
            label=f"跨数据蠕变补偿(残余{resid:.1f}%)")
    ax.set_title(f"{name} 主通道 ch{mc}\n标定参数: a={a_cal:.2f}, τ={tau_cal:.0f}s")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "d6_cross_calibration.png"), dpi=140)
plt.close(fig)

pd.DataFrame(rows, columns=["数据集", "标定a", "标定tau", "主通道残余漂移%"]).to_csv(
    os.path.join(OUT, "cross_calibration.csv"), index=False, encoding="utf-8-sig")
print("saved d6 + cross_calibration.csv")
