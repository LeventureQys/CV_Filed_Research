# -*- coding: utf-8 -*-
"""指数蠕变拟合扣除的敏感性分析：
A. 初值鲁棒性（扰动 p0）
B. 因果化代价：只用负载段前 N 秒拟合，外推扣除整段 → 残余漂移退化曲线
C. 单指数 vs 双指数模型选择
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
    segs = sorted(zip(starts, ends), key=lambda s: s[1] - s[0], reverse=True)
    return segs[0]


def exp1(t, a, tau, c):
    return a * (1 - np.exp(-t / tau)) + c


def exp2(t, a1, t1, a2, t2, c):
    return a1 * (1 - np.exp(-t / t1)) + a2 * (1 - np.exp(-t / t2)) + c


data = {}
for name in DATASETS:
    df = load_csv(os.path.join(BASE, name, "device_001_seg000.csv"))
    t = df["elapsed"].to_numpy()
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    X = df[ch_cols].to_numpy()
    s0, s1 = segment(t, X.sum(axis=1))
    load_mean = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    mc = int(np.argmax(load_mean))
    sig = X[s0:s1, mc] - X[:s0, mc].mean()
    tl = t[s0:s1] - t[s0]
    n10 = max(len(sig) // 10, 10)
    step = sig[:n10].mean()
    data[name] = dict(tl=tl, sig=sig, step=step, n10=n10, mc=mc)


def resid_drift(corrected, n10, step):
    return abs(corrected[-n10:].mean() - corrected[:n10].mean()) / step * 100


# ---------- A. 初值鲁棒性 ----------
print("== A. 初值鲁棒性（扰动 p0，全段拟合）==")
rng = np.random.default_rng(0)
for name in DATASETS:
    d = data[name]
    res = []
    for trial in range(20):
        p0 = [d["sig"][-1] - d["sig"][0], max(d["tl"][-1] / 3, 1.0), d["sig"][0]]
        p0 = [p * (1 + rng.uniform(-0.5, 0.5)) for p in p0]
        p0[1] = max(p0[1], 0.5)
        try:
            popt, _ = curve_fit(exp1, d["tl"], d["sig"], p0=p0, maxfev=20000)
            corr = d["sig"] - (exp1(d["tl"], *popt) - exp1(0.0, *popt))
            res.append(resid_drift(corr, d["n10"], d["step"]))
        except Exception:
            res.append(np.nan)
    res = np.array(res)
    print(f"{name}: 20次随机初值 残余漂移 mean={np.nanmean(res):.2f}% "
          f"max={np.nanmax(res):.2f}% 失败={np.isnan(res).sum()}次")

# ---------- B. 因果化代价 ----------
print("\n== B. 只用负载段前 N 秒拟合（模拟在线早期定参），外推扣除整段 ==")
windows = [5, 10, 20, 40, 60, 90, 10**9]
rows_b = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    d = data[name]
    ax.plot(d["tl"], d["sig"], lw=0.4, color="gray", label="原始(扣固定基线)")
    for w in windows:
        m = d["tl"] <= w
        if m.sum() < 50:
            continue
        try:
            p0 = [d["sig"][m][-1] - d["sig"][m][0], max(d["tl"][m][-1] / 3, 1.0), d["sig"][m][0]]
            popt, _ = curve_fit(exp1, d["tl"][m], d["sig"][m], p0=p0, maxfev=20000)
            corr = d["sig"] - (exp1(d["tl"], *popt) - exp1(0.0, *popt))
            r = resid_drift(corr, d["n10"], d["step"])
            lbl = "全段(离线)" if w > 1e6 else f"前{w}s"
            rows_b.append([name, lbl, r, popt[1]])
            ax.plot(d["tl"], corr, lw=1.0,
                    label=f"{lbl}拟合→残余{r:.1f}%")
        except Exception as e:
            rows_b.append([name, f"前{w}s", np.nan, np.nan])
    ax.set_title(f"{name} 主通道 ch{d['mc']}")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIG, "e1_causality_cost.png"), dpi=140)
plt.close(fig)
df_b = pd.DataFrame(rows_b, columns=["数据集", "拟合窗口", "残余漂移%", "拟合τ(s)"])
print(df_b.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
df_b.to_csv(os.path.join(OUT, "causality_windows.csv"), index=False, encoding="utf-8-sig")

# ---------- C. 单指数 vs 双指数 ----------
print("\n== C. 单指数 vs 双指数（全段离线拟合）==")
for name in DATASETS:
    d = data[name]
    popt1, _ = curve_fit(exp1, d["tl"], d["sig"],
                         p0=[d["sig"][-1] - d["sig"][0], 50, d["sig"][0]], maxfev=20000)
    r1 = 1 - np.var(d["sig"] - exp1(d["tl"], *popt1)) / np.var(d["sig"])
    corr1 = d["sig"] - (exp1(d["tl"], *popt1) - exp1(0.0, *popt1))
    rd1 = resid_drift(corr1, d["n10"], d["step"])
    try:
        popt2, _ = curve_fit(exp2, d["tl"], d["sig"],
                             p0=[0.3, 5, 0.3, 60, d["sig"][0]], maxfev=40000)
        r2 = 1 - np.var(d["sig"] - exp2(d["tl"], *popt2)) / np.var(d["sig"])
        corr2 = d["sig"] - (exp2(d["tl"], *popt2) - exp2(0.0, *popt2))
        rd2 = resid_drift(corr2, d["n10"], d["step"])
        print(f"{name}: 单指数 R²={r1:.4f} 残余={rd1:.2f}% | 双指数 R²={r2:.4f} 残余={rd2:.2f}% "
              f"(τ1={popt2[1]:.1f}s τ2={popt2[3]:.1f}s)")
    except Exception as e:
        print(f"{name}: 单指数 R²={r1:.4f} 残余={rd1:.2f}% | 双指数拟合失败: {e}")

print("\nsaved e1_causality_cost.png + causality_windows.csv")
