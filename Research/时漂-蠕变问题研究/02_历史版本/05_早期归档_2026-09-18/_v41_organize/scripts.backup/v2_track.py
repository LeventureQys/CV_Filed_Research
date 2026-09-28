# -*- coding: utf-8 -*-
"""决定性指标：补偿器跟踪「真实蠕变曲线」的精度。

做法（不引入任何前视）：
  1. 用长窗移动平均（τ=2s）从原始主通道提取平滑曲线 Ẋ —— 蠕变是慢变量，平滑后
     保留蠕变、抹掉噪声与快相抖动；它只用于**评估**，不是算法组成部分；
  2. 定义理想补偿曲线 Creep*(t) = Ẋ(t) − X(t_ref)，其中 t_ref 取名义弹性参考时刻；
  3. 对每个算法，实际补偿曲线 Comp(t) = Ẋ(t) − Y(t)（用平滑后的原始对比显示值，
     避免显示端噪声污染评估）；
  4. 报告 Comp 与 Creep* 的偏差：平均偏差（系统性欠/过补偿）、RMS 偏差、
     以及「过补偿」与「欠补偿」的时间占比。

这样能直接把「免责区到底帮了什么忙」量化出来。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3  # noqa: E402
import importlib
_rf = importlib.import_module("r_fastphase")
GLM53v4 = _rf.GLM53v4

DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]


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


rows = []
print("=" * 110)
print("补偿曲线跟踪精度（9 组恒载）  ·  temp/v4.1flash/scripts/v2_track.py")
print("=" * 110)
print("Comp(t) = 平滑原始 − 显示；Creep*(t) = 平滑原始(t) − 平滑原始(t_ref)，t_ref = onset+0.15s")
print("偏差 = Comp − Creep*：正 = 过补偿（扣多了），负 = 欠补偿（扣少了）。\n")
for loc, name in DATASETS:
    t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    thr = 0.15 * tot.max()
    ld = tot > thr
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
    thr2 = base + 0.05 * (tot[s0:s1].max() - base)
    n_on = next((i for i in range(s0, s0 + 300) if tot[i] > thr2), s0)

    Xs = smooth(Xu[:, m], dt, 2.0)
    k_ref = n_on + max(1, int(round(0.15 / dt)))
    creep_star = Xs - Xs[k_ref]

    Yv3 = run(tu, Xu, GLM53v3)
    Yv4 = run(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)

    print(f"[{loc}/{name}] ch{m}  onset={tu[n_on]:.2f}s  负载段 {tu[s0]:.1f}~{tu[s1]:.1f}s  "
          f"真蠕变幅度={creep_star[s1-1]:.3f}  快相(到5s)={creep_star[n_on+int(5/dt)]:.3f}"
          f"（占 {100*creep_star[n_on+int(5/dt)]/creep_star[s1-1]:.0f}%）")
    for lab, Y in [("v3", Yv3), ("v4_fast5", Yv4)]:
        comp = Xs - smooth(Y[:, m], dt, 2.0)
        seg = slice(n_on + int(2.0 / dt), s1)      # 看 onset+2s 之后
        dev = comp[seg] - creep_star[seg]
        cs = creep_star[seg]
        over = float(np.mean(dev > 0.05 * max(cs.max(), 1e-9)))
        under = float(np.mean(dev < -0.05 * max(cs.max(), 1e-9)))
        rows.append(dict(location=loc, dataset=name, algo=lab, main_ch=m,
                         creep_amp=float(cs.max()),
                         dev_mean=float(dev.mean()), dev_rms=float(np.sqrt(np.mean(dev ** 2))),
                         dev_rms_pct=100 * float(np.sqrt(np.mean(dev ** 2))) / amp[m],
                         dev_max=float(np.abs(dev).max()),
                         over_frac=over, under_frac=under))
        print(f"    {lab:<9} 平均偏差={dev.mean():+.4f}  RMS偏差={np.sqrt(np.mean(dev**2)):.4f} "
              f"({100*np.sqrt(np.mean(dev**2))/amp[m]:.2f}% 幅)  最大偏差={np.abs(dev).max():.4f}  "
              f"过补偿时间占比={100*over:.0f}% 欠补偿={100*under:.0f}%")

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "track_accuracy.csv"), index=False, encoding="utf-8-sig")
agg = dfm.groupby("algo").agg(
    平均偏差=("dev_mean", "mean"),
    偏差RMS=("dev_rms", "mean"),
    偏差RMS_占幅pct=("dev_rms_pct", "mean"),
    最大偏差=("dev_max", "mean"),
    过补偿时间占比=("over_frac", "mean"),
    欠补偿时间占比=("under_frac", "mean"),
).round(4)
print("\n===== 9 组汇总 =====")
print(agg.to_string())
print("\nsaved: results/track_accuracy.csv")
