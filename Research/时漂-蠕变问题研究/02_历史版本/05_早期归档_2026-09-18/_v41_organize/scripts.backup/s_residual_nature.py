# -*- coding: utf-8 -*-
"""慢相残差的性质：是「线性剩余」（可标定/可接受）还是「非线性蠕变」（算法必须处理）？

对免责期结束后的慢相段，把显示值减去原始值得到「残差轨迹」，拆成：
  线性分量  —— 去除后剩下的 RMS（= 非线性残余，算法真正要消掉的东西）
  偏置分量  —— 段首残差（免责区留下的平滑偏移，可标定）
输出每个算法的：残差 RMS、去线性后 RMS、偏置、线性斜率。
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
from r_fastphase import GLM53v4  # noqa: E402

DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]
FAST_S = 5.0


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def segs(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


rows = []
print("=" * 104)
print("慢相残差分解（9 组恒载，主通道）  ·  temp/v4.1flash/scripts/s_residual_nature.py")
print("=" * 104)
print("残差 = 补偿后显示 − 原始；线性分量可由出厂标定吸收，非线性分量才是算法缺陷。\n")
for loc, name in DATASETS:
    t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    s0r, s1r = segs(X.sum(axis=1))[0]
    s0, s1 = int(np.searchsorted(tu, t[s0r])), int(np.searchsorted(tu, t[s1r]))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    tot = Xu.sum(axis=1)
    base = tot[:s0].mean()
    thr = base + 0.05 * (tot[s0:s1].max() - base)
    n_on = next((i for i in range(s0, s0 + 300) if tot[i] > thr), s0)
    a5 = int(np.searchsorted(tu, tu[n_on] + FAST_S))
    a5 = max(a5, s0)
    Yv3 = run(tu, Xu, GLM53v3)
    Yv4 = run(tu, Xu, GLM53v4, FAST_S=FAST_S, A_W0_V4=3.5, A_W1_V4=5.0)
    print(f"[{loc}/{name}] ch{m}  慢相段 {tu[a5]-tu[n_on]:.0f}s~{tu[s1]-tu[n_on]:.0f}s（长 {(s1-a5)*dt:.0f}s） 幅度={amp[m]:.3f}")
    for lab, Y in [("v3", Yv3), ("v4_fast5", Yv4)]:
        res = (Y[a5:s1, m] - Xu[a5:s1, m])
        tt = tu[a5:s1] - tu[a5]
        k, b0 = np.polyfit(tt, res, 1)
        lin = k * tt + b0
        nonlin = res - lin
        rows.append(dict(location=loc, dataset=name, algo=lab, main_ch=m,
                         slow_len_s=(s1 - a5) * dt, amp=amp[m],
                         resid_bias=float(res[0]), resid_slope_per10s=float(k * 10),
                         resid_rms=float(np.sqrt(np.mean(res ** 2))),
                         resid_lin_pct=100 * abs(k * (s1 - a5) * dt) / amp[m],
                         resid_nonlin_rms=float(np.sqrt(np.mean(nonlin ** 2))),
                         resid_nonlin_pct=100 * np.sqrt(np.mean(nonlin ** 2)) / amp[m]))
        print(f"    {lab:<9} 段首偏置={res[0]:+.4f}  线性斜率={k*10:+.4f}/10s  "
              f"线性占比={100*abs(k*(s1-a5)*dt)/amp[m]:5.2f}%  "
              f"残差RMS={np.sqrt(np.mean(res**2)):.4f}  去线性后RMS={np.sqrt(np.mean(nonlin**2)):.4f} "
              f"({100*np.sqrt(np.mean(nonlin**2))/amp[m]:.2f}% 幅)")

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "residual_nature.csv"), index=False, encoding="utf-8-sig")
agg = dfm.groupby("algo").agg(
    段首偏置=("resid_bias", "mean"),
    线性漂移_占幅pct=("resid_lin_pct", "mean"),
    残差RMS=("resid_rms", "mean"),
    去线性后RMS=("resid_nonlin_rms", "mean"),
    非线性残差_占幅pct=("resid_nonlin_pct", "mean"),
).round(4)
print("\n===== 9 组汇总 =====")
print(agg.to_string())
print("\n【读法】「线性漂移」是免责期留下的平滑偏移，可用出厂标定吸收；")
print("        「非线性残差」是算法必须消掉的蠕变残余 —— 越小越好。")
print("saved: results/residual_nature.csv")
