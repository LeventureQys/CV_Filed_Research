# -*- coding: utf-8 -*-
"""变化负载（含中途加重）上的快相免责期检验：免责期会不会妨碍阶跃检测？

v3 的切换跳变缺陷来自「检测滞后期的偏差被记成蠕变，再被一次性清零」。
免责期把 onset 后 0~5s 的补偿冻结，理论上让检测器工作在一段干净信号上。
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
import importlib
rf = importlib.import_module("r_fastphase")
GLM53v4 = rf.GLM53v4
from glm53_v3 import GLM53v3  # noqa: E402

CASES = [("变化负载", "零负载-切换负载-零负载-再切换负载", "数据A"),
         ("变化负载", "零负载-中途切换负载-零负载-切换负载", "数据B")]


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


print("=" * 108)
print("变化负载：快相免责期对阶跃检测的影响  ·  temp/v4.1flash/scripts/u_varying_fast.py")
print("=" * 108)
rows = []
for loc, name, tag in CASES:
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
    segl = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)
    s0, s1 = segl[0]
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    Yv3 = run(tu, Xu, GLM53v3)
    Yv4 = run(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)
    nL = s1 - s0
    segl_s = ", ".join(f"{tu[a]:.1f}~{tu[min(b, len(tu)-1)]:.1f}s" for a, b in segl)
    print(f"\n=== {tag}（{name}）ch{m}  {tu[-1]:.1f}s  负载区间: " + segl_s + " ===")
    for lab, Y in [("v3", Yv3), ("v4_fast5", Yv4)]:
        dr = (Y[s0 + int(0.9 * nL):s1, m].mean() - Y[s0:s0 + int(0.1 * nL), m].mean())
        dY, dX = np.diff(Y[:, m]), np.diff(Xu[:, m])
        exc = np.abs(dY - dX)
        kmax = int(np.argmax(exc[s0:s1])) + s0
        # 负载段内最大单帧跳变（显示相对原始的超额）
        print(f"  {lab:<9} 负载段电平漂移={dr:+.3f}  最大事件跳变超额={exc[kmax]:.1f} ADC @ t={tu[kmax]:.2f}s"
              f"  (原始同帧跳变={dX[kmax]:+.1f}, 显示跳变={dY[kmax]:+.1f})")
        rows.append(dict(dataset=tag, algo=lab, level_drift=dr,
                         jump_excess=float(exc[kmax]), jump_t=float(tu[kmax]),
                         raw_jump=float(dX[kmax]), disp_jump=float(dY[kmax])))
    # 显示总量时间线抽样（中途加重附近）
    print(f"  {'t(s)':>7} {'原始总量':>12} {'v3 显示':>12} {'v4 显示':>12} {'v3-v4':>10}")
    for tt in [8, 12, 16, 20, 21, 22, 23, 24, 26, 28, 32, 40, 55, 62, 70]:
        k = int(np.searchsorted(tu, tt))
        if k < len(tu):
            print(f"  {tt:7.1f} {tot[k]:12.0f} {Yv3[k].sum():12.0f} {Yv4[k].sum():12.0f} "
                  f"{Yv3[k].sum()-Yv4[k].sum():10.0f}")

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "varying_fastphase.csv"), index=False, encoding="utf-8-sig")
print("\nsaved: results/varying_fastphase.csv")
