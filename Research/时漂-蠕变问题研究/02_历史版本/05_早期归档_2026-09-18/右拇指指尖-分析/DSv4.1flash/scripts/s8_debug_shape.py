# -*- coding: utf-8 -*-
"""步骤8：调试阵列共享形态估计。"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, DATASETS
import tac_algorithms as AL

np.set_printoptions(precision=4, suppress=True, linewidth=150)

for name in DATASETS[:1]:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), D["t"], fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    base = X[a:bb].mean(axis=0)
    Xn = X - base
    s = c0
    Xl = Xn[s:]
    tt = t[s:] - t[s]
    w = lambda x: int(round(x * fs))
    t_ref = 0.5
    ir = int(t_ref * fs)
    hw = w(0.25)
    m = Xl[max(0, ir - hw):min(len(tt), ir + hw)].mean(axis=0)
    alive = (Xl.max(axis=0) - Xl.min(axis=0)) > 1e-6
    resp = Xl[w(0.05):w(0.5)].mean(axis=0)
    use = alive & (np.abs(resp) > 0.05) & (np.abs(m) > 1e-6)
    print(f"{name}: alive={alive.sum()} use={use.sum()}  有效通道={[D['ch_cols'][i] for i in np.where(use)[0]]}")
    print(f"  m (锚点窗口均值, mN) = {np.round(m[use]*1000,2)}")
    print(f"  resp (0.05~0.5s, mN) = {np.round(resp[use]*1000,2)}")
    wts = np.abs(Xl[-w(2.0):].mean(axis=0)) * use
    wts = np.where(use, np.clip(wts, 0, None), 0.0)
    print(f"  weights(late) = {np.round(wts[use],4)}")
    U = Xl[:, use] / m[use][None, :]
    W = wts[use] / wts[use].sum()
    h_emp = U @ W
    print(f"  h_emp: min={h_emp.min():.4f} max={h_emp.max():.4f} "
          f"h_emp[0]={h_emp[0]:.4f} h_emp[ir]={h_emp[ir]:.4f} h_emp[-1]={h_emp[-1]:.4f}")
    print("  h_emp 采样:")
    for frac in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        i = min(int(frac * (len(h_emp) - 1)), len(h_emp) - 1)
        print(f"    t={tt[i]:7.2f}s  h_emp={h_emp[i]:.4f}")
    # 逐通道相对漂移比（末端/锚点）
    ratio = Xl[-w(2):].mean(axis=0)[use] / m[use]
    print(f"  逐通道 末端/锚点 比值 = {np.round(ratio,3)}")
    print(f"    中位={np.median(ratio):.3f} 加权={np.average(ratio, weights=wts[use]):.3f}")
    # 拟合
    sm = AL.ShapeModel("power", t_floor=0.05)
    ok = sm.fit(tt - t_ref, h_emp)
    print(f"  拟合 power: ok={ok} params={sm.params} R²={sm.r2}")
    if ok:
        hh = sm(tt - t_ref)
        print(f"    fitted h: h[0]={hh[0]:.4f} h[-1]={hh[-1]:.4f} 比值={hh[-1]/hh[0]:.3f}")
    sm2 = AL.ShapeModel("log", t_floor=0.05)
    ok2 = sm2.fit(tt - t_ref, h_emp)
    print(f"  拟合 log: ok={ok2} params={sm2.params} R²={sm2.r2}")
    if ok2:
        hh = sm2(tt - t_ref)
        print(f"    fitted h: h[0]={hh[0]:.4f} h[-1]={hh[-1]:.4f} 比值={hh[-1]/hh[0]:.3f}")

    # 直接用 31 通道总和来做形态（更简单直观）
    tot = Xn[:, use].sum(axis=1)
    h_tot = tot / tot[ir]
    print(f"  基于总和的形态: h[0]={h_tot[s]:.4f} h[ir]={h_tot[s+ir]:.4f} "
          f"h[-1]={h_tot[-1]:.4f}  (负载起点处应为0)")
    # 用 t>=t_ref 段归一化到 t_ref
    tt2 = tt.copy()
    m2 = tt2 >= t_ref
    sm3 = AL.ShapeModel("power", t_floor=0.0)
    z = h_emp / h_emp[ir]
    ok3 = sm3.fit(tt2[m2] - t_ref, z[m2])
    print(f"  仅在 t>=t_ref 拟合 power: ok={ok3} params={sm3.params} R²={sm3.r2}")
    if ok3:
        hh = sm3(tt - t_ref)
        print(f"    fitted h: h[0]={hh[0]:.4f} h[-1]={hh[-1]:.4f} 比值={hh[-1]/hh[0]:.3f}")
