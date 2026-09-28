# -*- coding: utf-8 -*-
"""步骤2b：修正版深挖诊断。

修正点：
1. 时间轴：用 frame_index 回归得到的均匀时间轴（硬件 ~100.5Hz），CSV 的 timestamp 存在
   约 40% 重复（上位机接收抖动），不可直接用于求导/滤波。
2. 通道分类：用"负载全段最大响应"而不是"首 1s 响应"，避免漏掉慢响应通道。
3. 共模回归用 lstsq 并对病态情形做了保护。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import (load_dataset, detect_segments, spatial_map, polyfit_slope,
                        r2, savefig, dump_json, DATASETS, FIG, RES, ROWS, COLS)
import matplotlib.pyplot as plt

lines = []
out = {}


def P(s=""):
    print(s)
    lines.append(s)


def lstsq_slope(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    (k, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return k, b


INFO = {}
for name in DATASETS:
    D = load_dataset(name)
    t, X, fi = D["t"], D["X"], D["frame_index"]
    # 均匀时间轴
    k, b = np.polyfit(fi, t, 1)
    tu = fi * k                        # 均匀等价时间轴
    fs = 1.0 / k
    seg = detect_segments(X.sum(axis=1), t, fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    tl = tu[c0:d0] - tu[c0]
    tpre = tu[a:bb] - tu[a]
    tpost = tu[d0:] - tu[d0]

    base = X[a:bb].mean(axis=0)
    resp_fast = X[c0:c0 + int(1 * fs)].mean(axis=0) - base
    resp_slow = X[d0 - int(5 * fs):d0].mean(axis=0) - base
    resp_max = X[c0:d0].max(axis=0) - base
    act = np.where(np.abs(resp_max) > 0.05)[0]
    qui = np.where(np.abs(resp_max) <= 0.05)[0]
    INFO[name] = dict(D=D, tu=tu, fs=fs, seg=seg, a=a, b=bb, c=c0, d=d0,
                      act=act, qui=qui, base=base, resp_max=resp_max,
                      resp_fast=resp_fast, resp_slow=resp_slow)

    P("=" * 80)
    P(f"### {name}   均匀时间轴 fs={fs:.4f}Hz  时长={tu[-1]:.3f}s  N={D['n']}")
    P(f"    分段: 前空载 0~{tu[bb]:.2f}s | 负载 {tu[c0]:.2f}~{tu[d0]:.2f}s "
      f"({tu[d0]-tu[c0]:.2f}s) | 后空载 {tu[d0]:.2f}~{tu[-1]:.2f}s")
    P(f"    通道分类(全段最大响应>0.05N): 响应 {len(act)} 个 {[D['ch_cols'][i] for i in act]}")
    P(f"                                  静默 {len(qui)} 个 {[D['ch_cols'][i] for i in qui]}")
    P(f"    死通道(全时段 std=0): {[D['ch_cols'][i] for i in np.where(X.std(axis=0)==0)[0]]}")

    w = int(2 * fs)
    drift = X[d0 - w:d0].mean(axis=0) - X[c0:c0 + w].mean(axis=0)
    zero = X[d0:].mean(axis=0) - X[a:bb].mean(axis=0)
    P(f"    受载通道 响应最大={resp_max[act].max():.4f}N 漂移={drift[act].max():.4f}N "
      f"=> 最大漂移率={drift[act].max()/resp_max[act].max()*100:.1f}%")
    P(f"    漂移/阶跃 比: 中位={np.median(drift[act]/resp_max[act]):.3f} "
      f"范围={ (drift[act]/resp_max[act]).min():.3f}~{(drift[act]/resp_max[act]).max():.3f}")
    P(f"    响应 vs 漂移 空间相关 r={np.corrcoef(resp_max[act], drift[act])[0,1]:.4f}")
    P(f"    静默通道漂移: max|.|={np.abs(drift[qui]).max():.4f}N  "
      f"max|零漂|={np.abs(zero[qui]).max():.4f}N")
    P(f"    静默通道零漂明细: " + " ".join(f"{D['ch_cols'][i]}={zero[i]:+.4f}" for i in qui))
    P(f"    受载通道零漂: 中位={np.median(zero[act]):+.4f} max={zero[act].max():+.4f} "
      f"min={zero[act].min():+.4f}  正漂移 {int((zero[act]>0).sum())}/{len(act)}")

    # ---------- 共模回归 ----------
    P(f"    -- 共模结构 --")
    for label, idxs in [("静默通道", qui), ("全部通道", np.arange(31))]:
        if len(idxs) < 2:
            continue
        ref = X[:, idxs].mean(axis=1)
        k_ref = lstsq_slope(tl, ref[c0:d0])[0]
        P(f"      [{label}] {len(idxs)}ch 均值: 负载段斜率={k_ref*1000:+.4f}e-3 N/s  "
          f"前空载斜率={lstsq_slope(tpre, ref[a:bb])[0]*1000:+.4f}  "
          f"后空载斜率={lstsq_slope(tpost, ref[d0:])[0]*1000:+.4f}")
        # 各受载通道拟合 α（前空载段，ref 几乎为 0 -> 用负载段整体最小二乘）
        alphas = []
        for j in act:
            A = np.vstack([ref[c0:d0], np.ones(len(tl))]).T
            (al, bl), *_ = np.linalg.lstsq(A, X[c0:d0, j], rcond=None)
            alphas.append(al)
        P(f"        受载通道 α(相对{label}均值) 中位={np.median(alphas):.3f} "
          f"范围={min(alphas):.3f}~{max(alphas):.3f} (离散度 σ/μ="
          f"{np.std(alphas)/abs(np.mean(alphas)):.3f})")

    # ---------- PCA ----------
    Xl = X[c0:d0]
    Xc = Xl - Xl.mean(axis=0, keepdims=True)
    # 归一化后 PCA（消除通道幅度差异，看形态一致性）
    sc = Xc.std(axis=0); sc[sc == 0] = 1.0
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Un, Sn, Vtn = np.linalg.svd(Xc / sc, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)
    evn = Sn ** 2 / np.sum(Sn ** 2)
    P(f"    PCA(原始): PC1..PC3 = " + " ".join(f"{e*100:.2f}%" for e in ev[:3]))
    P(f"    PCA(列归一化): PC1..PC3 = " + " ".join(f"{e*100:.2f}%" for e in evn[:3]))
    P(f"    PC1载荷 受载均值={Vtn[0][act].mean():+.4f} 静默均值={Vtn[0][qui].mean():+.4f}"
      if len(qui) else "")
    # PC1 score 与主通道漂移形态对比
    s1 = Un[:, 0] * Sn[0]
    if np.corrcoef(s1, X[c0:d0, act[np.argmax(resp_max[act])]])[0, 1] < 0:
        s1 = -s1
    cc = np.corrcoef(s1, X[c0:d0, act[np.argmax(resp_max[act])]])[0, 1]
    P(f"    PC1 score vs 主通道 相关 r={cc:.5f}")

    # ---------- 每通道漂移形态一致性（形状是否一致） ----------
    P(f"    -- 漂移形态一致性 --")
    shapes = []
    for j in act:
        y = X[c0:d0, j] - base[j]
        yy = y - y[:int(2 * fs)].mean()
        amp = yy[-w:].mean()
        if abs(amp) > 0.05:
            shapes.append(yy / amp)
    if shapes:
        Smat = np.vstack(shapes)
        shape_mean = Smat.mean(axis=0)
        shape_dev = Smat.std(axis=0)
        P(f"      归一化漂移形态: 通道间平均相对散度={np.mean(shape_dev/np.abs(shape_mean)[None,:]+1e-9):.4f} "
          f"(在 10%,50%,90% 时刻的形态值="
          f"{shape_mean[int(0.1*len(shape_mean))]:.3f},{shape_mean[int(0.5*len(shape_mean))]:.3f},"
          f"{shape_mean[int(0.9*len(shape_mean))]:.3f})")

    out[name] = dict(
        fs=fs, t_load=float(tu[d0] - tu[c0]), t_pre=float(tu[bb]), t_post=float(tu[-1] - tu[d0]),
        act=[int(i) for i in act], qui=[int(i) for i in qui],
        drift=drift, zero=zero, resp_max=resp_max, base=base,
        ev=ev[:5], evn=evn[:5],
    )

dump_json(out, "B_diagnostics.json")
with open(os.path.join(RES, "B_diagnostics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[ok] 步骤2b 完成")
