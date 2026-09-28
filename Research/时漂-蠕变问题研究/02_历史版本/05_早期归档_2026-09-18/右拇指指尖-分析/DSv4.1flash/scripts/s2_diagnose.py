# -*- coding: utf-8 -*-
"""步骤2：深挖诊断——时间轴质量、共模结构、通道漂移一致性、死通道、共模补偿可行性。"""
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


for name in DATASETS:
    D = load_dataset(name)
    t, X = D["t"], D["X"]
    P("=" * 78)
    P(f"### {name} 时间轴质量诊断")

    # timestamp 间隔分布
    dt = np.diff(t)
    hist, edges = np.histogram(dt * 1000, bins=[0, 0.4, 3, 8, 13, 18, 23, 30,
                                                40, 50, 70, 100, 1e9])
    P("    帧间隔直方图 (ms):")
    for i, h in enumerate(hist):
        P(f"      [{edges[i]:>7.1f},{edges[i+1]:>8.1f})  {h:>7d}  {100*h/len(dt):>6.2f}%")
    # elapsed 与 timestamp 差异
    tr = D["t_raw"]
    dtr = np.diff(tr)
    P(f"    elapsed 间隔: 唯一值={sorted(set(np.round(dtr, 6)))[:8]} ... "
      f"零间隔占比={100*np.mean(dtr == 0):.1f}%")
    P(f"    timestamp 零间隔占比={100*np.mean(dt == 0):.2f}%  负间隔={int((dt<0).sum())}")
    # 用 frame_index 与实际时间做回归，估计真实采样率
    fi = D["frame_index"]
    k, b = np.polyfit(fi, t, 1)
    P(f"    frame_index→时间 回归: 斜率={k*1000:.4f} ms/frame  => 名义采样率={1/k:.3f} Hz "
      f"(R²={r2(t, k*fi+b):.6f})")
    P(f"    名义帧数/时长 = {D['n']/t[-1]:.3f} fps")
    # 有效独立采样率：按实际不同时间戳
    uniq = np.unique(t)
    P(f"    去重后唯一时间戳数={len(uniq)} => 有效上限采样率={len(uniq)/t[-1]:.3f} Hz")
    # 用 3 阶多项式看是否有时间轴压缩
    for deg in (1, 2, 3):
        c = np.polyfit(fi, t, deg)
        P(f"      {deg}阶多项式拟合 frame_index→t 的 R²={r2(t, np.polyval(c, fi)):.8f}")

    seg = detect_segments(X.sum(axis=1), t, fs=D["fs"])
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    tl = t[c0:d0] - t[c0]
    activ = X[c0:c0 + int(1 * D["fs"])].mean(axis=0) - X[a:bb].mean(axis=0)
    act = np.where(np.abs(activ) > 0.05)[0]
    qui = np.where(np.abs(activ) <= 0.05)[0]

    P(f"### {name} 死通道/静默通道检查")
    allstd = X.std(axis=0)
    P(f"    全时段 std=0 的通道: {[D['ch_cols'][i] for i in np.where(allstd == 0)[0]]}")
    P(f"    前空载 std<0.002 的通道: {[D['ch_cols'][i] for i in np.where(X[a:bb].std(axis=0) < 0.002)[0]]}")
    P(f"    受载通道数={len(act)}  静默通道数={len(qui)}")

    P(f"### {name} 通道漂移一致性 (. 漂移量 = 负载末段均值 - 负载首段均值)")
    w = int(2 * D["fs"])
    drift = X[d0 - w:d0].mean(axis=0) - X[c0:c0 + w].mean(axis=0)
    P(f"    受载通道漂移: " + " ".join(
        f"{D['ch_cols'][i]}={drift[i]:+.3f}" for i in act))
    P(f"    静默通道漂移: " + " ".join(
        f"{D['ch_cols'][i]}={drift[i]:+.4f}" for i in qui))
    P(f"    受载通道漂移/阶跃比: " + " ".join(
        f"{drift[i]/activ[i]:+.3f}" for i in act if abs(activ[i]) > 1e-6))
    P(f"    漂移符号: 受载通道正漂移 {int((drift[act] > 0).sum())}/{len(act)}  "
      f"静默通道正漂移 {int((drift[qui] > 0).sum())}/{len(qui)}")

    P(f"### {name} 共模补偿可行性（静默通道当参考）")
    for kk in ("mean", "median", "pca"):
        if kk == "mean":
            ref = X[:, qui].mean(axis=1)
            label = f"静默通道均值({len(qui)})"
        elif kk == "median":
            ref = np.median(X[:, qui], axis=1)
            label = f"静默通道中位({len(qui)})"
        else:
            continue
        k_ref = polyfit_slope(tl, ref[c0:d0])[0]
        # 主通道残差
        # 用回归系数把 ref 映射到主通道
        alpha = np.polyfit(ref[a:bb], X[a:bb, act[0]], 1)[0]
        resid = X[:, act[0]] - alpha * ref
        k_res = polyfit_slope(tl, resid[c0:d0])[0]
        P(f"    [{label}] 参考斜率={k_ref*1000:+.4f}e-3  "
          f"主通道α={alpha:.3f}  残差斜率={k_res*1000:+.4f}e-3 "
          f"(改善 {100*(1-abs(k_res)/abs(polyfit_slope(tl, X[c0:d0, act[0]])[0])):.1f}%)")

    # PCA 主成分与漂移形态的关系
    P(f"### {name} 负载段共模子空间")
    Xl = X[c0:d0]
    Xc = Xl - Xl.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P(f"    第一主成分载荷(全31通道): ")
    for i in range(0, 31, 8):
        P("      " + " ".join(f"{v:+.3f}" for v in Vt[0][i:i + 8]))
    P(f"    PC1 载荷 受载通道均值={Vt[0][act].mean():+.4f} 静默通道均值={Vt[0][qui].mean():+.4f}")
    P(f"    PC1 载荷符号: 受载通道正 {int((Vt[0][act]>0).sum())}/{len(act)}  "
      f"静默通道正 {int((Vt[0][qui]>0).sum())}/{len(qui)}")

    # 漂移与响应的空间相关：是不是"响应越大漂移越大"
    if len(act) > 3:
        cc = np.corrcoef(activ[act], drift[act])[0, 1]
        P(f"    受载通道: 响应幅度 vs 漂移量 空间相关 r={cc:.4f}")

    # 全阵列 PCA 投影时间序列（PC1 分数）
    P(f"    负载段 PC1 分数 分段均值(5等分): " +
      " ".join(f"{U[:,0][int(i*len(U)/5):int((i+1)*len(U)/5)].mean():+.3f}" for i in range(5)))

    out[name] = dict(
        seg=seg, act=[int(i) for i in act], qui=[int(i) for i in qui],
        drift=drift, activ=activ,
        fs_reg=float(1 / k), dt_hist=hist,
        pc1_act_mean=float(Vt[0][act].mean()), pc1_qui_mean=float(Vt[0][qui].mean()),
    )

dump_json(out, "B_diagnostics.json")
with open(os.path.join(RES, "B_diagnostics.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[ok] 步骤2 完成")
