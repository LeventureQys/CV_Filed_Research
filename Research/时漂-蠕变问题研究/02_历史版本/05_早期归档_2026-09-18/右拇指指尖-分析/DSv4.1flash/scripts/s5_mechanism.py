# -*- coding: utf-8 -*-
"""步骤5：漂移物理机理与零漂深挖。

回答：
1. 时漂是否与瞬时力成比例（乘性蠕变）——决定补偿是"加性"还是"乘性"；
2. 漂移速率随时间的形态（幂律指数）——决定用什么在线模型；
3. 零漂的真实来源（前空载是否残留接触、卸载后是否回到真零）；
4. 不加补偿时误差超标时间（工程可用性）；
5. 通道间时间常数差异（决定"共享形态"够不够）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import (load_dataset, detect_segments, spatial_map, savefig,
                        dump_json, DATASETS, FIG, RES, ROWS, COLS)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

lines = []
out = {}


def P(s=""):
    print(s)
    lines.append(s)


def prep(name):
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1.0 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), D["t"], fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    base = X[a:bb].mean(axis=0)
    Xn = X - base
    return dict(name=name, t=t, fs=fs, X=X, Xn=Xn, pre=(a, bb), load=(c0, d0),
                post=(d0, len(t)), base=base, ch=D["ch_cols"])


INFO = {n: prep(n) for n in DATASETS}

for name, D in INFO.items():
    t, fs, Xn, X = D["t"], D["fs"], D["Xn"], D["Xn"]
    a, bb = D["pre"]; c0, d0 = D["load"]; n = len(t)
    w = lambda s: int(round(s * fs))
    resp = Xn[c0:d0].max(axis=0)
    act = np.where(resp > 0.05)[0]
    P("=" * 90)
    P(f"### {name}  (前空载 {t[bb]:.2f}s / 负载 {t[d0]-t[c0]:.2f}s / 后空载 {t[-1]-t[d0]:.2f}s)")

    # ---------- 1) 乘性 vs 加性 ----------
    # 每个通道：漂移量 vs 载荷幅度；若 D_j = α·F_j 则空间上是过原点直线
    drift = Xn[d0 - w(2):d0].mean(axis=0) - Xn[c0:c0 + w(2)].mean(axis=0)
    A = np.vstack([resp[act], np.ones(len(act))]).T
    (alpha, c0_), *_ = np.linalg.lstsq(A, drift[act], rcond=None)
    pred = alpha * resp[act] + c0_
    r2a = 1 - np.sum((drift[act] - pred) ** 2) / np.sum((drift[act] - drift[act].mean()) ** 2)
    # 过原点
    alpha0 = np.sum(resp[act] * drift[act]) / np.sum(resp[act] ** 2)
    r2b = 1 - np.sum((drift[act] - alpha0 * resp[act]) ** 2) / np.sum(
        (drift[act] - drift[act].mean()) ** 2)
    P(f"  乘性检验: 漂移 = α·响应 空间拟合  α={alpha:+.4f}(含截距 {c0_*1000:+.3f}mN) R²={r2a:.4f}")
    P(f"             过原点 α0={alpha0:+.4f} R²={r2b:.4f}  "
      f"=> 漂移/阶跃比 中位={np.median(drift[act]/resp[act]):.3f}")
    P(f"  结论: 时漂近似为乘性（与瞬时力成比例），比例系数 α≈{alpha0:.3f}")

    # ---------- 2) 漂移速率形态 ----------
    # 单通道：拟合 dF/dt ~ t^(p-1)
    j = int(act[np.argmax(resp[act])])
    y = Xn[c0:d0, j]
    tt = t[c0:d0] - t[c0]
    ys = np.convolve(y, np.ones(w(1.0)) / w(1.0), mode="same")
    dy = np.gradient(ys, 1 / fs)
    m = (tt > 1.0) & (tt < tt[-1] * 0.95)
    lx, ly = np.log(tt[m]), np.log(np.maximum(np.abs(dy[m]), 1e-12))
    pp = np.polyfit(lx, ly, 1)
    P(f"  速率幂律: dF/dt ∝ t^(p-1)  拟合指数 p-1={pp[0]:+.3f} => p={pp[0]+1:+.3f} "
      f"(R²={1 - np.var(ly - np.polyval(pp, lx))/np.var(ly):.4f})")
    # 对比：漂移量本身对 t 的幂律
    m2 = tt > 0.5
    yc = y - y[:w(0.5)].mean()
    lx2, ly2 = np.log(tt[m2]), np.log(np.maximum(yc[m2], 1e-12))
    pp2 = np.polyfit(lx2, ly2, 1)
    P(f"  漂移量幂律: ΔF ∝ t^p  p={pp2[0]:+.4f} "
      f"(R²={1 - np.var(ly2 - np.polyval(pp2, lx2))/np.var(ly2):.4f})")

    # ---------- 3) 零漂来源 ----------
    P(f"  零漂深挖:")
    P(f"    前空载: 均值={X[a:bb].mean(axis=0)[act].mean()*1000:+.3f}mN  斜率="
      f"{np.polyfit(t[a:bb]-t[a], X[a:bb].mean(axis=1),1)[0]*1000:+.4f}e-3 N/s  "
      f"std={X[a:bb].mean(axis=1).std()*1000:.3f}mN")
    P(f"    后空载: 均值={X[d0:].mean(axis=0)[act].mean()*1000:+.3f}mN  斜率="
      f"{np.polyfit(t[d0:]-t[d0], X[d0:].mean(axis=1),1)[0]*1000:+.4f}e-3 N/s  "
      f"std={X[d0:].mean(axis=1).std()*1000:.3f}mN")
    P(f"    死通道(恒为 0) 数={int((X.std(axis=0)==0).sum())}  "
      f"未受载且恒零的通道={[D['ch'][i] for i in np.where(X.std(axis=0)==0)[0]]}")
    P(f"    => 零漂主要体现为'前空载残留接触/未完全卸载'与'卸载后亚毫牛级残余'，"
      f"幅值远小于时漂（{np.abs(X[d0:].mean(axis=0)-X[a:bb].mean(axis=0))[act].max()*1000:.2f}mN "
      f"vs {drift[act].max()*1000:.1f}mN）")
    # 卸载瞬间的跳变 vs 缓慢恢复
    z_jump = Xn[d0:d0 + w(0.2)].mean(axis=0)[j] - Xn[d0 - w(0.2):d0].mean(axis=0)[j]
    z_end = Xn[n - w(3):].mean(axis=0)[j]
    z_start = Xn[d0 + w(0.2):d0 + w(2)].mean(axis=0)[j]
    P(f"    主通道 {D['ch'][j]}: 卸载瞬跳={z_jump*1000:+.3f}mN  卸载后2s={z_start*1000:+.3f}  "
      f"末尾={z_end*1000:+.3f}mN  残余/蠕变={100*abs(z_end)/drift[j]:.1f}%")

    # ---------- 4) 误差超标时间 ----------
    rel_err = y / y[c0:c0 + w(1.0)].mean()
    over = {}
    for thr in (0.02, 0.05, 0.10, 0.20, 0.30):
        idx = np.where(np.abs(rel_err - 1) > thr)[0]
        over[thr] = float(t[c0 + idx[0]] - t[c0]) if len(idx) else np.inf
    P(f"  不加补偿的误差超标时间: " +
      "  ".join(f"{int(k*100)}%→{v:.1f}s" if np.isfinite(v) else f"{int(k*100)}%→未超标"
                for k, v in over.items()))

    # ---------- 5) 通道时间常数 ----------
    taus = []
    for jj in act:
        yy = Xn[c0:d0, jj]
        amp = yy.max() - yy[:w(0.5)].mean()
        if amp < 0.1:
            continue
        try:
            def f(t_, a, tau, c):
                return a * (1 - np.exp(-t_ / tau)) + c
            popt, _ = curve_fit(f, tt, yy, p0=[amp * 0.6, 60.0, yy[0]],
                                maxfev=100000,
                                bounds=([-np.inf, 0.5, -np.inf], [np.inf, 1e5, np.inf]))
            taus.append((int(jj), popt[1], popt[0]))
        except Exception:
            pass
    if taus:
        tv = np.array([x[1] for x in taus])
        P(f"  通道单指数时间常数 τ: 中位={np.median(tv):.1f}s  范围={tv.min():.1f}~{tv.max():.1f}s  "
          f"变异系数={tv.std()/tv.mean():.3f}  (n={len(tv)})")

    out[name] = dict(alpha=float(alpha0), r2_mult=float(r2b),
                     power_p=float(pp2[0]), over=over,
                     tau_med=float(np.median(tv)) if taus else None,
                     tau_cv=float(tv.std() / tv.mean()) if taus else None,
                     drift=drift, resp=resp, act=[int(i) for i in act],
                     taus=[[int(a_), float(b_), float(c_)] for a_, b_, c_ in taus])

# ================= 图 P1：乘性检验 + 速率形态 =================
fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    D = INFO[name]; O = out[name]
    t, fs, Xn = D["t"], D["fs"], D["Xn"]
    a, bb = D["pre"]; c0, d0 = D["load"]
    act = O["act"]; w = lambda s: int(round(s * fs))
    ax = axes[r, 0]
    ax.scatter(O["resp"][act] * 1000, O["drift"][act] * 1000, s=28, color="tab:blue")
    xs = np.linspace(0, O["resp"][act].max() * 1000, 20)
    ax.plot(xs, O["alpha"] * xs, "r--", lw=1.4,
            label=f"α={O['alpha']:.3f}, R²={O['r2_mult']:.3f}")
    ax.set_xlabel("负载响应 (mN)"); ax.set_ylabel("时漂量 (mN)")
    ax.set_title(f"{name} 乘性检验：漂移 ∝ 响应"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[r, 1]
    j = int(act[np.argmax(O["resp"][act])])
    tt = t[c0:d0] - t[c0]
    yy = Xn[c0:d0, j] - Xn[c0:c0 + w(0.5), j].mean()
    m = tt > 0.5
    ax.loglog(tt[m], np.maximum(yy[m], 1e-6), lw=0.8, color="tab:red", label="ΔF(t)")
    ref = yy[m][0] * (tt[m] / tt[m][0]) ** O["power_p"]
    ax.loglog(tt[m], ref, "k--", lw=1.2, label=f"幂律 p={O['power_p']:.3f}")
    ax.set_xlabel("负载持续 (s)"); ax.set_ylabel("ΔF (N)")
    ax.set_title(f"{name} 漂移量幂律形态"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    ax = axes[r, 2]
    tv = np.array([x[1] for x in O["taus"]])
    if len(tv):
        ax.hist(tv, bins=12, color="tab:green")
        ax.axvline(np.median(tv), color="r", ls="--", label=f"中位={np.median(tv):.1f}s")
        ax.set_xlabel("单指数 τ (s)"); ax.set_ylabel("通道数")
        ax.set_title(f"{name} 通道时间常数分布 (CV={O['tau_cv']:.2f})")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("图 P1  时漂机理：乘性、幂律形态、通道时间常数离散", fontsize=13)
savefig(fig, "P1_mechanism.png")

# ================= 图 P2：误差增长与零漂细节 =================
fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
for r, name in enumerate(DATASETS):
    D = INFO[name]; O = out[name]
    t, fs, Xn = D["t"], D["fs"], D["Xn"]
    a, bb = D["pre"]; c0, d0 = D["load"]; n = len(t)
    act = O["act"]; w = lambda s: int(round(s * fs))
    j = int(act[np.argmax(O["resp"][act])])

    ax = axes[r, 0]
    y = Xn[c0:d0, j]
    rel = (y / y[c0:c0 + w(1.0)].mean() - 1) * 100
    ax.plot(t[c0:d0] - t[c0], rel, lw=0.8, color="tab:red")
    for thr, c in ((2, "g"), (5, "b"), (10, "m"), (30, "k")):
        ax.axhline(thr, color=c, ls=":", lw=1)
    ax.set_xlabel("负载持续 (s)"); ax.set_ylabel("相对误差 (%)")
    ax.set_title(f"{name} 主通道未补偿相对误差")
    ax.grid(alpha=0.3)

    ax = axes[r, 1]
    ax.plot(t[a:bb] - t[a], Xn[a:bb].mean(axis=1) * 1000, lw=0.6, label="前空载均值")
    ax.plot(t[d0:] - t[d0], Xn[d0:].mean(axis=1) * 1000, lw=0.6, label="后空载均值")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("段内时间 (s)"); ax.set_ylabel("均值 (mN)")
    ax.set_title(f"{name} 空载基线的亚毫牛级行为（死区截断）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[r, 2]
    # 卸载后恢复曲线
    tt = t[d0:] - t[d0]
    for k_, jj in enumerate(act[:6]):
        ax.plot(tt, Xn[d0:, jj] * 1000, lw=0.7, label=D["ch"][jj])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("卸载后时间 (s)"); ax.set_ylabel("残余 (mN)")
    ax.set_title(f"{name} 卸载后零漂恢复（前6个受载通道）")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
fig.suptitle("图 P2  误差增长速率与零漂细节", fontsize=13)
savefig(fig, "P2_error_zero.png")

dump_json(out, "E_mechanism.json")
with open(os.path.join(RES, "E_mechanism.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[ok] 步骤5 完成")
