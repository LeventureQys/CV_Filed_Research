# -*- coding: utf-8 -*-
"""步骤1：数据体检与特征刻画（分段、时漂形态、零漂、噪声、空间/共模结构）。"""
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import (load_dataset, detect_segments, spatial_map, polyfit_slope,
                        r2, savefig, dump_json, DATASETS, FIG, RES, MASK_ON,
                        ROWS, COLS)
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)


def exp_model(t, a, tau, c):
    return a * (1.0 - np.exp(-t / tau)) + c


def biexp_model(t, a1, tau1, a2, tau2, c):
    return a1 * (1.0 - np.exp(-t / tau1)) + a2 * (1.0 - np.exp(-t / tau2)) + c


def power_model(t, a, p, c):
    return a * np.power(np.maximum(t, 1e-6), p) + c


report = {}
lines = []


def P(s=""):
    print(s)
    lines.append(s)


for name in DATASETS:
    D = load_dataset(name)
    t, X, fs = D["t"], D["X"], D["fs"]
    total = X.sum(axis=1)
    seg = detect_segments(total, t, fs=fs)
    a, b = seg["pre"]; c, d = seg["load"]
    tl = t[c:d] - t[c]
    tpre = t[a:b] - t[a]
    tpost = t[d:] - t[d]

    P("=" * 78)
    P(f"### {name}   帧数={D['n']}  时长={t[-1]:.2f}s  有效采样率≈{fs:.3f}Hz")
    P(f"    分段: 前空载 0.00~{t[b]:.2f}s | 负载 {t[c]:.2f}~{t[d]:.2f}s "
      f"(持续 {t[d]-t[c]:.2f}s) | 后空载 {t[d]:.2f}~{t[-1]:.2f}s")

    # ---------- 时间轴质量 ----------
    dt = np.diff(t)
    P(f"    帧间隔: 中位={np.median(dt)*1000:.3f}ms 均值={dt.mean()*1000:.3f}ms "
      f"最大={dt.max()*1000:.1f}ms  丢帧间隔>2倍中位数的次数={int((dt > 2.5*np.median(dt)).sum())}")
    P(f"    frame_index 连续性: 期望={D['n']} 实际末值={int(D['frame_index'][-1])} "
      f"缺口={int(D['frame_index'][-1]) - (D['n']-1)}")

    # ---------- 通道活跃度 ----------
    pre_mean = X[a:b].mean(axis=0)
    pre_std = X[a:b].std(axis=0)
    load_mean = X[c:d].mean(axis=0)
    load_step0 = X[c:c + int(1.0 * fs)].mean(axis=0)
    response = load_step0 - pre_mean

    active = np.where(np.abs(response) > 0.05)[0]
    quiet = np.where(np.abs(response) <= 0.05)[0]
    P(f"    通道分级: 有响应(|Δ|>0.05N) {len(active)} 个 -> {[D['ch_cols'][i] for i in active]}")
    P(f"              近零响应 {len(quiet)} 个 -> {[D['ch_cols'][i] for i in quiet]}")

    # ---------- 时漂形态（在受载通道上拟合） ----------
    if len(active) == 0:
        P("    !! 无受载通道，跳过时漂拟合")
        continue
    main_ch = int(active[np.argmax(np.abs(response[active]))])
    sig = X[:, main_ch]
    y = sig[c:d]
    shift = y.min()

    forms = {}
    # 线性
    k1, b1 = np.polyfit(tl, y, 1)
    forms["线性 k·t+b"] = (r2(y, k1 * tl + b1), {"k": k1, "b": b1})
    # 单指数 a(1-exp(-t/tau))+c
    try:
        p0 = [y[-1] - y[0], max(tl[-1] / 3, 1.0), y[0]]
        popt, _ = curve_fit(exp_model, tl, y, p0=p0, maxfev=60000)
        forms["单指数 a(1-e^{-t/τ})+c"] = (r2(y, exp_model(tl, *popt)),
                                     {"a": popt[0], "tau": popt[1], "c": popt[2]})
    except Exception as e:
        forms["单指数 a(1-e^{-t/τ})+c"] = (np.nan, {"err": str(e)})
    # 双指数
    try:
        p0 = [y[-1] - y[0], max(tl[-1] / 8, 1.0), 0.0, max(tl[-1] / 2, 5.0), y[0]]
        popt2, _ = curve_fit(biexp_model, tl, y, p0=p0, maxfev=200000,
                             bounds=([-np.inf, 0.5, -np.inf, 5.0, -np.inf],
                                     [np.inf, 1e5, np.inf, 1e5, np.inf]))
        forms["双指数"] = (r2(y, biexp_model(tl, *popt2)),
                        {"a1": popt2[0], "tau1": popt2[1], "a2": popt2[2],
                         "tau2": popt2[3], "c": popt2[4]})
    except Exception as e:
        forms["双指数"] = (np.nan, {"err": str(e)})
    # 对数 a·ln(1+t)+b
    A = np.vstack([np.log1p(tl), np.ones_like(tl)]).T
    (al, bl), *_ = np.linalg.lstsq(A, y, rcond=None)
    forms["对数 a·ln(1+t)+b"] = (r2(y, al * np.log1p(tl) + bl), {"a": al, "b": bl})
    # 幂律 a·t^p+c
    try:
        popt3, _ = curve_fit(power_model, tl, y, p0=[1.0, 0.3, y[0]], maxfev=200000)
        forms["幂律 a·t^p+c"] = (r2(y, power_model(tl, *popt3)),
                              {"a": popt3[0], "p": popt3[1], "c": popt3[2]})
    except Exception as e:
        forms["幂律 a·t^p+c"] = (np.nan, {"err": str(e)})

    P(f"    主受载通道 = {D['ch_cols'][main_ch]}  (阶跃幅度 {response[main_ch]:.4f}N)")
    P("    时漂形态拟合（负载段整段）:")
    for kk, (rr, pp) in forms.items():
        P(f"      {kk:<26s} R²={rr:.5f}   {pp}")

    # ---------- 时漂速率分段 ----------
    nseg = 5
    edges = np.linspace(c, d, nseg + 1).astype(int)
    slope_evo = []
    for i in range(nseg):
        s, e = edges[i], edges[i + 1]
        kk, _ = polyfit_slope(t[s:e] - t[s], sig[s:e])
        slope_evo.append(kk)
    P(f"    负载段分段斜率(N/s, 5等分): " +
      " ".join(f"{v*1000:+.3f}e-3" for v in slope_evo))
    P(f"    负载段总漂移 = {y[-int(fs*2):].mean() - y[:int(fs*2)].mean():+.4f}N "
      f"({100*(y[-int(fs*2):].mean()-y[:int(fs*2)].mean())/response[main_ch]:+.1f}% of step)")

    # ---------- 零漂 ----------
    zpre = sig[a:b].mean(); zpost = sig[d:].mean()
    kpre, _ = polyfit_slope(tpre, sig[a:b])
    kpost, _ = polyfit_slope(tpost, sig[d:])
    P(f"    零漂: 前空载均值={zpre:+.4f}N±{sig[a:b].std():.4f}  后空载均值={zpost:+.4f}N"
      f"±{sig[d:].std():.4f}  差={zpost-zpre:+.4f}N ({100*(zpost-zpre)/response[main_ch]:+.2f}% of step)")
    P(f"    空载段自身斜率: 前={kpre*1000:+.4f}e-3 N/s  后={kpost*1000:+.4f}e-3 N/s")
    # 零漂是否也是"指数恢复"
    tail_ratio = (sig[d:][-int(fs*5):].mean() - sig[d:d + int(fs*1)].mean())

    # ---------- 每个通道的零漂 ----------
    zero_shift_all = X[d:].mean(axis=0) - X[a:b].mean(axis=0)
    P(f"    全通道零漂: 均值={zero_shift_all.mean():+.4f}N 中位={np.median(zero_shift_all):+.4f}N "
      f"max={zero_shift_all.max():+.4f} min={zero_shift_all.min():+.4f}")
    P(f"      受载通道零漂均值={zero_shift_all[active].mean():+.4f}N  "
      f"未受载通道零漂均值={zero_shift_all[quiet].mean():+.4f}N" if len(quiet) else "")
    # 各通道零漂/响应 比值
    ratio = np.divide(zero_shift_all, response, out=np.full_like(zero_shift_all, np.nan),
                      where=np.abs(response) > 1e-6)
    P(f"      零漂/阶跃 比值: 中位={np.nanmedian(ratio):+.3f} "
      f"受载通道={np.nanmedian(ratio[active]):+.3f}")

    # ---------- 噪声 ----------
    # 空载段去趋势后的高频残差
    def noise_of(yy, tt):
        k, bb = np.polyfit(tt, yy, 1)
        res = yy - (k * tt + bb)
        # 高频成分（一阶差分）
        return res.std(), (np.diff(res).std() / np.sqrt(2))

    ns_pre = np.array([noise_of(X[a:b, j], tpre)[0] for j in range(X.shape[1])])
    ns_hf = np.array([noise_of(X[a:b, j], tpre)[1] for j in range(X.shape[1])])
    P(f"    前空载噪声: 去趋势std 中位={np.median(ns_pre):.5f}N max={ns_pre.max():.5f}N; "
      f"高频std(Δ/√2) 中位={np.median(ns_hf):.5f}N")
    P(f"    量化台阶: 唯一值个数(全数据)={len(np.unique(X))}  最小非零间隔≈"
      f"{np.min(np.diff(np.unique(X))[np.diff(np.unique(X))>0]):.4f}N")

    # ---------- 空间结构 ----------
    sm_amp = spatial_map(response)
    slope_all = np.array([polyfit_slope(tl, X[c:d, j])[0] for j in range(X.shape[1])])
    sm_slope = spatial_map(slope_all * 1000.0)  # e-3 N/s
    zero_all = zero_shift_all
    sm_zero = spatial_map(zero_all)

    # 空间自相关（相邻有效单元 响应差的统计）
    def neighbor_pairs(mat):
        out = []
        for i in range(ROWS):
            for j in range(COLS):
                if not np.isfinite(mat[i, j]):
                    continue
                for di, dj in ((0, 1), (1, 0)):
                    ii, jj = i + di, j + dj
                    if ii < ROWS and jj < COLS and np.isfinite(mat[ii, jj]):
                        out.append((mat[i, j], mat[ii, jj]))
        return np.array(out)

    pairs = neighbor_pairs(sm_slope)
    if len(pairs) > 5:
        cc = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
        P(f"    漂移斜率空间相邻相关性(Slope同/跨行相邻) r={cc:.3f}  "
          f"梯度中位={np.median(np.abs(pairs[:,0]-pairs[:,1])):.4f}e-3 N/s")

    # ---------- 共模性（关键：能不能用参考通道/共模剔除） ----------
    # 用未受载通道做参考
    if len(quiet) >= 3:
        ref = X[:, quiet].mean(axis=1)
        k_ref_load = polyfit_slope(tl, ref[c:d])[0]
        k_ref_post = polyfit_slope(tpost, ref[d:])[0]
        k_ref_pre = polyfit_slope(tpre, ref[a:b])[0]
        # 受载通道漂移斜率 & 与参考斜率的关系
        k_act = slope_all[active]
        corr_slope = np.corrcoef(ref[c:d], X[c:d, main_ch])[0, 1]
        P(f"    未受载参考通道({len(quiet)}个) 斜率: 前空载={k_ref_pre*1000:+.4f} "
          f"负载段={k_ref_load*1000:+.4f} 后空载={k_ref_post*1000:+.4f} e-3 N/s")
        P(f"    主通道 vs 参考 负载段相关 r={corr_slope:.4f}  "
          f"受载通道斜率中位={np.median(k_act)*1000:+.4f}e-3 N/s")
        P(f"    受载/未受载 漂移斜率比 = {np.median(k_act)/k_ref_load if k_ref_load != 0 else np.nan:.2f}")
    else:
        ref = np.zeros(len(t)); k_ref_load = np.nan

    # 全通道共模：取全部通道在负载段减去各自均值的部分？
    # 更直接：负载段 各通道信号 = 响应 + 共模漂移。用最小值-最大值的 PCA
    Xc = X[c:d] - X[c:d].mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)
    P(f"    负载段 PCA 前5主成分能量占比: " + " ".join(f"{e*100:.2f}%" for e in ev[:5]))
    P(f"    第一主成分载荷(前10通道) = {Vt[0][:10]}")

    report[name] = dict(
        n=D["n"], dur=float(t[-1]), fs=float(fs),
        seg=dict(pre=[int(a), int(b)], load=[int(c), int(d)], post=[int(d), int(D['n'])]),
        t_pre=float(t[b]), t_load0=float(t[c]), t_load1=float(t[d]),
        main_ch=int(main_ch), main_ch_name=D["ch_cols"][main_ch],
        active=[int(i) for i in active], quiet=[int(i) for i in quiet],
        response=response, slope_all=slope_all, zero_shift_all=zero_shift_all,
        noise_pre=ns_pre, forms={k: float(v[0]) for k, v in forms.items()},
        zpre=float(zpre), zpost=float(zpost), kpre=float(kpre), kpost=float(kpost),
        k_ref_load=float(k_ref_load), ev=ev[:5],
        sm_amp=sm_amp, sm_slope=sm_slope, sm_zero=sm_zero,
        ref=ref, slope_evo=np.array(slope_evo),
    )

# ==================== 图 A1：总览 ====================
fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    D = load_dataset(name)
    t, X = D["t"], D["X"]
    total = X.sum(axis=1)
    a, b = R["seg"]["pre"][0], R["seg"]["load"][0]
    c, d = R["seg"]["load"]
    ax = axes[r, 0]
    ax.plot(t, total, lw=0.5, color="#1f77b4")
    ax.axvspan(t[c], t[d], color="orange", alpha=0.15)
    ax.set_title(f"{name}  31通道总和  ΣF={X.sum(axis=1).max():.3f}N  max")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Σ Force (N)"); ax.grid(alpha=0.3)
    ax = axes[r, 1]
    for j in range(X.shape[1]):
        ax.plot(t, X[:, j], lw=0.4, alpha=0.75)
    ax.axvspan(t[c], t[d], color="orange", alpha=0.15)
    ax.plot(t, X[:, R["main_ch"]], lw=0.9, color="k", label=f"主通道 {R['main_ch_name']}")
    ax.set_title(f"{name}  全部31通道 + 主通道")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Force (N)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("图 A1  三组数据总览（橙区 = 恒定负载段）", fontsize=13)
savefig(fig, "A1_overview.png")

# ==================== 图 A2：时漂形态拟合 ====================
fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    D = load_dataset(name)
    t, X, fs = D["t"], D["X"], R["fs"]
    c, d = R["seg"]["load"]
    tl = t[c:d] - t[c]
    y = X[c:d, R["main_ch"]]
    ax = axes[r, 0]
    ax.plot(tl, y, lw=0.5, color="gray", label="原始")
    k, b = np.polyfit(tl, y, 1)
    ax.plot(tl, k * tl + b, lw=1.3, color="tab:red", label=f"线性 R²={R['forms']['线性 k·t+b']:.4f}")
    A = np.vstack([np.log1p(tl), np.ones_like(tl)]).T
    (al, bl), *_ = np.linalg.lstsq(A, y, rcond=None)
    ax.plot(tl, al * np.log1p(tl) + bl, lw=1.3, color="tab:green", ls="--",
            label=f"对数 R²={R['forms']['对数 a·ln(1+t)+b']:.4f}")
    ax.set_title(f"{name} 主通道 {R['main_ch_name']} 负载段时漂")
    ax.set_xlabel("负载持续 (s)"); ax.set_ylabel("Force (N)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[r, 1]
    try:
        p0 = [y[-1] - y[0], max(tl[-1] / 3, 1.0), y[0]]
        popt, _ = curve_fit(exp_model, tl, y, p0=p0, maxfev=60000)
        ax.plot(tl, y, lw=0.4, color="gray")
        ax.plot(tl, exp_model(tl, *popt), lw=1.4, color="tab:blue",
                label=f"单指数 τ={popt[1]:.1f}s R²={R['forms']['单指数 a(1-e^{-t/τ})+c']:.4f}")
    except Exception:
        pass
    try:
        p0 = [y[-1] - y[0], max(tl[-1] / 8, 1.0), 0.0, max(tl[-1] / 2, 5.0), y[0]]
        popt2, _ = curve_fit(biexp_model, tl, y, p0=p0, maxfev=200000,
                             bounds=([-np.inf, 0.5, -np.inf, 5.0, -np.inf],
                                     [np.inf, 1e5, np.inf, 1e5, np.inf]))
        ax.plot(tl, biexp_model(tl, *popt2), lw=1.4, color="tab:purple", ls="-.",
                label=f"双指数 τ1={popt2[1]:.0f}s τ2={popt2[3]:.0f}s R²={R['forms']['双指数']:.4f}")
    except Exception:
        pass
    ax.set_title(f"{name} 指数族拟合"); ax.set_xlabel("负载持续 (s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[r, 2]
    se = R["slope_evo"] * 1000
    ax.bar(np.arange(len(se)), se, color="tab:orange")
    ax.set_title(f"{name} 负载段5等分漂移速率 (e-3 N/s)")
    ax.set_xticks(range(len(se)))
    ax.set_xticklabels([f"{i+1}/5" for i in range(len(se))])
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("图 A2  时漂形态：线性/对数/指数族拟合与速率衰减", fontsize=13)
savefig(fig, "A2_drift_forms.png")

# ==================== 图 A3：零漂 ====================
fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    D = load_dataset(name)
    t, X, fs = D["t"], D["X"], R["fs"]
    a, b = R["seg"]["pre"][0], R["seg"]["load"][0]
    c, d = R["seg"]["load"]
    n = D["n"]
    sig = X[:, R["main_ch"]]
    ax = axes[r, 0]
    ax.plot(t[:b], sig[:b], lw=0.5, color="tab:blue", label="前空载")
    ax.plot(t[d:], sig[d:], lw=0.5, color="tab:green", label="后空载")
    ax.axhline(R["zpre"], color="tab:blue", ls="--", lw=1)
    ax.axhline(R["zpost"], color="tab:green", ls="--", lw=1)
    ax.set_title(f"{name} 主通道 零漂={R['zpost']-R['zpre']:+.4f}N")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Force (N)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[r, 1]
    zs = R["zero_shift_all"]
    ax.bar(np.arange(len(zs)), zs * 1000, color=["tab:red" if i in R["active"] else "tab:gray"
                                                 for i in range(len(zs))])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{name} 各通道零漂 (e-3 N)  红=受载通道")
    ax.set_xlabel("通道序号"); ax.grid(alpha=0.3, axis="y")

    ax = axes[r, 2]
    ax.plot(t[:b], X[:b].mean(axis=1), lw=0.6, label="全通道均值(前空载)")
    ax.plot(t[d:], X[d:].mean(axis=1), lw=0.6, label="全通道均值(后空载)")
    ax.plot(t, X.mean(axis=1), lw=0.4, color="gray", alpha=0.6, label="全通道均值(全程)")
    ax.axvspan(t[c], t[d], color="orange", alpha=0.12)
    ax.set_title(f"{name} 全通道均值（共模量）")
    ax.set_xlabel("t (s)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("图 A3  零漂特征（前空载 vs 后空载基线）", fontsize=13)
savefig(fig, "A3_zero_drift.png")

# ==================== 图 A4：空间分布 ====================
fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    for k, (mat, title, cmap) in enumerate([
        (R["sm_amp"], "负载响应 Δ (N)", "hot"),
        (R["sm_slope"], "负载段漂移速率 (e-3 N/s)", "coolwarm"),
        (R["sm_zero"], "零漂 (N)", "coolwarm"),
        (spatial_map(np.log10(np.maximum(np.abs(R["slope_all"] * 1000), 1e-4))),
         "log10|漂移速率| ", "viridis"),
    ]):
        ax = axes[r, k]
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_title(f"{name} {title}", fontsize=9)
        ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
        plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("图 A4  空间分布：响应 / 漂移速率 / 零漂（9×7 阵列，灰=无效位）", fontsize=13)
savefig(fig, "A4_spatial.png")

# ==================== 图 A5：共模与 PCA ====================
fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    D = load_dataset(name)
    t, X, fs = D["t"], D["X"], R["fs"]
    c, d = R["seg"]["load"]
    act, qui = R["active"], R["quiet"]
    ax = axes[r, 0]
    if len(act):
        ax.plot(t, X[:, act].mean(axis=1), lw=0.6, color="tab:red",
                label=f"受载组均值({len(act)}ch)")
    if len(qui):
        ax.plot(t, X[:, qui].mean(axis=1), lw=0.8, color="tab:blue",
                label=f"未受载组均值({len(qui)}ch)")
    ax.axvspan(t[c], t[d], color="orange", alpha=0.12)
    ax.set_title(f"{name} 共模检验  k_ref={R['k_ref_load']*1000:+.3f}e-3 N/s")
    ax.set_xlabel("t (s)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[r, 1]
    ax.bar(np.arange(5), R["ev"] * 100, color="tab:purple")
    ax.set_title(f"{name} 负载段 PCA 能量占比")
    ax.set_xlabel("主成分"); ax.set_ylabel("%"); ax.grid(alpha=0.3, axis="y")

    # 各通道归一化漂移形态是否一致（把负载段归一化后叠加）
    ax = axes[r, 2]
    if len(act):
        Y = X[c:d][:, act]
        Y0 = Y[:int(2 * fs)].mean(axis=0)
        Yy = Y - Y0
        scale = Yy[-int(2 * fs):].mean(axis=0)
        for i, j in enumerate(act):
            if abs(scale[i]) > 1e-4:
                ax.plot(t[c:d] - t[c], Yy[:, i] / scale[i], lw=0.7)
        ax.set_title(f"{name} 受载通道漂移形态归一化对齐")
        ax.set_xlabel("负载持续 (s)"); ax.grid(alpha=0.3)
fig.suptitle("图 A5  共模结构检验（参考通道法可行性）与 PCA", fontsize=13)
savefig(fig, "A5_common_mode.png")

# ==================== 图 A6：噪声与量化 ====================
fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
for r, name in enumerate(DATASETS):
    R = report[name]
    D = load_dataset(name)
    t, X, fs = D["t"], D["X"], R["fs"]
    a, b = R["seg"]["pre"][0], R["seg"]["load"][0]
    ax = axes[r, 0]
    ax.bar(np.arange(X.shape[1]), R["noise_pre"] * 1000, color="tab:cyan")
    ax.set_title(f"{name} 前空载各通道噪声 σ (e-3 N)")
    ax.set_xlabel("通道序号"); ax.grid(alpha=0.3, axis="y")

    ax = axes[r, 1]
    sig = X[:b, R["main_ch"]]
    ax.plot(t[:min(b, int(5 * fs))], sig[:min(b, int(5 * fs))] * 1000, lw=0.8)
    ax.set_title(f"{name} 主通道前5s原始（e-3 N，看量化台阶）")
    ax.set_xlabel("t (s)"); ax.grid(alpha=0.3)

    ax = axes[r, 2]
    f, Pxx = sps.welch(X[:b, R["main_ch"]], fs=fs, nperseg=min(4096, b))
    ax.semilogy(f, Pxx)
    ax.set_title(f"{name} 主通道前空载 PSD")
    ax.set_xlabel("Hz"); ax.set_ylabel("N²/Hz"); ax.grid(alpha=0.3, which="both")
fig.suptitle("图 A6  噪声与量化特征", fontsize=13)
savefig(fig, "A6_noise.png")

# 保存报告
out = {}
for name in DATASETS:
    R = {k: v for k, v in report[name].items()
         if k not in ("sm_amp", "sm_slope", "sm_zero", "ref")}
    out[name] = R
dump_json(out, "A_characterization.json")

with open(os.path.join(RES, "A_characterization.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[ok] 步骤1 完成")
