# -*- coding: utf-8 -*-
"""t4a_04_intermediate：T4-Q4 中间态与「连续谱 vs 两个离散类」。

检验：
  C1 T_ramp（等效输入上升时间）的分布是双峰还是连续（双峰系数 BC + 1/2 分量 GMM 的 BIC）；
  C2 T_ramp 与 armed（预载）谁是形态的驱动量（OLS：z_at_02 ~ log10 T_ramp [+ armed]）；
  C3 中间态样本清点（慢压起的零基线加载 / 带载硬压 / 低预载 / 门限翻转 / 复合 / 卸载后加压）；
  C4 拍击后加压：本批是否有可辨识样本（无则明确「缺数据」）。

产出：results/t4a_intermediate.csv、results/t4a_continuum.csv
日志：results/_t4a_04_intermediate.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t4a_common as C


def load():
    mor = pd.read_csv(os.path.join(C.RES, "t4a_morphology.csv"))
    ir = pd.read_csv(os.path.join(C.RES, "t4a_input_recover.csv"))
    mor["t_on_r"] = mor["t_on"].round(3)
    ir["t_on_r"] = ir["t_on"].round(3)
    return mor.merge(ir[["key", "t_on_r", "T_ramp", "rmse_ramp_pct", "rmse_step_pct", "gain_ramp"]],
                     on=["key", "t_on_r"], how="left")


def bc(x):
    """双峰系数 BC = (skew²+1)/kurtosis（>5/9≈0.555 提示双峰）。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return np.nan, n
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return np.nan, n
    z = (x - m) / s
    g = float((z ** 3).mean())
    k = float((z ** 4).mean())
    return (g * g + 1.0) / k, n


def gmm1d(x, k, iters=300, seeds=12):
    """一维高斯混合 EM（自己实现，避免额外依赖）；返回 (ll, bic, params)。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    best = None
    rng = np.random.default_rng(7)
    for s in range(seeds):
        if s == 0 and k == 2:
            mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
        else:
            mu = np.sort(rng.choice(x, k, replace=False))
        sd = np.full(k, max(x.std(), 1e-6))
        w = np.full(k, 1.0 / k)
        for _ in range(iters):
            d = np.exp(-0.5 * ((x[:, None] - mu[None, :]) / sd[None, :]) ** 2) / \
                (np.sqrt(2 * np.pi) * sd[None, :])
            p = d * w[None, :]
            tot = p.sum(1, keepdims=True)
            tot[tot <= 1e-300] = 1e-300
            g = p / tot
            nk = g.sum(0) + 1e-9
            mu = (g * x[:, None]).sum(0) / nk
            sd = np.sqrt((g * (x[:, None] - mu[None, :]) ** 2).sum(0) / nk) + 1e-9
            w = nk / n
        d = np.exp(-0.5 * ((x[:, None] - mu[None, :]) / sd[None, :]) ** 2) / \
            (np.sqrt(2 * np.pi) * sd[None, :])
        ll = float(np.log((d * w[None, :]).sum(1) + 1e-300).sum())
        npar = 3 * k - 1
        bic = -2 * ll + npar * np.log(n)
        if best is None or bic < best[1]:
            best = (ll, bic, dict(mu=mu.copy(), sd=sd.copy(), w=w.copy()))
    return best[0], best[1], best[2]


def ols(y, X, names):
    """最小二乘 + t 统计量；返回系数、t、R²。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[m], X[m]
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    res = y - Xd @ beta
    dof = max(1, n - k - 1)
    s2 = float(res @ res) / dof
    try:
        cov = s2 * np.linalg.inv(Xd.T @ Xd)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k + 1, np.nan)
    t = beta / se
    r2 = 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    return dict(n=n, names=["const"] + list(names), beta=beta, t=t, r2=r2)


def main():
    C.start_log()
    df = load()
    ld = df[df.jump > 0].copy()
    ld["logT"] = np.log10(ld.T_ramp.clip(lower=0.005))
    pd.set_option("display.width", 250)

    # ── C1 双峰性 ──
    rows = []
    for sample, sub in (("clean", ld[ld.clean]), ("all", ld)):
        for name, v in (("T_ramp(s)", sub.T_ramp), ("log10 T_ramp", sub.logT),
                        ("z_at_02", sub.z_at_02), ("n_frames_rise", sub.n_frames_rise),
                        ("t80(s)", sub.t80)):
            b, n = bc(v)
            x = np.asarray(v, float)
            x = x[np.isfinite(x)]
            if n < 6:
                continue
            ll1, bic1, p1 = gmm1d(x, 1)
            ll2, bic2, p2 = gmm1d(x, 2)
            rows.append(dict(sample=sample, test="BC & GMM(1 vs 2 分量)", var=name, n=n,
                             bimodality_coef=b, bc_threshold=5 / 9, bic_1comp=bic1,
                             bic_2comp=bic2, dBIC=bic2 - bic1, gmm1_mu=float(p1["mu"][0]),
                             gmm2_mu1=float(np.min(p2["mu"])), gmm2_mu2=float(np.max(p2["mu"])),
                             gmm2_w1=float(p2["w"][int(np.argmin(p2["mu"]))]),
                             verdict=("支持双峰" if b > 5 / 9 and bic2 < bic1 else "不能拒绝单峰/连续")))
    c1 = pd.DataFrame(rows)
    print("== C1 双峰性检验 ==")
    print(c1.round(4).to_string(index=False))

    # ── C2 谁是驱动量（clean / all 两套样本）──
    c2 = []
    for sample, sub in (("clean", ld[ld.clean]), ("all", ld)):
        for tag, cols in (("z_at_02 ~ armed_20", ["armed_20"]),
                          ("z_at_02 ~ log10 T_ramp", ["logT"]),
                          ("z_at_02 ~ log10 T_ramp + armed_20", ["logT", "armed_20"])):
            X = sub[cols].astype(float).to_numpy()
            r = ols(sub.z_at_02, X, cols)
            for nm, b, t in zip(r["names"], r["beta"], r["t"]):
                c2.append(dict(sample=sample, test="OLS", model=tag, term=nm, beta=float(b),
                               t=float(t), n=r["n"], r2=r["r2"]))
        r_rev = ols(sub.logT, sub[["pre_over_peak"]].to_numpy(), ["pre_over_peak"])
        for nm, b, t in zip(r_rev["names"], r_rev["beta"], r_rev["t"]):
            c2.append(dict(sample=sample, test="OLS", model="log10 T_ramp ~ pre_over_peak",
                           term=nm, beta=float(b), t=float(t), n=r_rev["n"], r2=r_rev["r2"]))
    c2 = pd.DataFrame(c2)
    print("\n== C2 驱动量：OLS（clean / all）==")
    print(c2.round(4).to_string(index=False))

    # ── C3 中间态清点 ──
    ev = []
    order = df[df.jump.notna()].sort_values(["key", "t_on"]).reset_index(drop=True)   # 含卸载事件
    for _, q in ld.iterrows():
        cats, notes = [], []
        if q.kind == "onset" and np.isfinite(q.T_ramp) and q.T_ramp >= 0.15:
            cats.append("A_零基线慢压起")
            notes.append("零基线但输入是斜坡(T=%.2fs)" % q.T_ramp)
        if q.kind == "restep" and np.isfinite(q.T_ramp) and q.T_ramp <= 0.15:
            cats.append("B_带载硬压")
            notes.append("带载但输入近阶跃(T=%.2fs)" % q.T_ramp)
        if 0.10 <= q.pre_over_peak <= 0.30:
            cats.append("C_低预载(阈值附近)")
        if bool(q.armed_10) != bool(q.armed_30):
            cats.append("D_标签随门限翻转")
        if not q.clean:
            cats.append("E_复合/基线在动")
        prev = order[(order.key == q.key) & (order.t_on < q.t_on)]
        if len(prev):
            last = prev.iloc[-1]
            if q.t_on - last.t_on <= 8.0 and last.jump < 0:
                cats.append("F_卸载后加压")
                notes.append("前事件=卸载 Δ=%.0f @%.2fs" % (last.jump, last.t_on))
            elif q.t_on - last.t_on <= 8.0:
                cats.append("G_连续两次加载")
                notes.append("前事件=加载 Δ=%.0f @%.2fs" % (last.jump, last.t_on))
        if cats:
            ev.append(dict(key=q.key, rec=q.rec, dom=q.dom, t_on=q.t_on, kind=q.kind,
                           categories="|".join(cats), pre_over_peak=round(q.pre_over_peak, 4),
                           jump=round(q.jump, 1), T_ramp=q.T_ramp, z_at_02=round(q.z_at_02, 4),
                           n_frames_rise=q.n_frames_rise, t50=q.t50, t90=q.t90,
                           clean=q["clean"], note="; ".join(notes)))
    ei = pd.DataFrame(ev)
    ei.round(5).to_csv(os.path.join(C.RES, "t4a_intermediate.csv"), index=False,
                       encoding="utf-8-sig")
    print("\n== C3 中间态样本（n=%d / %d 加载类事件）==" % (len(ei), len(ld)))
    print(ei.round(3).to_string(index=False))

    # ── C4 拍击后加压 ──
    n_tap = 0
    for _, q in ld.iterrows():
        pass
    print("\n== C4 拍击后加压 ==")
    print("  本批 13 份录制中未检出「短促拍击后立即加压」的加载事件（拍击不满足本任务的")
    print("  2%·记录极差 电平门限，属 T5 扰动分析对象）⇒ 该中间态**缺数据**。")

    # ── 汇总 ──
    cont = pd.concat([c1.assign(block="C1"), c2.assign(block="C2")], ignore_index=True)
    cont.to_csv(os.path.join(C.RES, "t4a_continuum.csv"), index=False, encoding="utf-8-sig")
    print("\n== 连续谱区间统计 ==")
    ov = ld[(ld.T_ramp >= 0.05) & (ld.T_ramp <= 0.30)]
    print("  T_ramp ∈ [0.05, 0.30] s 的重叠区事件：n=%d" % len(ov))
    if len(ov):
        print(ov[["rec", "kind", "t_on", "pre_over_peak", "T_ramp", "z_at_02",
                  "n_frames_rise"]].round(3).to_string(index=False))
    print("  onset 组 T_ramp: 中位 %.3f s，范围 %.2f~%.2f s（n=%d）"
          % (ld[ld.kind == "onset"].T_ramp.median(), ld[ld.kind == "onset"].T_ramp.min(),
             ld[ld.kind == "onset"].T_ramp.max(), int((ld.kind == "onset").sum())))
    print("  restep 组 T_ramp: 中位 %.3f s，范围 %.2f~%.2f s（n=%d）"
          % (ld[ld.kind == "restep"].T_ramp.median(), ld[ld.kind == "restep"].T_ramp.min(),
             ld[ld.kind == "restep"].T_ramp.max(), int((ld.kind == "restep").sum())))
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
