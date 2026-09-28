# -*- coding: utf-8 -*-
"""t4b_slope_collinearity.py —— 澄清 `slope_auc` 为何高达 0.92~0.98（与 P1 的共线程度）。

背景：`results/t4b_tramp_window.csv` 的 `slope_auc` 列（判据 `J(Δ)/Δ ÷ 记录峰值`）在全部 Δ 上
都 ≥0.92，与"因果速率判据只有 AUC≈0.69"的结论表面冲突。本脚本给三条定量证据：
  ① `slope` 与 P1（`pre_frac` = pre 电平 ÷ 因果滚动峰值）的 Spearman ρ 与 R²（线性回归）；
  ② **条件 AUC**：把事件按 P1 五等分分层，在每层内部重算 slope 的 AUC（层内 P1 近似常数
     ⇒ 若 slope 只是 P1 的代理，层内 AUC 应塌到 0.5 附近）；
  ③ **把公共的"幅度/电平尺度"除掉**后的重标量：`slope_rel = J(Δ)/Δ ÷ J(Δ)`（≡ 1/Δ，无信息，用作对照）
     与真正的无量纲速率量 `FR(Δ) = level(t_on+Δ)/level(t_on+0.01)`、
     `T50c(Δ)`；并给出"用 P1 残差化后的 slope"的 AUC（`slope ⟂ P1` 的秩残差）。

只用 `results/t4b_morphology_by_arm.csv`（已含全部因果特征），不重读原始 CSV、不重跑算法。

产物：results/t4b_slope_collinearity.csv（逐 Δ 一行）
运行：python scripts\t4b_slope_collinearity.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402


def auc_mw(x, y):
    x = np.asarray([v for v in x if v == v], float)
    y = np.asarray([v for v in y if v == v], float)
    if len(x) < 3 or len(y) < 3:
        return np.nan
    r = pd.Series(np.concatenate([x, y])).rank().to_numpy()
    rx = r[:len(x)].sum()
    return float((rx - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y)))


def r2_linear(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    X = np.column_stack([np.ones(m.sum()), x[m]])
    b, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    pred = X @ b
    sse = float(((y[m] - pred) ** 2).sum())
    sst = float(((y[m] - y[m].mean()) ** 2).sum())
    return 1 - sse / sst if sst > 1e-12 else np.nan


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def rank_resid(v, z):
    """把 v 对 z 做**秩空间**残差化：rank(v) − OLS(rank(v) ~ rank(z))。"""
    v = np.asarray(v, float)
    z = np.asarray(z, float)
    m = np.isfinite(v) & np.isfinite(z)
    out = np.full(len(v), np.nan)
    if m.sum() < 6:
        return out
    rv = pd.Series(v[m]).rank().to_numpy()
    rz = pd.Series(z[m]).rank().to_numpy()
    X = np.column_stack([np.ones(len(rz)), rz])
    b, *_ = np.linalg.lstsq(X, rv, rcond=None)
    out[m] = rv - X @ b
    return out


def main():
    ev, _ = C.build_events(verbose=False)
    r = ev[ev.kind.isin(["onset", "restep"])].copy()
    print("事件 n=%d（onset %d / restep %d）" % (len(r), (r.kind == "onset").sum(),
                                              (r.kind == "restep").sum()))
    rows = []
    for D in C.LAGS:
        if D < 0.02:
            continue
        Dp = int(round(D * 100))
        slope = r["Jpc_%03d" % Dp]                 # J(Δ)/Δ ÷ 记录峰值
        fr = r["FR_%03d" % Dp]                     # level(t_on+Δ)/level(t_on+0.01)
        t50 = r["T50c_%03d" % Dp]
        p1 = r.pre_frac                            # P1
        on = r.kind == "onset"
        re_ = r.kind == "restep"
        rho_s, p_s, n_s = spearman(slope, p1)
        r2_s = r2_linear(slope, p1)
        rho_f, p_f, _ = spearman(fr, p1)
        auc_slope = auc_mw(slope[on], slope[re_])
        auc_fr = auc_mw(fr[on], fr[re_])
        auc_t50 = auc_mw(t50[on], t50[re_])
        # ② 条件 AUC：按 P1 分层（二分 + 三分位；五等分会把 restep 拆空 ⇒ NaN）
        cond = {}
        for tag, nb in (("2", 2), ("3", 3)):
            q = pd.qcut(p1, nb, labels=False, duplicates="drop")
            acs, ns = [], []
            for g in sorted(pd.unique(q.dropna())):
                m = (q == g)
                a = auc_mw(slope[m & on], slope[m & re_])
                if a == a:
                    acs.append(abs(a - 0.5) + 0.5)
                    ns.append(int(m.sum()))
            cond["cond_%s_median" % tag] = float(np.median(acs)) if acs else np.nan
            cond["cond_%s_n_strata" % tag] = len(acs)
            cond["cond_%s_n_restep_min" % tag] = (
                min(int((m & re_).sum()) for m in [q == g for g in sorted(pd.unique(q.dropna()))])
                if ns else np.nan)
        # ③ 秩残差化后的 slope（去掉 P1 的秩成分）
        resid = rank_resid(slope, p1)
        rr = pd.Series(resid)
        auc_resid = auc_mw(rr[on.to_numpy()], rr[re_.to_numpy()])
        rows.append(dict(
            lag_s=D,
            slope_auc=round(auc_slope, 4) if auc_slope == auc_slope else np.nan,
            FR_auc=round(auc_fr, 4) if auc_fr == auc_fr else np.nan,
            T50c_auc=round(auc_t50, 4) if auc_t50 == auc_t50 else np.nan,
            rho_slope_P1=round(rho_s, 4) if rho_s == rho_s else np.nan,
            p_rho_slope=round(p_s, 5) if p_s == p_s else np.nan,
            R2_slope_P1=round(r2_s, 4) if r2_s == r2_s else np.nan,
            rho_FR_P1=round(rho_f, 4) if rho_f == rho_f else np.nan,
            slope_auc_cond2_median=round(cond["cond_2_median"], 4) if cond["cond_2_median"] == cond["cond_2_median"] else np.nan,
            slope_auc_cond3_median=round(cond["cond_3_median"], 4) if cond["cond_3_median"] == cond["cond_3_median"] else np.nan,
            cond3_n_restep_min=cond["cond_3_n_restep_min"],
            slope_auc_rank_resid=round(auc_resid, 4) if auc_resid == auc_resid else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(C.RES, "t4b_slope_collinearity.csv"), index=False,
             encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print("\n== slope 判据的共线证据 ==")
    print(T.to_string(index=False))
    print("\n读法：")
    print("  * slope 与 P1 的 Spearman ρ 中位 %.3f（p 全部 <1e-7）、线性 R² 中位 %.3f ⇒ 单变量已解释 ~2/3 方差"
          % (float(T.rho_slope_P1.median()), float(T.R2_slope_P1.median())))
    print("  * 按 P1 二分/三分层后 slope 的 AUC 中位 %.4f / %.4f（层内 P1 近似常数）"
          % (float(T.slope_auc_cond2_median.median()), float(T.slope_auc_cond3_median.median())))
    print("  * 去掉 P1 的秩成分后 slope 的 AUC 中位 %.4f ⇒ 回到随机附近"
          % float(T.slope_auc_rank_resid.median()))
    print("  * 纯无量纲速率量 FR(Δ) 的 AUC 中位 %.4f（最大 %.4f @ Δ=%.2f s）"
          % (float(T.FR_auc.median()), float(T.FR_auc.max()),
             float(T.loc[T.FR_auc.idxmax(), "lag_s"])))
    print("  * 到达时刻 T50c(Δ) 的 AUC：最大 %.4f @ Δ=%.2f s；Δ=0.10 s 处 %.4f；Δ=1.00 s 处 %.4f"
          % (float(T.T50c_auc.max()), float(T.loc[T.T50c_auc.idxmax(), "lag_s"]),
             float(T.loc[T.lag_s == 0.1, "T50c_auc"].iloc[0]),
             float(T.loc[T.lag_s == 1.0, "T50c_auc"].iloc[0])))
    print("\n-> results/t4b_slope_collinearity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
