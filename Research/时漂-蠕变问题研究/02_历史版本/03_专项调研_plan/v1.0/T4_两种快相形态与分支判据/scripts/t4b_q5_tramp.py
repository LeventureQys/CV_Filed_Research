# -*- coding: utf-8 -*-
"""t4b_q5_tramp.py —— T4-B × T4-A 接口增量：**输入速率 vs 电平门限**的判据对照。

回答三个问题（用户/席位给出的接口要求）：
  1. 用「电平门限」还是用「输入速率」？—— 离线可分性 + 因果可得性 + 在环收益三者一起看；
  2. 用「输入速率」需要多长的观测窗？—— 逐 Δ 的因果累计 AUC 曲线（Δ = 0.05…1.00 s）；
  3. 我的判据（P1）与 T4-A 的 `T_ramp`（反卷积、**非因果**）相比，谁是代理变量。

T4-A 的冻结接口（**只读，不重算**）：
  `results/t4a_morphology.csv`（60×51，含 `z_at_*` / `step_frame_frac` / `clean` / `armed*`）、
  `results/t4a_input_recover.csv`（`T_ramp` 等，用 `key` + `t_on` 关联）、
  `results/t4a_channel_consistency.csv`（`t50_ch_iqr`）。
本脚本用 (rec, t_on±0.6 s) 把两张表并到本任务的因果特征表上。

产物：results/t4b_tramp_join.csv（事件级并表）、results/t4b_tramp_window.csv（逐 Δ 的因果速率可分性）、
      results/t4b_tramp_vs_p1.csv（代理变量检验：秩相关 + Wishart 层次/净解释力）、
      figures/T4B_06_tramp_window.png
运行：python scripts/t4b_q5_tramp.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402

T4A_MORPH = os.path.join(C.TASK, "results", "t4a_morphology.csv")
T4A_IR = os.path.join(C.TASK, "results", "t4a_input_recover.csv")
T4A_CH = os.path.join(C.TASK, "results", "t4a_channel_consistency.csv")


def auc_mw(x, y):
    x = np.asarray([v for v in x if v == v], float)
    y = np.asarray([v for v in y if v == v], float)
    if len(x) < 3 or len(y) < 3:
        return np.nan, len(x), len(y)
    r = pd.Series(np.concatenate([x, y])).rank().to_numpy()
    rx = r[:len(x)].sum()
    a = float((rx - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y)))
    return a, len(x), len(y)


def bal_acc(x, y, thr, higher_is_onset):
    x = np.asarray([v for v in x if v == v], float)
    y = np.asarray([v for v in y if v == v], float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    sx = float((x >= thr).mean() if higher_is_onset else (x <= thr).mean())
    sy = float((y >= thr).mean() if higher_is_onset else (y <= thr).mean())
    return 0.5 * (sx + (1 - sy))


def best_rule(x, y):
    """给 (AUC, 方向, 最佳阈值, balanced acc)：阈值网格 = 观测值中点。"""
    a, n1, n2 = auc_mw(x, y)
    if a != a:
        return np.nan, np.nan, np.nan, n1, n2
    hi = a >= 0.5
    xx = np.asarray([v for v in x if v == v], float)
    yy = np.asarray([v for v in y if v == v], float)
    vals = np.unique(np.concatenate([xx, yy]))
    grid = np.concatenate([vals, (vals[:-1] + vals[1:]) / 2.0]) if len(vals) > 1 else vals
    best = (np.nan, -1.0)
    for t in grid:
        b = bal_acc(xx, yy, t, hi)
        if b == b and b > best[1]:
            best = (float(t), float(b))
    return a, best[0], best[1], n1, n2


def spearman(a, b):
    from scipy.stats import spearmanr
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def ols_r2(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    X = np.column_stack([np.ones(m.sum()), x[m]])
    beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    pred = X @ beta
    ss_res = float(((y[m] - pred) ** 2).sum())
    ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan


def main():
    ev, _ = C.build_events(verbose=False)
    rise = ev[ev.kind.isin(["onset", "restep"])].copy()

    mor = pd.read_csv(T4A_MORPH)
    mor = mor[mor.kind.isin(["onset", "restep"])].copy()
    ir = pd.read_csv(T4A_IR) if os.path.isfile(T4A_IR) else pd.DataFrame()
    ch = pd.read_csv(T4A_CH) if os.path.isfile(T4A_CH) else pd.DataFrame()

    # ── 并表：(rec, t4a t_on ±0.6 s) ──
    rows = []
    for _, r in rise.iterrows():
        cand = mor[(mor.rec == r.rec) & (np.abs(mor.t_on - r.t_on) <= 0.6)]
        if not len(cand):
            continue
        j = cand.iloc[(cand.t_on - r.t_on).abs().argmin()]
        rec = dict(rec=r.rec, t_on=round(r.t_on, 3), t_on_t4a=round(j.t_on, 3),
                   kind=r.kind, kind_t4a=j.kind, clean=bool(j.clean), armed=bool(j.armed),
                   pre_frac=r.pre_frac, p1s_ratio=r.p1s_ratio, sh_020b=r.sh_020b,
                   z_at_02=j.z_at_02, z_at_01=j.z_at_01, z_at_005=j.z_at_005,
                   step_frame_frac=j.step_frame_frac, frames_to_50=j.frames_to_50,
                   t50=j.t50, t90=j.t90, n_frames_rise=j.n_frames_rise,
                   exp_tau=j.exp_tau, key=j.key, t4a_t_on_raw=j.t_on)
        if len(ir):
            k = ir[(ir.key == j.key) & (np.abs(ir.t_on - j.t_on) <= 0.6)]
            rec["T_ramp"] = float(k.iloc[(k.t_on - j.t_on).abs().argmin()].T_ramp) if len(k) else np.nan
        else:
            rec["T_ramp"] = np.nan
        if len(ch):
            k = ch[(ch.key == j.key) & (ch.kind == r.kind)]
            rec["t50_ch_iqr"] = float(k.iloc[0].t50_ch_iqr) if len(k) else np.nan
        for D in C.LAGS:
            Dp = int(round(D * 100))
            for pre in ("inc_", "T50c_", "T90c_", "R50c_", "FR_", "Jpc_"):
                col = "%s%03d" % (pre, Dp)
                if col in r:
                    rec["%s%03d" % (pre, Dp)] = r[col]
        rows.append(rec)
    J = pd.DataFrame(rows)
    J.to_csv(os.path.join(C.RES, "t4b_tramp_join.csv"), index=False, encoding="utf-8-sig")
    print("== 并表：本任务 %d 个加载事件 → 与 T4-A 配到 %d 个 ==" % (len(rise), len(J)))
    print("   其中 T4-A clean=True 的 %d 个（onset %d / restep %d）"
          % (int(J.clean.sum()),
             int(((J.clean) & (J.kind == "onset")).sum()),
             int(((J.clean) & (J.kind == "restep")).sum())))

    on = J[J.kind == "onset"]
    re_ = J[J.kind == "restep"]
    onc, rec_ = on[on.clean], re_[re_.clean]

    # ── ① 三个"参考判据"的离线可分性（同一批并表事件上）──
    print("\n== ① 判据离线可分性（全部并表事件 n=%d；clean 子集 n=%d）==" % (len(J), len(onc) + len(rec_)))
    ref_rows = []
    for tag, col, hi in (("P1 电平门限 pre/滚动峰值", "pre_frac", False),
                         ("P1b 冻结 V 实现", "p1s_ratio", False),
                         ("T4-A armed（pre/记录峰值 20%）", "armed", False),
                         ("T4-A T_ramp（反卷积，非因果）", "T_ramp", False),
                         ("T4-A t50_ch_iqr（通道间起步离散）", "t50_ch_iqr", False),
                         ("T4-A step_frame_frac", "step_frame_frac", True),
                         ("T4-A frames_to_50", "frames_to_50", True)):
        for sub, n1, n2 in ((J, len(on), len(re_)), (pd.concat([onc, rec_]), len(onc), len(rec_))):
            x = sub[sub.kind == "onset"][col]
            y = sub[sub.kind == "restep"][col]
            if col == "armed":
                x = x.astype(float)
                y = y.astype(float)
            a, thr, ba, n1, n2 = best_rule(x, y)
            ref_rows.append(dict(crit=tag, sample="clean" if len(sub) < len(J) else "all",
                                 n_onset=n1, n_restep=n2,
                                 auc=round(a, 4) if a == a else np.nan,
                                 auc_eff=round(0.5 + abs(a - 0.5), 4) if a == a else np.nan,
                                 thr=round(thr, 4) if thr == thr else np.nan,
                                 balanced_acc=round(ba, 4) if ba == ba else np.nan))
    ref = pd.DataFrame(ref_rows)
    print(ref.to_string(index=False))
    ref.to_csv(os.path.join(C.RES, "t4b_tramp_ref_crit.csv"), index=False, encoding="utf-8-sig")

    # ── ② 因果速率判据需要多长观测窗 ──
    print("\n== ② 因果输入速率判据：逐 Δ 的可分性（判据只用 t ≤ t_on+Δ）==")
    wrows = []
    for D in C.LAGS:
        if D < 0.02:
            continue
        Dp = int(round(D * 100))
        # (a) 到达比例 FR(Δ)：level(t_on+Δ) 相对 level(t_on+0.01) —— 越大越快
        a1, t1, b1, n1, n2 = best_rule(on["FR_%03d" % Dp], re_["FR_%03d" % Dp])
        # (b) T50c(Δ)：达到 0.5·J(Δ) 的时刻（NaN = Δ 内没到 50%）—— 越大越慢
        a2, t2, b2, _, _ = best_rule(on["T50c_%03d" % Dp], re_["T50c_%03d" % Dp])
        # (c) 因果速率量 slope = J(Δ)/Δ
        a3, t3, b3, _, _ = best_rule(on["Jpc_%03d" % Dp], re_["Jpc_%03d" % Dp])
        wrows.append(dict(
            lag_s=D,
            FR_auc=round(a1, 4) if a1 == a1 else np.nan,
            FR_auc_eff=round(0.5 + abs(a1 - 0.5), 4) if a1 == a1 else np.nan,
            FR_bal=round(b1, 4) if b1 == b1 else np.nan, FR_thr=round(t1, 4) if t1 == t1 else np.nan,
            T50c_nan_onset=int(on["T50c_%03d" % Dp].isna().sum()),
            T50c_nan_restep=int(re_["T50c_%03d" % Dp].isna().sum()),
            T50c_auc=round(a2, 4) if a2 == a2 else np.nan,
            T50c_bal=round(b2, 4) if b2 == b2 else np.nan,
            slope_auc=round(a3, 4) if a3 == a3 else np.nan,
            slope_bal=round(b3, 4) if b3 == b3 else np.nan))
    W = pd.DataFrame(wrows)
    W.to_csv(os.path.join(C.RES, "t4b_tramp_window.csv"), index=False, encoding="utf-8-sig")
    print(W.to_string(index=False))

    # ── ③ 代理变量检验：T_ramp 是驱动量吗？P1 是它的代理吗？──
    print("\n== ③ 代理变量检验（参考量 = T4-A 的 z_at_02，干净的形态轴）==")
    vrows = []
    for tag, col in (("P1 pre/峰值（因果、Δ=0）", "pre_frac"),
                     ("因果斜率 J(0.10)/0.10（Δ=0.10）", "Jpc_010"),
                     ("因果到达比例 FR(0.20)（Δ=0.20）", "FR_020"),
                     ("T4-A armed（因果、Δ=0）", "armed"),
                     ("T4-A T_ramp（非因果）", "T_ramp"),
                     ("T4-A t50_ch_iqr（非因果）", "t50_ch_iqr")):
        rho, p, n = spearman(J[col], J.z_at_02)
        r2 = ols_r2(J[col], J.z_at_02)
        vrows.append(dict(var=tag, n=n, spearman_vs_z_at_02=round(rho, 4) if rho == rho else np.nan,
                          p=round(p, 5) if p == p else np.nan,
                          R2_z_at_02=round(r2, 4) if r2 == r2 else np.nan))
    V = pd.DataFrame(vrows)
    # 净解释力：z_at_02 ~ log10 T_ramp (+ P1)
    ltr = np.log10(np.maximum(J.T_ramp.to_numpy(float), 1e-3))
    z = J.z_at_02.to_numpy(float)
    X1 = np.column_stack([np.ones(len(ltr)), ltr])
    m = np.isfinite(ltr) & np.isfinite(z)
    b1, *_ = np.linalg.lstsq(X1[m], z[m], rcond=None)
    r2_1 = ols_r2(ltr, z)
    X2 = np.column_stack([np.ones(len(ltr)), ltr, J.pre_frac.to_numpy(float)])
    m2 = m & np.isfinite(J.pre_frac.to_numpy(float))
    b2, *_ = np.linalg.lstsq(X2[m2], z[m2], rcond=None)
    pred2 = X2[m2] @ b2
    r2_2 = 1 - float(((z[m2] - pred2) ** 2).sum()) / float(((z[m2] - z[m2].mean()) ** 2).sum())
    print(V.to_string(index=False))
    print("\n  z_at_02 ~ log10 T_ramp                      R² = %.4f (n=%d)" % (r2_1, int(m.sum())))
    print("  z_at_02 ~ log10 T_ramp + P1(pre_frac)        R² = %.4f (n=%d)  ΔR² = %+.4f"
          % (r2_2, int(m2.sum()), r2_2 - r2_1))
    print("  系数（标准化前）：const %.4f  logT %.4f  P1 %.4f" % (b2[0], b2[1], b2[2]))
    V.to_csv(os.path.join(C.RES, "t4b_tramp_vs_p1.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(model="z_at_02 ~ log10 T_ramp", R2=round(r2_1, 4), n=int(m.sum())),
                  dict(model="z_at_02 ~ log10 T_ramp + P1", R2=round(r2_2, 4), n=int(m2.sum())),
                  dict(model="ΔR²(P1 | T_ramp)", R2=round(r2_2 - r2_1, 4), n=int(m2.sum())),
                  dict(model="const/logT/P1 系数", R2=np.nan,
                       n=int(m2.sum()))]).to_csv(
        os.path.join(C.RES, "t4b_tramp_r2.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(C.RES, "t4b_tramp_r2.csv"), "a", encoding="utf-8-sig") as f:
        f.write("# const=%.6f logT=%.6f P1=%.6f\n" % (b2[0], b2[1], b2[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
