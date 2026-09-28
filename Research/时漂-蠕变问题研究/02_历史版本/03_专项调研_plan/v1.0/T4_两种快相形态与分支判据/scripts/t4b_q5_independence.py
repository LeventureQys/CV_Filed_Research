# -*- coding: utf-8 -*-
"""t4b_q5_independence.py —— T4-Q5 / T4-Q2 支撑：因果判据候选的可分性（AUC + 重叠度 + 所需窗长）。

对每个候选判据（① pre 电平 ② 上升沿斜率/时长 ③ 台阶幅度相对电平比 ④ 上升沿形状单帧占比），
在多个 Δ（0.00~1.00 s）上计算 onset vs restep 的：
  AUC（Mann-Whitney U / 秩和，方向已归一到 ≥0.5）、最佳阈值、准确率、balanced accuracy、
  p10~p90 区间重叠率、中位差与 Cliff's δ。
判据**只用 t ≤ t_on + Δ 的帧**；Δ 写在 `lag_s` 列，越小越好。

产物：results/t4b_morphology_by_arm.csv（每事件一行：真值标签 + 因果特征 + 事后形状）
      results/t4b_separability.csv（每判据 × 每 Δ 一行）
      results/t4b_shape_by_arm.csv（两形态归一化形状对比 + 效应量）
运行：python scripts/t4b_q5_independence.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402


def auc_mw(x, y):
    """AUC = P(x>y) + 0.5·P(x=y)，秩和公式（无外部依赖）。"""
    x = np.asarray([v for v in x if v == v], float)
    y = np.asarray([v for v in y if v == v], float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    r = pd.Series(np.concatenate([x, y])).rank().to_numpy()
    rx = r[:len(x)].sum()
    return float((rx - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y)))


def best_threshold(x, y, higher_is_onset=True):
    """在候选阈值上最大化 balanced accuracy；返回 (thr, bal, sens, spec, acc)。"""
    x = np.asarray([v for v in x if v == v], float)
    y = np.asarray([v for v in y if v == v], float)
    if len(x) == 0 or len(y) == 0:
        return (np.nan,) * 5
    vals = np.unique(np.concatenate([x, y]))
    grid = np.concatenate([vals, (vals[:-1] + vals[1:]) / 2.0]) if len(vals) > 1 else vals
    best = (np.nan, -1.0)
    for t in grid:
        sx = (x >= t).mean() if higher_is_onset else (x <= t).mean()
        sy = (y >= t).mean() if higher_is_onset else (y <= t).mean()
        bal = 0.5 * (sx + (1 - sy))
        if bal > best[1]:
            best = (float(t), float(bal))
    t, bal = best
    if t != t:
        return (np.nan,) * 5
    if higher_is_onset:
        sens, spec = float((x >= t).mean()), float(1 - (y >= t).mean())
    else:
        sens, spec = float((x <= t).mean()), float(1 - (y <= t).mean())
    tp = int((x >= t).sum()) if higher_is_onset else int((x <= t).sum())
    fp = int((y >= t).sum()) if higher_is_onset else int((y <= t).sum())
    acc = (tp + len(y) - fp) / (len(x) + len(y))
    return t, bal, sens, spec, float(acc)


def overlap_coef(x, y):
    """p10~p90 区间重叠率（1 = 完全重叠，0 = 完全分离）与该区间宽度。"""
    x = np.asarray([v for v in x if v == v], float)
    y = np.asarray([v for v in y if v == v], float)
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    xa, xb = np.percentile(x, 10), np.percentile(x, 90)
    ya, yb = np.percentile(y, 10), np.percentile(y, 90)
    inter = max(0.0, min(xb, yb) - max(xa, ya))
    union = max(xb, yb) - min(xa, ya)
    return (inter / union if union > 1e-12 else 1.0), float(xb - xa)


def cliff_delta(x, y):
    a = auc_mw(x, y)
    return np.nan if a != a else 2 * a - 1


REFMAP = {           # 第一轮 events.csv 的 rec 列名 → 本任务键名（逐条核对过）
    "中途切换-13ffca": "中途切换-13ffca",
    "中途切换-1d9493": "中途切换-1d9493",
    "切换负载-快相无责": "切换负载-快相无责",
}


def main():
    ev, cache = C.build_events()
    ev.to_csv(os.path.join(C.RES, "t4b_morphology_by_arm.csv"), index=False,
              encoding="utf-8-sig")
    print("事件真值分布：", dict(ev.kind.value_counts()))
    print("臂分布：", dict(ev.arm.value_counts()))

    rise = ev[ev.kind.isin(["onset", "restep"])].copy()
    on = rise[rise.kind == "onset"]
    re = rise[rise.kind == "restep"]
    onc = rise[(rise.kind == "onset") & rise.label_confident]
    rec_ = rise[(rise.kind == "restep") & rise.label_confident]
    print("\nonset n=%d  restep n=%d（样本不平衡 ⇒ AUC 用秩和，另给 balanced accuracy）"
          % (len(on), len(re)))
    print("剔除 20% 门限 ±5 pt 边界带后：onset n={}  restep n={}".format(len(onc), len(rec_)))

    refp = os.path.join(C.PROG_REF, "events.csv")
    if os.path.isfile(refp):
        r = pd.read_csv(refp)
        m = []
        for _, row in rise.iterrows():
            c = r[(r.rec == REFMAP.get(row.rec, row.rec))
                  & (np.abs(r.t_edge - row.t_on) <= 0.5)]
            if len(c):
                m.append(row.kind == c.iloc[(c.t_edge - row.t_on).abs().argmin()].kind)
        if m:
            print("与第一轮 events.csv 的标签一致率：%.3f（n=%d，仅匹配到的事件）"
                  % (float(np.mean(m)), len(m)))

    rows = []

    def add(cid, desc, xa, ya, lag, note=""):
        x = np.asarray([v for v in xa if v == v], float)
        y = np.asarray([v for v in ya if v == v], float)
        if len(x) < 4 or len(y) < 4:
            print("  (跳过 %s：有效样本 onset=%d restep=%d)" % (cid, len(x), len(y)))
            return
        a = auc_mw(x, y)
        hi_on = a >= 0.5
        thr, bal, sens, spec, acc = best_threshold(x, y, higher_is_onset=hi_on)
        ov, w = overlap_coef(x, y)
        rows.append(dict(crit=cid, desc=desc, lag_s=lag, causal=True,
                         n_onset=len(x), n_restep=len(y),
                         auc=round(float(a), 4),
                         auc_dir="higher_onset" if hi_on else "lower_onset",
                         auc_eff=round(0.5 + abs(a - 0.5), 4),
                         cliff_delta=round(float(cliff_delta(x, y)), 4),
                         thr=round(float(thr), 4) if thr == thr else np.nan,
                         balanced_acc=round(float(bal), 4), sens_onset=round(sens, 4),
                         spec_restep=round(spec, 4), acc=round(acc, 4),
                         med_onset=round(float(np.median(x)), 4),
                         med_restep=round(float(np.median(y)), 4),
                         p10_onset=round(float(np.percentile(x, 10)), 4),
                         p90_onset=round(float(np.percentile(x, 90)), 4),
                         p10_restep=round(float(np.percentile(y, 10)), 4),
                         p90_restep=round(float(np.percentile(y, 90)), 4),
                         p10p90_overlap=round(float(ov), 4) if ov == ov else np.nan,
                         band_width_onset=round(float(w), 4) if w == w else np.nan,
                         note=note))

    # ── ① pre 电平（两个实现口径）──
    add("P1_pre_frac", "① pre 电平：pre 因果 0.5 s 电平 ÷ 全历史因果滚动峰值（Q5 主口径）",
        on.pre_frac, re.pre_frac, 0.00,
        note="Δ=0；分母用记录内可得的历史峰值上限")
    add("P1b_pre_frac_frozenV", "① pre 电平：同分子 ÷ 检测帧冻结峰值 V=max(zc[:k+1])（Q8 实现口径）",
        on.p1s_ratio, re.p1s_ratio, 0.00,
        note="Δ=0；工程实现必须冻结 V，否则判据自我指涉")
    for thr_lbl, frac in (("10%", 0.10), ("20%", 0.20), ("30%", 0.30), ("40%", 0.40)):
        pred_on = rise.pre_frac <= frac
        rows.append(dict(crit="P1_pre_frac@%s" % thr_lbl,
                         desc="① pre 门限固定值敏感性 %s（无调参）" % thr_lbl,
                         lag_s=0.00, causal=True, n_onset=len(on), n_restep=len(re),
                         auc=np.nan, auc_dir="", auc_eff=np.nan, cliff_delta=np.nan,
                         thr=frac, balanced_acc=np.nan,
                         sens_onset=round(float((on.pre_frac <= frac).mean()), 4),
                         spec_restep=round(float((re.pre_frac > frac).mean()), 4),
                         acc=round(float((pred_on == (rise.kind == "onset")).mean()), 4),
                         med_onset=round(float(on.pre_frac.median()), 4),
                         med_restep=round(float(re.pre_frac.median()), 4),
                         p10_onset=np.nan, p90_onset=np.nan, p10_restep=np.nan,
                         p90_restep=np.nan, p10p90_overlap=np.nan,
                         band_width_onset=np.nan, note="固定门限"))

    # ── ② 上升沿斜率 ──
    for D in ("010", "020", "030", "050", "100"):
        add("P2_slope@%ss" % (int(D) / 100.0), "② 上升沿斜率 J(Δ)/Δ ÷ 记录峰值",
            on["Jpc_%s" % D], re["Jpc_%s" % D], int(D) / 100.0)
    add("P2_slope_rel", "② 斜率 ÷ 自身 J(5 s)（无量纲）",
        on.Jpc_020 / np.maximum(np.abs(on.A5_self), 1e-9),
        re.Jpc_020 / np.maximum(np.abs(re.A5_self), 1e-9), 0.20)

    # ── ② 上升沿时长（对 Δ 敏感）──
    KEYS = ["inc_%03d" % k for k in (0, 1, 2, 3, 5, 10, 20, 30, 50, 100)]

    def rise_dur(sub, D):
        out = []
        Dp = int(round(D * 100))
        keys = [k for k in KEYS if int(k[4:]) <= Dp]
        for _, r in sub.iterrows():
            if len(keys) < 3 or not (r["inc_%03d" % Dp] > 0):
                out.append(np.nan)
                continue
            tgt = 0.9 * r["inc_%03d" % Dp]
            hit = [int(k[4:]) / 100.0 for k in keys if r[k] >= tgt]
            out.append(hit[0] if hit else np.nan)
        return np.array(out, float)

    for D in (0.20, 0.50, 1.00):
        add("P2_dur@%0.2fs" % D, "② 上升沿时长（0.9·J(Δ) 到达时刻，Δ=%.2f s）" % D,
            rise_dur(on, D), rise_dur(re, D), D,
            note="Δ 内未达 90% 记 NaN ⇒ 该判据对 Δ 更敏感")

    # ── ③ 台阶幅度相对电平比 ──
    for D in ("020", "050", "100"):
        add("P3_J_over_pre@%ss" % (int(D) / 100.0), "③ 台阶幅度 / pre 电平  J(Δ)/pre",
            on["inc_%s" % D] / np.maximum(on.pre, 1e-9),
            re["inc_%s" % D] / np.maximum(re.pre, 1e-9), int(D) / 100.0)
    add("P3_J_frac_peak", "③ 台阶幅度 ÷ 记录峰值 J(0.20)/peak",
        on.Jpc_020, re.Jpc_020, 0.20,
        note="与 P2_slope@0.2s 同源（差常数倍）⇒ 列出以说明'相对幅度'无额外信息")

    # ── ④ 上升沿形状 ──
    add("P4_singleframe@020", "④ 单帧跃变占比 J(0.00)/J(0.20)",
        on.JF_020, re.JF_020, 0.20)
    add("P4_shape020", "④ J(0.02)/J(0.05)", on.shape_020, re.shape_020, 0.05)
    add("P4_shape010@020", "④ J(0.01)/J(0.02)", on.shape_010, re.shape_010, 0.02)
    for D in ("030", "050", "100"):
        add("P4_ratio020@%ss" % (int(D) / 100.0), "④ 早期完成度 J(0.20)/J(Δ)",
            on.inc_020 / np.maximum(on["inc_%s" % D], 1e-9),
            re.inc_020 / np.maximum(re["inc_%s" % D], 1e-9), int(D) / 100.0)

    sep = pd.DataFrame(rows).sort_values(["crit", "lag_s"]).reset_index(drop=True)
    sep.to_csv(os.path.join(C.RES, "t4b_separability.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print("\n== 判据可分性（auc_eff 越接近 1 越可分；p10p90_overlap 越小越可分）==")
    print(sep[["crit", "lag_s", "auc_eff", "balanced_acc", "acc", "thr", "p10p90_overlap",
               "med_onset", "med_restep"]].to_string(index=False))

    # ── 两形态归一化形状（两套归一）──
    print("\n== 两形态归一化形状（J(τ)/J(5 s)，与既有 shape_stats.csv 口径可比）==")
    st = []
    for c in [x for x in rise.columns if x.startswith("sh_") and x.endswith("b")]:
        a, b = on[c].dropna(), re[c].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        st.append(dict(tau=int(c[3:6]) / 100, n_on=len(a), n_re=len(b),
                       med_on=a.median(), med_re=b.median(),
                       d_pt=100 * (a.median() - b.median()),
                       p10_on=a.quantile(.1), p90_on=a.quantile(.9),
                       p10_re=b.quantile(.1), p90_re=b.quantile(.9),
                       auc=auc_mw(a, b), overlap=overlap_coef(a, b)[0]))
    stdf = pd.DataFrame(st)
    stdf.to_csv(os.path.join(C.RES, "t4b_shape_by_arm.csv"), index=False, encoding="utf-8-sig")
    print(stdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
