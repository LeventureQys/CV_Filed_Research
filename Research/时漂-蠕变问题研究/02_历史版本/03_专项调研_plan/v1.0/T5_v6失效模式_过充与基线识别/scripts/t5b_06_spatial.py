# -*- coding: utf-8 -*-
"""T5-B / 06：空间模式线索（第一轮未开发）—— 拍击 vs 真实加载的通道空间模式差异

要回答：**「活跃通道数 / 空间分布」能不能把"人手拍在负载上"和"真实加载"区分开？**

必须写在前面的诚实声明（否则整节会误导）
    本任务所用的**注入拍击**（`t1_common.add_tap`）把扰动峰值按**当前逐通道电平 `|X[i0]|` 分摊**，
    即"拍击的空间模式 = 加载的空间模式"是**构造出来的**，不是实测的。因此：
      · 用**注入拍击**评估空间判据，得到的必然是"无区分度"——这只能证明"注入模型不支持该判据"，
        不能证明"实拍击不可区分"；
      · 真正有信息量的是 **实录真实瞬态**（SW4 @99.56 s 一例）与 **真实加载** 的对比；
      · 以及**假设性评估**：若实拍击是局部的（只压到 1/3/5 个通道），空间判据有多少区分度。

产物
    results/t5b_spatial_pattern.csv   逐事件的空间特征（真实加载 / 实录瞬态 / 三类注入拍击）
    results/t5b_spatial_roc.csv       空间判据（active_frac / coup）的 ROC 与 AUC
    results/t5b_spatial_summary.csv   分组统计
    results/_t5b_06.log
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_core import REC, load_gt, gt_in_window, WIN, DT, load_window    # noqa: E402
from t5b_ad_lib import load_rec, med_smooth, channel_activity, find_transients  # noqa: E402
from t5b_a_common import load_uniform, add_tap                     # noqa: E402

K_TOP = (1, 3, 5, 8)
TAP_SITES = [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]


def prof_cos(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def feat(dv, group, name, t, extra=None):
    d = channel_activity(dv, frac=0.10, floor=1.0)
    row = dict(group=group, name=name, t=round(float(t), 3),
               dv_max=round(float(np.abs(dv).max()), 1), dv_sum=round(float(np.abs(dv).sum()), 1),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
    if extra:
        row.update(extra)
    return row


def auc_mw(vals, y, direction="ge"):
    """Mann-Whitney U 形式的 AUC（对并列值取平均秩，避免 ROC 排序/重复点带来的偏差）。"""
    from scipy.stats import rankdata
    v = np.asarray(vals, float).copy()
    if direction == "le":
        v = -v
    y = np.asarray(y, int)
    r = rankdata(v)
    P = int(y.sum())
    N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    return float((r[y == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def main():
    gt = load_gt()
    rows = []
    # ── 1. 真实加载：T4-A 冻结事件表里的 4 份变载实录（39 个事件）──
    for key, rec in (("SW1", "切换负载-快相无责"), ("SW2", "零负载-切换负载-零负载-再切换负载"),
                     ("SW3", "零负载-中途切换负载-零负载-切换负载"),
                     ("SW4", "中途切换-最终测试目标")):
        d = load_uniform(REC[rec])
        tu, X = d["tu"], d["Xu"]
        g = gt[gt["key"] == key]
        for _, r in g.iterrows():
            t0 = float(r["t_on"])
            a0 = max(0, int(round((t0 - 2.0) / DT)))
            a1 = max(1, int(round(t0 / DT)))
            b0 = int(round((t0 + 4.0) / DT))
            b1 = int(round((t0 + 6.0) / DT))
            if b1 >= len(X) or a1 - a0 < 2 or b1 - b0 < 2:
                continue
            dv = X[b0:b1].mean(axis=0) - X[a0:a1].mean(axis=0)
            rows.append(feat(dv, f"real_load_{key}", f"{key}@{t0:.2f}", t0,
                             dict(kind=str(r["kind"]), jump=round(float(r["jump"]), 1),
                                  level=round(float(r["pre"]), 1))))
    # ── 2. 实录真实瞬态（放宽阈值重扫；推断为"人手拍/冲击"类）──
    tr_rows = []
    for rec in REC:
        d = load_uniform(REC[rec])
        tu, X = d["tu"], d["Xu"]
        Z = X.sum(axis=1)
        for i0, peak, back, pre in find_transients(Z, DT, up_frac=0.02, rec_s=0.60,
                                                   pre_s=0.30, min_peak=200.0, dedup_s=1.0):
            a0 = max(0, i0 - 40)
            dv = X[i0] - X[a0]
            rows.append(feat(dv, "real_transient", f"{rec}@{tu[i0]:.2f}", tu[i0],
                             dict(kind="transient", jump=round(peak, 1), level=round(pre, 1),
                                  back=round(back, 1),
                                  ratio=round(abs(peak) / max(abs(pre), 1.0), 4))))
            tr_rows.append(dict(rec=rec, t=round(float(tu[i0]), 3), peak=round(peak, 1),
                                back=round(back, 1), level=round(pre, 1), idx=int(i0)))
    print(f"真实瞬态候选 n={len(tr_rows)}")
    for r in tr_rows:
        print("  ", r)
    # ── 3. 注入拍击：三种空间模式（按电平分摊 / 顶部 k 通道焦点）──
    #    注意：必须在**拍击峰值帧**取样，且 dv = Xp[i] − Xu[i]（同一帧），
    #    否则会把"不同帧的自然起伏"当成空间模式（本脚本第一版就踩了这个坑）。
    tu, Xu, W = load_window(WIN["W1_1w"])
    lvl = float(np.median(Xu.sum(axis=1)))
    RISE, HOLD, FALL = 25, 50, 25
    nr, nh = int(round(RISE / 10.0)), int(round(HOLD / 10.0))
    all_ch = Xu.shape[1]
    for k in list(K_TOP) + [all_ch]:
        grp = "inj_tap_all" if k >= all_ch else f"inj_tap_top{k}"
        for site in TAP_SITES:
            i0 = int(round(site / DT))
            base_prof = np.abs(Xu[i0])
            order = np.argsort(-base_prof)
            mask = np.zeros(all_ch)
            mask[order[:k]] = 1.0
            i_pk = i0 + nr + max(1, nh // 2)
            for amp_pct in (30, 50, 100):
                Xp, _ = add_tap(Xu, i0, amp_pct / 100.0 * lvl, fs=100.0, rise_ms=RISE,
                                hold_ms=HOLD, fall_ms=FALL, ch_mask=mask)
                dv = Xp[i_pk] - Xu[i_pk]
                rows.append(feat(dv, grp, f"top{k}_{amp_pct}pct@{site:.0f}", site,
                                 dict(kind="inj_tap", jump=round(amp_pct / 100.0 * lvl, 1),
                                      level=round(lvl, 1), n_mask=int(mask.sum()),
                                      i_peak=i_pk, rise_ms=RISE, hold_ms=HOLD, fall_ms=FALL)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t5b_spatial_pattern.csv"), index=False, encoding="utf-8-sig")

    # ── 4. 空间判据 ROC：real_load(正) vs 各类拍击(负) ──
    roc, summ = [], []
    for neg in ("inj_tap_all", "inj_tap_top8", "inj_tap_top5", "inj_tap_top3", "inj_tap_top1",
                "real_transient"):
        sub = df[df["group"].isin(["real_load_SW1", "real_load_SW2", "real_load_SW3",
                                   "real_load_SW4", neg])].copy()
        sub["y"] = (sub["group"].str.startswith("real_load")).astype(int)
        P, N = int(sub["y"].sum()), int((1 - sub["y"]).sum())
        for featname, mode in (("coup", "ge"), ("active_frac", "ge"), ("participation", "ge"),
                               ("n_active", "ge"), ("hhi", "le")):
            vals = sub[featname].to_numpy(float)
            grid = np.unique(np.round(np.linspace(np.nanmin(vals), np.nanmax(vals), 61), 4))
            pts = []
            for thr in grid:
                acc = vals >= thr if mode == "ge" else vals <= thr
                tp = int(np.sum(acc & (sub["y"] == 1)))
                fp = int(np.sum(acc & (sub["y"] == 0)))
                pts.append(dict(thr=float(thr), tp=tp, fp=fp, tpr=tp / P, fpr=fp / N,
                                precision=(tp / (tp + fp)) if (tp + fp) else np.nan))
            # 锚点必须与判据方向一致：mode=ge 时 thr=+inf 全拒、thr=-inf 全收
            if mode == "ge":
                pts.append(dict(thr=np.inf, tp=0, fp=0, tpr=0.0, fpr=0.0, precision=np.nan))
                pts.append(dict(thr=-np.inf, tp=P, fp=N, tpr=1.0, fpr=1.0,
                                precision=P / (P + N)))
            else:
                pts.append(dict(thr=-np.inf, tp=0, fp=0, tpr=0.0, fpr=0.0, precision=np.nan))
                pts.append(dict(thr=np.inf, tp=P, fp=N, tpr=1.0, fpr=1.0,
                                precision=P / (P + N)))
            q = pd.DataFrame(pts).drop_duplicates(subset=["tpr", "fpr"]) \
                .sort_values(["fpr", "tpr"], kind="mergesort")
            auc = auc_mw(vals, sub["y"].to_numpy(int), direction=mode)
            q["neg_group"] = neg
            q["feature"] = featname
            q["auc"] = round(auc, 4)
            roc.append(q)
            b = q[q["fpr"] <= 0.05]
            summ.append(dict(neg_group=neg, feature=featname, n_pos=P, n_neg=N, auc=round(auc, 4),
                             best_tpr_at_fpr05=(round(float(b["tpr"].max()), 3) if len(b) else np.nan),
                             thr_at_fpr05=(float(b.loc[b["tpr"].idxmax(), "thr"]) if len(b) else np.nan)))
    pd.concat(roc, ignore_index=True).to_csv(os.path.join(RES, "t5b_spatial_roc.csv"),
                                            index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(RES, "t5b_spatial_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 分组统计 ===")
    print(df.groupby("group")[["n_ch", "n_active", "active_frac", "coup", "hhi",
                               "participation"]].median().round(3).to_string())
    print("\n=== 空间判据 AUC ===")
    print(sm.to_string())


if __name__ == "__main__":
    main()
