# -*- coding: utf-8 -*-
"""T5-B / 04：T5B-Q4 判据改进的可解性 —— 候选判据的 ROC/PR。

问题：现行判据（短滞后电平差 `d` + σ 门限）为什么被拍击骗？能不能加"平台/保持"维
（第一轮已指出判据里缺这一维）与"空间模式"维来区分？

做法（写死）
    候选 = `raw_hit` 连续段里首次满足「连续 ≥3 帧」的那一帧（= 现行判据的决策时刻）。
    标签 = 该时刻是否落在某个真实沿 ±1.5 s 内（真值 = T4-A 冻结事件表，只读引用）。
    判据 = `base ∧ cond(param)`；对每个判据扫 param → ROC/PR 点；AUC = 梯形积分。

产物
    results/t5b_criteria_features.csv  逐候选特征 + 标签（含 dwell_s）
    results/t5b_criteria_roc.csv       判据 × 参数 → TPR/FPR/precision/recall
    results/t5b_criteria_summary.csv   每判据的 AUC / 最佳工作点 / 因果性标注
    results/_t5b_04.log
并把 `family='criteria'` 行追加进 results/t5b_detector_roc.csv（与 Q3 的 boundary 行同表）。

因果性标注（上线可行性）
    K0/K1/K2/K5/K6/K7/K8 = **因果**（只用当前帧及过去）；K3/K4 = **非因果**（需 +0.5~0.6 s 未来样本，
    上线等于给判据加 0.5~0.6 s 延迟）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_core import (TraceV6, run_full, load_window, perturb_window,   # noqa: E402
                      gt_in_window, load_gt, WIN, DT)

GRID = [
    # (win, kind, cls, param, seeds)
    ("W1_1w", "noise", "white", 500.0, list(range(6))),
    ("W1_1w", "noise", "white", 1000.0, list(range(6))),
    ("W1_1w", "noise", "white", 2000.0, list(range(6))),
    ("W1_1w", "noise", "common", 1000.0, list(range(4))),
    ("W1_1w", "noise", "common", 2000.0, list(range(4))),
    ("W1_1w", "noise", "band5", 1000.0, list(range(4))),
    ("W1_1w", "noise", "band5", 2000.0, list(range(4))),
]
TAPS = [(20, 100), (50, 100), (100, 100), (50, 50), (50, 200), (50, 500)]
SITES = [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]


def build():
    out = []
    for win, kind, cls, amp, seeds in GRID:
        out.append(dict(win=win, kind=kind, cls=cls, amp=amp, seeds=seeds, sites=[None]))
    for pct, dur in TAPS:
        out.append(dict(win="W1_1w", kind="tap", cls="tap", amp_pct=pct, dur_ms=dur,
                        seeds=[0], sites=SITES))
    return out


def candidates(tu, Xu, tr, epochs, gt_w, w_t0, cls, param, seed, wname):
    rh = np.asarray(tr["raw_hit"], bool)
    d = np.asarray(tr["d"], float)
    ts = np.asarray(tr["ts"], float)
    idx = np.where(rh)[0]
    if len(idx) == 0:
        return []
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    gt_arr = np.array([t - w_t0 for t in gt_w["t_on"]], float) if len(gt_w) else np.array([])
    Z = Xu.sum(axis=1)
    n_ch = Xu.shape[1]
    out = []
    for s in segs:
        cand = None
        for j in s:
            if tr["hit_run"][j] >= 3:
                cand = int(j)
                break
        if cand is None:
            continue
        i_r0, i_r1 = max(0, cand - int(0.65 / DT)), max(0, cand - int(0.35 / DT))
        if i_r1 <= i_r0:
            continue
        dvv = Xu[max(0, cand - int(0.20 / DT)):cand + 1].mean(axis=0) - Xu[i_r0:i_r1 + 1].mean(axis=0)
        ad = np.abs(dvv)
        mx = float(ad.max())
        act = ad > max(0.10 * mx, 1.0)
        p = ad / ad.sum() if ad.sum() > 1e-12 else np.zeros_like(ad)
        hhi = float((p ** 2).sum())
        ih = cand + int(0.50 / DT)
        zref = float(np.median(Z[i_r0:max(1, i_r1)]))
        hold = rev = np.nan
        if ih < len(Z):
            hold = float(np.sign(d[cand]) * (float(np.median(Z[ih - 5:ih + 6])) - zref) /
                         max(abs(d[cand]), 1e-9))
            iw = Z[cand:min(len(Z), cand + int(0.60 / DT))]
            rev = float(np.any(np.sign(iw - zref) * np.sign(d[cand]) < 0))
        out.append(dict(
            win=wname, cls=cls, param=param, seed=seed, t_c=round(float(ts[cand]), 3),
            d=round(float(d[cand]), 1), lv_ref=round(float(tr["lv_ref"][cand]), 1),
            gate=round(float(tr["gate"][cand]), 1),
            ratio=round(abs(float(d[cand])) / max(float(tr["gate"][cand]), 1e-9), 3),
            ratio_rel=round(abs(float(d[cand])) /
                            max(0.05 * abs(float(tr["lv_ref"][cand])), 1e-9), 3),
            sig_d=round(float(tr["sig_d"][cand]), 1),
            dwell_s=round((int(s[-1]) - int(s[0]) + 1) * DT, 3),
            n_ch=int(n_ch), n_active=int(act.sum()),
            active_frac=round(float(act.sum()) / n_ch, 3),
            coup=round(float(ad.sum() / mx), 2) if mx > 0 else np.nan,
            hhi=round(hhi, 4) if hhi == hhi else np.nan,
            f_mono=round(float(np.mean(np.sign(d[max(0, cand - 50):cand + 1]) ==
                                       np.sign(d[cand]))), 3),
            f_hold=(round(hold, 3) if hold == hold else np.nan),
            f_rev=(int(rev) if rev == rev else np.nan),
            label=int(len(gt_arr) > 0 and np.min(np.abs(gt_arr - float(ts[cand]))) <= 1.5)))
    return out


CRIT = {
    "K0_gate":       ("ratio",            [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],        "causal", "θ"),
    "K1_dwell":      ("dwell_s",          [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
                      "causal", "τ_dwell(s)"),
    "K2_mono":       ("f_mono",           [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00],
                      "causal", "f_mono≥"),
    "K3_hold":       ("f_hold",           [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],             "非因果(+0.5s)", "f_hold≥"),
    "K5_actfrac":    ("active_frac",      [1.01, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
                      "causal", "active_frac≤"),
    "K6_coup":       ("coup",             [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
                      "causal", "coup≥"),
}


def roc_from(df, feat, params, mode="ge"):
    P = int(df["label"].sum())
    N = int(len(df) - P)
    pts = []
    for t in params:
        acc = (df[feat] >= t).to_numpy() if mode == "ge" else (df[feat] <= t).to_numpy()
        tp = int((acc & (df["label"] == 1)).sum())
        fp = int((acc & (df["label"] == 0)).sum())
        pts.append(dict(param=t, tp=tp, fp=fp, fn=P - tp, tn=N - fp,
                        tpr=tp / P if P else np.nan, fpr=fp / N if N else np.nan,
                        precision=(tp / (tp + fp)) if (tp + fp) else np.nan,
                        recall=tp / P if P else np.nan))
    pts.append(dict(param="-inf", tp=0, fp=0, fn=P, tn=N, tpr=0.0, fpr=0.0,
                    precision=np.nan, recall=0.0))
    pts.append(dict(param="+inf", tp=P, fp=N, fn=0, tn=0, tpr=1.0, fpr=1.0,
                    precision=P / (P + N), recall=1.0))
    r = pd.DataFrame(pts).drop_duplicates(subset=["tpr", "fpr"])
    r = r.sort_values("fpr")
    auc = float(np.trapezoid(r["tpr"].to_numpy(), r["fpr"].to_numpy()))
    return r, auc


def main():
    gt = load_gt()
    rows = []
    for cell in build():
        tu, Xu, W = load_window(WIN[cell["win"]])
        gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
        lvl = float(np.median(Xu.sum(axis=1)))
        for seed in cell["seeds"]:
            for site in cell["sites"]:
                if cell["kind"] == "noise":
                    kd = "white" if cell["cls"] == "white" else (
                        "common" if cell["cls"] == "common" else "band")
                    kw = dict(f_lo=0.3, f_hi=5.0) if kd == "band" else {}
                    Xp, _ = perturb_window(tu, Xu, kd, cell["amp"], seed * 1000 + 7, **kw)
                    tag = f"{cell['cls']}{cell['amp']:.0f}"
                else:
                    amp = cell["amp_pct"] / 100.0 * lvl
                    Xp, _ = perturb_window(tu, Xu, "tap", amp, seed * 1000 + 7, t_inj=site,
                                           rise_ms=max(10, cell["dur_ms"] // 4),
                                           hold_ms=max(10, cell["dur_ms"] // 2),
                                           fall_ms=max(10, cell["dur_ms"] // 4))
                    tag = f"tap{cell['amp_pct']}pct_{cell['dur_ms']}ms"
                r = run_full(tu, Xp, TraceV6)
                rows += candidates(tu, Xu, r["tr"], r["epoch"], gt_w, W["t0"], tag, None,
                                   seed, cell["win"])
        print(f"  [{cell['win']}] {cell['cls']} amp={cell.get('amp', cell.get('amp_pct'))} "
              f"累计候选={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(int)
    df.to_csv(os.path.join(RES, "t5b_criteria_features.csv"), index=False, encoding="utf-8-sig")
    print(f"\n候选总数={len(df)} 正例={int(df['label'].sum())} 负例={int((1-df['label']).sum())}")

    # 正例/负例按"候选时刻是否落在真实沿 ±1.5 s"划分；一个真实沿附近可能出现多个候选，
    # 这会让 AUC 略偏乐观（同一事件被重复计入），报告里已标注该口径。
    out_rows, summ = [], []
    for name, (feat, params, causal, pname) in CRIT.items():
        mode = "le" if name == "K5_actfrac" else "ge"
        r, auc = roc_from(df, feat, params, mode)
        r["criterion"] = name
        r["param_name"] = pname
        r["causality"] = causal
        r["auc"] = round(auc, 4)
        r["n_pos"] = int(df["label"].sum())
        r["n_neg"] = int((1 - df["label"]).sum())
        out_rows.append(r)
        rp = r[r["param"].apply(lambda z: isinstance(z, (int, float)) and np.isfinite(z))]
        best = {}
        for fpr_cap in (0.01, 0.05, 0.10, 0.20):
            sub = rp[rp["fpr"] <= fpr_cap]
            if len(sub):
                b2 = sub.loc[sub["tpr"].idxmax()]
                best[f"tpr@fpr<={fpr_cap}"] = round(float(b2["tpr"]), 3)
                best[f"param@fpr<={fpr_cap}"] = b2["param"]
                best[f"prec@fpr<={fpr_cap}"] = (round(float(b2["precision"]), 3)
                                                if b2["precision"] == b2["precision"] else np.nan)
            else:
                best[f"tpr@fpr<={fpr_cap}"] = np.nan
                best[f"param@fpr<={fpr_cap}"] = np.nan
                best[f"prec@fpr<={fpr_cap}"] = np.nan
        summ.append(dict(criterion=name, feature=feat, causality=causal, auc=round(auc, 4),
                         **best))
    # 组合判据：dwell × mono × spatial（全部因果）
    combos = []
    for tau in (0.03, 0.05, 0.10, 0.15):
        for mono in (0.0, 0.8, 0.95):
            for af in (1.01, 0.6, 0.4):
                acc = ((df["ratio"] > 1.0) & (df["dwell_s"] >= tau) &
                       (df["f_mono"] >= mono) & (df["active_frac"] <= af)).to_numpy()
                P, N = int(df["label"].sum()), int((1 - df["label"]).sum())
                tp = int((acc & (df["label"] == 1)).sum())
                fp = int((acc & (df["label"] == 0)).sum())
                combos.append(dict(criterion="K7_dwell_mono_spatial", tau=tau, mono=mono,
                                   active_frac_max=af, tp=tp, fp=fp,
                                   tpr=tp / P if P else np.nan, fpr=fp / N if N else np.nan,
                                   precision=(tp / (tp + fp)) if (tp + fp) else np.nan,
                                   causality="causal"))
    combos = pd.DataFrame(combos)
    # 非因果组合（上界参照）
    nc = []
    for tau in (0.03, 0.10, 0.20):
        for hold in (0.0, 0.4, 0.7):
            acc = ((df["ratio"] > 1.0) & (df["dwell_s"] >= tau) & (df["f_hold"] >= hold) &
                   (df["f_rev"] == 0)).to_numpy()
            P, N = int(df["label"].sum()), int((1 - df["label"]).sum())
            tp = int((acc & (df["label"] == 1)).sum())
            fp = int((acc & (df["label"] == 0)).sum())
            nc.append(dict(criterion="K9_dwell_hold_norev", tau=tau, f_hold_min=hold,
                           tp=tp, fp=fp, tpr=tp / P if P else np.nan,
                           fpr=fp / N if N else np.nan,
                           precision=(tp / (tp + fp)) if (tp + fp) else np.nan,
                           causality="非因果(+0.6s)"))
    combos = pd.concat([combos, pd.DataFrame(nc)], ignore_index=True)
    allroc = pd.concat(out_rows, ignore_index=True)
    allroc.to_csv(os.path.join(RES, "t5b_criteria_roc.csv"), index=False, encoding="utf-8-sig")
    combos.to_csv(os.path.join(RES, "t5b_criteria_combos.csv"), index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(RES, "t5b_criteria_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 单维判据 ===")
    print(sm.to_string())
    print("\n=== 组合判据（按 FPR 升序前 20）===")
    print(combos.sort_values("fpr").head(20).to_string())
    # 追加进 t5b_detector_roc.csv
    rocp = os.path.join(RES, "t5b_detector_roc.csv")
    if os.path.exists(rocp):
        old = pd.read_csv(rocp, encoding="utf-8-sig")
        old = old[old.get("family", "boundary") == "boundary"]
        add = allroc.copy()
        add["family"] = "criteria"
        new = pd.concat([old, add], ignore_index=True)
        new.to_csv(rocp, index=False, encoding="utf-8-sig")
        print(f"\n已把 {len(add)} 行 family=criteria 追加进 t5b_detector_roc.csv")


if __name__ == "__main__":
    main()
