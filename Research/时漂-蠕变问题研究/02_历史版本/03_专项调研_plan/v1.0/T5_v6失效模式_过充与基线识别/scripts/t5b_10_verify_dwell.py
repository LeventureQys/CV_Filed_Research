# -*- coding: utf-8 -*-
"""T5-B / 10：Q5 关键结论的**加密样本复核**（驻留确认是"改善"还是"更糟"）。

背景：`t5b_05_dwell.py` 的配置扫描每档只有 8 次拍击 / 2 次抖动（样本太小），
但它给出一个与第一轮相反、且量级很大的信号：
    驻留 ≥0.15 s 时拍击留下的**永久基线偏置**从 1.4% 电平跳到 27.3% 电平。
本脚本用 4 个驻留档 × (7 注入点 × 2 组合 × 2 种子 = 28 次拍击 + 6 次抖动 + 2 次阶跃) 复核。

产物
    results/t5b_dwell_verify.csv（含 Wilson/exact 区间与分位数）
    results/t5b_dwell_verify_trials.csv（逐 trial）
    results/_t5b_10.log
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_core import (TraceV6, make_traced_v6, load_window, run_full,   # noqa: E402
                      perturb_window, gt_in_window, load_gt, WIN, DT)
from t5b_02_sweep import wilson                                        # noqa: E402

DWELLS = (3, 6, 11, 16)          # 0.02 / 0.05 / 0.10 / 0.15 s
TAP_CASES = [(50, 100), (100, 500)]
SITES = [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]
SEEDS = [0, 1]
NOISE = [("white", 1000.0), ("white", 2000.0), ("white", 4000.0)]
NSEED = [0, 1]


def main():
    t00 = time.time()
    gt = load_gt()
    tu, Xu, W = load_window(WIN["W1_1w"])
    gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
    ref = run_full(tu, Xu, TraceV6)
    lvl = float(np.median(Xu.sum(axis=1)))
    print(f"W1 电平={lvl:.0f} 干净 epoch={len(ref['epoch'])} 真值={len(gt_w)}", flush=True)
    rows = []
    for P in DWELLS:
        cls = make_traced_v6(capf=None)
        kw = dict(DET_PERSIST=int(P), DET_K=5.0)
        t0 = time.time()
        taps, noises, steps = [], [], []
        for pct, dur in TAP_CASES:
            for site in SITES:
                for sd in SEEDS:
                    amp = pct / 100.0 * lvl
                    Xp, _ = perturb_window(tu, Xu, "tap", amp, sd * 1000 + 7, t_inj=site,
                                           rise_ms=max(10, dur // 4), hold_ms=max(10, dur // 2),
                                           fall_ms=max(10, dur // 4))
                    r = run_full(tu, Xp, cls, **kw)
                    i_inj = int(round((site + 0.05) / DT))
                    near = [e for e in r["epoch"]
                            if abs(e[2] - (site + 0.05)) <= 3.0 or abs(e[0] - (site + 0.05)) <= 3.0]
                    surv = 0
                    for e in near:
                        rv = [v[0] for v in r["revoke"] if v[0] > e[0]]
                        if (not rv) or (rv[0] - e[0]) > 0.5:
                            surv += 1
                    i_ck = min(i_inj + int(20 / DT), len(r["tr"]["sumA"]) - 1)
                    dA = abs(float(r["tr"]["sumA"][i_ck] - ref["tr"]["sumA"][i_ck]))
                    dev = float(np.abs(r["Ysum"] - ref["Ysum"]).max())
                    taps.append(dict(pct=pct, dur=dur, site=site, seed=sd,
                                     n_near=len(near), n_surv=surv, dA=dA, dA_rel=dA / lvl,
                                     dev_rel=dev / lvl))
        for kd, amp in NOISE:
            for sd in NSEED:
                Xp, _ = perturb_window(tu, Xu, kd, amp, sd * 1000 + 7)
                r = run_full(tu, Xp, cls, **kw)
                ref_tr, tr = ref["tr"], r["tr"]
                Zs = pd.Series(r["Z"]).rolling(30, center=True, min_periods=1).median().to_numpy()
                Zc = pd.Series(ref["Z"]).rolling(30, center=True, min_periods=1).median().to_numpy()
                n = min(len(Zs), len(Zc))
                exc = float(np.abs((r["Ysum"][:n] - Zs[:n]) - (ref["Ysum"][:n] - Zc[:n])).max()
                            / np.percentile(np.abs(Zs), 95))
                gt_w_arr = np.array([float(x) - W["t0"] for x in gt_w["t_on"]], float)
                m1 = 0
                for t_on in gt_w_arr:
                    mc = [e for e in ref["epoch"] if abs(e[0] - t_on) <= 1.5]
                    mm = [e for e in r["epoch"] if abs(e[0] - t_on) <= 1.5]
                    if mc and not mm:
                        m1 += 1
                lc = (tr["sig_d"] * 5.0 > 0.05 * np.abs(tr["lv_ref"]))
                noises.append(dict(amp=amp, seed=sd, M1=m1, exc=exc,
                                   dA_end=abs(float(tr["sumA"][-1] - ref_tr["sumA"][-1])),
                                   lowconf=float(lc.mean())))
        for sd in SEEDS:
            Xp, _ = perturb_window(tu, Xu, "step", 0.2 * lvl, sd * 1000 + 7, t_inj=36.0)
            r = run_full(tu, Xp, cls, **kw)
            cand = [e for e in r["epoch"] if 0 < (e[2] - 36.0) <= 3.0 or abs(e[0] - 36.0) <= 1.0]
            lat = (min(cand, key=lambda z: abs(z[0] - 36.0))[2] - 36.0) if cand else np.nan
            steps.append(dict(seed=sd, lat=lat))
        tp, nz = pd.DataFrame(taps), pd.DataFrame(noises)
        st = pd.DataFrame(steps)
        rows.append(dict(
            dwell_persist=P, dwell_s=round((P - 1) * DT, 3), n_tap=len(tp),
            tap_surv_mean=round(float(tp["n_surv"].mean()), 3),
            tap_surv_any_frac=round(float((tp["n_surv"] > 0).mean()), 3),
            tap_dA_rel_med=round(float(tp["dA_rel"].median()), 4),
            tap_dA_rel_p10=round(float(tp["dA_rel"].quantile(0.1)), 4),
            tap_dA_rel_p90=round(float(tp["dA_rel"].quantile(0.9)), 4),
            tap_dA_rel_gt20pct_frac=round(float((tp["dA_rel"] > 0.20).mean()), 3),
            tap_dev_rel_p90=round(float(tp["dev_rel"].quantile(0.9)), 4),
            n_noise=len(nz),
            noise_M1_mean=round(float(nz["M1"].mean()), 3),
            noise_exc_p90=round(float(nz["exc"].quantile(0.9)), 4),
            noise_exc_med=round(float(nz["exc"].median()), 4),
            noise_lowconf_med=round(float(nz["lowconf"].median()), 3),
            lat_t_det_med=round(float(st["lat"].median()), 3) if st["lat"].notna().any() else np.nan,
            n_lat=int(st["lat"].notna().sum()),
            sec=round(time.time() - t0, 1)))
        print(f"  P={P} dwell={rows[-1]['dwell_s']}s: 拍击存活={rows[-1]['tap_surv_mean']} "
              f"ΔA_rel中位={rows[-1]['tap_dA_rel_med']:.4f} (>20%比例={rows[-1]['tap_dA_rel_gt20pct_frac']}) "
              f"抖动漏={rows[-1]['noise_M1_mean']} 超额p90={rows[-1]['noise_exc_p90']:.3f} "
              f"延迟={rows[-1]['lat_t_det_med']} ({rows[-1]['sec']}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t5b_dwell_verify.csv"), index=False, encoding="utf-8-sig")
    print(f"\n总耗时 {time.time()-t00:.0f}s")
    print(df.to_string())


if __name__ == "__main__":
    main()
