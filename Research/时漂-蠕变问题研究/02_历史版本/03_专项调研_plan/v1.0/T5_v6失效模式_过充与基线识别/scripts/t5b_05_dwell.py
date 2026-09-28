# -*- coding: utf-8 -*-
"""T5-B / 05：T5B-Q5 误触发率 ↔ 检测延迟的权衡（细粒度系统扫描 + Pareto）

第一轮只做了**单点**（驻留 0.15/0.25/0.40 s 三档、固定 DET_K=5、无封顶）。本脚本做**三维系统扫描**：
    驻留确认 `DET_PERSIST`（= 连续帧数，dwell = (P−1)·10 ms）
    σ 门限系数 `DET_K`
    封顶策略 `CAPF`（5σ_d 用 CAPF × 电平项封顶；None = 不封顶 = 原型）
全部通过**参数化/属性补丁**实现，默认取值下与原型逐帧零差（见 `t5b_patch_ab_zero.csv`）。

产物
    results/t5b_dwell_sweep.csv       阶段 A：驻留细扫（11 档）
    results/t5b_config_sweep.csv      阶段 B/C：K × dwell、CAPF × dwell
    results/t5b_detector_pareto.csv   全部配置的 Pareto 表（含非支配标记）
    results/_t5b_05.log
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
                      perturb_window, gt_in_window, load_gt, med_smooth, WIN, DT)
from t5b_02_sweep import eval_trial2, step_latency                      # noqa: E402

WN = "W1_1w"
# 每个配置只跑 14 个**同一批** trial（配对比较）：拍击 2 组合 × 4 点 + 抖动 4 + 阶跃 2。
# 样本量在多配置间一致，故 Pareto 的相对关系可靠；绝对样本量偏小已在报告里标注。
TAP_CASES = [(50, 100), (100, 500)]
SITES = [6.0, 18.0, 30.0, 42.0]
NOISE = [("white", 1000.0, [0]), ("white", 2000.0, [0])]
STEP = [(20, [0, 1])]


def dwell_s(p):
    return round((int(p) - 1) * DT, 3)


def run_config(tu, Xu, W, gt_w, ref, lvl, dwell_p, k, capf):
    cls = make_traced_v6(capf=capf)
    kw = dict(DET_PERSIST=int(dwell_p), DET_K=float(k))
    out = dict(dwell_persist=int(dwell_p), dwell_s=dwell_s(dwell_p), DET_K=float(k),
               CAPF=("None" if capf is None else float(capf)))
    # —— 拍击 ——
    tap_false, tap_false_surv, tap_dA, tap_exc = [], [], [], []
    for pct, dur in TAP_CASES:
        for site in SITES:
            amp = pct / 100.0 * lvl
            Xp, _ = perturb_window(tu, Xu, "tap", amp, 7, t_inj=site,
                                   rise_ms=max(10, dur // 4), hold_ms=max(10, dur // 2),
                                   fall_ms=max(10, dur // 4))
            res = eval_trial2(tu, Xp, ref, gt_w, W["t0"], cls=cls, **kw)
            eps = res["r"]["epoch"]
            i_inj = int(round((site + 0.05) / DT))
            near = [e for e in eps if abs(e[2] - (site + 0.05)) <= 3.0 or abs(e[0] - (site + 0.05)) <= 3.0]
            surv = 0
            for e in near:
                rv = [v[0] for v in res["r"]["revoke"] if v[0] > e[0]]
                if (not rv) or (rv[0] - e[0]) > 0.5:
                    surv += 1
            tap_false.append(len(near))
            tap_false_surv.append(surv)
            i_ck = min(i_inj + int(20 / DT), len(res["r"]["tr"]["sumA"]) - 1)
            tap_dA.append(abs(float(res["r"]["tr"]["sumA"][i_ck] - ref["tr"]["sumA"][i_ck])))
            tap_exc.append(res["excess_rel"])
    out.update(tap_false_mean=round(float(np.mean(tap_false)), 3),
               tap_false_max=int(np.max(tap_false)),
               tap_surv_mean=round(float(np.mean(tap_false_surv)), 3),
               tap_surv_any=int(np.sum(np.array(tap_false_surv) > 0)),
               tap_dA_med=round(float(np.median(tap_dA)), 1),
               tap_dA_rel_med=round(float(np.median(tap_dA)) / max(lvl, 1.0), 4),
               tap_excess_p90=round(float(np.percentile(tap_exc, 90)), 4))
    # —— 抖动 ——
    nz = []
    for kd, amp, seeds in NOISE:
        for sd in seeds:
            Xp, _ = perturb_window(tu, Xu, kd, amp, sd * 1000 + 7)
            res = eval_trial2(tu, Xp, ref, gt_w, W["t0"], cls=cls, **kw)
            nz.append(dict(amp=amp, M1=res["flags"]["M1_miss"], M3=res["flags"]["M3_false_capture"],
                           M2=res["flags"]["M2_wrong_anchor"], exc=res["excess_rel"],
                           fail=res["fail"], dA_end=abs(res["dA_end"]),
                           lo=res["n_lowconf_frac"]))
    nzd = pd.DataFrame(nz)
    out.update(noise_M1_mean=round(float(nzd["M1"].mean()), 3),
               noise_M2_mean=round(float(nzd["M2"].mean()), 3),
               noise_M3_mean=round(float(nzd["M3"].mean()), 3),
               noise_fail_rate=round(float(nzd["fail"].mean()), 3),
               noise_exc_p90=round(float(nzd["exc"].quantile(0.9)), 4),
               noise_exc_med=round(float(nzd["exc"].median()), 4))
    hi = nzd[nzd["amp"] == 2000.0]
    out.update(noise2000_exc_p90=round(float(hi["exc"].quantile(0.9)), 4) if len(hi) else np.nan,
               noise2000_fail=round(float(hi["fail"].mean()), 3) if len(hi) else np.nan)
    # —— 真实阶跃延迟 ——
    lat, t0r = [], []
    for pct, seeds in STEP:
        for sd in seeds:
            amp = pct / 100.0 * lvl
            Xp, _ = perturb_window(tu, Xu, "step", amp, sd * 1000 + 7, t_inj=36.0)
            rr = run_full(tu, Xp, cls, **kw)
            cand = [e for e in rr["epoch"] if 0 < (e[2] - 36.0) <= 3.0 or abs(e[0] - 36.0) <= 1.0]
            if cand:
                e = min(cand, key=lambda z: abs(z[0] - 36.0))
                lat.append(e[2] - 36.0)
                t0r.append(e[0] - 36.0)
    out.update(lat_t_det_med=round(float(np.median(lat)), 3) if lat else np.nan,
               lat_t_det_max=round(float(np.max(lat)), 3) if lat else np.nan,
               lat_t0_med=round(float(np.median(t0r)), 3) if t0r else np.nan,
               n_lat=len(lat))
    return out


def pareto_flags(df, xcol, ycol):
    """y 越小越好、x 越小越好 ⇒ 非支配标记。"""
    xs = df[xcol].to_numpy(float)
    ys = df[ycol].to_numpy(float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    flag = np.zeros(len(df), bool)
    for i in range(len(df)):
        if not ok[i]:
            continue
        dom = ((xs <= xs[i]) & (ys <= ys[i]) & ((xs < xs[i]) | (ys < ys[i])) & ok)
        flag[i] = not dom.any()
    return flag


def main():
    t00 = time.time()
    gt = load_gt()
    tu, Xu, W = load_window(WIN[WN])
    gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
    ref = run_full(tu, Xu, TraceV6)
    lvl = float(np.median(Xu.sum(axis=1)))
    print(f"[{WN}] 帧={len(tu)} 电平={lvl:.0f} 干净 epoch={len(ref['epoch'])} "
          f"真值事件={len(gt_w)}", flush=True)

    rowsA = []
    print("\n=== 阶段 A：驻留细扫（DET_K=5, CAPF=None）===", flush=True)
    for p in (3, 4, 6, 8, 11, 13, 16, 21, 26, 31, 41, 51):
        t0 = time.time()
        r = run_config(tu, Xu, W, gt_w, ref, lvl, p, 5.0, None)
        rowsA.append(r)
        print(f"  dwell={r['dwell_s']:.2f}s(P={p}) 阶跃延迟={r['lat_t_det_med']} "
              f"拍击存活伪epoch均值={r['tap_surv_mean']} 抖动漏={r['noise_M1_mean']} "
              f"抖动过冲p90={r['noise_exc_p90']} ({time.time()-t0:.0f}s)", flush=True)
    dA = pd.DataFrame(rowsA)
    dA.to_csv(os.path.join(RES, "t5b_dwell_sweep.csv"), index=False, encoding="utf-8-sig")

    rowsB = []
    print("\n=== 阶段 B：DET_K × 驻留 ===", flush=True)
    for k in (3.0, 5.0, 8.0):
        for p in (3, 11, 21):
            t0 = time.time()
            r = run_config(tu, Xu, W, gt_w, ref, lvl, p, k, None)
            rowsB.append(r)
            print(f"  K={k} dwell={r['dwell_s']:.2f} 延迟={r['lat_t_det_med']} "
                  f"拍击存活={r['tap_surv_mean']} 抖动漏={r['noise_M1_mean']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("\n=== 阶段 C：CAPF × 驻留 ===", flush=True)
    for capf in (None, 1.0):
        for p in (3, 11, 21):
            t0 = time.time()
            r = run_config(tu, Xu, W, gt_w, ref, lvl, p, 5.0, capf)
            rowsB.append(r)
            print(f"  CAPF={capf} dwell={r['dwell_s']:.2f} 延迟={r['lat_t_det_med']} "
                  f"拍击存活={r['tap_surv_mean']} 抖动漏={r['noise_M1_mean']} "
                  f"抖动误触发={r['noise_M3_mean']} ({time.time()-t0:.0f}s)", flush=True)
    dfB = pd.DataFrame(rowsB)
    dfB.to_csv(os.path.join(RES, "t5b_config_sweep.csv"), index=False, encoding="utf-8-sig")

    allc = pd.concat([dA, dfB], ignore_index=True)
    allc["is_pareto_lat_tap"] = pareto_flags(allc, "lat_t_det_med", "tap_surv_mean")
    allc["is_pareto_lat_noise"] = pareto_flags(allc, "lat_t_det_med", "noise_exc_p90")
    allc["is_pareto_lat_fail"] = pareto_flags(allc, "lat_t_det_med", "noise_fail_rate")
    allc.to_csv(os.path.join(RES, "t5b_detector_pareto.csv"), index=False, encoding="utf-8-sig")
    print(f"\n总耗时 {time.time()-t00:.0f}s")
    print("\n=== 驻留细扫表 ===")
    print(dA.to_string())
    print("\n=== 全部配置的 Pareto 非支配点（lat × tap 存活伪 epoch）===")
    print(allc[allc["is_pareto_lat_tap"]][
        ["dwell_s", "DET_K", "CAPF", "lat_t_det_med", "tap_surv_mean", "tap_false_mean",
         "noise_M1_mean", "noise_exc_p90", "noise_fail_rate"]].to_string())


if __name__ == "__main__":
    main()
