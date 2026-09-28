# -*- coding: utf-8 -*-
"""T5-B / 02：T5B-Q1（复现与失败率）+ T5B-Q3（扰动扫描与失效边界）+ Q4 候选判据特征。

产出
    results/t5b_baseline_failure.csv   每档失败率（Wilson 95% CI + 分量 M1~M4 + n）→ **Q1**
    results/t5b_trials_raw.csv         逐 trial 原始行（win×class×param×seed×site）
    results/t5b_detector_roc.csv       幅值扫描 + logistic A50（bootstrap CI）→ **Q3**
    results/t5b_event_detail.csv       逐事件 M1/M2 明细（带时间戳）
    results/t5b_candidates.csv         Q4 候选判据的逐候选特征 + 标签 → 供 t5b_05
    results/t5b_step_latency.csv       真实阶跃的建事件/交接延迟（Q5 的"代价"参照）
    results/_t5b_02.log

失效定义（**写死**，报告 §2 逐字复述）
    M1 事件未建立：|jump| ≥ max(5%·窗内电平, 300 ADC) 的真实沿，在 [t_on−0.60, t_on+1.50] 内
       没有任何 v6 epoch ⇒ 漏
    M2 基线锚点错误：检查点 t_ck = min(t_on+20 s, 窗末)（要求 ≥ t_on+8 s）处
       |ΣA_扰动 − ΣA_干净| / 电平 > **20%**（用户口径"Â 偏离真实最终电平"）
    M3 错误重捕获：与任何真实沿相距 >1.5 s 的 epoch，且未被撤销或存活 ≥0.50 s
    M4 显示越界（**算法可归因部分**）：max_t |[(Y−Z̄)_扰动 − (Y−Z̄)_干净]| / 电平 > **20%**
       （先各自扣掉输入自身的 0.65 s 中值电平，避免把"输入真的变了"算成算法失效）
    **trial 失败 = M1~M4 任一成立**；`step` 类只测延迟、不参与失效统计（它是控制组）。
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_core import (TraceV6, run_full, load_window, perturb_window,   # noqa: E402
                      gt_in_window, load_gt, med_smooth, WIN, DT, FAIL_DEF)

RNG_SEEDS = list(range(10))


# ───────────────────────── 统计工具 ─────────────────────────

def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def logit_fit(A, y):
    """logistic 拟合 p(A)。用 IRLS 解析求解（scipy 的 Nelder-Mead 在 100+ 组 × bootstrap 下太慢）。
    返回 (A50, beta, nll)。A 做标准化以改善条件数。"""
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    if len(np.unique(y)) < 2 or len(A) < 4:
        return (np.nan, np.nan, np.nan)
    mu, s = float(A.mean()), float(A.std())
    if not np.isfinite(s) or s < 1e-12:
        return (np.nan, np.nan, np.nan)
    z = (A - mu) / s
    X = np.column_stack([np.ones_like(z), z])
    b = np.zeros(2)
    for _ in range(60):
        eta = np.clip(X @ b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        zz = eta + (y - p) / w
        H = X.T @ (X * w[:, None])
        g = X.T @ (w * zz)
        try:
            bn = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return (np.nan, np.nan, np.nan)
        if not np.all(np.isfinite(bn)):
            return (np.nan, np.nan, np.nan)
        if np.max(np.abs(bn - b)) < 1e-10:
            b = bn
            break
        b = bn
    if abs(b[1]) < 1e-12:
        return (np.nan, float(b[1]), np.nan)
    a50_z = -b[0] / b[1]
    nll = -np.sum(y * np.clip(X @ b, -30, 30) - np.log1p(np.exp(np.clip(X @ b, -30, 30))))
    return (mu + s * a50_z, float(b[1] / s), float(nll))


def logit_boot(A, y, n_boot=150, seed=0):
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    out, n = [], len(A)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a50, _, _ = logit_fit(A[idx], y[idx])
        if np.isfinite(a50):
            out.append(a50)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out \
        else (np.nan, np.nan)


# ───────────────────────── 失效判定（本脚本版） ─────────────────────────
#
# 与 `t5b_core.FAIL_DEF` 的差异（三处修正，均在报告 §2 写明理由）：
#   1. M1 改为**增量口径**：只有"干净运行检出了、扰动运行没检出"才算漏
#      （否则 W3 里 v6 干净状态本来就漏的那个 13.7 s restep 会让所有档 100% 失败）；
#      配对用 `t0 ∈ [t_on−1.2, t_on+1.5]` **或** `t_det ∈ [t_on−0.10, t_on+2.0]`
#      （t0 是回溯真沿，本身有 ±0.7 s 抖动，只用 t0 会把"检出但回溯偏移"误判成漏）。
#   2. M2 的分母改用**事件自己的平台电平** `L_e = median Zs[t_on+2, t_on+8]`，
#      并加 300 ADC 绝对地板；原实现用检查点瞬时电平，在"检查点恰好落在空载段"时
#      会把 37 ADC 的锚点差算成 20.6%（分母 ≈180）。
#   3. M3 也改为增量口径：只统计"扰动运行有、干净运行（±1.0 s）没有"的存活伪 epoch。

def _matched(ep, t_on, lo=-1.2, hi=1.5):
    """epoch 是否对应真实沿：t0 窗口 或 t_det 窗口。"""
    return (lo <= (ep[0] - t_on) <= hi) or (-0.10 <= (ep[2] - t_on) <= 2.00)


def eval_trial2(tu, Xp, ref, gt_win, w_t0, fail_def=None, cls=TraceV6, **kw):
    fd = dict(FAIL_DEF, **(fail_def or {}))
    r = run_full(tu, Xp, cls, **kw)
    tr, trr = r["tr"], ref["tr"]
    Z = r["Z"]
    Zs = med_smooth(Z, 30)
    Zsc = med_smooth(ref["Z"], 30)
    n = min(len(Zs), len(Zsc))
    excess = (r["Ysum"][:n] - Zs[:n]) - (ref["Ysum"][:n] - Zsc[:n])
    rows, flags = [], dict(M1_miss=0, M2_wrong_anchor=0, M3_false_capture=0, M4_disp=0,
                           M1_miss_abs=0, M2_measured=0)
    gt_w = np.array([float(x) - w_t0 for x in gt_win["t_on"]], float) if len(gt_win) else np.array([])
    for _, g in gt_win.iterrows():
        t_on_w = float(g["t_on"]) - w_t0
        i_on = int(np.argmin(np.abs(tu - t_on_w)))
        lvl = float(np.median(Zs[max(0, i_on - 50):max(1, i_on)]))
        should = int(abs(float(g["jump"])) >= max(fd["rel_gate"] * abs(lvl), fd["abs_gate"]))
        m = [e for e in r["epoch"] if _matched(e, t_on_w)]
        mc = [e for e in ref["epoch"] if _matched(e, t_on_w)]
        miss_abs = int(should and len(m) == 0)
        miss = int(should and len(mc) > 0 and len(m) == 0)      # 增量口径
        flags["M1_miss"] += miss
        flags["M1_miss_abs"] += miss_abs
        # 事件平台电平（post 窗 2~8 s 中值），并设 500 ADC 下限
        i_a, i_b = int(round((t_on_w + 2) / DT)), int(round((t_on_w + 8) / DT))
        L_e = float(np.median(Zs[i_a:i_b])) if i_b > i_a and i_b <= len(Zs) else np.nan
        L_e = max(abs(L_e), 500.0) if L_e == L_e else 500.0
        i_ck = min(int(round((t_on_w + fd["ck_dt"]) / DT)), len(tr["sumA"]) - 1)
        meas = int((i_ck * DT - t_on_w) >= 8.0)
        dA = rel = np.nan
        bad = 0
        if meas:
            flags["M2_measured"] += 1
            dA = float(tr["sumA"][i_ck] - trr["sumA"][i_ck])
            rel = abs(dA) / L_e
            bad = int(abs(dA) > max(fd["thr_anchor"] * L_e, fd["abs_anchor"]))
            flags["M2_wrong_anchor"] += bad
        rows.append(dict(kind_gt=str(g["kind"]), t_on=round(float(g["t_on"]), 2),
                         t_on_w=round(t_on_w, 2), jump=round(float(g["jump"]), 1),
                         lvl_pre=round(lvl, 1), lvl_event=round(L_e, 1),
                         should_detect=should, n_match=len(m), n_match_clean=len(mc),
                         match_kinds="|".join(sorted({e[1] for e in m})), M1_miss=miss,
                         M1_miss_abs=miss_abs, ck_meas=meas,
                         dA_ck=(round(dA, 1) if dA == dA else np.nan),
                         dA_rel=(round(rel, 4) if rel == rel else np.nan), M2=bad))
    n_ep_clean = np.array([e[0] for e in ref["epoch"]], float)
    n_false = 0
    for e in r["epoch"]:
        t0e = e[0]
        if len(gt_w) and np.min(np.abs(gt_w - t0e)) <= 1.5:
            continue
        if len(n_ep_clean) and np.min(np.abs(n_ep_clean - t0e)) <= 1.0:
            continue                                            # 增量口径
        rev_later = [v[0] for v in r["revoke"] if v[0] > t0e]
        if (not rev_later) or (rev_later[0] - t0e) > fd["survive_s"]:
            n_false += 1
    flags["M3_false_capture"] = n_false
    lv_max = float(np.percentile(np.abs(Zs), 95))
    mdev = float(np.abs(excess).max())
    flags["M4_disp"] = int(mdev > fd["thr_disp"] * max(lv_max, 1.0))
    lc = (tr["sig_d"] * 5.0 > 0.05 * np.abs(tr["lv_ref"]))
    res = dict(flags=flags, n_epoch=len(r["epoch"]), n_revoke=len(r["revoke"]),
               n_handoff=len(r["handoff"]), n_epoch_clean=len(ref["epoch"]),
               excess_max=mdev, excess_rel=mdev / max(lv_max, 1.0), lv_max=lv_max,
               maxdev=float(np.abs(r["Ysum"][:n] - ref["Ysum"][:n]).max()),
               n_lowconf=int(lc.sum()), n_lowconf_frac=float(lc.mean()),
               dA_end=float(tr["sumA"][-1] - trr["sumA"][-1]),
               det_rows=rows, r=r, ref=ref)
    res["fail"] = int(any(flags[k] > 0 for k in
                          ("M1_miss", "M2_wrong_anchor", "M3_false_capture", "M4_disp")))
    res["fail_det"] = int(flags["M1_miss"] + flags["M3_false_capture"] > 0)
    res["fail_anchor"] = int(flags["M2_wrong_anchor"] > 0)
    return res


def step_latency(tu, Xp, t_inj, ref=None):
    """真实阶跃的建事件/交接延迟（Q5 的"代价"参照）。"""
    r = run_full(tu, Xp, TraceV6)
    out = dict(t_inj=t_inj, n_epoch=len(r["epoch"]), t_det=np.nan, t_ho=np.nan,
               dA_end=float(r["tr"]["sumA"][-1] - ref["tr"]["sumA"][-1]) if ref else np.nan)
    cand = [e for e in r["epoch"] if abs(e[0] - t_inj) <= 1.0 or 0 < (e[2] - t_inj) <= 3.0]
    if cand:
        e = min(cand, key=lambda z: abs(z[0] - t_inj))
        out["t_det"] = round(e[2] - t_inj, 3)
        out["t0_rel"] = round(e[0] - t_inj, 3)
        out["kind"] = e[1]
    ho = [h for h in r["handoff"] if h[0] - t_inj > 0]
    if ho:
        out["t_ho"] = round(ho[0][0] - t_inj, 3)
    return out


# ───────────────────────── 候选判据特征 ─────────────────────────

def candidates_from_run(tr, Xp, epochs, gt_w, cls_name, param, seed, wname, W):
    """候选 = `raw_hit` 连续段里首次满足「连续 ≥3 帧」的那一帧（= 现行判据的决策时刻）。"""
    rh = np.asarray(tr["raw_hit"], bool)
    hr = np.asarray(tr["hit_run"], int)
    d = np.asarray(tr["d"], float)
    ts = np.asarray(tr["ts"], float)
    idx = np.where(rh)[0]
    if len(idx) == 0:
        return []
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    gt_arr = np.array([t - W["t0"] for t in gt_w["t_on"]], float) if len(gt_w) else np.array([])
    Z = Xp.sum(axis=1)
    out = []
    for s in segs:
        cand = None
        for j in s:
            if hr[j] >= 3:
                cand = int(j)
                break
        if cand is None:
            continue
        t_c = float(ts[cand])
        dc = float(d[cand])
        gate = float(tr["gate"][cand])
        i0 = max(0, cand - 50)
        dd = d[i0:cand + 1]
        i_r0, i_r1 = max(0, cand - int(0.65 / DT)), max(0, cand - int(0.35 / DT))
        if i_r1 > i_r0:
            dvv = Xp[max(0, cand - int(0.20 / DT)):cand + 1].mean(axis=0) - \
                Xp[i_r0:i_r1 + 1].mean(axis=0)
        else:
            dvv = np.zeros(Xp.shape[1])
        ad = np.abs(dvv)
        mx = float(ad.max()) if len(ad) else 0.0
        act = ad > max(0.10 * mx, 1.0)
        p = ad / ad.sum() if ad.sum() > 1e-12 else np.zeros_like(ad)
        hhi = float((p ** 2).sum()) if ad.sum() > 1e-12 else np.nan
        ih = cand + int(0.50 / DT)
        hold, rev = np.nan, np.nan
        if ih < len(Z):
            zref = float(np.median(Z[i_r0:max(1, i_r1)]))
            hold = float(np.sign(dc) * (float(np.median(Z[ih - 5:ih + 6])) - zref) / max(abs(dc), 1e-9))
            iw = Z[cand:min(len(Z), cand + int(0.60 / DT))]
            if len(iw):
                rev = float(np.any(np.sign(iw - zref) * np.sign(dc) < 0))
        out.append(dict(win=wname, cls=cls_name, param=param, seed=int(seed), t_c=round(t_c, 3),
                        d=round(dc, 2), lv_ref=round(float(tr["lv_ref"][cand]), 1),
                        gate=round(gate, 1), ratio=round(abs(dc) / max(gate, 1e-9), 3),
                        sig_d=round(float(tr["sig_d"][cand]), 2),
                        state_pre=str(tr["state_pre"][cand]),
                        armed_pre=bool(tr["armed_pre"][cand]),
                        n_ch=int(Xp.shape[1]), n_active=int(act.sum()),
                        active_frac=round(float(act.sum()) / Xp.shape[1], 3),
                        coup=round(float(ad.sum() / mx), 2) if mx > 0 else np.nan,
                        hhi=round(hhi, 4) if hhi == hhi else np.nan,
                        f_mono=round(float(np.mean(np.sign(dd) == np.sign(dc))) if len(dd) else np.nan, 3),
                        f_hold=(round(hold, 3) if hold == hold else np.nan),
                        f_rev=(int(rev) if rev == rev else np.nan),
                        label=int(len(gt_arr) > 0 and np.min(np.abs(gt_arr - t_c)) <= 1.5),
                        is_epoch=int(any(abs(e[0] - t_c) <= 0.30 for e in epochs))))
    return out


# ───────────────────────── 扫描单元 ─────────────────────────

def build_cells():
    cells = []
    wins = ["W1_1w", "W2_3w", "W3_1p5w", "W4_1p9w"]
    for w in wins:
        for amp in (200, 500, 1000, 2000, 4000, 8000):
            cells.append(dict(win=w, cls="white", amp=amp, seeds=RNG_SEEDS, kind="noise"))
    for w in wins:
        for amp in (200, 500, 1000, 2000, 4000, 8000):
            cells.append(dict(win=w, cls="common", amp=amp, seeds=RNG_SEEDS, kind="noise"))
    for f_hi in (1.0, 2.0, 5.0, 15.0, 40.0):
        for amp in (500, 1000, 2000, 4000):
            cells.append(dict(win="W1_1w", cls=f"band{f_hi:g}", amp=amp, seeds=list(range(8)),
                              kind="noise", f_lo=0.3, f_hi=f_hi))
    for w in ("W2_3w", "W3_1p5w", "W4_1p9w"):
        for amp in (500, 1000, 2000, 4000):
            cells.append(dict(win=w, cls="band5", amp=amp, seeds=list(range(6)),
                              kind="noise", f_lo=0.3, f_hi=5.0))
    sites = [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0]
    for pct in (10, 20, 30, 50, 100):
        for dur in (20, 50, 100, 200, 500):
            cells.append(dict(win="W1_1w", cls="tap", amp_pct=pct, dur_ms=dur,
                              seeds=[0, 1], sites=sites, kind="tap"))
    for pct in (5, 20):
        cells.append(dict(win="W1_1w", cls="step", amp_pct=pct, dur_ms=50,
                          seeds=RNG_SEEDS, sites=[36.0], kind="step"))
    return cells


def pct_level(pct, lvl):
    return float(pct) / 100.0 * float(lvl)


def run_cells(cells):
    gt = load_gt()
    trials, details, cands, steps = [], [], [], []
    for wname in sorted({c["win"] for c in cells}):
        W = WIN[wname]
        tu, Xu, W = load_window(W)
        gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
        ref = run_full(tu, Xu, TraceV6)
        lvl = float(np.median(Xu.sum(axis=1)))
        print(f"\n### {wname} {W['rec']} 窗 [{W['t0']},{W['t1']}]s 帧={len(tu)} "
              f"电平中位={lvl:.0f} 干净 epoch={len(ref['epoch'])} 真值事件={len(gt_w)}",
              flush=True)
        for c in [c for c in cells if c["win"] == wname]:
            t_start, n0 = time.time(), len(trials)
            for seed in c["seeds"]:
                sites = c.get("sites", [None]) if c["kind"] in ("tap", "step") else [None]
                for site in sites:
                    if c["kind"] == "noise":
                        kd = "white" if c["cls"] == "white" else (
                            "common" if c["cls"] == "common" else "band")
                        Xp, meta = perturb_window(tu, Xu, kd, c["amp"], seed * 1000 + 7,
                                                  **{k: c[k] for k in ("f_lo", "f_hi") if k in c})
                    elif c["kind"] == "tap":
                        Xp, meta = perturb_window(
                            tu, Xu, "tap", pct_level(c["amp_pct"], lvl), seed * 1000 + 7,
                            t_inj=site, rise_ms=max(10, c["dur_ms"] // 4),
                            hold_ms=max(10, c["dur_ms"] // 2), fall_ms=max(10, c["dur_ms"] // 4))
                    else:
                        Xp, meta = perturb_window(tu, Xu, "step",
                                                  pct_level(c["amp_pct"], lvl), seed * 1000 + 7,
                                                  t_inj=site)
                    if c["kind"] == "step":
                        sl = step_latency(tu, Xp, site, ref)
                        steps.append(dict(win=wname, amp_pct=c["amp_pct"], level=lvl,
                                          amp_adc=pct_level(c["amp_pct"], lvl),
                                          seed=seed, site=site, **sl))
                        continue
                    res = eval_trial2(tu, Xp, ref, gt_w, W["t0"])
                    trials.append(dict(win=wname, rec=W["rec"], level=lvl, cls=c["cls"],
                                       kind=c["kind"], amp=c.get("amp", np.nan),
                                       amp_pct=c.get("amp_pct", np.nan),
                                       dur_ms=c.get("dur_ms", np.nan),
                                       f_lo=c.get("f_lo", np.nan), f_hi=c.get("f_hi", np.nan),
                                       seed=seed, site=site, fail=res["fail"],
                                       fail_det=res["fail_det"], fail_anchor=res["fail_anchor"],
                                       **res["flags"], n_epoch=res["n_epoch"],
                                       n_epoch_clean=res["n_epoch_clean"],
                                       n_revoke=res["n_revoke"], n_handoff=res["n_handoff"],
                                       excess_rel=round(res["excess_rel"], 4),
                                       maxdev=round(res["maxdev"], 1),
                                       dA_end=round(res["dA_end"], 1),
                                       lowconf_frac=round(res["n_lowconf_frac"], 4)))
                    for dr in res["det_rows"]:
                        details.append(dict(win=wname, cls=c["cls"], amp=c.get("amp", np.nan),
                                            amp_pct=c.get("amp_pct", np.nan),
                                            dur_ms=c.get("dur_ms", np.nan), seed=seed, site=site,
                                            **dr))
                    cands += candidates_from_run(res["r"]["tr"], Xp, res["r"]["epoch"],
                                                 gt_w, c["cls"], c.get("amp", c.get("amp_pct")),
                                                 seed, wname, W)
            sub = trials[n0:]
            print(f"  [{wname}] {c['cls']} amp={c.get('amp', c.get('amp_pct'))} "
                  f"dur={c.get('dur_ms', '-')} n={len(sub)} 失败={sum(r['fail'] for r in sub)} "
                  f"耗时={time.time()-t_start:.1f}s", flush=True)
    return (pd.DataFrame(trials), pd.DataFrame(details), pd.DataFrame(cands), pd.DataFrame(steps))


def aggregate(tr):
    keys = ["win", "cls", "kind", "amp", "amp_pct", "dur_ms"]
    rows = []
    for k, g in tr.groupby(keys, dropna=False):
        n = len(g)
        kf = int(g["fail"].sum())
        lo, hi = wilson(kf, n)
        row = dict(zip(keys, k), n=n, n_fail=kf, p_fail=round(kf / n, 4),
                    ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                    p_fail_anchor=round(g["fail_anchor"].mean(), 4),
                    p_fail_det=round(g["fail_det"].mean(), 4),
                    p_M1=round(g["M1_miss"].gt(0).mean(), 4),
                    p_M2=round(g["M2_wrong_anchor"].gt(0).mean(), 4),
                    p_M3=round(g["M3_false_capture"].gt(0).mean(), 4),
                    p_M4=round(g["M4_disp"].gt(0).mean(), 4),
                    p_M1_abs=round(g["M1_miss_abs"].gt(0).mean(), 4),
                    m2_measured_mean=round(float(g["M2_measured"].mean()), 2),
                    excess_rel_med=round(float(g["excess_rel"].median()), 4),
                    excess_rel_p90=round(float(g["excess_rel"].quantile(0.9)), 4),
                    n_epoch_med=float(g["n_epoch"].median()),
                    lowconf_frac_med=round(float(g["lowconf_frac"].median()), 4))
        lvlin = float(g["level"].median())
        if np.isfinite(row["amp"]) and row["amp"] > 0:
            row["amp_pct_level"] = round(row["amp"] / lvlin, 4)
        if np.isfinite(row["amp_pct"]):
            row["amp_adc"] = round(row["amp_pct"] / 100.0 * lvlin, 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    fits = []
    for (w, c, kd), g in tr.groupby(["win", "cls", "kind"]):
        x = g["amp"].to_numpy(float) if kd == "noise" else g["amp_pct"].to_numpy(float)
        for comp in ("fail", "fail_anchor", "fail_det", "M1_miss", "M2_wrong_anchor",
                     "M3_false_capture", "M4_disp"):
            yv = g[comp].to_numpy(float)
            if comp != "fail":
                yv = (yv > 0).astype(float)
            if len(np.unique(yv)) < 2:
                fits.append(dict(win=w, cls=c, kind=kd, component=comp, n=len(g), A50=np.nan,
                                 A50_ci_lo=np.nan, A50_ci_hi=np.nan, p_at_min=np.nan))
                continue
            a50, b, _ = logit_fit(x, yv)
            blo, bhi = logit_boot(x, yv, n_boot=300, seed=11)
            fits.append(dict(win=w, cls=c, kind=kd, component=comp, n=len(g),
                             A50=round(a50, 2) if np.isfinite(a50) else np.nan,
                             A50_ci_lo=round(blo, 2), A50_ci_hi=round(bhi, 2),
                             p_at_min=round(float(yv[np.argmin(x)]), 3)))
    return df, pd.DataFrame(fits)


def merge_parts():
    """把 w1 / rest 两个分片的产物合并成正式产物。"""
    import glob
    for base in ("t5b_trials_raw", "t5b_event_detail", "t5b_candidates", "t5b_step_latency"):
        parts = []
        for suf in ("_w1", "_rest"):
            p = os.path.join(RES, f"{base}{suf}.csv")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    parts.append(pd.read_csv(p, encoding="utf-8-sig"))
                except pd.errors.EmptyDataError:
                    continue
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(RES, f"{base}.csv"), index=False, encoding="utf-8-sig")
        print(f"[merge] {base}: {len(df)} 行")
    tr = pd.read_csv(os.path.join(RES, "t5b_trials_raw.csv"), encoding="utf-8-sig")
    agg, fits = aggregate(tr)
    agg.to_csv(os.path.join(RES, "t5b_baseline_failure.csv"), index=False, encoding="utf-8-sig")
    roc = agg.copy()
    roc["family"] = "boundary"
    roc = roc.merge(fits[fits["component"] == "fail"].drop(columns=["component"]),
                    on=["win", "cls", "kind", "n"], how="left")
    roc.to_csv(os.path.join(RES, "t5b_detector_roc.csv"), index=False, encoding="utf-8-sig")
    fits.to_csv(os.path.join(RES, "t5b_boundary_a50.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 全档失败率 ===")
    print(agg.to_string())
    print("\n=== 50% 失效边界（logistic A50 + bootstrap 95% CI）===")
    print(fits.to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "merge":
        merge_parts()
        sys.exit(0)
    t0 = time.time()
    cells = build_cells()
    if mode == "w1":
        cells = [c for c in cells if c["win"] == "W1_1w"]
    elif mode == "rest":
        cells = [c for c in cells if c["win"] != "W1_1w"]
    suf = "" if mode == "all" else "_" + mode
    n_trial = sum(len(c["seeds"]) * len(c.get("sites", [None])) for c in cells)
    print(f"[计划] mode={mode} 单元格 {len(cells)}；trial {n_trial}", flush=True)
    tr, det, cand, steps = run_cells(cells)
    tr.to_csv(os.path.join(RES, f"t5b_trials_raw{suf}.csv"), index=False, encoding="utf-8-sig")
    det.to_csv(os.path.join(RES, f"t5b_event_detail{suf}.csv"), index=False, encoding="utf-8-sig")
    cand.to_csv(os.path.join(RES, f"t5b_candidates{suf}.csv"), index=False, encoding="utf-8-sig")
    steps.to_csv(os.path.join(RES, f"t5b_step_latency{suf}.csv"), index=False, encoding="utf-8-sig")
    print(f"\n分片完成 {mode}：trial={len(tr)} 候选={len(cand)} 耗时 {time.time()-t0:.1f}s")
