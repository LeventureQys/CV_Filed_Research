# -*- coding: utf-8 -*-
"""paper_v6 公共层：数据清单、口径（指标定义）、缓存。

与 `paper/` （无责 3 s/5 s 路线）**互不影响**：本目录只写 v6 抗蠕变部分。
所有口径与 `scripts/ci_v6_vs_1s_3s.py` 逐条一致（同一时间轴 ad_lib.prep、
同一指标函数），使本文数字可与既有 I1/I2、H1/H2 报告直接对照。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # paper_v6/scripts
ROOT = os.path.dirname(HERE)                               # paper_v6
FLASH = os.path.dirname(os.path.dirname(ROOT))                              # temp/v4.1flash
TEMP = os.path.dirname(FLASH)                              # temp
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "docs", "figures")
CACHE = os.path.join(RES, "cache")
for _d in (RES, FIG, CACHE):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, HERE)
import ad_lib as L                                          # noqa: E402

# ── 数据清单（13 份：恒载 9 组 + 变载实录 4 份）────────────────────
B = os.path.join(TEMP, "变化负载")
HOLD = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
VARY = [
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                       "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv")),
]
ALL = HOLD + VARY
KIND = {t: ("恒载" if t in [a for a, _ in HOLD] else "实采") for t, _ in ALL}
MAIN_CH = {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}

# ── 算法臂（本次复算的全部对照）──────────────────────────────────
ARMS = ["raw", "e1s", "e3s", "v6", "v6trim"]
ARM_LABEL = {"raw": "原始（无补偿）", "e1s": "无责 1 s", "e3s": "无责 3 s（现役 v5.1）",
             "v6": "v6（pin，trim 关）", "v6trim": "v6 + A 慢修正（trim 2.5% 死区）"}
COL = {"raw": "0.62", "e1s": "#1f77b4", "e3s": "#ff7f0e", "v6": "#2ca02c", "v6trim": "#d62728"}
LS = {"raw": "-", "e1s": "--", "e3s": "-.", "v6": "-", "v6trim": "-"}


# ── 指标（逐条抄自 ci_v6_vs_1s_3s.py，保证与既有报告同口径）────────
def find_segment(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    dd = np.diff(ld.astype(int))
    s = list(np.where(dd == 1)[0] + 1)
    e = list(np.where(dd == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


def hold_metrics(Y, c, tu, Xu, dtm, tot_s, peak):
    s0r, s1r = find_segment(Xu.sum(axis=1))[0]
    s0 = int(s0r)
    s1 = min(int(s1r), len(tu) - 1)
    amp_v = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp_v))
    amp = amp_v[m]
    loaded = amp_v > 0.10 * amp_v.max()
    base = float(np.median(tot_s[max(0, s0 - int(2 / dtm)):s0]))
    n_on = next((i for i in range(s0, min(s0 + int(5 / dtm), s1))
                 if tot_s[i] > base + 0.05 * (tot_s[s0:s1].max() - base)), s0)
    nL = max(1, s1 - s0)
    seg = Y[s0:s1]
    by, bx = Y[:s0, m].mean(), Xu[:s0, m].mean()
    dr = seg[-nL // 10:].mean(axis=0) - seg[:nL // 10].mean(axis=0)
    a5 = max(s0, n_on + int(5.0 / dtm))
    n5 = max(1, s1 - a5)
    dr5 = Y[a5:s1][-n5 // 10:].mean(axis=0) - Y[a5:s1][:n5 // 10].mean(axis=0)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    sx = Xu[i1:i2, m].mean() - bx
    step = (Y[i1:i2, m].mean() - by) / sx if abs(sx) > 1e-9 else np.nan
    tq = tu[s0:s1] - tu[s0]

    def dstd(sig, t):
        k, b0 = np.polyfit(t, sig, 1)
        return (sig - (k * t + b0)).std()

    ny = dstd(seg[10:, m], tq[10:])
    nx = dstd(Xu[s0 + 10:s1, m], tq[10:])
    ded = (Xu - Y).sum(axis=1)
    hit = np.where(ded[s0:s1] > 0.005 * abs(tot_s[min(s1, len(tu) - 1)] - base))[0]
    return dict(drift_main=100 * dr[m] / amp, drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=ny / nx if nx > 1e-12 else np.nan, flat=100 * ny / amp,
                step_ratio=step, ded_delay=float(hit[0] * dtm) if len(hit) else np.nan,
                a_max=float(c.A.max()), g_end=float(c.g), epoch=len(c.epoch_t),
                max_gap=np.nan, pct=np.nan, cap_med=np.nan, cap_min=np.nan,
                n_event=0, gap_med=np.nan, gap_max=np.nan,
                s0=s0, s1=s1, main_ch=m)


def vary_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d):
    d["Ys"] = {"a": Y}
    d["periods"] = L.find_periods(Xu.sum(axis=1), dtm)
    d["events"] = [e for e, _ in L.detect_events(Xu.sum(axis=1), dtm)]
    y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    gap = np.abs(y - tot_s)
    ev = L.event_table(d, d["events"], {"a": Y}, [], gain_s=6.0, algos=["a"])
    ev["big"] = ev["jump"].abs() >= 2000.0
    thr = 0.30 * ev["pre"].max() if len(ev) else 0
    ev["mid"] = (ev["pre"] > thr) & (ev["post"] > thr) if len(ev) else False
    ml = ev[ev.mid & ev.big] if len(ev) else ev
    cap = (ml["gain_a"] / ml["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna() \
        if len(ml) else pd.Series(dtype=float)
    return dict(drift_main=np.nan, drift_slow=np.nan, drift_loaded=np.nan, noise_ratio=np.nan,
                flat=np.nan, step_ratio=np.nan, ded_delay=np.nan, a_max=float(c.A.max()),
                g_end=float(c.g), epoch=len(c.epoch_t), max_gap=float(gap.max()),
                pct=100 * float(gap.max()) / peak,
                cap_med=float(cap.median()) if len(cap) else np.nan,
                cap_min=float(cap.min()) if len(cap) else np.nan, n_event=int(len(ml)),
                gap_med=float(ml["gap_a"].median()) if len(ml) else np.nan,
                gap_max=float(ml["gap_a"].max()) if len(ml) else np.nan)


def first_onset(tu, tot, dtm):
    peak = float(np.percentile(tot, 99.5))
    idx = np.where(tot > 0.5 * peak)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    pre = float(np.median(tot[max(0, i - int(1.5 / dtm)):max(1, i - int(0.3 / dtm))]))
    j = i
    while j > 0 and tot[j] > pre + 0.05 * (tot[i] - pre):
        j -= 1
    return j + 1, pre


def settle_time(tu, Ytot, i0, target, step, dtm, hold_s=30.0):
    """T_band：进入 ±5%×阶跃带（相对真值）且此后 hold_s 内不再离开（混了"准"与"快"）。"""
    if step <= 0:
        return np.nan
    lo, hi = target - 0.05 * step, target + 0.05 * step
    ok = (Ytot >= lo) & (Ytot <= hi)
    n = len(Ytot)
    H = int(hold_s / dtm)
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            if ok[k:e].all():
                return float(tu[k] - tu[i0])
            break
        if ok[k:e].all():
            return float(tu[k] - tu[i0])
    return np.nan


def stable_time(tu, Ytot, i0, step, dtm, hold_s=30.0, tol_frac=0.05):
    """T_stable：显示首次"停下"（此后 hold_s 内相对该时刻自身的漂移 ≤tol×阶跃）。"""
    if step <= 0:
        return np.nan
    n = len(Ytot)
    H = int(hold_s / dtm)
    tol = tol_frac * step
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            break
        if np.max(np.abs(Ytot[k:e] - Ytot[k])) <= tol:
            return float(tu[k] - tu[i0])
    return np.nan


def cache_path(tag):
    return os.path.join(CACHE, tag.replace("/", "_") + ".npz")


def load_npz(tag):
    return np.load(cache_path(tag), allow_pickle=True)


def save_table(df, name):
    p = os.path.join(RES, name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  -> {os.path.relpath(p, FLASH)}")
    return p
