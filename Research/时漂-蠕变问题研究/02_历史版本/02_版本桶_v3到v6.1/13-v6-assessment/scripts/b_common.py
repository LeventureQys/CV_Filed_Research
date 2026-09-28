# -*- coding: utf-8 -*-
"""13-v6-assessment / b_* 离线分析公共层（只读复用既有产物，只写本目录 b_*）。

复用来源（算法臂配置与 11-paper-v6/results/metrics_all.csv 完全一致）：
  temp/v4.1flash/progress/11-paper-v6/scripts/{ad_lib,pv_common,pv_run,glm53_v51,glm53_v6}.py
  temp/v4.1flash/progress/11-paper-v6/results/cache/*.npz   逐帧曲线（25 Hz，逐通道原始输出）
  temp/v4.1flash/progress/13-v6-assessment/results/{phase_map,phases}.csv

口径约定（全脚本统一）：
  t_dn  卸载沿时刻（0.5 s 中值平滑后的原始总量下穿「空载底 + 50%·本段幅度」的最后一帧）
  pre   加载前空载电平（卸载前邻接空载窗内「低于 底+10%·本段幅度」帧的中值）
  post  卸载后稳态电平（t_dn+8~t_dn+18 s 中值）
  plat  卸载前负载电平（t_dn−5~t_dn−0.5 s 中值）；step 为加载沿 +0.2 s 的台阶代理
  下冲  dip_vs_post = post − min(显示[t_dn, t_dn+8])   （低于「卸载后稳态」的深度）
        dip_vs_pre  = pre  − min(显示[t_dn, t_dn+8])   （低于「加载前空载电平」的深度）
  恢复  t_rec_in / t_rec_keep：显示回到 post ±5%·drop 内（后者要求此后 5 s 不再离开）
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                                   # 13-v6-assessment
FLASH = os.path.dirname(os.path.dirname(OUT))                 # temp/v4.1flash
TEMP = os.path.dirname(FLASH)                                 # temp
PROG = os.path.join(FLASH, "progress")
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
DOC = os.path.join(OUT, "docs")
PV = os.path.join(PROG, "11-paper-v6")
PAPER_RES = os.path.join(PV, "results")
CACHE = os.path.join(PAPER_RES, "cache")
for _d in (RES, FIG, DOC):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, os.path.join(PV, "scripts"))
import ad_lib as L                                            # noqa: E402
import pv_common as C                                         # noqa: E402
import pv_run as PR                                           # noqa: E402
import glm53_v6 as V6M                                        # noqa: E402

ALL = C.ALL
KIND = C.KIND
ARMS = ("raw", "e3s", "v6", "v6trim")
ARM_LABEL = {"raw": "raw（原始读数）", "e3s": "v5.1（免责 3 s，现役）",
             "v6": "v6（pin，trim 关）", "v6trim": "v6 + trim（2.5% 死区）"}
COL = {"raw": "0.45", "e3s": "#ff7f0e", "v6": "#2ca02c", "v6trim": "#d62728"}
ROM_TAU = V6M.ROM_TAU
ROM_G = V6M.ROM_G


def log_reconfigure():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                                       # noqa: BLE001
        pass


def cache_path(tag):
    return os.path.join(CACHE, tag.replace("/", "_") + ".npz")


def load_csv(tag, path):
    d = L.prep(path)
    d.update(tag=tag, kind=KIND[tag], path=path)
    return d


def med(x, dt, sec):
    return L.med_smooth(x, max(1, int(round(sec / max(dt, 1e-9)))))


def run_arm(arm, tu, Xu):
    """与 pv_run.run_arm 等价，但不覆盖 c.A（保留交接时的逐通道 A 供诊断）。"""
    if arm == "raw":
        return Xu.copy(), None
    c = PR.build(arm, Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


# ────────────────────────── 受载段 / 卸载沿定位 ──────────────────────────
def _hysteresis(ts, dt, idle, amp, hi_frac, lo_frac, min_len_s):
    hi, lo = idle + hi_frac * amp, idle + lo_frac * amp
    segs = []
    state = bool(ts[0] > hi)
    rise = 0 if state else -1
    for i in range(1, len(ts)):
        if not state and ts[i] > hi:
            state, rise = True, i
        elif state and ts[i] < lo:
            state = False
            if (i - rise) * dt >= min_len_s:
                segs.append((rise, i))
    if state and (len(ts) - 1 - rise) * dt >= min_len_s:
        segs.append((rise, len(ts) - 1))
    return segs


def load_segments(tot, dt, smooth_s=0.5):
    """多尺度迟滞：4 档阈值取并集，保证小负载段（实录里 ~10% 峰值的台阶）也被捕获。"""
    ts = med(tot, dt, smooth_s)
    idle = float(np.percentile(ts, 5))
    peak = float(np.percentile(ts, 99.5))
    amp = peak - idle
    if amp <= 1e-9:
        return [], idle, peak, ts
    cand = []
    for hi_f, lo_f in ((0.55, 0.20), (0.25, 0.10), (0.10, 0.04), (0.05, 0.02)):
        cand += _hysteresis(ts, dt, idle, amp, hi_f, lo_f, 2.0)
    cand.sort()
    merged = []
    for a, b in cand:
        if merged and a <= merged[-1][1] + int(0.5 / dt):
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged, idle, peak, ts


def _win_med(x, tu, t0, t1):
    m = (tu >= t0) & (tu <= t1)
    return float(np.median(x[m])) if m.any() else np.nan


def fall_time(tu, ts, i_fall, plat, post, dt):
    """90%→10% 下降时间（0.5 s 中值平滑后的原始总量）。"""
    if not (np.isfinite(plat) and np.isfinite(post)) or abs(plat - post) < 1e-9:
        return np.nan
    y_hi, y_lo = post + 0.90 * (plat - post), post + 0.10 * (plat - post)
    a = max(0, i_fall - int(4.0 / dt))
    b = min(len(tu) - 1, i_fall + int(4.0 / dt))
    idx = np.where(ts[a:i_fall + 1] >= y_hi)[0]
    i90 = a + idx[-1] if len(idx) else None
    idx = np.where(ts[i_fall:b + 1] <= y_lo)[0]
    i10 = i_fall + idx[0] if len(idx) else None
    if i90 is None or i10 is None or i10 < i90:
        return np.nan
    return float((i10 - i90) * dt)


def _floor_runs(ds, lvl, min_run):
    """返回「连续处于空载底」的 (起, 止) 帧对（止 = 最后一帧 + 1）。"""
    at = ds <= lvl
    d = np.diff(at.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if at[0]:
        starts = [0] + starts
    if at[-1]:
        ends = ends + [len(at)]
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_run]


def unload_events(d, min_step_frac=0.06, min_local_frac=0.12, nms_s=2.0):
    """卸载沿检测（阶跃式，含部分卸载）；加载沿 = 前一个「空载底连续段」的结束帧。

    第 1 遍：Δ(i) = med(ds[i+0.3,i+1.5]) − med(ds[i−1.5,i−0.3])，|Δ| ≥ max(0.06·全record幅值,
    0.12·本电平) 记候选，按 |Δ| 做 2 s 非极大抑制 → 下降沿集合与上升沿集合。
    第 2 遍：为每个下降沿定位加载沿 t_up（沿用底段结束）与卸载后稳态窗（上界 = 下一个
    加载事件的起点，避免窗口落进后一段负载）。
    """
    tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    n = len(tu)
    ds = med(tot, dt, 0.5)
    gfloor = float(np.percentile(ds, 5))
    gpeak = float(np.percentile(ds, 99.5))
    amp = gpeak - gfloor
    if amp <= 1e-9:
        return [], gfloor, gpeak, ds
    k1, k2 = max(3, int(round(1.5 / dt))), max(1, int(round(0.3 / dt)))
    rm = pd.Series(ds).rolling(max(2, k1 - k2), min_periods=1).median().to_numpy()
    pre_m = np.full(n, np.nan)
    post_m = np.full(n, np.nan)
    pre_m[k2:] = rm[:n - k2]
    post_m[:n - k1] = rm[k1:]
    dlt = post_m - pre_m
    thr = np.maximum(min_step_frac * amp, min_local_frac * np.maximum(pre_m, 1e-9))
    cand = [(i, dlt[i]) for i in range(n) if np.isfinite(dlt[i]) and abs(dlt[i]) >= thr[i]]
    cand.sort(key=lambda z: -abs(z[1]))
    keep = []
    for i, v in cand:
        if all(abs(i - j) > int(nms_s / dt) or np.sign(v) != np.sign(w) for j, w in keep):
            keep.append((i, v))
    keep.sort()
    falls = [i for i, v in keep if v < 0]
    rises = [i for i, v in keep if v > 0]
    ts1 = med(tot, dt, 0.1)

    # ── 第 1 遍：加载沿 ──
    info = []
    for i_fall in falls:
        plat = _win_med(ds, tu, float(tu[i_fall]) - 5.0, float(tu[i_fall]) - 0.5)
        if not np.isfinite(plat):
            plat = float(ds[i_fall])
        lvl = gfloor + 0.08 * (plat - gfloor)
        runs = _floor_runs(ds, lvl, max(3, int(0.3 / dt)))
        runs = [(a, b) for a, b in runs if b <= i_fall - max(1, int(0.5 / dt))]
        i_up = runs[-1][1] if runs else 0
        info.append(dict(i_fall=i_fall, plat=plat, runs=runs, i_up=i_up))
    # ── 第 2 遍：窗口 ──
    out = []
    for k, e in enumerate(info):
        i_fall, i_up, plat = e["i_fall"], e["i_up"], e["plat"]
        m = ((tu >= max(0.0, float(tu[i_up]) - 45.0)) & (tu <= float(tu[i_up]) - 3.0)
             & (ds <= gfloor + 0.10 * (plat - gfloor)))
        if int(m.sum()) >= 20:
            pre, pre_src = float(np.median(ds[m])), "local"
        else:
            pre, pre_src = gfloor, "global"
        nxt = [float(tu[i]) for i in rises if i > i_fall + max(1, int(0.5 / dt))]
        nxt += [float(tu[j["i_up"]]) for j in info[k + 1:] if j["i_up"] > i_fall]
        post_end_t = min(float(tu[-1]), float(tu[i_fall]) + 18.0,
                         (min(nxt) - 0.5) if nxt else float(tu[-1]))
        b0 = float(tu[i_fall]) + 2.0
        if post_end_t - b0 >= 1.0:
            post = float(np.median(ds[(tu >= b0) & (tu <= post_end_t)]))
            post_src = "long" if (post_end_t - b0) >= 6.0 else "short"
        else:
            post, post_src = np.nan, "end"
        i_step = min(n - 1, i_up + int(0.2 / dt))
        step_prox = float(ts1[i_step] - pre)
        drop = plat - pre
        # 加载沿速度：10%→90% 上升时间（判定「快加载」，慢速手工加载的 creep 代理不可用）
        y10, y90 = pre + 0.10 * drop, pre + 0.90 * drop
        win = ts1[i_up:i_fall + 1]
        w10, w90 = np.where(win >= y10)[0], np.where(win >= y90)[0]
        rise_t = float((w90[0] - w10[0]) * dt) if (len(w10) and len(w90) and w90[0] >= w10[0]) \
            else np.nan
        cls = "eof" if post_src == "end" else (
            "full" if post <= pre + 0.15 * drop else "partial")
        out.append(dict(rec=d["tag"], kind=d["kind"], i_rise=i_up, i_fall=i_fall,
                        t_up=float(tu[i_up]), t_dn=float(tu[i_fall]),
                        pre=pre, pre_src=pre_src, plat=plat, post=post, post_src=post_src,
                        post_end=post_end_t,
                        post_limit=min(float(tu[-1]),
                                       (min(nxt) - 0.5) if nxt else float(tu[-1])),
                        drop=drop, step_prox=step_prox,
                        hold_s=float(tu[i_fall] - tu[i_up]), rise_t=rise_t,
                        fast_load=bool(np.isfinite(rise_t) and rise_t <= 3.0),
                        creep_acc=plat - pre - step_prox,
                        creep_pct=(100.0 * (plat - pre - step_prox) / step_prox
                                   if abs(step_prox) > 1e-12 else np.nan),
                        fall_t=fall_time(tu, ts1, i_fall, plat, post, dt),
                        ev_class=cls))
    return [e for e in out if e["ev_class"] != "eof"], gfloor, gpeak, ds


def event_axes(ev, tu, dis, dt, band_frac=0.05, keep_s=5.0, dip_win_s=8.0):
    """卸载沿附近的显示响应（raw 与各算法臂用同一口径；窗口与事件几何一致）。"""
    n = len(tu)
    if ev["i_fall"] >= n - 1:
        return None
    ds = med(dis, dt, 0.5)
    t_dn, t_up = float(tu[ev["i_fall"]]), float(tu[ev["i_rise"]])
    floor = float(np.percentile(ds, 5))
    plat = _win_med(ds, tu, t_dn - 5.0, t_dn - 0.5)
    if not np.isfinite(plat):
        plat = float(np.median(ds[max(0, ev["i_fall"] - int(3.0 / dt)):ev["i_fall"]]))
    m = ((tu >= max(0.0, t_up - 45.0)) & (tu <= t_up - 3.0)
         & (ds <= floor + 0.10 * (plat - floor)))
    if ev["pre_src"] == "local" and int(m.sum()) >= 10:
        pre = float(np.median(ds[m]))
    else:
        pre = floor
    b0 = t_dn + 2.0
    b1 = min(ev["post_end"], t_dn + 18.0)
    if b1 - b0 >= 1.0:
        post = float(np.median(ds[(tu >= b0) & (tu <= b1)]))
    else:
        post = float(np.median(ds[max(ev["i_fall"], n - int(3.0 / dt)):]))
    j0 = ev["i_fall"] + int(0.05 / dt)
    j1 = min(n - 1, ev["i_fall"] + int(dip_win_s / dt))
    if j1 <= j0 or not np.isfinite(post):
        return None
    kk = j0 + int(np.argmin(ds[j0:j1]))
    dmin = float(ds[kk])
    post_late = np.nan
    if b1 - b0 >= 12.0:
        post_late = float(np.median(ds[(tu >= b1 - 10.0) & (tu <= b1)]))
    elif ev["post_src"] == "long" and b1 - b0 > 6.0:
        post_late = float(np.median(ds[(tu >= b1 - 4.0) & (tu <= b1)]))
    drop = plat - post
    band = band_frac * abs(drop)
    inside = (ds >= post - band) & (ds <= post + band)
    K = max(1, int(keep_s / dt))
    t_in = t_keep = np.nan
    for i in range(ev["i_fall"], n):
        if inside[i]:
            if not np.isfinite(t_in):
                t_in = float(tu[i] - t_dn)
            if i + K < n and inside[i:i + K].all():
                t_keep = float(tu[i] - t_dn)
                break
    tau1 = rms1 = tauf = taus = rms2 = np.nan
    if b1 - t_dn >= 6.0:
        tau1, rms1, tauf, taus, rms2 = fit_taus(tu, ds, t_dn + 0.10, b1, post)
    return dict(pre_dis=pre, plat_dis=plat, post_dis=post, post_late=post_late, drop_dis=drop,
                min_dis=dmin, trough_s=float(tu[kk] - t_dn),
                dip_vs_post=post - dmin, dip_vs_pre=pre - dmin,
                below_post_s=float(np.sum(ds[j0:j1] < post) * dt),
                t_rec_in=t_in, t_rec_keep=t_keep, resid=post - pre,
                resid_late=(post_late - pre) if np.isfinite(post_late) else np.nan,
                tau1=tau1, rms1=rms1, tau_f=tauf, tau_s=taus, rms2=rms2)


def fit_tail3(tu, x, t0, t1):
    """三参数尾拟合 x(t) = c + a·exp(−t/τ)：直接给出「残余渐近值 c」与「衰减常数 τ」。"""
    from scipy.optimize import curve_fit
    m = (tu >= t0) & (tu <= t1)
    if m.sum() < 30:
        return np.nan, np.nan, np.nan, np.nan
    t = tu[m] - t0
    y = x[m]
    span = float(t[-1] - t[0])
    try:
        p, _ = curve_fit(lambda t, c, a, tau: c + a * np.exp(-t / tau), t, y,
                         p0=[float(y[-1]), float(y[0] - y[-1]), max(1.0, span / 5)],
                         bounds=([-np.inf, -np.inf, 0.2], [np.inf, np.inf, 5000.0]),
                         maxfev=40000)
        rms = float(np.sqrt(np.mean((y - (p[0] + p[1] * np.exp(-t / p[2]))) ** 2)))
        return float(p[0]), float(p[1]), float(p[2]), rms
    except Exception:                                                       # noqa: BLE001
        return np.nan, np.nan, np.nan, np.nan


def fit_taus(tu, x, t0, t_end, x_inf):
    """对 x(t)−x_inf 做单/双指数拟合，返回 (tau1, rms1, tauf, taus, rms2)。"""
    from scipy.optimize import curve_fit
    m = (tu >= t0) & (tu <= t_end)
    if m.sum() < 20:
        return (np.nan,) * 5
    t = tu[m] - t0
    y = x[m] - x_inf
    if not np.isfinite(y).all() or np.nanmax(np.abs(y)) < 1e-12:
        return (np.nan,) * 5
    y0 = float(y[0])
    tau1 = rms1 = tauf = taus = rms2 = np.nan
    try:
        p, _ = curve_fit(lambda t, a, tau: a * np.exp(-t / tau), t, y,
                         p0=[y0, 10.0], bounds=([-np.inf, 0.2], [np.inf, 5000.0]),
                         maxfev=20000)
        tau1 = float(p[1])
        rms1 = float(np.sqrt(np.mean((y - p[0] * np.exp(-t / p[1])) ** 2)))
    except Exception:                                                       # noqa: BLE001
        pass
    try:
        f = lambda t, a1, t1, a2, t2: a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)
        p, _ = curve_fit(f, t, y, p0=[0.7 * y0, 1.0, 0.3 * y0, 60.0],
                         bounds=([-np.inf, 0.2, -np.inf, 20.0],
                                 [np.inf, 20.0, np.inf, 5000.0]), maxfev=40000)
        tauf, taus = sorted([float(p[1]), float(p[3])])
        rms2 = float(np.sqrt(np.mean((y - f(t, *p)) ** 2)))
    except Exception:                                                       # noqa: BLE001
        pass
    return tau1, rms1, tauf, taus, rms2


# ────────────────────────── v6 形状先验证据 ──────────────────────────
def shape_mismatch(tu, tot, i_e, dt, grid=None, g=None):
    """复算 v6 的逆模型：Â = Σy·g/Σg²（窗 [0.2, 0.8] s）；返回 (Ahat, inc5, Ahat/inc5)。"""
    grid = ROM_TAU if grid is None else np.asarray(grid, float)
    g = ROM_G if g is None else np.asarray(g, float)
    ts1 = med(tot, dt, 0.1)
    base = float(np.median(ts1[max(0, i_e - int(0.30 / dt)):max(1, i_e - int(0.05 / dt))]))
    tau = tu - tu[i_e]
    inc = ts1 - base
    sel = (tau >= 0.20) & (tau <= 0.80)
    if sel.sum() < 5:
        return np.nan, np.nan, np.nan
    gg = np.interp(tau[sel], grid, g, left=0.0, right=1.0)
    den = float((gg * gg).sum())
    if den < 1e-12:
        return np.nan, np.nan, np.nan
    Ahat = float((inc[sel] * gg).sum()) / den
    inc5 = float(np.interp(5.0, tau, inc))
    return Ahat, inc5, (Ahat / inc5 if abs(inc5) > 1e-12 else np.nan)


def shape_of(tu, tot, i_e, dt, grid=None):
    """某份录制自身的快相形状 f(τ)=inc(τ)/inc(5 s)，τ 取给定网格（默认 ROM 网格 ≥0.2 s）。"""
    grid = ROM_TAU if grid is None else np.asarray(grid, float)
    ts1 = med(tot, dt, 0.1)
    base = float(np.median(ts1[max(0, i_e - int(0.30 / dt)):max(1, i_e - int(0.05 / dt))]))
    tau = tu - tu[i_e]
    inc = ts1 - base
    inc5 = float(np.interp(5.0, tau, inc))
    if abs(inc5) < 1e-12:
        return None
    gs = grid[grid >= 0.2]
    return gs, np.array([float(np.interp(x, tau, inc)) / inc5 for x in gs])
