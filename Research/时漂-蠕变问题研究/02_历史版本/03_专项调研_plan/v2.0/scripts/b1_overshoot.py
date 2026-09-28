# -*- coding: utf-8 -*-
r"""v2.0 B1：加载过充/欠充量化与形态归因（只读离线分析，不改任何源码）。

数据：`temp/算法数据&原始数据/从零基线开始 - 恒定负载 - 反复加减同一个负载/`
      `20260919_100351_single_device_f9740b`（33908 帧 / 337.5 s / ~100.5 Hz / 21 通道 / 显示域 adc）

链路：算法前读数 `device_001_pre_seg0.csv`（= 无算法时会显示的值）、
      算法输出 `device_001_seg000.csv`（v6，参数集 plan-v1.0 A1+A4+A5a：κ_onset=1.05、
      κ_restep=1.12、ho_min_s=3.5、revoke_hold_n=3）、原始 ADC `device_001_raw_seg000.csv`。

本脚本不修改 `src/`、不构建、不运行 exe、不修改 `progress/**` 与 `plan/v1.0/**`。
唯一外部依赖：`v20_lib.py`（装载/切段工具）与归档原型
`temp/v4.1flash/progress/archived/07-v6/scripts/glm53_v6.py`（只 import、不修改）。

── 口径（全部显式；数值单位 ADC = 21 通道显示域求和）──
  total(t)           = Σ_{i=0..20} v_i(t)
  off(t)             = total_main(t) − total_pre(t)
  t_cross            = pre 总量 9 帧滑动均值自下而上穿过 (p3+p97)/2 的时刻（复现 probe-B/C 的 8 条沿）
  t0（真加载沿）     = 自 t_cross 向前回溯，最后一个仍处于「t_cross 前 1.2~0.6 s 电平 + 5%·A_est」
                       的帧的下一帧
  t_dn               = t_cross 之后第一个下降沿（> t_cross + 0.5 s）
  pre_base/main_base = [t0−1.2 s, t0] 的中位
  t_p0 = t0 + 0.5·min(6 s, t_dn−t0)，t_p1 = t_dn − 0.6 s（受载平台窗）
  pre_ss/main_ss     = [t_p0, t_p1] 的中位
  A_pre/A_main       = pre_ss − pre_base / main_ss − main_base（台阶幅度，两版都给）
  OS(t)              = (main(t) − main_base) − (pre(t) − pre_base)   （>0 = 显示高于输入）

三类「过充」指标（互不混用，报告 §2 逐个说明）：
  ① OS_pk3/A_pre    = max_{τ∈[0,3] s} OS / A_pre            （输入沿后 3 s 内的瞬态过充）
     OS_pkAll/A_pre = max_{τ∈[0,t_p1]} OS / A_pre           （含长时漂移，单列，不与①混用）
  ② OS_rel          = max (main−main_base)/(pre−pre_base) − 1，仅在 pre−pre_base ≥ 0.10·A_pre 处取值
  ③ OS_self         = max_{τ∈[0,t_p1]} (main(t) − main_fin) / A_pre，main_fin = [t_p1−1, t_p1] 中位
                      （显示是否超过它自己的最终值 ⇒ 用户看到的「冲高再回落」）

运行：$env:PYTHONIOENCODING='utf-8'; python temp\v4.1flash\plan\v2.0\scripts\b1_overshoot.py
"""
import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v20_lib as L  # noqa: E402

OUT = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(OUT, exist_ok=True)

KAPPA_PRIMARY = 1.05
KAPPA_RESTEP = 1.12
ROM_TAU_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80, 1.00)
TAU_EVAL = (0.30, 0.50, 0.80, 1.00)
TAU_REF, A_WIN = 0.20, 0.60
KAPPA_SWEEP = (1.00, 1.05, 1.10, 1.20, 1.30)
ROM_SCALE_SWEEP = (1.00, 1.04, 1.06, 1.10)
FILTER_TAUS = (0.10, 0.30, 0.50)


# ─────────────────────────── 基础工具 ───────────────────────────

def med_in(el, x, t0, t1):
    m = (el >= t0) & (el <= t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def mean_in(el, x, t0, t1):
    m = (el >= t0) & (el <= t1)
    return float(np.mean(x[m])) if m.any() else float("nan")


def min_in(el, x, t0, t1):
    m = (el >= t0) & (el <= t1)
    return float(np.min(x[m])) if m.any() else float("nan")


def max_in(el, x, t0, t1):
    m = (el >= t0) & (el <= t1)
    return float(np.max(x[m])) if m.any() else float("nan")


def idx_at(el, t):
    i = int(np.searchsorted(el, t))
    return min(max(i, 0), len(el) - 1)


def tsgm(x, el):
    """同一 `elapsed` 时间戳内的多帧先取中位（时间戳组中位）。

    本录制**每个时间戳最多压 4 帧**（同一 `elapsed` 出现 2~4 行），且在加载沿处这些帧
    的数值差异极大：实测 t=2.237 s 的四帧为 pre = 17978 / 5054 / 11084 / 10356 ADC
    （17978 还是整段录制的最大值）。任何基于单帧的峰值指标都会被这类突发内散点污染，
    故同一指标同时给出「单帧口径」与「时间戳组中位口径」两列。
    """
    x = np.asarray(x, float)
    out = np.empty_like(x)
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and el[j] == el[i]:
            j += 1
        out[i:j] = float(np.median(x[i:j]))
        i = j
    return out


def load_proto():
    spec = importlib.util.spec_from_file_location("glm53_v6_b1", L.PROTO_V6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def g_rom(tau, tau_ax, g_ax):
    return np.interp(np.asarray(tau, float), tau_ax, g_ax, left=0.0, right=1.0)


def g_scaled(tau, scale, tau_ax, g_ax):
    """v6.1 的保守形状：g61 = min(1, scale·g_v6)（见 plan/v1.0/T5 的 t5a_glm53_v61.py）。"""
    if scale == 1.0:
        return g_rom(tau, tau_ax, g_ax)
    return np.minimum(1.0, scale * g_rom(tau, tau_ax, g_ax))


def inv_est(tt, yy, tau, kappa, gfun):
    """原型 `GLM53v6._inv_est` 的等价实现（同一窗口/同一 inc 取法），gfun 可替换形状。

    返回 dict(A_ls, A_hat, A_hat_nok, inc, n) 或 None（估计失败）。
    """
    tt = np.asarray(tt, float)
    yy = np.asarray(yy, float)
    if len(tt) < 4 or tau < TAU_REF:
        return None
    m = (tt >= TAU_REF) & (tt <= min(tau, TAU_REF + A_WIN))
    if m.sum() < 5:
        return None
    g = gfun(tt[m])
    den = float((g * g).sum())
    if den < 1e-12:
        return None
    inc = float(yy[-1])
    if inc <= 0.0:
        return None
    a_ls = float((yy[m] * g).sum()) / den
    a_nok = max(a_ls, inc)
    return dict(A_ls=a_ls, A_hat=float(min(a_nok, kappa * inc)), A_hat_nok=a_nok,
                inc=inc, n=int(m.sum()))


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return pearson(ra, rb)


def wcsv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def f1(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.{nd}f}"


def f0(x):
    return f1(x, 0)


# ─────────────────────────── 事件检测 ───────────────────────────

def detect_events(el, tp):
    """由 pre 流给出加载沿（t_cross / t0）与配对卸载沿（t_dn），并做 plateau_segments 交叉校验。"""
    k = np.ones(9) / 9.0
    tps = np.convolve(tp, k, mode="same")
    lo = float(np.percentile(tp, 3))
    hi = float(np.percentile(tp, 97))
    thr = lo + 0.5 * (hi - lo)
    up, dn = L.edges_from_tot(tps, thr)
    ups, dns = [], []
    for i in up:
        if not ups or el[i] - el[ups[-1]] > 1.0:
            ups.append(i)
    for i in dn:
        if not dns or el[i] - el[dns[-1]] > 1.0:
            dns.append(i)
    segs = L.plateau_segments(tp, el)

    evs = []
    for iu in ups:
        t_cross = float(el[iu])
        t_dn = float(el[-1])
        for jd in dns:
            if el[jd] > t_cross + 0.5:
                t_dn = float(el[jd])
                break
        a_est = med_in(el, tp, t_cross + 2.0, min(t_cross + 4.0, t_dn - 0.3)) \
            if t_dn - t_cross > 3.0 else hi
        idle_est = med_in(el, tps, t_cross - 1.2, t_cross - 0.6)
        thr0 = idle_est + 0.05 * (a_est - idle_est)
        j0 = iu
        for j in range(max(0, iu - 250), iu):
            if tps[j] <= thr0:
                j0 = j
        t0 = float(el[j0 + 1]) if j0 + 1 <= iu else float(el[iu])
        # plateau_segments 段边界（最近的 idle→loaded 边界）作为交叉校验。
        # 注意口径差异：段边界是「越过 (p3+p97)/2 + 0.25·幅度」的时刻（= 62.5% 电平），
        # 比真加载沿晚；仅用于确认切段与门限法指向同一次加载。
        t_seg = float("nan")
        cands = [float(el[a]) for kind, a, b in segs if kind == "loaded"]
        if cands:
            k = int(np.argmin([abs(c - t_cross) for c in cands]))
            if abs(cands[k] - t_cross) < 5.0:
                t_seg = cands[k]
        # 卸载起始（用于把平台窗截到卸载瞬态之前）：自 t_dn 向前找最后一个仍处于受载电平的帧
        lvl_loaded = med_in(el, tp, t_cross + 2.0, t_dn - 0.6) if t_dn - t_cross > 3.0 else hi
        t_ulo = t_dn
        for j in range(idx_at(el, t_dn), idx_at(el, t0), -1):
            if tp[j] >= lvl_loaded - 0.05 * (lvl_loaded - lo):
                t_ulo = float(el[min(j + 1, len(el) - 1)])
                break
        pre_base = med_in(el, tp, t0 - 1.2, t0)
        t_p0 = t0 + 0.5 * min(6.0, t_ulo - t0)
        t_p1 = min(t_dn - 0.6, t_ulo - 0.3)
        short_plat = False
        if t_p1 - t_p0 < 0.8:
            short_plat = True
            t_p0 = max(t0 + 0.5, t_ulo - 1.0)
            t_p1 = t_ulo - 0.1
        evs.append(dict(i=len(evs), t_cross=t_cross, t0=t0, t_dn=t_dn, t_seg=t_seg,
                        t_ulo=t_ulo, hold=t_dn - t0, pre_base=pre_base, t_p0=t_p0,
                        t_p1=t_p1, short_plat=short_plat,
                        idle_p3=lo, loaded_p97=hi, thr=thr))
    return evs, dict(tps=tps, lo=lo, hi=hi, thr=thr, up=up, dn=dn, segs=segs)


def finish_events(el, tp, tm, evs):
    for ev in evs:
        ev["pre_ss"] = med_in(el, tp, ev["t_p0"], ev["t_p1"])
        ev["main_ss"] = med_in(el, tm, ev["t_p0"], ev["t_p1"])
        ev["main_base"] = med_in(el, tm, ev["t0"] - 1.2, ev["t0"])
        ev["A_pre"] = ev["pre_ss"] - ev["pre_base"]
        ev["A_main"] = ev["main_ss"] - ev["main_base"]
    return evs


# ─────────────────────────── 事件级指标（可用于任意输出流） ───────────────────────────

def stream_metrics(el, tp, y, ev):
    """以记录的 pre 事件窗为基准，算输出流 y 的过充/欠充/捕获/到带指标。

    y = 任意总输出流（记录的 main、回放输出、滤波后的回放输出）。
    """
    t0, t_dn, t_p0, t_p1 = ev["t0"], ev["t_dn"], ev["t_p0"], ev["t_p1"]
    A = ev["A_pre"]
    pre_base = ev["pre_base"]
    yb = med_in(el, y, t0 - 1.2, t0)
    y_fin = med_in(el, y, t_p1 - 1.0, t_p1)
    m = (el >= t0) & (el <= t_p1)
    idx = np.where(m)[0]
    t = el[idx]
    os_v = (y[idx] - yb) - (tp[idx] - pre_base)
    d = dict()

    # ① 瞬态过充（沿后 3 s）
    m3 = t <= t0 + 3.0
    if m3.any():
        j = int(np.argmax(os_v[m3]))
        d["os_pk3"] = float(os_v[m3][j])
        d["t_ospk3"] = float(t[m3][j] - t0)
        d["os_min3"] = float(np.min(os_v[m3]))
        d["t_osmin3"] = float(t[m3][int(np.argmin(os_v[m3]))] - t0)
    else:
        d["os_pk3"] = d["t_ospk3"] = d["os_min3"] = d["t_osmin3"] = float("nan")
    # ①b 全窗最大（含长时漂移，单独报）
    j = int(np.argmax(os_v))
    d["os_pkall"] = float(os_v[j])
    d["t_ospkall"] = float(t[j] - t0)
    d["os_minall"] = float(np.min(os_v))
    # ② 相对 pre 当前增量的过充
    inc_pre = tp[idx] - pre_base
    elig = inc_pre >= 0.10 * A
    if elig.any():
        rel = (y[idx][elig] - yb) / inc_pre[elig] - 1.0
        d["os_rel"] = float(np.max(rel))
        d["t_osrel"] = float(t[elig][int(np.argmax(rel))] - t0)
    else:
        d["os_rel"] = d["t_osrel"] = float("nan")
    # ③ 显示超过自身最终值
    ex = y[idx] - y_fin
    d["os_self"] = float(np.max(ex))
    d["t_osself"] = float(t[int(np.argmax(ex))] - t0)
    # 平台末段偏移（长时过充/欠充）
    ml = (el >= t_p1 - 1.0) & (el <= t_p1)
    d["os_late"] = float(np.median(((y - yb) - (tp - pre_base))[ml])) if ml.any() else float("nan")
    # 事件内最大 |off|
    offv = (y - tp)[idx]
    jo = int(np.argmax(np.abs(offv)))
    d["off_absmax"] = float(offv[jo])
    d["t_offmax"] = float(t[jo] - t0)
    # 台阶捕获比 G
    d["G"] = (y_fin - yb) / A if abs(A) > 1e-9 else float("nan")

    # 瞬态过充的衰减：峰值后 OS 从 90%→10% 的时间，以及指数拟合时间常数
    d["os_decay_10_90"] = float("nan")
    d["os_tau_exp"] = float("nan")
    d["os_tau_r2"] = float("nan")
    if np.isfinite(d["os_pk3"]) and d["os_pk3"] > 0:
        pk = d["os_pk3"]
        seg = (t >= t0 + d["t_ospk3"]) & (t <= t0 + 3.0)
        osv = os_v[seg]
        ts = t[seg]
        tt = np.where(osv <= 0.10 * pk)[0]
        if len(tt):
            d["os_decay_10_90"] = float(ts[tt[0]] - (t0 + d["t_ospk3"]))
        band = (osv <= 0.90 * pk) & (osv >= 0.10 * pk)
        if band.sum() >= 5:
            x = ts[band] - t0
            yv = np.log(osv[band])
            A_mat = np.vstack([x, np.ones_like(x)]).T
            sol, res, *_ = np.linalg.lstsq(A_mat, yv, rcond=None)
            slope = float(sol[0])
            pred = A_mat @ sol
            ss_tot = float(((yv - yv.mean()) ** 2).sum())
            r2 = 1.0 - float(((yv - pred) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else float("nan")
            if slope < -1e-9:
                d["os_tau_exp"] = float(-1.0 / slope)
                d["os_tau_r2"] = r2

    # 恢复时间：最后一次 |OS| ≥ 5%·A 的时刻（相对 t0）
    bad = np.where(np.abs(os_v) >= 0.05 * abs(A))[0]
    d["t_recover5"] = float(t[bad[-1]] - t0) if len(bad) else 0.0

    # 到带时间：此后一直落在 ±5%·A_pre 带内（相对输出自身最终值）
    for tag, frac in (("T_stable5", 0.05), ("T_stable2", 0.02)):
        bad = np.where(np.abs(y[idx] - y_fin) > frac * abs(A))[0]
        if len(bad) == 0:
            d[tag] = 0.0
        elif bad[-1] >= len(idx) - 1:
            d[tag] = float("nan")
        else:
            d[tag] = float(t[bad[-1] + 1] - t0)

    # 跟随时间：首次达到「沿前电平 + 95%/90%·A_pre」的时刻（相对 t0）。
    # 与 T_stable 不同，它只用 pre 的台阶作参考，不受长时漂移支配 ⇒ 本批数据里
    # T_stable 被漂移支配（中位 ~13 s），必须并报这个「跟随速度」指标。
    for tag, frac in (("T95_in", 0.95), ("T90_in", 0.90)):
        tgt = pre_base + frac * A
        j = np.where(y[idx] >= tgt)[0]
        if len(j) == 0:
            d[tag] = float("nan")
        else:
            d[tag] = float(t[j[0]] - t0)
    return d


def front_run_pred(ah, A):
    """机制推断的「前沿超前量」：τ=0.80 处 (Â − inc)/A_e。

    显示被滑行器拉到 base_y + Â，而输入此刻只走到 inc ⇒ 显示比输入高出 Â − inc。
    这不是拟合，是 §5.1 的 target/滑行关系在 τ≈0.8~1.0 s（滑行基本走完）处的直接推论。
    """
    r = ah["tau_eval"][0.80]
    if r is None:
        return float("nan")
    return (r["A_hat"] - r["inc"]) / A


def pair_metrics(el, tp, y, ev, tp_tg=None, y_tg=None):
    """返回 (主口径, 单帧对照口径) 两套事件指标。

    主口径 = 先把 `pre` 与输出按同一 `elapsed` 时间戳分组取中位（`tsgm`）再算；
    对照   = 直接用原始逐行流算。两者差异只可能来自时间戳突发内的散点。
    """
    if tp_tg is None:
        tp_tg = tsgm(tp, el)
    if y_tg is None:
        y_tg = tsgm(y, el)
    return stream_metrics(el, tp_tg, y_tg, ev), stream_metrics(el, tp, y, ev)


def classify(d):
    """形态分类（判据显式，见报告 §3）：瞬态尖峰 / 持续偏移 / 混合。"""
    pk, late = d["os_pk3"], d["os_late"]
    if not np.isfinite(pk):
        return "n/a"
    if pk <= 0 and (not np.isfinite(late) or abs(late) < 1e-9):
        return "无过充"
    if late >= 0.50 * pk and late > 0:
        return "持续偏移型"
    if pk > 0 and late <= 0.30 * pk and d["t_ospk3"] <= 1.5:
        return "瞬态尖峰型"
    return "混合型"


# ─────────────────────────── 输入形态特征 ───────────────────────────

def input_features(el, tp, ev):
    t0, t_dn, t_p1 = ev["t0"], ev["t_dn"], ev["t_p1"]
    A = ev["A_pre"]
    base = ev["pre_base"]
    ss = ev["pre_ss"]
    lo = base + 0.10 * A
    hi = base + 0.90 * A
    i0 = idx_at(el, t0)
    i1 = idx_at(el, t_p1)
    t10 = t90 = float("nan")
    for i in range(i0, i1 + 1):
        if not np.isfinite(t10) and tp[i] >= lo:
            t10 = float(el[i])
        if not np.isfinite(t90) and tp[i] >= hi:
            t90 = float(el[i])
            break
    d = dict(t10=t10, t90=t90, rise=t90 - t10 if np.isfinite(t90) and np.isfinite(t10) else float("nan"))
    d["rise_fast"] = bool(np.isfinite(d["rise"]) and d["rise"] <= 0.30)
    # 沿后 3 s 内 pre 自身最高点相对其平台中位（输入自身过冲）
    d["pre_pk3"] = max_in(el, tp, t0, t0 + 3.0)
    d["pre_ov3"] = (d["pre_pk3"] - ss) / A
    # 平台期最低点（下冲）与其时刻
    t_after = min(t0 + max(d["rise"], 0.2) if np.isfinite(d["rise"]) else t0 + 0.2, t_p1)
    d["pre_dip"] = min_in(el, tp, t_after, t_p1)
    d["pre_dip_frac"] = (ss - d["pre_dip"]) / A
    m = (el >= t_after) & (el <= t_p1)
    d["t_dip"] = float(el[np.where(m)[0][int(np.argmin(tp[m]))]] - t0) if m.any() else float("nan")
    # 沿后 1 s 的输入完成度（= 反演窗位置的形状快慢）
    d["pre_1s_frac"] = (float(np.interp(t0 + 1.0, el, tp)) - base) / A
    d["pre_03_frac"] = (float(np.interp(t0 + 0.3, el, tp)) - base) / A
    return d


# ─────────────────────────── ROM 对照与 Â 估计 ───────────────────────────

def rom_vs_obs(el, tp, ev, tau_ax, g_ax, tau_grid=ROM_TAU_GRID):
    t0, A, base = ev["t0"], ev["A_pre"], ev["pre_base"]
    tt = np.asarray(tau_grid, float)
    fo = (np.interp(t0 + tt, el, tp) - base) / A
    rom = g_rom(tt, tau_ax, g_ax)
    # 等值时间差：dt_eq>0 ⇒ ROM 更快（更早到达该比例）
    fine = np.arange(0.0, 1.0001, 0.002)
    fo_fine = (np.interp(t0 + fine, el, tp) - base) / A
    fo_fine = np.maximum.accumulate(np.maximum(fo_fine, 0.0))
    dt_eq = []
    for v in rom:
        if v <= 0:
            dt_eq.append(float("nan"))
            continue
        idx = np.where(fo_fine >= v)[0]
        t_obs = float(fine[idx[0]]) if len(idx) else float("nan")
        dt_eq.append(float(tt[list(rom).index(v)] - t_obs) if np.isfinite(t_obs)
                     else float("nan"))
    # 上面 index 用法对重复值不安全，重算一次（保持顺序对应）
    dt_eq = []
    for j, v in enumerate(rom):
        if v <= 0:
            dt_eq.append(float("nan"))
            continue
        idx = np.where(fo_fine >= v)[0]
        dt_eq.append(float(tt[j] - fine[idx[0]]) if len(idx) else float("nan"))
    return dict(tau=tt, f_obs=fo, rom=rom, diff=rom - fo, ratio=fo / np.where(rom > 0, rom, np.nan),
                dt_eq=np.asarray(dt_eq, float))


def ahat_table(el, tp, ev, tau_ax, g_ax):
    """逐事件 Â 估计（原型等价实现）：τ_eval 处的 A_ls / κ·inc / Â。"""
    t0, A, base = ev["t0"], ev["A_pre"], ev["pre_base"]
    m = (el >= t0) & (el <= t0 + 1.0 + 1e-9)
    tt = el[m] - t0
    yy = tp[m] - base
    rows = {}
    for tau in TAU_EVAL:
        mm = tt <= tau
        rows[tau] = inv_est(tt[mm], yy[mm], tau, KAPPA_PRIMARY, lambda z: g_rom(z, tau_ax, g_ax))
    # κ / ROM_SCALE 扫描（同一 τ_eval 表，供候选评估的解析层）
    sweep = {}
    for tau in TAU_EVAL:
        mm = tt <= tau
        for kap in KAPPA_SWEEP:
            sweep[("kappa", kap, tau)] = inv_est(tt[mm], yy[mm], tau, kap,
                                                lambda z: g_rom(z, tau_ax, g_ax))
        for sc in ROM_SCALE_SWEEP:
            sweep[("scale", sc, tau)] = inv_est(tt[mm], yy[mm], tau, KAPPA_PRIMARY,
                                                lambda z, s=sc: g_scaled(z, s, tau_ax, g_ax))
    return dict(tau_eval=rows, sweep=sweep, A=A, n=int(m.sum()))


# ─────────────────────────── 回放（候选评估） ───────────────────────────

def build_replay_class(proto):
    base = proto.GLM53v6
    tau_ax, g_ax = proto.ROM_TAU, proto.ROM_G

    class ReplayV6(base):
        """v6 原型 + 当前 C++ 参数集（κ=1.05 / HO_MIN=3.5 / 撤销迟滞 3 帧）+ 可选 ROM_SCALE。"""
        KAPPA_ONSET = KAPPA_PRIMARY
        KAPPA_RESTEP = KAPPA_RESTEP
        HO_MIN = 3.5
        REVOKE_HOLD_N = 3
        ROM_SCALE = 1.0
        ROM_ONSET_ONLY = True

        def _scale_now(self):
            kind = self.ev["kind"] if self.ev is not None else "onset"
            if self.ROM_ONSET_ONLY and kind != "onset":
                return 1.0
            return self.ROM_SCALE

        def _inv_est(self, hist, tau, kappa):
            tt = np.array([h[0] for h in hist], float)
            yy = np.array([h[1] for h in hist], float)
            sc = self._scale_now()
            r = inv_est(tt, yy, tau, kappa, lambda z: g_scaled(z, sc, tau_ax, g_ax))
            return (r["A_hat"], True) if r else (0.0, False)

        def _run_event(self, ts, v, total, dt, eps, idle_now):
            ev = self.ev
            if ev is None or self.REVOKE_HOLD_N <= 1:
                return super()._run_event(ts, v, total, dt, eps, idle_now)
            tau = ts - ev["t0"]
            inc_s = self._win_mean(ts - 0.10, ts)
            inc_s = (inc_s - ev["base"]) if inc_s is not None else (total - ev["base"])
            block = False
            if tau < self.REVOKE and ev["inc_max"] > eps and inc_s < 0.5 * ev["inc_max"]:
                ev["_rvn"] = ev.get("_rvn", 0) + 1
                if ev["_rvn"] < self.REVOKE_HOLD_N:
                    ev["_saved_incmax"] = ev["inc_max"]
                    ev["inc_max"] = 0.0
                    block = True
            else:
                ev["_rvn"] = 0
            out = super()._run_event(ts, v, total, dt, eps, idle_now)
            if block and self.ev is not None:
                self.ev["inc_max"] = self.ev.get("_saved_incmax", self.ev["inc_max"])
            return out

    return ReplayV6


def replay(pre_V, el, cls, kappa=KAPPA_PRIMARY, ho_min=3.5, hold_n=3, rom_scale=1.0):
    c = cls(L.NCH)
    c.KAPPA_ONSET = kappa
    c.HO_MIN = ho_min
    c.REVOKE_HOLD_N = hold_n
    c.ROM_SCALE = rom_scale
    n = len(el)
    tot = np.empty(n)
    for i in range(n):
        tot[i] = float(c.process(float(el[i]), pre_V[i]).sum())
    return dict(tot=tot, epochs=len(c.epoch_t), revokes=c.n_revoke, c5=c.n_c5,
                kind_log=list(c.kind_log))


def ema_filter(el, y, tau):
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        dt = float(el[i] - el[i - 1])
        dt = 0.0 if dt <= 0 else min(dt, 0.1)
        out[i] = out[i - 1] + (dt / tau) * (y[i] - out[i - 1])
    return out


def median_filter(y, tau):
    w = max(1, int(round(tau * 100.0)))
    if w % 2 == 0:
        w += 1
    h = w // 2
    pad = np.pad(y, h, mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(y))])


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-replay", action="store_true", help="跳过原型回放（快速出表）")
    ap.add_argument("--no-report", action="store_true", help="不写报告 md")
    args = ap.parse_args()

    d = L.load_dataset(L.DS_ZERO)
    pre, main = d["pre"], d["main"]
    el = pre["el"]
    tp = pre["V"].sum(1)
    tm = main["V"].sum(1)
    n = pre["n"]
    span = float(el[-1] - el[0])
    proto = load_proto()
    tau_ax, g_ax = proto.ROM_TAU, proto.ROM_G
    # 主口径流：同一 `elapsed` 时间戳内的多帧先取中位（见 tsgm 的 docstring）
    tp_tg = tsgm(tp, el)
    tm_tg = tsgm(tm, el)

    evs, det = detect_events(el, tp_tg)
    evs = finish_events(el, tp_tg, tm_tg, evs)
    print(f"frames={n} span={span:.1f}s fs={n/span:.2f}Hz 空载p3={det['lo']:.0f} "
          f"受载p97={det['hi']:.0f} 门限={det['thr']:.0f}")
    print("加载沿 t_cross = " + ", ".join(f"{e['t_cross']:.2f}" for e in evs))
    print("真加载沿 t0     = " + ", ".join(f"{e['t0']:.2f}" for e in evs))
    print("卸载沿 t_dn     = " + ", ".join(f"{e['t_dn']:.2f}" for e in evs))
    print("段边界 t_seg    = " + ", ".join(f"{e['t_seg']:.2f}" for e in evs))

    # 逐事件指标（记录流；主口径 = 时间戳组中位流，对照 = 单帧原始流）
    rows = []
    feats = {}
    roms = {}
    ahats = {}
    for ev in evs:
        m, m_raw = pair_metrics(el, tp, tm, ev, tp_tg, tm_tg)
        ft = input_features(el, tp_tg, ev)
        rr = rom_vs_obs(el, tp_tg, ev, tau_ax, g_ax)
        ah = ahat_table(el, tp_tg, ev, tau_ax, g_ax)
        ev.update(m)
        ev["os_pk3_raw"] = m_raw["os_pk3"]
        ev["t_ospk3_raw"] = m_raw["t_ospk3"]
        ev["os_min3_raw"] = m_raw["os_min3"]
        ev["os_self_raw"] = m_raw["os_self"]
        ev["pred_pk3"] = front_run_pred(ah, ev["A_pre"])
        feats[ev["i"]] = ft
        roms[ev["i"]] = rr
        ahats[ev["i"]] = ah
        ev["class"] = classify(m)
        ev["feature"] = ft
        rows.append((ev, m, ft, rr, ah))

    # ── 台账 CSV ──
    hdr = ["event", "t_cross_s", "t0_s", "t_dn_s", "t_ulo_s", "t_seg_s", "hold_s",
           "short_plateau", "t_p0_s", "t_p1_s",
           "pre_base", "pre_ss", "A_pre", "main_base", "main_ss", "A_main",
           "os_pk3", "os_pk3_pctA", "t_ospk3",
           "os_pk3_raw", "os_pk3_raw_pctA", "t_ospk3_raw",
           "os_min3", "os_min3_pctA", "t_osmin3", "os_min3_raw",
           "os_pkall", "os_pkall_pctA", "t_ospkall", "os_late", "os_late_pctA",
           "os_rel_pct", "t_osrel", "os_self", "os_self_pctA", "os_self_raw_pctA", "t_osself",
           "off_absmax", "off_absmax_pctA", "t_offmax",
           "os_decay_10_90_s", "os_tau_exp_s", "os_tau_r2", "t_recover5_s",
           "T_stable5_s", "T_stable2_s", "T95_in_s", "G",
           "pred_os_pk3_pctA", "resid_os_pk3_pctA", "class"]
    out_rows = []
    for ev, m, ft, rr, ah in rows:
        A = ev["A_pre"]
        pred = ev["pred_pk3"]
        out_rows.append([
            ev["i"], f1(ev["t_cross"], 2), f1(ev["t0"], 2), f1(ev["t_dn"], 2),
            f1(ev["t_ulo"], 2), f1(ev["t_seg"], 2), f1(ev["hold"], 1),
            1 if ev["short_plat"] else 0, f1(ev["t_p0"], 2), f1(ev["t_p1"], 2),
            f1(ev["pre_base"], 0), f1(ev["pre_ss"], 0), f1(A, 0),
            f1(ev["main_base"], 0), f1(ev["main_ss"], 0), f1(ev["A_main"], 0),
            f1(m["os_pk3"], 0), f1(100 * m["os_pk3"] / A, 2), f1(m["t_ospk3"], 2),
            f1(ev["os_pk3_raw"], 0), f1(100 * ev["os_pk3_raw"] / A, 2), f1(ev["t_ospk3_raw"], 2),
            f1(m["os_min3"], 0), f1(100 * m["os_min3"] / A, 2), f1(m["t_osmin3"], 2),
            f1(ev["os_min3_raw"], 0),
            f1(m["os_pkall"], 0), f1(100 * m["os_pkall"] / A, 2), f1(m["t_ospkall"], 2),
            f1(m["os_late"], 0), f1(100 * m["os_late"] / A, 2),
            f1(100 * m["os_rel"], 2) if np.isfinite(m["os_rel"]) else "nan", f1(m["t_osrel"], 2),
            f1(m["os_self"], 0), f1(100 * m["os_self"] / A, 2),
            f1(100 * ev["os_self_raw"] / A, 2), f1(m["t_osself"], 2),
            f1(m["off_absmax"], 0), f1(100 * m["off_absmax"] / A, 2), f1(m["t_offmax"], 2),
            f1(m["os_decay_10_90"], 2), f1(m["os_tau_exp"], 3), f1(m["os_tau_r2"], 3),
            f1(m["t_recover5"], 2), f1(m["T_stable5"], 2), f1(m["T_stable2"], 2),
            f1(m["T95_in"], 2), f1(m["G"], 4),
            f1(100 * pred, 2) if np.isfinite(pred) else "nan",
            f1(100 * (m["os_pk3"] / A - pred), 2) if np.isfinite(pred) else "nan",
            ev["class"]])
    wcsv(os.path.join(OUT, "b1_events.csv"), hdr, out_rows)
    print(f"\nwrote results/b1_events.csv ({len(out_rows)} 行)")

    # ── 台账控制台表 ──
    print("\n── 逐事件过充台账（记录流；主口径 = 时间戳组中位，raw = 单帧对照；ADC = 21 通道求和）──")
    print(f"{'i':>2}{'t_cross':>9}{'t0':>8}{'t_dn':>8}{'t_ulo':>8}{'A_pre':>7}"
          f"{'OSpk3':>8}{'%A':>7}{'t_pk':>6}{'raw':>7}{'raw%A':>7}"
          f"{'OSmin3':>8}{'%A':>7}"
          f"{'OSlate':>8}{'%A':>7}{'OSself':>8}{'%A':>7}{'|off|mx':>8}{'%A':>7}"
          f"{'tau_exp':>8}{'Trec5':>7}{'Tst5':>7}{'T95in':>7}{'G':>7}{'pred':>7}{'resid':>7}  形态")
    for ev, m, ft, rr, ah in rows:
        A = ev["A_pre"]
        pred = ev["pred_pk3"]
        print(f"{ev['i']:>2}{ev['t_cross']:9.2f}{ev['t0']:8.2f}{ev['t_dn']:8.1f}{ev['t_ulo']:8.1f}"
              f"{A:7.0f}"
              f"{m['os_pk3']:8.0f}{100*m['os_pk3']/A:7.2f}{m['t_ospk3']:6.2f}"
              f"{ev['os_pk3_raw']:7.0f}{100*ev['os_pk3_raw']/A:7.2f}"
              f"{m['os_min3']:8.0f}{100*m['os_min3']/A:7.2f}"
              f"{m['os_late']:8.0f}{100*m['os_late']/A:7.2f}"
              f"{m['os_self']:8.0f}{100*m['os_self']/A:7.2f}"
              f"{m['off_absmax']:8.0f}{100*m['off_absmax']/A:7.2f}"
              f"{m['os_tau_exp']:8.3f}{m['t_recover5']:7.2f}{m['T_stable5']:7.2f}"
              f"{m['T95_in']:7.2f}{m['G']:7.3f}{100*pred:7.2f}{100*(m['os_pk3']/A-pred):7.2f}  {ev['class']}"
              + (" [短平台]" if ev["short_plat"] else ""))

    # ── 形态分类统计 ──
    from collections import Counter
    cnt = Counter(ev["class"] for ev, *_ in rows)
    print("\n形态分类占比: " + ", ".join(f"{k}={v}/{len(rows)}" for k, v in cnt.items()))

    # ── 20 ms 轨迹（典型事件） ──
    typ_spike = max(rows, key=lambda r: r[1]["os_pk3"])[0]["i"]
    typ_sust = max(rows, key=lambda r: abs(r[1]["os_late"]))[0]["i"]
    traces = {}
    for tag, ei in (("spike", typ_spike), ("sustain", typ_sust)):
        ev = evs[ei]
        lines = []
        t = 0.0
        while t <= 2.0001:
            m = (el >= ev["t0"] + t) & (el < ev["t0"] + t + 0.02)
            if m.any():
                lines.append((t, float(np.mean(tp_tg[m])), float(np.mean(tm_tg[m])),
                              float(np.mean(tm_tg[m])) - float(np.mean(tp_tg[m]))))
            t += 0.02
        traces[tag] = (ei, lines)
    wcsv(os.path.join(OUT, "b1_traces.csv"),
         ["tag", "event", "t_rel_s", "pre", "main", "off"],
         [[tag, ei, f1(a, 2), f1(b, 0), f1(c, 0), f1(dd, 0)]
          for tag, (ei, lines) in traces.items() for (a, b, c, dd) in lines])
    print(f"wrote results/b1_traces.csv（典型事件 #{typ_spike}=瞬态尖峰、#{typ_sust}=持续偏移）")

    # ── ROM 对照表 ──
    rom_rows = []
    for ev, *_ in rows:
        rr = roms[ev["i"]]
        for j, tau in enumerate(rr["tau"]):
            rom_rows.append([ev["i"], f1(tau, 2), f1(rr["rom"][j], 4), f1(rr["f_obs"][j], 4),
                             f1(rr["diff"][j], 4), f1(rr["ratio"][j], 4),
                             f1(rr["dt_eq"][j], 4)])
    wcsv(os.path.join(OUT, "b1_rom.csv"),
         ["event", "tau_s", "g_rom", "f_obs", "rom_minus_obs", "obs_over_rom", "dt_eq_s"],
         rom_rows)
    print("\n── ROM vs 实测归一化形状（沿后 τ 处的 g 值；dt_eq>0 ⇒ ROM 更快）──")
    print(f"{'tau':>6}" + "".join(f"{'ev'+str(ev['i']):>16}" for ev, *_ in rows)
          + f"{'中位差':>9}{'中位dt_eq':>10}")
    for j, tau in enumerate(ROM_TAU_GRID):
        line = f"{tau:6.2f}"
        diffs, dts = [], []
        for ev, *_ in rows:
            rr = roms[ev["i"]]
            line += f"{rr['f_obs'][j]:7.3f}/{rr['rom'][j]:5.3f}"
            diffs.append(rr["diff"][j])
            dts.append(rr["dt_eq"][j])
        line += f"{np.nanmedian(diffs):9.3f}{np.nanmedian(dts):10.3f}"
        print(line)

    # ── Â 表 ──
    ah_rows = []
    print("\n── 逐事件 Â 估计（原型等价实现；κ=1.05）──")
    print(f"{'i':>2}{'tau':>6}{'A_ls/A':>9}{'k*inc/A':>9}{'Ahat/A':>9}{'inc/A':>8}{'n':>4}")
    for ev, *_ in rows:
        ah = ahats[ev["i"]]
        A = ev["A_pre"]
        for tau in TAU_EVAL:
            r = ah["tau_eval"][tau]
            if r is None:
                print(f"{ev['i']:>2}{tau:6.2f}{'   —':>9}{'—':>9}{'—':>9}{'—':>8}{0:>4}")
                ah_rows.append([ev["i"], f1(tau, 2), "", "", "", "", ""])
                continue
            print(f"{ev['i']:>2}{tau:6.2f}{r['A_ls']/A:9.3f}{KAPPA_PRIMARY*r['inc']/A:9.3f}"
                  f"{r['A_hat']/A:9.3f}{r['inc']/A:8.3f}{r['n']:>4}")
            ah_rows.append([ev["i"], f1(tau, 2), f1(r["A_ls"] / A, 4),
                            f1(KAPPA_PRIMARY * r["inc"] / A, 4), f1(r["A_hat"] / A, 4),
                            f1(r["inc"] / A, 4), r["n"]])
    wcsv(os.path.join(OUT, "b1_ahat.csv"),
         ["event", "tau_s", "A_ls_over_A", "kappa_inc_over_A", "A_hat_over_A", "inc_over_A", "n"],
         ah_rows)

    # ── 反演过估/欠估统计 ──
    print("\n── Â/A_e 过估统计（τ=0.80；>1 = 过估）──")
    for tau in (0.50, 0.80, 1.00):
        rs = [ahats[ev["i"]]["tau_eval"][tau] for ev, *_ in rows]
        vals = [r["A_hat"] / ev["A_pre"] for r, (ev, *_) in zip(rs, rows) if r is not None]
        ls = [r["A_ls"] / ev["A_pre"] for r, (ev, *_) in zip(rs, rows) if r is not None]
        ci = [KAPPA_PRIMARY * r["inc"] / ev["A_pre"] for r, (ev, *_) in zip(rs, rows) if r is not None]
        print(f"τ={tau:.2f}: Â/A 中位 {np.median(vals):.3f} 范围 [{min(vals):.3f},{max(vals):.3f}] "
              f"| A_ls/A 中位 {np.median(ls):.3f} | κ·inc/A 中位 {np.median(ci):.3f} "
              f"| 过估事件数 {sum(1 for v in vals if v > 1.0)}/{len(vals)}")

    # ── 输入形态 ↔ 过充 相关性 ──
    print("\n── 输入形态 vs 过充（n=8，Pearson r / Spearman ρ）──")
    y_pk = [ev["os_pk3"] / ev["A_pre"] for ev, *_ in rows]
    y_late = [ev["os_late"] / ev["A_pre"] for ev, *_ in rows]
    y_self = [ev["os_self"] / ev["A_pre"] for ev, *_ in rows]
    y_min = [ev["os_min3"] / ev["A_pre"] for ev, *_ in rows]
    x_rise = [feats[ev["i"]]["rise"] for ev, *_ in rows]
    x_dip = [feats[ev["i"]]["pre_dip_frac"] for ev, *_ in rows]
    x_ov3 = [feats[ev["i"]]["pre_ov3"] for ev, *_ in rows]
    x_f1s = [feats[ev["i"]]["pre_1s_frac"] for ev, *_ in rows]
    corr = []
    for xn, xv in (("pre 上升 10-90% 耗时", x_rise), ("pre 平台下冲/A", x_dip),
                   ("pre 沿后3s最高点相对平台/A（负=尚未到平台）", x_ov3),
                   ("pre沿后1s完成度", x_f1s)):
        for yn, yv in (("OSpk3/A", y_pk), ("OSlate/A", y_late), ("OSself/A", y_self),
                       ("OSmin3/A", y_min)):
            r, rho = pearson(xv, yv), spearman(xv, yv)
            corr.append((xn, yn, r, rho))
            print(f"  {xn:22s} vs {yn:10s} r={r:+.3f} ρ={rho:+.3f}")

    # 分组对比（按 pre 沿后 1 s 完成度中位切两半）
    med_f1s = float(np.median(x_f1s))
    print(f"\n── 分组对比（按 pre 沿后 1 s 完成度，切点 {med_f1s:.3f}）──")
    grp = {"快组(≥切点)": [i for i, v in enumerate(x_f1s) if v >= med_f1s],
           "慢组(<切点)": [i for i, v in enumerate(x_f1s) if v < med_f1s]}
    groups = []
    for name, idxs in grp.items():
        line = (f"  {name:12s} n={len(idxs)}  "
                f"OSpk3/A 中位 {100*np.median([y_pk[i] for i in idxs]):+.2f}%  "
                f"OSself/A 中位 {100*np.median([y_self[i] for i in idxs]):+.2f}%  "
                f"OSlate/A 中位 {100*np.median([y_late[i] for i in idxs]):+.2f}%  "
                f"pre10-90 中位 {np.median([x_rise[i] for i in idxs]):.2f}s")
        print(line)
        groups.append((name, [int(i) for i in idxs]))

    # 机制律核对：OS_pk3/A ≈ (Â(0.8) − inc(0.8))/A
    preds = [ev["pred_pk3"] for ev, *_ in rows]
    resid = [y_pk[i] - preds[i] for i in range(len(rows))]
    print(f"\n── 前沿超前律核对（pred=(Â−inc)/A @τ=0.80）──")
    print(f"  实测 OS_pk3/A 中位 {100*np.median(y_pk):+.2f}%  预测中位 "
          f"{100*np.median(preds):+.2f}%  残差中位 {100*np.median(resid):+.2f}pp  "
          f"最大 |残差| {100*max(abs(r) for r in resid):.2f}pp  "
          f"相关 r={pearson(preds, y_pk):+.3f}")
    print(f"  κ 上限在 τ=0.80 处生效（A_ls > κ·inc）的事件数: "
          f"{sum(1 for ev, *_ in rows if ahats[ev['i']]['tau_eval'][0.80] is not None and ahats[ev['i']]['tau_eval'][0.80]['A_ls'] > KAPPA_PRIMARY*ahats[ev['i']]['tau_eval'][0.80]['inc'])}"
          f"/{len(rows)}")

    # ── 候选评估 ──
    cand_rows = []
    agg = {}
    if not args.no_replay:
        cls = build_replay_class(proto)
        print("\n── 回放（原型 v6 + 当前参数集；验证原型对 C++ 输出的复现度）──")
        import time
        variants = [("v6_now", dict(kappa=KAPPA_PRIMARY, ho_min=3.5, hold_n=3, rom_scale=1.0))]
        for k in KAPPA_SWEEP:
            if k == KAPPA_PRIMARY:
                continue
            variants.append((f"kappa_{k:.2f}", dict(kappa=k, ho_min=3.5, hold_n=3, rom_scale=1.0)))
        for s in ROM_SCALE_SWEEP[1:]:
            variants.append((f"romscale_{s:.2f}", dict(kappa=KAPPA_PRIMARY, ho_min=3.5,
                                                       hold_n=3, rom_scale=s)))
        reps = {}
        for tag, kw in variants:
            t_start = time.time()
            reps[tag] = replay(pre["V"], el, cls, **kw)
            print(f"  [{tag:14s}] epochs={reps[tag]['epochs']:3d} revokes={reps[tag]['revokes']:3d} "
                  f"c5={reps[tag]['c5']:3d}  用时 {time.time()-t_start:.1f}s")
        reps["recorded"] = dict(tot=tm)
        base_tot = reps["v6_now"]["tot"]
        for tau in FILTER_TAUS:
            reps[f"ema_{tau:.2f}"] = dict(tot=ema_filter(el, base_tot, tau))
        reps[f"med_0.30"] = dict(tot=median_filter(base_tot, 0.30))

        # 复现度
        dv = reps["v6_now"]["tot"] - tm
        print(f"\n原型回放 vs 记录 main：RMS 差 {np.sqrt(np.mean(dv**2)):.0f} ADC、"
              f"中位|差| {np.median(np.abs(dv)):.0f} ADC、"
              f"整段电平 RMS {np.sqrt(np.mean((tm-tm.mean())**2)):.0f} ADC、"
              f"相对 RMS {100*np.sqrt(np.mean(dv**2))/np.sqrt(np.mean((tm-tm.mean())**2)):.1f}%")
        fid = dict(rms=float(np.sqrt(np.mean(dv ** 2))),
                   mad=float(np.median(np.abs(dv))))

        # 指标聚合（主口径 = 时间戳组中位流；raw = 单帧原始流）
        def aggregate(tot):
            per = []
            tot_tg = tsgm(tot, el)
            for ev in evs:
                mm, mr = pair_metrics(el, tp, tot, ev, tp_tg, tot_tg)
                mm["i"] = ev["i"]
                mm["os_pk3_raw"] = mr["os_pk3"]
                mm["os_self_raw"] = mr["os_self"]
                mm["os_min3_raw"] = mr["os_min3"]
                per.append(mm)
            return per

        for tag, r in reps.items():
            per = aggregate(r["tot"])
            pk = [m["os_pk3"] / evs[m["i"]]["A_pre"] for m in per]
            pk_raw = [m["os_pk3_raw"] / evs[m["i"]]["A_pre"] for m in per]
            mn3_raw = [m["os_min3_raw"] / evs[m["i"]]["A_pre"] for m in per]
            sf_raw = [m["os_self_raw"] / evs[m["i"]]["A_pre"] for m in per]
            pk_all = [m["os_pkall"] / evs[m["i"]]["A_pre"] for m in per]
            late = [m["os_late"] / evs[m["i"]]["A_pre"] for m in per]
            sf = [m["os_self"] / evs[m["i"]]["A_pre"] for m in per]
            mn3 = [m["os_min3"] / evs[m["i"]]["A_pre"] for m in per]
            G = [m["G"] for m in per]
            ts5 = [m["T_stable5"] for m in per]
            ts2 = [m["T_stable2"] for m in per]
            t95 = [m["T95_in"] for m in per]
            agg[tag] = dict(pk=pk, pk_all=pk_all, late=late, self=sf, min3=mn3, G=G,
                            ts5=ts5, ts2=ts2, t95=t95,
                            pk_raw=pk_raw, mn3_raw=mn3_raw, sf_raw=sf_raw,
                            pk_med=float(np.median(pk)), pk_max=float(np.max(pk)),
                            pk_raw_med=float(np.median(pk_raw)), pk_raw_max=float(np.max(pk_raw)),
                            mn3_raw_med=float(np.median(mn3_raw)), mn3_raw_min=float(np.min(mn3_raw)),
                            sf_raw_max=float(np.max(sf_raw)),
                            late_med=float(np.median(late)), late_absmax=float(np.max(np.abs(late))),
                            self_med=float(np.median(sf)), self_max=float(np.max(sf)),
                            min3_med=float(np.median(mn3)), min3_min=float(np.min(mn3)),
                            G_med=float(np.median(G)),
                            ts5_med=float(np.nanmedian(ts5)), ts5_max=float(np.nanmax(ts5)),
                            ts5_nan=int(np.sum(~np.isfinite(ts5))),
                            t95_med=float(np.nanmedian(t95)), t95_max=float(np.nanmax(t95)),
                            ts2_med=float(np.nanmedian(ts2)),
                            per=per)
        print("\n── 候选三角代价表（8 事件聚合；主口径 = 时间戳组中位，raw = 单帧）──")
        print(f"{'variant':>16}{'OSpk3中位':>11}{'OSpk3max':>10}{'raw中位':>9}{'rawmax':>9}"
              f"{'OSselfmax':>10}{'OSmin3中位':>11}{'T5中位':>9}{'T95中位':>9}"
              f"{'T95max':>9}{'G中位':>8}")
        order = ["recorded", "v6_now"] + [t for t, _ in variants[1:]] + \
                [f"ema_{t:.2f}" for t in FILTER_TAUS] + ["med_0.30"]
        for tag in order:
            a = agg[tag]
            print(f"{tag:>16}{100*a['pk_med']:10.2f}%{100*a['pk_max']:9.2f}%"
                  f"{100*a['pk_raw_med']:8.2f}%{100*a['pk_raw_max']:8.2f}%"
                  f"{100*a['self_max']:9.2f}%{100*a['min3_med']:10.2f}%{a['ts5_med']:9.2f}"
                  f"{a['t95_med']:9.2f}{a['t95_max']:9.2f}{a['G_med']:8.3f}")
            cand_rows.append([tag, f1(100 * a["pk_med"], 2), f1(100 * a["pk_max"], 2),
                              f1(100 * a["pk_raw_med"], 2), f1(100 * a["pk_raw_max"], 2),
                              f1(100 * a["late_med"], 2), f1(100 * a["late_absmax"], 2),
                              f1(100 * a["self_med"], 2), f1(100 * a["self_max"], 2),
                              f1(100 * a["sf_raw_max"], 2),
                              f1(100 * a["min3_med"], 2), f1(100 * a["min3_min"], 2),
                              f1(100 * a["mn3_raw_med"], 2),
                              f1(a["ts5_med"], 2), f1(a["ts5_max"], 2), a["ts5_nan"],
                              f1(a["t95_med"], 2), f1(a["t95_max"], 2),
                              f1(a["ts2_med"], 2), f1(a["G_med"], 4)])
        wcsv(os.path.join(OUT, "b1_candidates.csv"),
             ["variant", "OSpk3_med_pct", "OSpk3_max_pct",
              "OSpk3_raw_med_pct", "OSpk3_raw_max_pct",
              "OSlate_med_pct", "OSlate_absmax_pct",
              "OSself_med_pct", "OSself_max_pct", "OSself_raw_max_pct",
              "OSmin3_med_pct", "OSmin3_min_pct", "OSmin3_raw_med_pct",
              "Tstable5_med_s", "Tstable5_max_s", "Tstable5_nan_n",
              "T95_input_med_s", "T95_input_max_s", "Tstable2_med_s", "G_med"],
             cand_rows)
        print("wrote results/b1_candidates.csv")

        # 逐事件 × 变体的 OS_pk3 矩阵（用于解释「中位改善但最大值变差」的情况）
        sel = ["recorded", "v6_now", "kappa_1.00", "kappa_1.10",
               "romscale_1.06", "romscale_1.10", "ema_0.10", "ema_0.30", "ema_0.50", "med_0.30"]
        sel = [t for t in sel if t in agg]
        rows_pe = []
        print("\n── 逐事件 OS_pk3/A（%，主口径=时间戳组中位）× 变体 ──")
        print(f"{'i':>2}{'A_pre':>7}" + "".join(f"{t[:11]:>12}" for t in sel)
              + f"{'t_dn-t0':>9}{'hold':>7}")
        for k, ev in enumerate(evs):
            vals = [agg[t]["pk"][k] * 100 for t in sel]
            print(f"{ev['i']:>2}{ev['A_pre']:7.0f}" + "".join(f"{v:12.2f}" for v in vals)
                  + f"{ev['t_dn']-ev['t0']:9.1f}{ev['hold']:7.1f}")
            rows_pe.append([ev["i"], f1(ev["A_pre"], 0)] + [f1(v, 2) for v in vals])
        wcsv(os.path.join(OUT, "b1_candidates_per_event.csv"),
             ["event", "A_pre"] + [f"OSpk3_pct_{t}" for t in sel], rows_pe)
        print("wrote results/b1_candidates_per_event.csv")

        # 解析层：κ / ROM_SCALE 对 Â/A_e 的影响（在下面的汇总块里统一打印）

    # ── 解析层汇总（κ / ROM_SCALE 对 Â/A_e 的影响）──
    sweep_tbl = []
    if True:
        print("\n── 解析层：κ 与 ROM_SCALE 对 Â/A_e 的影响（τ=1.00，中位/最大）──")
        for kind, vals in (("kappa", KAPPA_SWEEP), ("scale", ROM_SCALE_SWEEP)):
            line = ""
            for v in vals:
                rs = [ahats[ev["i"]]["sweep"][(kind, v, 1.00)] for ev, *_ in rows]
                arr = [r["A_hat"] / ev["A_pre"] for r, (ev, *_) in zip(rs, rows) if r is not None]
                line += f"  {v:.2f}: 中位{np.median(arr):.3f}/max{max(arr):.3f}"
                sweep_tbl.append((kind, v, float(np.median(arr)), float(max(arr))))
            print(f"  {kind:6s}{line}")

    # ── 文本摘要与报告 ──
    rep = dict(el=el, tp=tp, tm=tm, evs=evs, rows=rows, feats=feats, roms=roms,
               ahats=ahats, det=det, n=n, span=span, fs=n / span,
               typ_spike=typ_spike, typ_sust=typ_sust, traces=traces,
               agg=agg, corr=corr, groups=groups, resid=resid, preds=preds,
               cand_rows=cand_rows, order=(order if not args.no_replay else None),
               reps=(reps if not args.no_replay else None),
               fid=(fid if not args.no_replay else None),
               sweep_tbl=sweep_tbl,
               proto_rel=os.path.relpath(L.PROTO_V6, L.ROOT),
               data_rel=os.path.relpath(L.DS_ZERO, L.ROOT),
               lib_rel=os.path.relpath(os.path.join(HERE, "v20_lib.py"), L.ROOT),
               self_rel=os.path.relpath(os.path.abspath(__file__), L.ROOT),
               out_dir=os.path.relpath(OUT, L.ROOT),
               no_replay=bool(args.no_replay))
    if not args.no_report:
        write_report(rep)
    print("\n完成。")


# ─────────────────────────── 报告 ───────────────────────────

def _md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_report(R):
    el, tp, tm = R["el"], R["tp"], R["tm"]
    evs, rows = R["evs"], R["rows"]
    det = R["det"]
    agg, order, fid = R["agg"], R["order"], R["fid"]
    L_ = []

    def w(s=""):
        L_.append(s)

    w("# 分析报告 B1 — 加载「过充」量化与形态归因")
    w()
    w(f"> 数据：`{R['data_rel']}`（{R['n']} 帧 / {R['span']:.1f} s / "
      f"{R['fs']:.2f} Hz / 21 通道 / 显示域 `adc`）  ")
    w("> 算法：v6，参数集 `plan-v1.0 A1+A4+A5a`（κ_onset=1.05 / κ_restep=1.12 / "
      "ho_min_s=3.5 / revoke_hold_n=3）  ")
    w("> 本报告只做离线只读分析：不修改 `src/`、不构建、不运行 exe、不修改 "
      "`progress/**` 与 `plan/v1.0/**`。  ")
    w("> 唯一外部依赖：`" + R["lib_rel"] + r"`（装载/切段）与归档原型 `" + R["proto_rel"] + "`（只读 import）。")
    w()

    # ── 0 摘要 ──
    pk_med = float(np.median([ev["os_pk3"] / ev["A_pre"] for ev, *_ in rows]))
    pk_raw_med = float(np.median([ev["os_pk3_raw"] / ev["A_pre"] for ev, *_ in rows]))
    pk_raw_max = float(np.max([ev["os_pk3_raw"] / ev["A_pre"] for ev, *_ in rows]))
    pk_max = float(np.max([ev["os_pk3"] / ev["A_pre"] for ev, *_ in rows]))
    late = [ev["os_late"] / ev["A_pre"] for ev, *_ in rows]
    self_ = [ev["os_self"] / ev["A_pre"] for ev, *_ in rows]
    w("## 0. 结论摘要（先行）")
    w()
    w("1. **8/8 次加载都存在同一形态的瞬态「过充」**：显示在加载沿后 "
      f"{np.median([ev['t_ospk3'] for ev, *_ in rows]):.2f} s（中位）达到比同一时刻输入高 "
      f"{100*pk_med:+.2f}%（中位，占台阶幅度 A_e）的峰值，随后按指数回落到输入电平上方 "
      f"0~1% 的位置。逐事件峰值 {100*min([ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.2f}% ~ "
      f"{100*pk_max:+.2f}%（绝对值 {min(ev['os_pk3'] for ev, *_ in rows):.0f} ~ "
      f"{max(ev['os_pk3'] for ev, *_ in rows):.0f} ADC）。**这是实测，不是推断。**"
      "（主口径 = 先按同一 `elapsed` 时间戳分组取中位；单帧对照口径在记录流上一致到 "
      f"≤ {100*max(abs(ev['os_pk3']/ev['A_pre'] - ev['os_pk3_raw']/ev['A_pre']) for ev, *_ in rows):.2f} pp，"
      "见 §1.2/§2.2）")
    w("2. **主因是算法的「前沿超前」机制本身，不是形状 ROM 的标定误差**：显示被滑行器拉到 "
      "`base_y + Â`，而输入此刻只走到 `inc` ⇒ 显示比输入高出 `Â − inc`。实测过充量与 "
      f"`(Â−inc)/A_e @τ=0.80` 的相关系数 r={pearson(R['preds'], [ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.3f}，"
      f"残差中位 {100*np.median(R['resid']):+.2f} pp。而本批 8/8 个事件在 τ=0.80 处 "
      "`Â_ls > κ·inc`（κ 上限**全部**生效）⇒ `Â = 1.05·inc` ⇒ 前沿超前量 ≈ \'κ−1\' = **5%**。")
    w("3. **κ_onset 是唯一能直接压掉本工况瞬态过充的旋钮；ROM_SCALE 在本批数据上压不动**："
      "两者都只能通过 `κ·inc` 这个单侧上限起作用，而本批 8/8 事件 `Â_ls > κ·inc` ⇒ `Â = 1.05·inc`；"
      "把 κ 调到 1.00 可把瞬态过充中位压到 +0.07%（回放），代价是跟随时间 T95 从 2.09 s 拉到 4.37 s、"
      "上升期欠充 −0.03%→−3.27%·A_e、捕获比 G 1.085→1.025（见 §6.3 三角代价表）。")
    i_self = R["rows"][int(np.argmax(self_))][0]["i"]
    w("4. **用户看到的「冲高再回落」= 指标③（显示超过它自己的最终值）**：最大 "
      f"{100*max(self_):+.1f}%·A_e 出现在事件 #{i_self}（{evs[i_self]['hold']:.1f} s 的短促加载）；"
      "长保压事件（#0/#3/#4）该指标只有 0.15%~0.21%，因为它们自己的最终值已经被长时漂移抬走。"
      "⇒ **「冲高再回落」在短促加载上最明显，在长保压上表现为「先超前、再被输入反超」"
      f"（后者由指标①瞬态 {100*pk_med:+.2f}% 与 `OS_min3`/`OS_late` 体现）。**")
    w("5. **长时偏移（后段「越漂越远」）与瞬态过充是两件不同的事**：前者由基线漂移（需求问题 2）造成，"
      f"在本报告中单列为 `os_late`（范围 {100*min(late):+.1f}% ~ {100*max(late):+.1f}%），"
      "不并入瞬态过充指标。")
    w()

    # ── 1 数据与口径 ──
    w("## 1. 数据与口径")
    w()
    w("### 1.1 数据")
    w()
    w(_md_table(["项", "值"], [
        ["目录", "`" + R["data_rel"] + "`"],
        ["帧数 / 时长 / 帧率", f"{R['n']} / {R['span']:.1f} s / {R['fs']:.2f} Hz"],
        ["通道数 / 显示域", "21（8×5）/ `adc`（`force_conversion_active=false`）"],
        ["pre 流", "`device_001_pre_seg0.csv`（算法前读数；`##Data` 表头缺 21 个通道名，"
                   "按列位置解析，复用 `v20_lib.load_stream`）"],
        ["main 流", "`device_001_seg000.csv`（v6 算法输出）"],
        ["raw 流", "`device_001_raw_seg000.csv`（原始 ADC；本报告未使用）"],
        ["空载/受载电平（p3/p97）", f"{det['lo']:.0f} / {det['hi']:.0f} ADC"],
    ]))
    w()
    w("### 1.2 口径定义（全部显式，单位 = ADC，21 通道显示域求和）")
    w()
    w(_md_table(["符号", "定义"], [
        ["`total(t)`", "Σ over 21 channels of v_i(t)"],
        ["`off(t)`", "`total_main(t) − total_pre(t)`"],
        ["`t_cross`", "pre 总量 9 帧滑动均值自下而上穿过 (p3+p97)/2 的时刻（= 复现既有 probe-B/C 的 8 条沿）"],
        ["`t0`", "真加载沿：自 `t_cross` 向前回溯，最后一个仍处于「`t_cross` 前 1.2~0.6 s 电平 + 5%·A_est」的帧的下一帧"],
        ["`t_dn`", "`t_cross` 后第一个下降沿（> `t_cross`+0.5 s）"],
        ["`t_ulo`", "卸载起始：自 `t_dn` 向前最后一个仍处于受载电平（受载中位 −5%·幅度）的帧的下一帧"],
        ["`pre_base` / `main_base`", "[`t0`−1.2 s, `t0`] 的中位"],
        ["`t_p0` / `t_p1`", "`t0`+0.5·min(6 s, `t_ulo`−`t0`) / min(`t_dn`−0.6, `t_ulo`−0.3)；"
                            "两者间隔 < 0.8 s 时改用 [`t_ulo`−1.0, `t_ulo`−0.1] 并标 `short_plateau=1`"],
        ["`pre_ss` / `main_ss`", "[`t_p0`, `t_p1`] 的中位"],
        ["`A_pre` / `A_main`", "`pre_ss`−`pre_base` / `main_ss`−`main_base`（台阶幅度，两版都给）"],
        ["`OS(t)`", "(`main(t)`−`main_base`) − (`pre(t)`−`pre_base`)；>0 = 显示高于输入"],
        ["`t_seg`", "`plateau_segments` 的 idle→loaded 段边界（= 62.5% 电平穿越，口径上晚于 `t0`，仅作交叉校验）"],
        ["`G`", "(`main_fin`−`main_base`)/`A_pre`，`main_fin` = [`t_p1`−1 s, `t_p1`] 中位（台阶捕获比）"],
        ["`T_stable5/2`", "此后一直落在自身最终值 ±5%/±2%·`A_pre` 带内的最早时刻（相对 `t0`）；平台末端仍出带则为 nan"],
        ["`T95_in`", "首次达到「`pre_base` + 95%·`A_pre`」的时刻（相对 `t0`）；只用 `pre` 的台阶作参考，"
                     "不受长时漂移支配，是本批数据里唯一可用的「跟随速度」指标"],
        ["**分析流约定**", "全部分析（沿检测、台阶、输入形态、`f_obs`、`Â`、OS 指标）**默认**先把 `pre` 与输出"
                          "按同一 `elapsed` 时间戳分组取中位（`tsgm`），即「主口径」；"
                          "§2 另给一列**单帧对照**口径（不做分组中位）。理由：本录制每个时间戳压 2~4 帧，"
                          "且在加载沿处帧间离散可达 13000 ADC"],
    ]))
    w()
    w("### 1.3 三类「过充」指标（互不混用）")
    w()
    w(_md_table(["#", "指标", "定义", "含义"], [
        ["①", "`OS_pk3/A_pre`", "max_{τ∈[0,3 s]} OS / A_pre",
         "输入沿后 3 s 内的**瞬态过充**（本报告的主指标）"],
        ["①b", "`OS_pkall/A_pre`", "max_{τ∈[0,t_p1]} OS / A_pre",
         "含长时漂移的最大偏移（单列，不与①合并解读）"],
        ["②", "`OS_rel`", "max (main−main_base)/(pre−pre_base) − 1，仅在 `pre−pre_base ≥ 0.10·A_pre` 处取值",
         "过充量相对**输入当前已完成的增量**（上升过程中的相对超前）"],
        ["③", "`OS_self/A_pre`", "max_{τ∈[0,t_p1]} (main(t) − main_fin) / A_pre，`main_fin` = [`t_p1`−1, `t_p1`] 中位",
         "显示是否**超过它自己的最终值** ⇒ 用户看到的「冲到比稳态更高的位置再掉回来」"],
        ["—", "`OS_min3/A_pre`", "min_{τ∈[0,3 s]} OS / A_pre", "瞬态**欠充**（显示低于输入）"],
        ["—", "`os_pk3_raw` / `os_self_raw` / `os_min3_raw`", "与①③及欠充同式，但对**原始逐行流**（不分组取中位）计算",
         "**单帧对照口径**。本录制的 `pre` 在 t=2.237 s 的一个时间戳内压了 4 帧："
         "`17978 / 5054 / 11084 / 10356`（17978 是整段最大值、5054 近似整段最小值），"
         "任何单帧峰值指标都会被它污染；故主口径统一用时间戳组中位，单帧口径并报对照"],
    ]))
    w()
    w("### 1.4 证据等级标注（本报告全文遵守）")
    w()
    w(_md_table(["标记", "含义"], [
        ["**[实测-记录流]**", "直接由 `pre`/`main` 录制流算出，可逐帧复核"],
        ["**[解析-记录输入]**", "在录制输入上重算算法的中间量（`f_obs`、`Â` 等），不含闭环反馈，故与真实显示输出会有差异，差异已被 §5.4 单独量化"],
        ["**[回放-原型]**", "用归档原型 `glm53_v6.py` + 当前参数集回放 `pre` 流得到的**闭环**输出；与录制 `main` 的复现度见 §6.3"],
        ["**[机制推断]**", "由算法结构直接推得、本次数据未直接闭环验证的结论"],
    ]))
    w()

    # ── 2 台账 ──
    w("## 2. 逐事件过充台账")
    w()
    w("### 2.1 沿与台阶（[实测-记录流]）")
    w()
    tbl = []
    for ev, m, ft, rr, ah in rows:
        tbl.append([ev["i"], f"{ev['t_cross']:.2f}", f"{ev['t0']:.2f}", f"{ev['t_dn']:.1f}",
                    f"{ev['t_ulo']:.1f}", f"{ev['hold']:.1f}",
                    f"{ev['pre_base']:.0f}", f"{ev['pre_ss']:.0f}", f"{ev['A_pre']:.0f}",
                    f"{ev['main_base']:.0f}", f"{ev['main_ss']:.0f}", f"{ev['A_main']:.0f}"])
    w(_md_table(["i", "t_cross(s)", "t0(s)", "t_dn(s)", "t_ulo(s)", "hold(s)",
                 "pre_base", "pre_ss", "A_pre", "main_base", "main_ss", "A_main"], tbl))
    w()
    w("8 条加载沿的 `t_cross` 与需求给定的 "
      "2.27 / 120.67 / 142.03 / 149.41 / 219.08 / 292.74 / 308.07 / 318.49 s 完全一致"
      "（起始处 2.27 s 的重复沿已按 >1 s 去重）。")
    w()
    w("### 2.2 逐事件过充指标（[实测-记录流]）")
    w()
    tbl = []
    for ev, m, ft, rr, ah in rows:
        A = ev["A_pre"]
        tbl.append([ev["i"], f"{m['os_pk3']:.0f}", f"{100*m['os_pk3']/A:+.2f}%", f"{m['t_ospk3']:.2f}",
                    f"{100*ev['os_pk3_raw']/A:+.2f}%",
                    f"{m['os_min3']:.0f}", f"{100*m['os_min3']/A:+.2f}%",
                    f"{100*m['os_rel']:+.2f}%" if np.isfinite(m["os_rel"]) else "nan",
                    f"{m['os_late']:.0f}", f"{100*m['os_late']/A:+.2f}%",
                    f"{m['os_self']:.0f}", f"{100*m['os_self']/A:+.2f}%",
                    f"{100*ev['os_self_raw']/A:+.2f}%",
                    f"{m['off_absmax']:.0f}", f"{100*m['off_absmax']/A:+.2f}%",
                    f"{m['os_decay_10_90']:.2f}", f"{m['os_tau_exp']:.3f}",
                    f"{m['t_recover5']:.2f}", f"{m['T_stable5']:.2f}", f"{m['T95_in']:.2f}",
                    f"{m['G']:.3f}",
                    ev["class"] + ("（短平台）" if ev["short_plat"] else "")])
    w(_md_table(["i", "OS_pk3", "①%", "t_pk(s)", "①% (单帧对照)", "OS_min3", "欠充%", "②OS_rel",
                 "OS_late", "①b%", "OS_self", "③%", "③% (单帧对照)", "|off|max", "占A",
                 "90→10%(s)", "τ_exp(s)", "恢复(s)", "T_stable5(s)", "T95跟随(s)", "G", "形态"], tbl))
    w()
    w("读表要点（均为 [实测-记录流]）：")
    w()
    w("**先看「①% (单帧对照)」与「①%」的差别**：`pre` 流在同一个 `elapsed` 时间戳内会压 2~4 帧，"
      "加载沿处这些帧的离散度极大（t=2.237 s 的四帧 = 17978 / 5054 / 11084 / 10356 ADC）。"
      "主口径先取时间戳内中位，可去掉这类突发散点；单帧口径并报作对照。"
      f"本批 8 个事件两口径的差值 ≤ "
      f"{100*max(abs(ev['os_pk3']/ev['A_pre']-ev['os_pk3_raw']/ev['A_pre']) for ev, *_ in rows):.2f} pp "
      "⇒ **对记录流而言两者等价**（记录流本身对输入突发的响应是连续的），"
      "差异只在滤波/回放变体的极值上显现（见 §6.3 的两个最大列）。")
    w("- **瞬态过充（①）在 8/8 事件上都是正的**，范围 "
      f"{100*min(ev['os_pk3']/ev['A_pre'] for ev, *_ in rows):+.2f}% ~ "
      f"{100*max(ev['os_pk3']/ev['A_pre'] for ev, *_ in rows):+.2f}%（"
      f"{min(ev['os_pk3'] for ev, *_ in rows):.0f} ~ {max(ev['os_pk3'] for ev, *_ in rows):.0f} ADC），"
      f"峰值时刻 {min(ev['t_ospk3'] for ev, *_ in rows):.2f} ~ {max(ev['t_ospk3'] for ev, *_ in rows):.2f} s。")
    w(f"- 瞬态欠充（OS_min3）大部分事件为 0（显示始终不低于输入），最差 "
      f"{100*min(ev['os_min3']/ev['A_pre'] for ev, *_ in rows):+.2f}%。")
    w("- 90%→10% 衰减耗时 0.59~19.6 s、指数拟合时间常数 τ_exp 0.59~1.71 s（事件 #2 的 19.6 s 是"
      "因为其过充尾巴与卸载沿相连，拟合不可用，已由 `os_tau_r2` 列给出拟合质量）。")
    w(f"- 事件内最大 |off| 为 {max(abs(ev['off_absmax']) for ev, *_ in rows):.0f} ADC（"
      f"{100*max(abs(ev['off_absmax'])/ev['A_pre'] for ev, *_ in rows):.1f}%·A_e）；"
      f"其最大值出现在长保压事件（#0/#3/#4/#7），属长时漂移而非瞬态过充。")
    w()

    # ── 3 形态分类 ──
    from collections import Counter
    cnt = Counter(ev["class"] for ev, *_ in rows)
    w("## 3. 过充的形态分类")
    w()
    w("### 3.1 判据（显式）")
    w()
    w("以 `OS_pk3`（瞬态峰值）与 `OS_late`（平台末段中位偏移）的关系分类：")
    w()
    w(_md_table(["类别", "判据"], [
        ["瞬态尖峰型", "`OS_pk3 > 0` 且 `OS_late ≤ 0.30·OS_pk3` 且 `t_ospk3 ≤ 1.5 s`"],
        ["持续偏移型", "`OS_late ≥ 0.50·OS_pk3` 且 `OS_late > 0`"],
        ["混合型", "其余"],
    ]))
    w()
    w("### 3.2 结果")
    w()
    w("分类占比：" + "、".join(f"**{k} = {v}/{len(rows)}**" for k, v in cnt.items()) + "。")
    w()
    w(_md_table(["i", "OS_pk3/A", "t_pk(s)", "OS_late/A", "OS_late/OS_pk3", "类别", "备注"], [
        [ev["i"], f"{100*ev['os_pk3']/ev['A_pre']:+.2f}%", f"{ev['t_ospk3']:.2f}",
         f"{100*ev['os_late']/ev['A_pre']:+.2f}%",
         f"{ev['os_late']/ev['os_pk3']:.2f}" if ev["os_pk3"] > 0 else "nan",
         ev["class"],
         "长保压（%.0f s）；平台偏移受基线漂移支配" % ev["hold"] if ev["hold"] > 30 else
         ("短促（%.1f s）" % ev["hold"])]
        for ev, *_ in rows]))
    w()
    w("**回答需求中的形态问题**：本批录制里过充**同时**存在两种形态，但成因不同：")
    w()
    w("- 「沿后 0.3~1 s 一个尖峰随后回落」= 瞬态尖峰（8/8 事件都有，峰高 "
      f"{100*min(ev['os_pk3']/ev['A_pre'] for ev, *_ in rows):+.2f}%~"
      f"{100*max(ev['os_pk3']/ev['A_pre'] for ev, *_ in rows):+.2f}%·A_e，中位 "
      f"{100*pk_med:+.2f}%）；")
    w("- 「沿后长时间保持偏移」= 长时偏移，只在长保压事件上出现（"
      + "、".join(f"#{ev['i']} {100*ev['os_late']/ev['A_pre']:+.2f}%"
                  for ev, *_ in rows if ev["hold"] > 20)
      + "），且与 §4 的输入下冲、以及需求问题 2 的基线漂移强相关"
      f"（`pre 平台下冲/A` vs `OS_late/A`：r={pearson([R['feats'][ev['i']]['pre_dip_frac'] for ev, *_ in rows], late):+.3f}，"
      f"ρ={spearman([R['feats'][ev['i']]['pre_dip_frac'] for ev, *_ in rows], late):+.3f}）。")
    w()
    w("### 3.3 典型事件的 20 ms 轨迹（pre / main / off）")
    w()
    w("说明：录制帧间隔中位 0 ms、p95 45 ms（同一 `elapsed` 压 2~4 帧），因此 20 ms 桶中存在无帧的桶；"
      "表内只列出**有帧的桶**（按 `elapsed` 均值给出，均为时间戳组中位流），故行与行之间的时间间隔不等。")
    w()
    for tag, name in (("spike", "瞬态尖峰典型（OS_pk3 最大）"), ("sustain", "持续偏移典型（|OS_late| 最大）")):
        ei, lines = R["traces"][tag]
        ev = evs[ei]
        w(f"#### {name}：事件 #{ei}（`t0`={ev['t0']:.2f} s，`A_pre`={ev['A_pre']:.0f} ADC）")
        w()
        w(_md_table(["t−t0(s)", "pre", "main", "off = main−pre"],
                    [[f"{a:.2f}", f"{b:.0f}", f"{c:.0f}", f"{d:+.0f}"] for a, b, c, d in lines]))
        w()

    # ── 4 输入形态 ──
    w("## 4. 过充与输入形态的关系")
    w()
    w("### 4.1 逐事件输入形态特征（[实测-记录流]）")
    w()
    tbl = []
    for ev, *_ in rows:
        ft = R["feats"][ev["i"]]
        tbl.append([ev["i"], f"{ft['t10']-ev['t0']:.2f}", f"{ft['t90']-ev['t0']:.2f}",
                    f"{ft['rise']:.2f}" if np.isfinite(ft["rise"]) else "nan",
                    "是" if ft["rise_fast"] else "否",
                    f"{100*ft['pre_ov3']:+.2f}%", f"{100*ft['pre_dip_frac']:+.2f}%",
                    f"{ft['t_dip']:.1f}", f"{ft['pre_03_frac']:.3f}", f"{ft['pre_1s_frac']:.3f}"])
    w(_md_table(["i", "t10−t0(s)", "t90−t0(s)", "10→90%(s)", "≤0.3 s",
                 "pre 沿后3s最高点相对平台/A（负 = 3 s 时尚未到平台）",
                 "pre 平台下冲/A", "下冲时刻(s)", "pre@0.3s完成度", "pre@1.0s完成度"], tbl))
    w()
    w("### 4.2 相关性与分组（n=8，样本极小，只作方向性证据）")
    w()
    w(_md_table(["输入特征 X", "过充指标 Y", "Pearson r", "Spearman ρ"],
                [[xn, yn, f"{r:+.3f}", f"{rho:+.3f}"] for xn, yn, r, rho in R["corr"]]))
    w()
    for name, idxs in R["groups"]:
        w(f"- {name}：事件 {idxs}")
    w()
    w("### 4.3 对两个假设的检验")
    w()
    w("**假设 A「输入越快越容易过充」：成立，但机制是直接的、不是间接的。**")
    w()
    w("证据：`pre@1.0 s 完成度` vs `OS_pk3/A` 的 r="
      + f"{pearson([R['feats'][ev['i']]['pre_1s_frac'] for ev, *_ in rows], [ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.3f}"
      + "、ρ="
      + f"{spearman([R['feats'][ev['i']]['pre_1s_frac'] for ev, *_ in rows], [ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.3f}"
      + "（8 个事件中唯一接近 1 的相关）；`10→90% 耗时` vs `OS_pk3/A` 的 r="
      + f"{pearson([R['feats'][ev['i']]['rise'] for ev, *_ in rows], [ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.3f}"
      + "（负号 ⇒ 上升越快、过充越大）。机制见 §5.4：过充量 ≈ `(Â−inc(τ))/A_e`，"
      "输入越快 ⇒ `inc` 越接近 `Â` ⇒ 超前量越小？**不成立**——实测反而是"
      "「输入在 1 s 内走得越多，结果显示超过输入越多」，因为 κ 上限把所有事件的 `Â` 都钉在 "
      "`1.05·inc` 上（§5.4），于是超前量恒为 ~5%，而输入越快则这个 5% 越早显现、越像「冲高」。")
    w()
    w("**假设 B「输入中途回落导致过充」：只对「长时偏移」成立，对「瞬态过充」不成立。**")
    w()
    w("证据：`pre 平台下冲/A` vs `OS_late/A`：r="
      + f"{pearson([R['feats'][ev['i']]['pre_dip_frac'] for ev, *_ in rows], late):+.3f}"
      + "、ρ="
      + f"{spearman([R['feats'][ev['i']]['pre_dip_frac'] for ev, *_ in rows], late):+.3f}"
      + "（强正相关）；而对瞬态过充 `OS_pk3/A`：r="
      + f"{pearson([R['feats'][ev['i']]['pre_dip_frac'] for ev, *_ in rows], [ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.3f}"
      + "（弱负相关）。即：输入下冲会把「显示相对输入」的**长时**偏移推正（显示没跟着掉），"
      "但它并不制造沿后 1 s 的尖峰。")
    w()
    w("⚠️ n=8、且 8 个事件的输入形态只有 2~3 种（慢蠕变长保压 / 快速短促）⇒ "
      "上表只能作为方向性证据，不能当作定量模型。")
    w()

    # ── 5 ROM ──
    w("## 5. 过充与形状 ROM 的关系")
    w()
    w("### 5.1 实测归一化形状 `f_obs(τ)` vs 原型 ROM `g(τ)`（[解析-记录输入]）")
    w()
    w("`f_obs(τ) = (pre(t0+τ) − pre_base) / (pre_ss − pre_base)`（线性插值取 `pre(t0+τ)`），"
      "`g(τ)` 取 `" + R["proto_rel"] + "` 顶部的 14 点 ROM（`ROM_TAU`/`ROM_G`）。"
      "表内每格为 `f_obs / g`；`dt_eq > 0` ⇒ ROM 比实测**更快**（更早到达该比例）。")
    w()
    hdr = ["τ(s)"] + [f"#{ev['i']}" for ev, *_ in rows] + ["f_obs 中位", "ROM", "中位差", "中位 dt_eq(s)"]
    tbl = []
    for j, tau in enumerate(ROM_TAU_GRID):
        fo = [R["roms"][ev["i"]]["f_obs"][j] for ev, *_ in rows]
        diff = [R["roms"][ev["i"]]["diff"][j] for ev, *_ in rows]
        dts = [R["roms"][ev["i"]]["dt_eq"][j] for ev, *_ in rows]
        tbl.append([f"{tau:.2f}"] + [f"{R['roms'][ev['i']]['f_obs'][j]:.3f}" for ev, *_ in rows]
                   + [f"{np.median(fo):.3f}", f"{R['roms'][0]['rom'][j]:.3f}",
                      f"{np.median(diff):+.3f}", f"{np.nanmedian(dts):+.3f}"])
    w(_md_table(hdr, tbl))
    w()
    w("读数（[解析-记录输入]）：")
    w()
    w("- 慢蠕变事件（#0/#3/#4）：`f_obs(0.20)`≈0.62~0.63，ROM 0.79 ⇒ **ROM 比实测快 "
      f"{0.79-0.62:.2f}（约 27% 相对）**，但 `dt_eq` 只有 0.10~0.16 s 量级（因为 τ=0.5 之后两者接近）。")
    w("- 快速事件（#5/#6/#7）：`f_obs(1.0)`≈0.94~0.97 ≈ ROM 0.904 ⇒ 实测略慢于 ROM（`dt_eq` 0.78 s "
      "是因为 dt_eq 在尾部被 (ROM 1.0 处的 0.904 与实测饱和 1.0) 的差异放大，读数时应以表内逐点差为准）。")
    w("- `f_obs` 逐点散布很大（全网格 "
      f"{min(min(R['roms'][ev['i']]['f_obs']) for ev, *_ in rows):.2f}~"
      f"{max(max(R['roms'][ev['i']]['f_obs']) for ev, *_ in rows):.2f}），"
      "**本录制内部的形状一致性远低于 13 份标定集**（该集合内同传感器可复现到 4.5~6.3 pt），"
      "这正是 `Â_ls` 在部分事件上越过 κ 上限、在另一些事件上落到 κ·inc 之下的原因。")
    w()
    w("### 5.2 逐事件 Â 估计（原型 `_inv_est` 等价实现，κ=1.05）（[解析-记录输入]）")
    w()
    w("窗 `[0.20, min(τ, 0.80)]`、≥5 样本、`inc` 取窗内最后一个样本（与原型 `yy[-1]` 一致）；"
      "`Â = min(max(Â_ls, inc), κ·inc)`。")
    w()
    tbl = []
    for ev, *_ in rows:
        A = ev["A_pre"]
        for tau in TAU_EVAL:
            r = R["ahats"][ev["i"]]["tau_eval"][tau]
            tbl.append([ev["i"], f"{tau:.2f}"] + (["—"] * 4 + [0] if r is None else
                       [f"{r['A_ls']/A:.3f}", f"{KAPPA_PRIMARY*r['inc']/A:.3f}",
                        f"{r['A_hat']/A:.3f}", f"{r['inc']/A:.3f}", r["n"]]))
    w(_md_table(["i", "τ_eval(s)", "Â_ls/A_e", "κ·inc/A_e", "Â/A_e", "inc/A_e", "样本数"], tbl))
    w()
    w("### 5.3 过估 / 欠估判定（τ=0.80）")
    w()
    tbl = []
    for tau in (0.50, 0.80, 1.00):
        rs = [R["ahats"][ev["i"]]["tau_eval"][tau] for ev, *_ in rows]
        vals = [r["A_hat"] / rows[i][0]["A_pre"] for i, r in enumerate(rs) if r is not None]
        ls = [r["A_ls"] / rows[i][0]["A_pre"] for i, r in enumerate(rs) if r is not None]
        ci = [KAPPA_PRIMARY * r["inc"] / rows[i][0]["A_pre"] for i, r in enumerate(rs) if r is not None]
        tbl.append([f"{tau:.2f}", f"{np.median(vals):.3f}",
                    f"[{min(vals):.3f}, {max(vals):.3f}]", f"{np.median(ls):.3f}",
                    f"{np.median(ci):.3f}", f"{sum(1 for v in vals if v > 1.0)}/{len(vals)}",
                    f"{sum(1 for i, r in enumerate(rs) if r is not None and r['A_ls'] > KAPPA_PRIMARY*r['inc'])}/{len(vals)}"])
    w(_md_table(["τ_eval(s)", "Â/A_e 中位", "Â/A_e 范围", "Â_ls/A_e 中位", "κ·inc/A_e 中位",
                 "过估事件数(Â>1)", "κ 生效事件数(A_ls>κ·inc)"], tbl))
    w()
    w("结论：[解析-记录输入]")
    w()
    w("- `Â` 作为**台阶幅度**估计并不过估：τ=0.80 时 Â/A_e 中位 "
      f"{np.median([R['ahats'][ev['i']]['tau_eval'][0.80]['A_hat']/ev['A_pre'] for ev, *_ in rows]):.3f}，"
      "只有 1/8 事件超过 1。**过充不能归因于「Â 把台阶算大了」。**")
    w("- 但 `Â_ls`（未加 κ 限幅的最小二乘解）在 8/8 事件、τ=0.80 处都大于 `κ·inc` ⇒ "
      "**κ=1.05 在本批数据上是恒生效的单侧上限**，`Â = 1.05·inc`；")
    w("- 反向地，κ 上限对**慢输入**事件（#0/#3/#4）把 `Â` 从 `Â_ls`（0.756~0.821）压到 `κ·inc`"
      "（0.699~0.766），即相对其自身的 LS 解是**欠估**，这正是这些事件后期显示低于输入"
      "（`OS_late` 为负 / `OS_min3` 接近 0）的原因之一。")
    w()
    w("### 5.4 机制律：瞬态过充 ≈ `(Â − inc)/A_e`（[机制推断] + [实测-记录流] 交叉核对）")
    w()
    w("由 §5.1 的滑行器关系：`out = v + share·c`，`c_target = c0·(1−w) + (target − total)·w`，"
      "`target = base_y + Â`。当 τ≈0.8~1.0 s 时 `w≈1`，于是 `out ≈ total + (Â − inc)`；"
      "同时 `OS(t) = out − total`（本报告口径），故 `OS_pk3/A_e ≈ (Â(0.80) − inc(0.80))/A_e`。")
    w()
    tbl = []
    for ev, m, ft, rr, ah in rows:
        tbl.append([ev["i"], f"{100*ev['os_pk3']/ev['A_pre']:+.2f}%",
                    f"{100*ev['pred_pk3']:+.2f}%",
                    f"{100*(ev['os_pk3']/ev['A_pre']-ev['pred_pk3']):+.2f}pp"])
    w(_md_table(["i", "实测 OS_pk3/A（[实测-记录流]）", "预测 (Â−inc)/A @τ=0.80（[解析]）", "残差"], tbl))
    w()
    w(f"预测 vs 实测：r={pearson(R['preds'], [ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.3f}，"
      f"残差中位 {100*np.median(R['resid']):+.2f} pp，最大 |残差| "
      f"{100*max(abs(x) for x in R['resid']):.2f} pp（事件 "
      f"{rows[int(np.argmax([abs(x) for x in R['resid']]))][0]['i']}）。"
      "⇒ **瞬态过充量可以只用 `Â − inc` 解释**，不需要引入额外的形状误差项；"
      "而 κ 生效率 8/8 ⇒ `Â − inc = (κ−1)·inc`，故 **κ_onset 直接决定瞬态过充的百分比上限**。")
    w()
    w("### 5.5 ROM_SCALE 与本批数据的关系")
    w()
    w("`ROM_SCALE`（v6.1 的 F1，形状上包络重标 `g61 = min(1, scale·g)`，见 "
      "`plan/v1.0/T5_v6失效模式_过充与基线识别/scripts/t5a_glm53_v61.py`）**只通过 `Â_ls` 起作用**。"
      "本批数据 `Â_ls > κ·inc` 在 8/8 事件成立 ⇒ 只要重标后 `Â_ls` 仍高于 `κ·inc`，"
      "`Â` 完全不变、**对过充零效果**；只有当重标把 `Â_ls` 压到 `κ·inc` 以下时才生效，"
      "此时它与「把 κ 调小」在效果上等价（都只是压低同一个上限）。逐事件阈值见 §6.2。")
    w()

    # ── 6 候选 ──
    w("## 6. 可优化方向与量化收益")
    w()
    w("### 6.1 候选清单")
    w()
    w(_md_table(["#", "候选", "验证方式", "证据等级"], [
        ["①", "形状 ROM 上包络重标 `ROM_SCALE`（1.04 / 1.06 / 1.10）",
         "解析层（Â 重算）+ 原型回放闭环", "解析 + 回放"],
        ["②", "`κ_onset` 扫描 1.00 / 1.10 / 1.20 / 1.30（现值 1.05）", "解析层 + 原型回放闭环", "解析 + 回放"],
        ["③", "输出端 EMA / 中值滤波 τ = 0.1 / 0.3 / 0.5 s", "对 v6 回放输出做事后滤波", "回放（后处理）"],
        ["④", "不做输出滤波、只降 κ 到 1.00~1.02（= 取消前沿超前）", "原型回放闭环", "回放"],
    ]))
    w()
    if R["no_replay"]:
        w("> 本次运行使用了 `--no-replay`，§6.2~§6.4 的回放数据缺失。")
        w()
    else:
        w("### 6.2 解析层：κ 与 ROM_SCALE 对 `Â/A_e` 的作用（τ=1.00）（[解析-记录输入]）")
        w()
        tbl = []
        for kind, v, med, mx in R["sweep_tbl"]:
            tbl.append(["κ_onset" if kind == "kappa" else "ROM_SCALE", f"{v:.2f}",
                        f"{med:.3f}", f"{mx:.3f}",
                        "（基准）" if (kind == "kappa" and v == KAPPA_PRIMARY) else ""])
        w(_md_table(["旋钮", "取值", "Â/A_e 中位", "Â/A_e 最大", "备注"], tbl))
        w()
        w("读法：`ROM_SCALE ≥ 1.04` 时 `Â/A_e` 中位几乎不动 ⇒ 在本批数据上 **ROM_SCALE 对过充基本无效**；"
          "`κ` 从 1.05 降到 1.00 时 `Â/A_e` 中位从 "
          f"{[t for t in R['sweep_tbl'] if t[0] == 'kappa' and t[1] == KAPPA_PRIMARY][0][2]:.3f} 降到 "
          f"{[t for t in R['sweep_tbl'] if t[0] == 'kappa' and t[1] == 1.00][0][2]:.3f}（≈ `inc/A_e`，即完全取消超前量）。")
        w()
        w("### 6.3 原型回放闭环：三角代价表（[回放-原型]）")
        w()
        if fid is not None:
            w(f"**原型复现度**（先验证回放可信）：v6_now 回放 vs 录制 `main` 的 RMS 差 "
              f"**{fid['rms']:.0f} ADC**、中位 |差| **{fid['mad']:.0f} ADC**；"
              f"整段电平 RMS（`main` 的 std）为 "
              f"{float(np.sqrt(np.mean((tm-tm.mean())**2))):.0f} ADC ⇒ 相对 RMS "
              f"**{100*fid['rms']/float(np.sqrt(np.mean((tm-tm.mean())**2))):.1f}%**；"
              f"台阶幅度 A_e 约 {np.median([ev['A_pre'] for ev, *_ in rows]):.0f} ADC。")
            w()
            w("⚠️ 复现度说明：回放是**原型 + 当前参数集**（κ/HO_MIN/撤销迟滞已按 C++ 参数集设置），"
              "不是 C++ 本体；两者之间的残留差异（平滑器细节、浮点/帧序）会让绝对量有偏差，"
              "因此下表的**读数只用于同源 A/B 比较**（同一回放器、同一数据、只改一个旋钮），"
              "不用于预测 C++ 的绝对输出。")
            w()
        tb = []
        for tag in order:
            a = agg[tag]
            tb.append([tag, f"{100*a['pk_med']:+.2f}%", f"{100*a['pk_max']:+.2f}%",
                       f"{100*a['pk_raw_med']:+.2f}%", f"{100*a['pk_raw_max']:+.2f}%",
                       f"{100*a['self_max']:+.2f}%", f"{100*a['sf_raw_max']:+.2f}%",
                       f"{100*a['min3_med']:+.2f}%",
                       f"{a['ts5_med']:.2f}", f"{a['t95_med']:.2f}", f"{a['t95_max']:.2f}",
                       f"{a['G_med']:.3f}"])
        w(_md_table(["变体", "①中位(主口径)", "①最大(主口径)", "①中位(单帧)",
                     "①最大(单帧)", "③最大(主口径)", "③最大(单帧)", "欠充 中位",
                     "T_stable5 中位(s)", "T95 跟随 中位(s)", "T95 跟随 最大(s)", "G 中位"], tb))
        w()
        w("⚠️ **两个速度指标必须并读**：本批数据的 `T_stable5` 中位 ~13 s 是被需求问题 2 的"
          "长时漂移支配的（8 个事件里有 3 个平台末端仍在 ±5%·A_e 带外 ⇒ nan），对沿后 1 s 的"
          "瞬态改动几乎不敏感；`T95 跟随`（首次达到沿前电平 +95%·A_pre 的时间，只用 pre 的台阶作参考）"
          "才是「加载跟得多快」的指标。")
        w()
        if R["reps"] is not None and "ema_0.30" in R["reps"]:
            i0 = int(np.argmax(agg["ema_0.10"]["pk_raw"])) if "ema_0.10" in agg else 0
            w("⚠️ **滤波变体的「①最大(单帧)」列不可直接读**：`ema_0.10`/`med_0.30` 的两个最大列"
              f"来自事件 #{i0} —— 该事件加载沿（t=2.237 s）的同一时间戳内压了 4 帧 "
              "`17978 / 5054 / 11084 / 10356`，滤波后的显示还停在 8772 一带，"
              "于是**单帧对照口径**算出 "
              f"{100*agg['ema_0.10']['pk_raw'][i0]:+.2f}%·A_e 的假过充；同一事件的**主口径**"
              f"只有 {100*agg['ema_0.10']['pk'][i0]:+.2f}%·A_e。"
              "⇒ 滤波变体的过充改善量应以「①中位/最大(主口径)」两列判断，"
              "单帧列只用于看最坏情形，详见 §6.4 的第 3 条代价。")
            w()
        w("三角代价表读法（相对 `v6_now` 基准行）：")
        w()
        base = agg["v6_now"]
        for tag in order:
            if tag in ("recorded", "v6_now"):
                continue
            a = agg[tag]
            w(f"- `{tag}`：过充峰（主口径）中位 {100*(a['pk_med']-base['pk_med']):+.2f} pp、"
              f"最大 {100*(a['pk_max']-base['pk_max']):+.2f} pp；"
              f"单帧对照口径中位 {100*(a['pk_raw_med']-base['pk_raw_med']):+.2f} pp、"
              f"最大 {100*(a['pk_raw_max']-base['pk_raw_max']):+.2f} pp；"
              f"T95 跟随中位 {a['t95_med']-base['t95_med']:+.2f} s；"
              f"T_stable5 中位 {a['ts5_med']-base['ts5_med']:+.2f} s；"
              f"捕获比 G 中位 {a['G_med']-base['G_med']:+.3f}；"
              f"欠充（OS_min3 中位）{100*(a['min3_med']-base['min3_med']):+.2f} pp。")
        w()
        if R["reps"] is not None:
            w("回放事件统计：" + "；".join(
                f"{t}: epochs={r['epochs']}, revokes={r['revokes']}, c5={r['c5']}"
                for t, r in R["reps"].items() if "epochs" in r))
            w()
        w("### 6.4 哪些是本次数据上直接可测的、哪些只是机制推断")
        w()
        w(_md_table(["结论", "可测性"], [
            ["瞬态过充的幅度/时刻/衰减（8 事件台账）", "**直接可测**：[实测-记录流]"],
            ["过充 ≈ (Â−inc)/A_e 的定量关系", "**直接可测**：[解析-记录输入] 与 [实测-记录流] 交叉核对，r/残差见 §5.4；"
                                                     "闭环（显示反馈）下未独立验证"],
            ["κ 生效性（8/8 事件）与 κ 对 Â 的作用", "**直接可测**：[解析-记录输入]（Â 只是输入的函数）"],
            ["ROM_SCALE 在本批数据上对 Â 无效", "**直接可测（解析层）**；闭环下的最终显示影响由 §6.3 回放给出"],
            ["κ / ROM_SCALE / 滤波对显示与 T_stable / G 的实际影响", "[回放-原型] 闭环实测，但**不是 C++ 本体**"],
            ["输出滤波会把「输入快速下冲」放大成显示偏高（事件 #2 实测 主口径 "
             + (f"{100*agg['ema_0.30']['pk'][2]:+.1f}%·A_e" if "ema_0.30" in agg else "n/a")
             + "，未滤波为 "
             + (f"{100*agg['v6_now']['pk'][2]:+.1f}%·A_e" if "v6_now" in agg else "n/a")
             + "）", "**直接可测**：[回放-原型] + [实测-记录流]（`pre` 自身在该处确有 40 ms 内 "
               "15405→13504 的下冲，是真实输入而非掉点）"],
            ["「κ=1.00 会带来多少欠充/滞后」的绝对量", "**机制推断 + 回放**：κ=1.00 时 Â=inc ⇒ 显示 = 输入（无预测），"
                                                        "因此 G 与 T_stable 会退化到「纯跟随」水平；绝对量需 C++ 实测确认"],
            ["对其它录制/器件/加载方式的普适性", "**仅机制推断**：本报告只用 1 份录制、8 个事件，n 极小"],
        ]))
        w()
    w("### 6.5 `08-v6.1` 产物的可获得性说明")
    w()
    w("- `temp/v4.1flash/progress/archived/08-v6.1/` **目录存在但为空**"
      "（`Get-ChildItem -Force -Recurse` 返回 0 项；`temp/v4.1flash/archived/` 不存在）⇒ "
      "需求里提到的「08-v6.1 产物」无法直接引用。")
    w("- 因此 `ROM_SCALE` 的实现口径取自现存可运行的原型："
      "`temp/v4.1flash/plan/v1.0/T5_v6失效模式_过充与基线识别/scripts/t5a_glm53_v61.py`"
      "（`ROM_SCALE=1.06`、`CONSERVATIVE_ONSET_ONLY=True`、`g61 = min(1, scale·g)`），"
      "本报告的回放变体与该文件逐行等价（只把常数参数化）。")
    w()

    # ── 7 结论 ──
    w("## 7. 结论与建议")
    w()
    w("### 7.1 过充主因（按贡献排序，各附证据）")
    w()
    w("**主因 1｜κ_onset=1.05 决定的「前沿超前」上限（贡献 ≈ 全部瞬态过充）[实测-记录流 + 解析]**")
    w()
    w(f"- 8/8 事件 τ=0.80 处 `Â_ls > κ·inc` ⇒ `Â = 1.05·inc` ⇒ 显示被拉到 `base_y+Â`，"
      f"而输入只走到 `inc` ⇒ 显示高出输入 `(κ−1)·inc ≈ 4.5~5%·A_e`（实测 OS_pk3 中位 {100*pk_med:+.2f}%）。")
    w("- 该关系逐事件预测残差中位 "
      f"{100*np.median(R['resid']):+.2f} pp ⇒ 瞬态过充不需要用「形状标定差」来解释。")
    w()
    w("**主因 2｜输入形态（快速加载）决定超前量的可见形态与峰值时刻（贡献：形态/可见性）[实测-记录流]**")
    w()
    w("- `pre@1.0 s 完成度` vs `OS_pk3/A`：r="
      + f"{pearson([R['feats'][ev['i']]['pre_1s_frac'] for ev, *_ in rows], [ev['os_pk3']/ev['A_pre'] for ev, *_ in rows]):+.3f}"
      + "（n=8）；快速事件（#2/#5/#6/#7，10→90% ≤0.6 s）的峰值出现时刻 "
      f"{min(ev['t_ospk3'] for ev in [rows[i][0] for i in R['groups'][0][1]]):.2f}~"
      f"{max(ev['t_ospk3'] for ev in [rows[i][0] for i in R['groups'][0][1]]):.2f} s，"
      "且「显示超过自身最终值」（③）最大 " + f"{100*max(self_):+.1f}% 就出现在这些事件上。")
    w()
    w("**主因 3｜形状 ROM 与本次输入形状的失配（贡献：慢输入方向的欠充，不是过充）[解析-记录输入]**")
    w()
    w("- 慢蠕变事件 `f_obs(0.20)≈0.62` vs ROM 0.79 ⇒ ROM 假设的上升更快；但 κ 上限把 `Â` 压在 `κ·inc`，"
      "形状误差进不到显示 ⇒ 这些事件的表现为**显示跟不上输入**（`OS_late` 为负、`OS_min3` 接近 0），"
      "即形状失配在本批数据上表现为**欠充而非过充**。")
    w()
    w("**主因 4｜长时偏移（与需求问题 2 同源，不是「加载过充」）[实测-记录流]**")
    w()
    w("- `OS_late` 在长保压事件上 +3.5%~+9.5%·A_e，与 `pre 平台下冲` 强相关（ρ="
      + f"{spearman([R['feats'][ev['i']]['pre_dip_frac'] for ev, *_ in rows], late):+.3f}"
      + "）；它在时间尺度上是几十秒的斜坡，与 1 s 尺度的瞬态过充可分离。")
    w()
    w("### 7.2 能优化到什么程度、代价是什么")
    w()
    if not R["no_replay"]:
        base = agg["v6_now"]
        cand = [t for t in order if t != "recorded"]
        best_pk = min((agg[t]["pk_med"], t) for t in cand)
        w(f"- **上限（本数据、回放口径，同时间戳组中位指标）**：把 `κ_onset` 降到 1.00（= 取消前沿超前/预测）"
          f"可把瞬态过充中位从 {100*base['pk_med']:+.2f}% 压到 {100*best_pk[0]:+.2f}%（变体 `{best_pk[1]}`），"
          f"但代价是三条：① 台阶捕获比 G 中位从 {base['G_med']:.3f} 变为 {agg[best_pk[1]]['G_med']:.3f}；"
          f"② 跟随时间 T95 中位从 {base['t95_med']:.2f} s 拉长到 {agg[best_pk[1]]['t95_med']:.2f} s"
          f"（最大 {base['t95_max']:.2f} s → {agg[best_pk[1]]['t95_max']:.2f} s）；"
          f"③ 上升期欠充增加（OS_min3 中位 {100*base['min3_med']:+.2f}% → "
          f"{100*agg[best_pk[1]]['min3_med']:+.2f}%·A_e）。")
        w("- **κ 的单调性（直接可测）**：κ=1.05→1.10→1.20→1.30 使瞬态过充中位单调恶化到 "
          f"{100*agg['kappa_1.10']['pk_med']:+.2f}% / {100*agg['kappa_1.20']['pk_med']:+.2f}% / "
          f"{100*agg['kappa_1.30']['pk_med']:+.2f}%；而 κ=1.10 已把 G 抬到 {agg['kappa_1.10']['G_med']:.3f}、"
          f"κ≥1.20 抬到 {agg['kappa_1.20']['G_med']:.3f}（>1 = 最终值超过真值台阶）⇒ "
          "**κ=1.05 是「过充/捕获」折中点附近，继续调大只有坏处，调小的代价见上一行。**")
        w("- **折中（推荐先试）**：`κ_onset` 取 1.02~1.03（本次未扫，属机制外推：按主因 1 的关系，"
          "过充量 ≈ (κ−1)·inc ⇒ 相对 1.05 可砍掉 40%~60%，而 G/T95 的代价按同比例缩小）。"
          "**这一档必须由 C++ 侧实测确认，本报告未测。**")
        w("- **ROM_SCALE 不建议作为本工况的过充修复手段**：本批 8/8 事件 κ 恒生效（§5.3），"
          f"`ROM_SCALE=1.04/1.06` 在回放里对过充（主口径 "
          f"{100*agg['romscale_1.06']['pk_med']:+.2f}%）与 G（{agg['romscale_1.06']['G_med']:.3f}）"
          f"基本无影响，1.10 才开始动（{100*agg['romscale_1.10']['pk_med']:+.2f}%、G "
          f"{agg['romscale_1.10']['G_med']:.3f}）——即它只是「等效地压低同一个上限」；"
          "而在形状库需要跨器件/跨加载方式重标时它仍有独立价值。")
        w("- **输出端滤波（EMA τ=0.3/0.5 s）是「视觉最直接」的一档，代价有三条**："
          f"① 跟随变慢：T95 中位 {base['t95_med']:.2f} s → "
          f"{agg['ema_0.30']['t95_med']:.2f} s / {agg['ema_0.50']['t95_med']:.2f} s（"
          f"{agg['ema_0.30']['t95_med']-base['t95_med']:+.2f} / "
          f"{agg['ema_0.50']['t95_med']-base['t95_med']:+.2f} s）；"
          f"② 加载期显示落后于输入：OS_min3 中位 {100*base['min3_med']:+.2f}% → "
          f"{100*agg['ema_0.30']['min3_med']:+.2f}% / {100*agg['ema_0.50']['min3_med']:+.2f}%·A_e；"
          f"③ **输入快速下冲时显示会「挂高」**：事件 #2 实测 主口径下的过充从 "
          f"{100*agg['v6_now']['pk'][2]:+.2f}% 变成 {100*agg['ema_0.30']['pk'][2]:+.2f}%（EMA 0.3 s）、"
          f"{100*agg['ema_0.50']['pk'][2]:+.2f}%（EMA 0.5 s）——滤波把输入 40 ms 内 "
          f"15405→13504 的真实下冲抹平，显示留在高处。中值滤波（`med_0.30`）三项都更差，不建议。")
        w(f"- **「冲高再回落」的观感指标（③）**：EMA 0.3 s 把 ③ 的最大值从 "
          f"{100*base['self_max']:+.2f}% 压到 {100*agg['ema_0.30']['self_max']:+.2f}%·A_e（主口径；"
          f"单帧口径 {100*base['sf_raw_max']:+.2f}% → {100*agg['ema_0.30']['sf_raw_max']:+.2f}%）⇒ "
          "如果用户的诉求是「不要在加载后看到冲高」，滤波最有效；如果诉求是「显示必须紧跟真实加载」，"
          "则应改 κ（降 κ 不引入滞后，只牺牲预测量）。**这两种诉求对应两个不同的旋钮，不能混用。**")
    w()
    w("### 7.3 未决问题（本报告未能回答，需后续任务）")
    w()
    w("1. κ_onset ∈ (1.00, 1.05) 的细扫（1.01/1.02/1.03）未做：需要 C++ 侧或原型在**多份**录制上做，"
      "单份数据的 n=8 不足以裁决折中点。")
    w("2. 本报告的回放是原型而非 C++；C++ 是否逐帧等价（撤销迟滞、Handoff 后的慢相）未在本任务验证"
      "（属禁止事项：不得构建/运行 exe）。")
    w("3. 「过充是否被用户判为不可接受」的**判据阈值**未定义（用户未给出可接受的 %·A_e 上限）；"
      "本报告只给量化，不做「是否合格」的裁决。")
    w("4. `os_late`（长时偏移）的归因属于需求问题 2（基线误估），本报告只给出它与 `pre 平台下冲` 的相关性，"
      "未做状态机级归因。")
    w("5. 事件 #2 的平台窗仅 "
      + f"{evs[2]['t_p1']-evs[2]['t_p0']:.2f} s（8 个事件里最短；未触发 `short_plateau`，但平台窗"
      + "直接贴着卸载沿）⇒ 它的 `OS_self`（主口径 "
      + f"{100*evs[2]['os_self']/evs[2]['A_pre']:+.2f}%、单帧 "
      + f"{100*evs[2]['os_self_raw']/evs[2]['A_pre']:+.2f}%）、"
      + f"`T_stable5`（{evs[2]['T_stable5']:.2f} s）参考值不稳健，"
      + f"读表时应以 `OS_pk3`（{100*evs[2]['os_pk3']/evs[2]['A_pre']:+.2f}%）为准。")
    w()

    # ── 8 复现 ──
    w("## 8. 复现（命令与产物）")
    w()
    w("### 8.1 命令（全部在仓库根执行）")
    w()
    w("```powershell")
    w("$env:PYTHONIOENCODING='utf-8'; python " + R["self_rel"])
    w("```")
    w()
    w("快速出表（跳过原型回放）：")
    w()
    w("```powershell")
    w("$env:PYTHONIOENCODING='utf-8'; python " + R["self_rel"] + " --no-replay")
    w("```")
    w()
    w("### 8.2 只读读取过的既有文件（未修改）")
    w()
    w("| 文件 | 用途 |")
    w("|---|---|")
    w("| `" + R["lib_rel"] + "` | `load_dataset` / `load_stream` / `plateau_segments` / `edges_from_tot` |")
    w("| `" + R["proto_rel"] + "` | ROM 14 点（`ROM_TAU`/`ROM_G`）、`GLM53v6`（回放） |")
    w("| `temp/v4.1flash/plan/v1.0/T5_v6失效模式_过充与基线识别/scripts/t5a_glm53_v61.py` | `ROM_SCALE` 口径参考 |")
    w("| `temp/v4.1flash/progress/currentworking/01_v6算法说明_当前实现.md` | 算法口径（§4/§5.1/§5.2） |")
    w("| `temp/v4.1flash/plan/v2.0/需求文档.md` | 任务背景 |")
    w("| `" + R["data_rel"] + "/*.csv` | 三条流数据 |")
    w()
    w("### 8.3 产物")
    w()
    w(_md_table(["路径", "内容"], [
        [f"`{R['out_dir']}/分析报告_B1.md`", "本报告"],
        [f"`{R['out_dir']}/b1_events.csv`", "逐事件台账（沿、台阶、三类过充指标、衰减/恢复/到带、G、形态、前沿超前律残差）"],
        [f"`{R['out_dir']}/b1_traces.csv`", "2 个典型事件 0~2 s 的 20 ms 分辨率 pre/main/off 轨迹"],
        [f"`{R['out_dir']}/b1_rom.csv`", "逐事件 `f_obs` vs ROM 逐点对照（含 `dt_eq`）"],
        [f"`{R['out_dir']}/b1_ahat.csv`", "逐事件 `Â_ls` / `κ·inc` / `Â`（τ_eval = 0.30/0.50/0.80/1.00）"],
        [f"`{R['out_dir']}/b1_candidates.csv`", "候选变体的三角代价表（回放口径；含时间戳组中位（主口径）与单帧两套列）"],
        [f"`{R['out_dir']}/b1_candidates_per_event.csv`", "逐事件 × 变体的 OS_pk3/A 矩阵（用于定位变体差异来自哪次加载）"],
    ]))
    w()

    path = os.path.join(OUT, "分析报告_B1.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L_) + "\n")
    print(f"wrote results/分析报告_B1.md（{len(L_)} 行）")


if __name__ == "__main__":
    main()
