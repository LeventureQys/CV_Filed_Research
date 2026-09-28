# -*- coding: utf-8 -*-
"""T1 形状研究共享层。

职责：
  1. 会话发现与 CSV 装载（pre = 算法输入 v；seg = 实机显示；raw = 原始 ADC）；
  2. 从通道总量里切出加载/卸载台阶（沿区间 [i0,i1]）；
  3. 逐通道快相窗口切分 + 单/拉伸/双指数形状拟合；
  4. 观测器回放（与 v34_observer_core3 逐帧一致）并导出 x1/x2/显示。

数据根不写死会话路径：只认「目录里有 session.json，且有 device_001_pre_seg0.csv
或 device_001_seg000.csv」。
"""
import os

import numpy as np
from scipy.ndimage import median_filter

NCH = 21
CH_NAME = ["ch%d" % i for i in range(NCH)]


# ---------------------------------------------------------------- 路径

def _find_root(start):
    p = os.path.abspath(start)
    for _ in range(16):
        if os.path.isfile(os.path.join(p, "AGENTS.md")):
            return p
        q = os.path.dirname(p)
        if q == p:
            break
        p = q
    raise RuntimeError("repo root not found from %s" % start)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _find_root(HERE)
DATA_ROOT = os.path.join(ROOT, "Document", "Update", "Dev-Version",
                         "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据")
WORKING = os.path.join(DATA_ROOT, "working")
ARCHIVED = os.path.join(DATA_ROOT, "archived")
OUT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(OUT_ROOT, "results")
FIGURE = os.path.join(OUT_ROOT, "figure")


# ---------------------------------------------------------------- 装载

def read_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\r\n") for r in fh if r.strip()]
    di = rows.index("##Data")
    cfg = {}
    for r in rows[1:di]:
        if "," in r:
            k, v = r.split(",", 1)
            cfg[k] = v
    return cfg, rows[di + 2:]


def load_stream(ds_dir, name):
    cfg, body = read_rows(os.path.join(ds_dir, name))
    n = len(body)
    arr = np.empty((n, 3 + NCH))
    for i, r in enumerate(body):
        f = r.split(",")
        arr[i] = [float(x) for x in f[:3 + NCH]]
    return dict(ts=arr[:, 0], el=arr[:, 1], fr=arr[:, 2].astype(np.int64),
                V=arr[:, 3:3 + NCH], n=n, cfg=cfg, name=name)


def discover_sessions(root=None, max_depth=5):
    root = root or WORKING
    out = []

    def has(d):
        if not os.path.isfile(os.path.join(d, "session.json")):
            return False
        return (os.path.isfile(os.path.join(d, "device_001_pre_seg0.csv"))
                or os.path.isfile(os.path.join(d, "device_001_seg000.csv")))

    def walk(d, depth):
        if has(d):
            out.append((os.path.relpath(d, root).replace("\\", "/"), d))
            return
        if depth <= 0:
            return
        try:
            subs = sorted(os.listdir(d))
        except OSError:
            return
        for s in subs:
            q = os.path.join(d, s)
            if os.path.isdir(q):
                walk(q, depth - 1)

    walk(root, max_depth)
    return out


def load_session(ds_dir):
    pre = load_stream(ds_dir, "device_001_pre_seg0.csv")
    rec = None
    if os.path.isfile(os.path.join(ds_dir, "device_001_seg000.csv")):
        rec = load_stream(ds_dir, "device_001_seg000.csv")
    for s in (pre, rec):
        if s is not None:
            uniform_time(s)
    return pre, rec


def uniform_time(s):
    """建立均匀时间基。

    录制的 elapsed 是「批量到达」时间戳：一次串口读入的若干帧共享同一 elapsed
    （实测 70% 帧与前帧重复），直接做差会得到 dt=0。设备是定频采集，
    因此用帧序号 × 平均帧间隔重建时间轴，并标出真实丢帧缺口。
    """
    el, fr = s["el"], s["fr"]
    span = float(el[-1] - el[0])
    nfr = float(fr[-1] - fr[0])
    mean_dt = span / nfr if nfr > 0 else 0.01
    s["dt_mean"] = mean_dt
    s["t"] = el[0] + (fr - fr[0]) * mean_dt
    d = np.diff(el)
    s["gap_idx"] = np.nonzero(d > max(0.3, 30 * mean_dt))[0]
    s["gap_s"] = d[s["gap_idx"]] if len(s["gap_idx"]) else np.zeros(0)
    return s


def no_gap(t, a, b, gaps_t, margin=0.2):
    """[a, b] 区间内是否没有丢帧缺口。"""
    for g in gaps_t:
        if a + margin < g < b - margin:
            return False
    return True


# ---------------------------------------------------------------- 基础工具

def smooth_ma(x, w):
    """居中滑动平均，端点用边缘值填充。"""
    if w <= 1:
        return np.asarray(x, dtype=float).copy()
    x = np.asarray(x, dtype=float)
    w = int(w)
    left = w // 2
    right = w - 1 - left
    xp = np.concatenate([np.full(left, x[0]), x, np.full(right, x[-1])])
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def robust_sigma(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return 1.4826 * float(np.median(np.abs(x - med)))


def med_smooth(x, w):
    if w <= 1:
        return np.asarray(x, dtype=float).copy()
    return median_filter(np.asarray(x, dtype=float), size=int(w), mode="nearest")


def total(s):
    return s["V"].sum(axis=1)


def band_med(t, y, dt=1.0):
    """按 dt 秒分桶取中位，返回 (t_bucket, med)。"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    tt, md = [], []
    i = 0
    n = len(t)
    while i < n:
        j = int(np.searchsorted(t, t[i] + dt))
        j = max(j, i + 1)
        tt.append(float(t[i]))
        md.append(float(np.median(y[i:j])))
        i = j
    return np.asarray(tt), np.asarray(md)


def dt_series(el):
    d = np.diff(el)
    d = np.concatenate([[d[0]], d])
    return np.clip(d, 0.0, 0.1)


# ---------------------------------------------------------------- 台阶检测

def detect_steps(el, sig, lo_frac=0.01, min_step_frac=0.02, gap_s=0.35):
    """在信号 sig 上检测加载/卸载沿。

    返回 list of dict(i0, i1, t0, t1, dur, d, level_pre, level_post, sign)
      i0 = 沿起始帧（最后一个「沿前电平」帧）
      i1 = 沿结束帧（第一个「沿后电平」帧）
      d  = level_post - level_pre（正 = 加载）
    做法：0.2 s 中值+均值平滑 -> 导数 -> 双门限（高门限找沿，低门限内禁止分裂）。
    噪声门限用「导数的残差」（去掉 0.5 s 平滑分量）估计，避免把真实多电平
    工况当成大噪声（第一版用全局 MAD 估计，多电平信号下门限被抬到 1e4 ADC 而漏检）。
    """
    sig = np.asarray(sig, dtype=float)
    dts = float(np.median(np.diff(el)))
    if not np.isfinite(dts) or dts <= 0:
        dts = 0.01
    sm = smooth_ma(med_smooth(sig, 5), max(1, int(round(0.2 / dts))))
    d = np.gradient(sm, el)
    d_res = d - smooth_ma(d, max(1, int(round(0.5 / dts))))
    sd = robust_sigma(d_res)
    a, b = float(sm.min()), float(sm.max())
    rng = max(b - a, 1.0)
    thr = max(8.0 * sd, lo_frac * rng)           # ADC/s
    active = np.abs(d) > thr
    # 合并间隔 <= gap_s 的活动段
    idx = np.nonzero(active)[0]
    groups = []
    if len(idx):
        s = idx[0]
        p = idx[0]
        for k in idx[1:]:
            if el[k] - el[p] <= gap_s:
                p = k
            else:
                groups.append((s, p))
                s = p = k
        groups.append((s, p))
    steps = []
    resid = sm - smooth_ma(sm, max(1, int(round(0.5 / dts))))
    min_step = max(min_step_frac * rng, 8.0 * robust_sigma(resid))
    for s, p in groups:
        lvl_pre = float(np.median(sm[max(0, s - 10):s + 1]))
        lvl_post = float(np.median(sm[p:p + 11]))
        dd = lvl_post - lvl_pre
        if abs(dd) < min_step:
            continue
        steps.append(dict(i0=int(s), i1=int(p), t0=float(el[s]), t1=float(el[p]),
                          dur=float(el[p] - el[s]), d=dd,
                          level_pre=lvl_pre, level_post=lvl_post,
                          sign=1 if dd > 0 else -1,
                          slope=abs(dd) / max(el[p] - el[s], 1e-6),
                          thr=thr, min_step=min_step))
    return steps, sm


# ---------------------------------------------------------------- 快相窗口

def fast_window(el, tc, i_next=None, span_s=30.0):
    """沿结束时刻 tc 之后的快相窗口索引 [k0, k1)。"""
    k0 = int(np.searchsorted(el, tc))
    k1 = int(np.searchsorted(el, tc + span_s))
    if i_next is not None:
        k1 = min(k1, i_next)
    return k0, k1


def landing_value(el, y, tc, ramp, sm=None):
    """沿后「瞬时落点」：跳过斜坡尾（1.5·斜坡时长，下限 0.4 s）后 0.5 s 中位。"""
    y = sm if sm is not None else y
    a = tc + max(1.5 * ramp, 0.4)
    k0 = int(np.searchsorted(el, a))
    k1 = int(np.searchsorted(el, a + 0.5))
    if k1 <= k0:
        k1 = min(k0 + 1, len(el))
    return float(np.median(y[k0:k1])), k0


# ---------------------------------------------------------------- 形状拟合

def _exp_model(u, P, C, tau):
    return P - C * np.exp(-u / tau)


def _stretch_model(u, P, C, tau, beta):
    return P - C * np.exp(-np.power(u / tau, beta))


def _double_model(u, P, C1, tau1, C2, tau2):
    return P - C1 * np.exp(-u / tau1) - C2 * np.exp(-u / tau2)


def fit_shape(t, y, tc, models=("fixed8", "single", "stretch", "double")):
    """对快相窗口做形状拟合，返回 {model: dict(params, rms, aic, ...)}。"""
    from scipy.optimize import least_squares
    u = np.asarray(t, dtype=float) - tc
    y = np.asarray(y, dtype=float)
    span = float(u[-1] - u[0]) if len(u) > 1 else 1.0
    y0, y1 = float(y[0]), float(y[-1])
    C0 = max(abs(y1 - y0), 1e-3)
    out = {}

    def rec(name, popt, fun, k):
        r = y - fun(u, *popt)
        rms = float(np.sqrt(np.mean(r ** 2)))
        n = len(y)
        aic = n * np.log(max(np.mean(r ** 2), 1e-12)) + 2 * k
        out[name] = dict(popt=[float(v) for v in popt], rms=rms, aic=float(aic),
                         resid=r)

    if "fixed8" in models:
        P0 = y1 + C0 * np.exp(-span / 8.0)
        ls = least_squares(lambda p: _exp_model(u, p[0], p[1], 8.0) - y,
                           [P0, C0], method="lm", max_nfev=4000)
        rec("fixed8", ls.x, lambda uu, P, C: _exp_model(uu, P, C, 8.0), 2)
    if "single" in models:
        best = None
        for tau_g in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            P0 = y1 + C0 * np.exp(-span / tau_g)
            ls = least_squares(lambda p: _exp_model(u, p[0], p[1], p[2]) - y,
                               [P0, C0, tau_g],
                               bounds=([-1e9, 0.0, 0.05], [1e9, 1e9, 5000.0]),
                               max_nfev=4000)
            if best is None or ls.cost < best.cost:
                best = ls
        rec("single", best.x, _exp_model, 3)
    if "stretch" in models:
        best = None
        for tau_g in (2.0, 8.0, 32.0):
            for b_g in (0.3, 0.6, 1.0):
                P0 = y1 + C0 * np.exp(-(span / tau_g) ** b_g)
                ls = least_squares(
                    lambda p: _stretch_model(u, p[0], p[1], p[2], p[3]) - y,
                    [P0, C0, tau_g, b_g],
                    bounds=([-1e9, 0.0, 0.05, 0.05], [1e9, 1e9, 5000.0, 3.0]),
                    max_nfev=6000)
                if best is None or ls.cost < best.cost:
                    best = ls
        rec("stretch", best.x, _stretch_model, 4)
    if "double" in models:
        best = None
        for (t1g, t2g, f) in ((1.0, 20.0, 0.5), (1.0, 60.0, 0.5),
                              (2.0, 20.0, 0.7), (2.0, 50.0, 0.3),
                              (4.0, 40.0, 0.5), (0.5, 8.0, 0.5)):
            P0 = y1 + C0
            ls = least_squares(
                lambda p: _double_model(u, p[0], p[1], p[2], p[3], p[4]) - y,
                [P0, f * C0, t1g, (1 - f) * C0, t2g],
                bounds=([-1e9, 0.0, 0.05, 0.0, 0.05], [1e9, 1e9, 5000.0, 1e9, 5000.0]),
                max_nfev=8000)
            if best is None or ls.cost < best.cost:
                best = ls
        rec("double", best.x, _double_model, 5)
    return out


# ---------------------------------------------------------------- 观测器回放

def replay(ts, V, p=None):
    """v34_observer_core3 的逐帧复刻，额外导出 x1/x2/显示逐通道。"""
    import sys
    CORE = os.path.join(ROOT, "Document", "Update", "Dev-Version",
                        "v2.7 - 抗蠕变补偿算法", "v4.1flash", "progress",
                        "currentworking", "scripts")
    if CORE not in sys.path:
        sys.path.insert(0, CORE)
    import v34_observer_core3 as C
    pp = dict(C.P3)
    if p:
        pp.update(p)
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    zero = V[0].copy()
    y_max = np.maximum(V[0] - zero, 0.0)
    v_lp = V[0].copy()
    X1 = np.zeros((n, ch))
    X2 = np.zeros((n, ch))
    D = np.zeros((n, ch))
    t_prev = ts[0]
    for i in range(n):
        dt = min(max(ts[i] - t_prev, 0.0), 0.1)
        t_prev = ts[i]
        v = V[i]
        if dt > 0.0:
            y = v - zero
            y_max = np.maximum(y_max * np.exp(-dt / pp["y_max_tau"]),
                               np.maximum(y, 0.0))
            idle = y < pp["idle_frac"] * np.maximum(y_max, 1.0)
            zero = np.where(idle, zero + (dt / pp["tau_zero"]) * (v - zero), zero)
            y = v - zero
            e_now = np.maximum(y - x1 - x2, 0.0)
            dx1_rate = np.where(e_now > 0.0,
                                (pp["r1"] * e_now - x1) / pp["tc1"],
                                -x1 / pp["tr1"])
            x1 = np.maximum(x1 + dt * dx1_rate, 0.0)
            slope = (v - v_lp) / pp["tau_slope"]
            v_lp = v_lp + (dt / pp["tau_slope"]) * (v - v_lp)
            e = np.maximum(y - x1 - x2, 0.0)
            rate_cap = pp["slope_cap"] * np.maximum(e, 1.0)
            gate = pp["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1_rate, -rate_cap, rate_cap)
            dx2 = np.where((e > 0.0) & (np.abs(slope) < gate), dx2, 0.0) * dt
            x2 = x2 + dx2
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, pp["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / pp["tr2"], 0.0))
        X1[i] = x1
        X2[i] = x2
        D[i] = v - x1 - x2
    return dict(D=D, X1=X1, X2=X2, p=pp)


# ---------------------------------------------------------------- 通道选择

def top_channels(pre, n=3, exclude_flat=True):
    """按「信号幅度 + 台阶清晰度」选代表通道。"""
    V = pre["V"]
    el = pre["el"]
    amp = V.max(axis=0) - np.percentile(V, 5, axis=0)
    std = V.std(axis=0)
    d = np.abs(np.diff(med_smooth(V, 5), axis=0)).max(axis=0)
    score = amp + 0.0 * std
    order = np.argsort(-(amp + 0.05 * d))
    out = []
    for c in order:
        if exclude_flat and amp[c] < 20:
            continue
        out.append(int(c))
        if len(out) >= n:
            break
    return out
