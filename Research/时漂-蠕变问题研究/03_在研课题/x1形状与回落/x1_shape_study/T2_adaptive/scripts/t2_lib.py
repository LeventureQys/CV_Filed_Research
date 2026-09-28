# -*- coding: utf-8 -*-
"""T2「x1 自适应形状」研究共享层：数据装载 + 会话枚举 + 指标口径。

数据源：`算法数据&原始数据/working|archived` 下的会话目录，算法输入取
`device_001_pre_seg0.csv`（CSV 前部为 ##Session 元数据头，##Data 后为
timestamp,elapsed,frame_index,ch0..ch20）。

本层只做三件事：
  1. 复用 v30_lib 的 CSV 解析，接到 Document 下的新数据根；
  2. 按「通道总量」做滞回沿切分，给出加载事件（沿 → 卸载）；
  3. 定义统一的评估指标：回落深度(overshoot)、保压 std、一致性、空载偏差、稳定时间。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def _find_root(start):
    cur = start
    for _ in range(12):
        if os.path.isfile(os.path.join(cur, "CMakeLists.txt")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise RuntimeError("未找到仓库根（CMakeLists.txt）: " + start)


ROOT = _find_root(HERE)

SCRIPTS = os.path.join(ROOT, "Document", "Update", "Dev-Version",
                      "v2.7 - 抗蠕变补偿算法", "v4.1flash", "progress",
                      "currentworking", "scripts")
sys.path.insert(0, SCRIPTS)
# v30_lib 在 import 期断言 DATA_ROOT 的祖先里存在 CMakeLists.txt（数据根已搬到
# Document 下，该断言必然失败）；import 期没法先改它的 ROOT，这里临时把
# os.path.isfile 指到真实仓库根，让断言通过，再恢复。
_is_file = os.path.isfile


def _patched(path):
    if os.path.normcase(os.path.abspath(path)).endswith(
            os.path.normcase(os.path.join("Document", "Update", "Dev-Version"))):
        return False
    if os.path.basename(path) == "CMakeLists.txt":
        return _is_file(os.path.join(ROOT, "CMakeLists.txt"))
    return _is_file(path)


os.path.isfile = _patched
try:
    import v30_lib as L  # noqa: E402
finally:
    os.path.isfile = _is_file

L.ROOT = ROOT

DOC_ROOT = os.path.join(ROOT, "Document", "Update", "Dev-Version",
                        "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据")
WORKING = os.path.join(DOC_ROOT, "working")
ARCHIVED = os.path.join(DOC_ROOT, "archived")
OUT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
NCH = 21


def sessions(root, max_depth=5):
    return L.discover_sessions(root, max_depth)


def all_sessions():
    out = []
    for tag, root in (("working", WORKING), ("archived", ARCHIVED)):
        for label, d in sessions(root, 5):
            out.append(("%s/%s" % (tag, label), d))
    return out


def pre_name(d):
    if os.path.isfile(os.path.join(d, "device_001_pre_seg0.csv")):
        return "device_001_pre_seg0.csv"
    return "device_001_seg000.csv"


def load_pre(d):
    return L.load_stream(d, pre_name(d))


def load_recorded(d):
    p = os.path.join(d, "device_001_seg000.csv")
    if not os.path.isfile(p):
        return None
    return L.load_stream(d, "device_001_seg000.csv")


def total(s):
    return s["V"].sum(axis=1)


def session_key(label):
    return label.replace("\\", "/").replace("/", "__")


# ---------------------------------------------------------------- 事件切分

def hysteresis_states(el, tot, rel_hi=0.35, rel_lo=0.15, min_frames=40):
    """滞回双门限状态机：空载<=>受载，返回 [(kind, i0, i1), ...]。"""
    a, b = float(tot.min()), float(tot.max())
    hi = a + rel_hi * (b - a)
    lo = a + rel_lo * (b - a)
    state = "loaded" if tot[0] > hi else "idle"
    start = 0
    segs = []
    for i in range(1, len(tot)):
        if state == "idle" and tot[i] > hi:
            segs.append((state, start, i - 1))
            state, start = "loaded", i
        elif state == "loaded" and tot[i] < lo:
            segs.append((state, start, i - 1))
            state, start = "idle", i
    segs.append((state, start, len(tot) - 1))
    return [(k, i0, i1) for (k, i0, i1) in segs if i1 - i0 >= min_frames]


def event_at(el, tot, i_up, i_dn, min_step_frac=0.10):
    """给定一个受载段的起止帧，构造事件（过滤掉台阶过小的段）。"""
    i_dn = min(int(i_dn), len(el) - 1)
    seg = tot[i_up:i_dn]
    if len(seg) < 40:
        return None
    rng = float(tot.max() - tot.min())
    base = float(np.median(tot[max(0, i_up - 60):max(1, i_up - 5)]))
    post = float(np.median(seg[-min(len(seg), 200):]))
    step = post - base
    if abs(step) < min_step_frac * max(rng, 1.0):
        return None
    return dict(i_up=int(i_up), i_dn=int(i_dn), t_up=float(el[i_up]),
                t_dn=float(el[i_dn]), base=base, step=step,
                hold=float(el[i_dn] - el[i_up]))


def load_events(el, tot, win=None, **kw):
    """滞回切分 → 事件列表。win 给定时，事件窗口截到 t_up+win（保证各变体同窗可比）。"""
    evs = []
    for kind, i0, i1 in hysteresis_states(el, tot, **kw):
        if kind != "loaded":
            continue
        ev = event_at(el, tot, i0, i1 + 1)
        if ev is None:
            continue
        if win is not None:
            i_lim = int(np.searchsorted(el, ev["t_up"] + win))
            ev["i_win"] = int(min(max(i_lim, ev["i_up"] + 20), ev["i_dn"]))
        else:
            ev["i_win"] = ev["i_dn"]
        ev["win"] = float(el[ev["i_win"]] - el[ev["i_up"]])
        evs.append(ev)
    return evs


def hold_slice(ev, dur=None, guards=(1.0, 0.0)):
    """保压窗口 [t_up+guards0, t_dn−guards1]，可再截到 dur 秒。"""
    return ev["t_up"] + guards[0], ev["t_dn"] - guards[1]


# ---------------------------------------------------------------- 指标

def _nanmean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


# 事件入选门槛：窗口足够长 + 保压段输入足够平稳（否则"回落/漂移"量的是载荷变化不是算法）
MIN_WIN = 25.0
MAX_IN_STD = 0.15
CLEAN_IN_STD = 0.05


def event_metrics(y, x, el, ev, t_peak0=2.0, t_peak1=10.0, t_pl0=20.0,
                  t_pl1=45.0, seg0=15.0, seg_len=15.0, tau_lp=1.0,
                  min_decline_s=5.0, settle_win=300, frac=0.02,
                  xlp=None, ylp=None, x1=None):
    """一次加载事件的核心指标（y=显示总量, x=输入总量，均为通道和）。

    主指标「回落」（用户主诉，只看显示自身）：
      peak     = 加载后 [2s, 10s] 显示最大值（冲高段；0~2s 含载荷建立过程不取）
      plateau  = [20s, 45s] 显示中位（长保压平台）
      drop     = peak − plateau      （正 = 冲高后回落；这就是"回落"）
      drop_rel = drop / 台阶

    观测器自身贡献（把输入自身的漂移影响扣掉，用来判断"x1 是不是补过头/不够"）：
      x1_exc   = Δx1[2,10s] − Δy[2,10s]：>0 表示这段时间 x1 涨得比实测蠕变还快
                 （直接把显示往下拽 —— 这就是"x1 补偿导致回落"的机理量）
      decline  = 保压窗内 (显示历史最高 − 当前显示) 的最大值（持续 ≥5s 才计入）
      decline_t= 该最低点时刻

    长保压 / 其他：
      tail     = 显示(沿+45~60s) − plateau：>0 = 平台后显示还在上爬（蠕变没补住）
      hold_std = [2s,10s] 显示 std
      seg_err  = [15s,30s] 内 (显示−输入低通) 均值（负 = 显示被压到输入之下）
      seg_drift= 该段 Δ显示 − Δ输入低通（0 = 显示恰好跟随蠕变）
      t_settle = 进入 plateau+frac·台阶 带且此后不再回升的时刻（s）
    """
    i, j = ev["i_up"], ev["i_win"]
    if j - i < 30:
        return None
    step = ev["step"]
    ylp = ylp if ylp is not None else lowpass(y, el, tau=tau_lp)
    xlp = xlp if xlp is not None else lowpass(x, el, tau=5.0)
    t = el - ev["t_up"]

    def med(arr, t0, t1):
        m = (t >= t0) & (t <= t1)
        m[:i] = False
        m[j:] = False
        return float(np.median(arr[m])) if m.sum() else float("nan")

    m_win = (t >= t_peak0) & (t <= t_peak1)
    m_win[:i] = False
    m_win[j:] = False
    if m_win.sum() < 30:
        return None
    peak = float(np.max(ylp[m_win]))
    plateau = med(ylp, t_pl0, t_pl1)
    if np.isnan(plateau):
        return None
    drop = peak - plateau
    tail = float(med(ylp, t_pl1, t_pl1 + 15.0) - plateau)

    # ---- 绝对下降（持续 ≥ min_decline_s）
    m_dec = (t >= t_peak0)
    m_dec[:i] = False
    m_dec[j:] = False
    if m_dec.sum() > 30:
        dec_v, dec_t = ylp[m_dec], t[m_dec]
        run = np.maximum.accumulate(dec_v)
        dec = run - dec_v
        dtf = max(float(np.mean(np.diff(dec_t))), 1e-6)
        fs = min(max(1, int(min_decline_s / dtf)), len(dec_v) - 1)
        keep = dec[:len(dec_v) - fs]
        if len(keep):
            k = int(np.argmax(keep))
            decline, decline_t = float(keep[k]), float(dec_t[k])
        else:
            decline, decline_t = 0.0, float("nan")
    else:
        decline, decline_t = float(0.0), float("nan")

    tol = frac * abs(step) if step else 1.0
    yseg = y[i:j]
    above = np.nonzero(yseg > plateau + tol)[0]
    t_settle = float(el[i + above[-1] + 1] - el[i]) if len(above) and above[-1] + 1 < len(yseg) else 0.0

    # ---- x1 相对实测蠕变的超调量（观测器自身贡献）
    dy_in = _delta(xlp, el, ev, t_peak0, t_peak1)
    if x1 is not None:
        dx1_in = _delta(x1, el, ev, t_peak0, t_peak1)
        x1_exc = dx1_in - dy_in
    else:
        dx1_in, x1_exc = float("nan"), float("nan")

    dy = med(ylp, seg0, seg0 + seg_len)
    dx = med(xlp, seg0, seg0 + seg_len)
    err_seg = med(ylp - xlp, seg0, seg0 + seg_len)
    dr = (dy - plateau) - (dx - med(xlp, t_pl0, t_pl1))
    in_std = float(np.std(x[i:j]))
    return dict(drop=drop, drop_rel=drop / abs(step) if step else 0.0,
                decline=decline, decline_rel=decline / abs(step) if step else 0.0,
                decline_t=decline_t, x1_exc=x1_exc, dy_in=dy_in, dx1_in=dx1_in,
                peak=peak, plateau=plateau, tail=tail, step=step,
                t_settle=t_settle, seg_err=err_seg, seg_drift=dr,
                hold_std=float(np.std(y[m_win][-min(m_win.sum(), settle_win):])),
                in_std_rel=in_std / max(abs(step), 1.0), win=ev["win"])


def _delta(arr, el, ev, t0, t1):
    """窗 [t_up+t0, t_up+t1] 内「末 20% 均值 − 首 20% 均值」。"""
    t = el - ev["t_up"]
    m = (t >= t0) & (t <= t1)
    m[:ev["i_up"]] = False
    m[ev["i_win"]:] = False
    if m.sum() < 20:
        return float("nan")
    v = arr[m]
    k = max(3, len(v) // 5)
    return float(np.mean(v[-k:]) - np.mean(v[:k]))


def lowpass(sig, el, tau=5.0):
    """因果一阶低通（用于把"输入"的抖动滤掉，只看蠕变趋势）。"""
    out = np.empty_like(sig, dtype=float)
    acc = float(sig[0])
    prev = el[0]
    for i in range(len(sig)):
        dt = min(max(el[i] - prev, 0.0), 0.1)
        prev = el[i]
        if dt > 0:
            acc += (dt / tau) * (float(sig[i]) - acc)
        out[i] = acc
    return out


def event_gate(m, ev, min_win=MIN_WIN, max_std=MAX_IN_STD):
    return ev["win"] >= min_win and m["in_std_rel"] <= max_std


def idle_bias(y, x, el, evs, pad=1.0):
    """空载段 显示−输入 均值（只在事件之外、且远离沿 pad 秒的时间上统计）。"""
    mask = np.ones(len(el), dtype=bool)
    for ev in evs:
        mask &= ~((el >= ev["t_up"] - pad) & (el <= ev["t_dn"] + pad))
    if mask.sum() < 20:
        return float("nan")
    return float(np.mean((y - x)[mask]))


def consistency(evs, y, el, tol_frac=0.10):
    """一致性：把事件按台阶高度分档（±10%·台阶 判据），比较同档事件的稳定电平。"""
    items = []
    for ev in evs:
        i, j = ev["i_up"], ev["i_win"]
        if j - i < 30:
            continue
        settle = float(np.median(y[i:j][-min(j - i, 300):]))
        items.append(dict(ev=ev, settle=settle, step=ev["step"]))
    groups = {}
    for it in items:
        placed = False
        for key in groups:
            if abs(it["step"] - key) <= tol_frac * max(abs(key), 1.0):
                groups[key].append(it)
                placed = True
                break
        if not placed:
            groups[it["step"]] = [it]
    ok, tot_, worst = 0, 0, 0.0
    detail = []
    for key, g in groups.items():
        if len(g) < 2:
            continue
        ref = float(np.median([it["settle"] for it in g]))
        for it in g:
            dev = it["settle"] - ref
            rel = abs(dev) / max(abs(key), 1.0)
            ok += abs(dev) <= tol_frac * abs(key)
            tot_ += 1
            worst = max(worst, rel)
            detail.append((key, it["ev"]["t_up"], it["settle"], dev, rel))
    return ok, tot_, worst, detail


def summarize(tag, y, x, el, evs, drec=None, subset=False, xlp=None, ylp=None,
              x1=None):
    """一个 (会话, 变体) 的整体指标汇总。

    subset=True 时只统计「干净保压」事件（输入std/台阶 < CLEAN_IN_STD），
    这一子集把"回落/持平"从载荷抖动里隔离出来。
    """
    ms, ev_used = [], []
    for ev in evs:
        m = event_metrics(y, x, el, ev, xlp=xlp, ylp=ylp, x1=x1)
        if not m:
            continue
        if not event_gate(m, ev):
            continue
        if subset and m["in_std_rel"] > CLEAN_IN_STD:
            continue
        ms.append(m)
        ev_used.append(ev)
    ok, tot_, worst, _ = consistency(ev_used, y, el) if ev_used else (0, 0, 0.0, [])
    arr = lambda k: np.array([m[k] for m in ms]) if ms else np.array([np.nan])  # noqa: E731
    fid = float(np.median(np.abs(y - drec))) if drec is not None else float("nan")
    return dict(
        tag=tag, n_ev=len(ms),
        drop_mean=float(np.mean(arr("drop"))), drop_max=float(np.max(arr("drop"))),
        drop_rel_mean=float(np.mean(arr("drop_rel"))),
        drop_rel_max=float(np.max(arr("drop_rel"))),
        x1_exc_mean=float(np.mean(arr("x1_exc"))),
        x1_exc_absmax=float(np.max(np.abs(arr("x1_exc")))),
        decline_mean=float(np.mean(arr("decline"))),
        decline_max=float(np.max(arr("decline"))),
        tail_mean=float(np.mean(arr("tail"))), tail_max=float(np.max(arr("tail"))),
        std_mean=float(np.mean(arr("hold_std"))), std_max=float(np.max(arr("hold_std"))),
        seg_err_mean=float(np.mean(arr("seg_err"))),
        seg_err_absmax=float(np.max(np.abs(arr("seg_err")))),
        seg_drift_mean=float(np.mean(np.abs(arr("seg_drift")))),
        seg_drift_max=float(np.max(np.abs(arr("seg_drift")))),
        cons_ok=ok, cons_tot=tot_, cons_worst=worst,
        idle=idle_bias(y, x, el, evs), fid=fid,
        t_settle_mean=float(np.mean(arr("t_settle"))), t_settle_max=float(np.max(arr("t_settle"))),
    )


ROW_COLS = [("会话", 46, "s"), ("N", 2, "d"), ("回落均", 7, "f"),
            ("回落峰", 7, "f"), ("回落/台阶", 9, "f"), ("x1超调", 8, "f"),
            ("绝对下降", 8, "f"), ("尾段爬升", 8, "f"), ("保证段偏差", 10, "f"),
            ("保压std", 7, "f"), ("保压漂移", 8, "f"), ("空载偏差", 8, "f"),
            ("稳时", 6, "f"), ("一致性", 7, "s")]


def _cell(v, w, kind):
    if kind == "s":
        return ("%-" + str(w) + "s") % str(v)[:w]
    if kind == "d":
        return ("%" + str(w) + "d") % v
    return ("%" + str(w) + ".1f") % v


def fmt_row(s):
    vals = [s["tag"][-46:], s["n_ev"], s["drop_mean"], s["drop_max"],
            s["drop_rel_mean"], s["x1_exc_mean"], s["decline_mean"],
            s["tail_mean"], s["seg_err_mean"], s["std_mean"],
            s["seg_drift_mean"], s["idle"], s["t_settle_mean"],
            "%d/%d" % (s["cons_ok"], s["cons_tot"])]
    return " ".join(_cell(v, w, k) for v, (_, w, k) in zip(vals, ROW_COLS))


HEADER = " ".join(_cell(t, w, "s") for t, w, _ in ROW_COLS)
