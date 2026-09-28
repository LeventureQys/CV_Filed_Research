# -*- coding: utf-8 -*-
"""T3（提前补偿/前馈）共享层：数据装载 + 沿检测器 + 前馈变体观测器 + 事件量测。

基线 = `scripts/v34_observer_core3.py::observe3`（observer-v3，与 C++ 逐帧一致）。
本模块把 core3 逐帧流程原样搬进来（便于插入前馈动作），并提供 `mode="base"` 时
与 core3 数值一致的自检（见 t3_parity_selfcheck）。

数据根已从旧 `temp/` 迁到
`Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/算法数据&原始数据/`。
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NCH = 21


def _find_repo_root():
    d = HERE
    for _ in range(12):
        if os.path.isfile(os.path.join(d, "CMakeLists.txt")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root not found from " + HERE)


ROOT = _find_repo_root()
DATA_ROOT = os.path.join(ROOT, "Document", "Update", "Dev-Version",
                         "v2.7 - 抗蠕变补偿算法", "算法数据&原始数据")
WORKING = os.path.join(DATA_ROOT, "working")
ARCHIVED = os.path.join(DATA_ROOT, "archived")

P3 = dict(r1=0.12, tc1=8.0, tr1=6.0,
          r2max=0.35, tr2=150.0,
          tau_slope=3.0, slope_gate=0.02, slope_cap=0.01,
          tau_zero=8.0, idle_frac=0.05, y_max_tau=600.0)

# 沿检测器默认参数（初值，由 t3_probe_detector2 标定）
DET = dict(thr=10.0,         # 绝对底噪门限（ADC/s，逐通道）
           rel_on=0.5,       # 相对项：k·slope_gate_frac·max(e,1)
           hyst=0.35,        # 迟滞：slope 回落到 hyst·门限以下才重新武装
           min_gap=0.6)      # 同通道两次触发最小间隔（s）


# ---------------------------------------------------------------- 数据装载
def discover_sessions(root, max_depth=6):
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


def read_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    cfg = {}
    for r in rows[1:di]:
        if "," in r:
            k, v = r.split(",", 1)
            cfg[k] = v
    data = [r.split(",") for r in rows[di + 2:] if r.strip()]
    return cfg, data


def load_stream(ds_dir, name):
    cfg, data = read_rows(os.path.join(ds_dir, name))
    n = len(data)
    ts = np.empty(n)
    el = np.empty(n)
    V = np.empty((n, NCH))
    for i, f in enumerate(data):
        if len(f) < 3 + NCH:
            raise ValueError("%s 行 %d 列数不足: %d" % (name, i, len(f)))
        ts[i] = float(f[0])
        el[i] = float(f[1])
        V[i] = [float(x) for x in f[3:3 + NCH]]
    return dict(ts=ts, el=el, V=V, n=n, cfg=cfg, name=name)


def pre_name(ds_dir):
    if os.path.isfile(os.path.join(ds_dir, "device_001_pre_seg0.csv")):
        return "device_001_pre_seg0.csv"
    return "device_001_seg000.csv"


def load_input(ds_dir):
    """算法输入流（pre 优先，缺则用主文件）——与 v34_observer_v2_eval 口径一致。"""
    return load_stream(ds_dir, pre_name(ds_dir))


def load_recorded(ds_dir):
    """实机录制显示（主文件），仅用于对照。"""
    try:
        return load_stream(ds_dir, "device_001_seg000.csv")
    except Exception:
        return None


def all_sessions():
    """返回 [(组, 标签, 目录)]，覆盖 working 与 archived。"""
    out = []
    for tag, root in (("working", WORKING), ("archived", ARCHIVED)):
        for label, d in discover_sessions(root, 6):
            out.append((tag, tag + "/" + label, d))
    return out


# ---------------------------------------------------------------- 观测器
def observe(ts, V, p=None, ff=None, want_tracks=False):
    """core3 逐帧流程 + 可选前馈分支。

    ff=None            -> 基线（应与 observe3 数值一致）
    ff=dict(mode='A'|'B'|'C', alpha=1.0, pred='now'|'lag',
            tau_boost=1.0, boost_s=1.5, thr=..., ...)

    返回 dict(D=显示(n,ch), X1, X2, Y, E, trig=(通道,帧,slope,e_now,x1_before))
    逐通道触发记录只记首次/每次触发点，用于误触发风险统计（不参与算法）。
    """
    pp = dict(P3 if p is None else p)
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    zero = V[0].copy()
    y_max = np.maximum(V[0] - zero, 0.0)
    v_lp = V[0].copy()
    D = np.empty((n, ch))
    X1 = np.empty((n, ch))
    X2 = np.empty((n, ch))
    EE = np.empty((n, ch))
    YY = np.empty((n, ch))
    # 检测器状态（逐通道，3 个标量，非事件账本）
    armed = np.ones(ch, dtype=bool)
    last_t = np.full(ch, -1e9)
    boost_left = np.zeros(ch)
    trigs = []
    diag_rows = []
    diag = ff.get("diag") if ff else None
    mode = None
    if ff:
        mode = ff.get("mode", "base")
        thr = ff.get("thr", DET["thr"])
        hyst = ff.get("hyst", DET["hyst"])
        min_gap = ff.get("min_gap", DET["min_gap"])
        rel = ff.get("rel_on", DET["rel_on"])
        alpha = ff.get("alpha", 1.0)
        pred = ff.get("pred", "now")
        tau_boost = ff.get("tau_boost", 1.0)
        boost_s = ff.get("boost_s", 1.5)
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
            # --- 低通导数（与 core3 同值：先用上一帧 v_lp 算 slope，再更新 v_lp）
            #     slope 只依赖输入 v，与 x1/x2 无关，因此提前到 x1 更新之前不改变基线数值
            slope = (v - v_lp) / pp["tau_slope"]
            v_lp = v_lp + (dt / pp["tau_slope"]) * (v - v_lp)
            e_pre = np.maximum(y - x1 - x2, 0.0)
            if mode in ("A", "B", "C") and ff.get("enabled", True):
                # 门限 = 绝对底噪 + k·(算法自己的沿门 0.02·e)
                thr_v = thr + rel * pp["slope_gate"] * np.maximum(e_pre, 1.0)
                fire = (armed & (slope > thr_v) & (e_pre > 0.0)
                        & ((ts[i] - last_t) > min_gap))
                fidx = np.nonzero(fire)[0]
                if len(fidx):
                    for c in fidx:
                        # (通道, 帧号, slope, e_pre, 置位前 x1)；帧号由调用方映射到时间
                        trigs.append((int(c), int(i), float(slope[c]),
                                      float(e_pre[c]), float(x1[c])))
                    armed[fidx] = False
                    last_t[fidx] = ts[i]
                    if mode in ("A", "C"):
                        e_pred = e_pre[fidx].copy()
                        if pred == "lag":
                            e_pred = e_pred + pp["tau_slope"] * slope[fidx]
                        e_pred = np.maximum(e_pred, 0.0)
                        x1[fidx] = np.maximum(alpha * pp["r1"] * e_pred, x1[fidx])
                    if mode in ("B", "C"):
                        boost_left[fidx] = boost_s
                rearm = (~armed) & (slope < hyst * thr_v)
                if rearm.any():
                    armed[rearm] = True
                if diag is not None and diag[0] <= i <= diag[1]:
                    diag_rows.append((i, float(np.max(slope)), float(np.max(thr_v)),
                                      float(np.max(e_pre)), float(np.median(e_pre)),
                                      int(armed.sum()),
                                      int(np.sum(slope > thr_v)),
                                      int(np.sum((slope > thr_v) & armed))))
            tc1 = pp["tc1"]
            if mode in ("B", "C"):
                tc1 = np.where(boost_left > 0.0, tau_boost, tc1)
            e_now = np.maximum(y - x1 - x2, 0.0)
            dx1_rate = np.where(e_now > 0.0,
                                (pp["r1"] * e_now - x1) / tc1,
                                -x1 / pp["tr1"])
            x1 = np.maximum(x1 + dt * dx1_rate, 0.0)
            if mode in ("B", "C"):
                boost_left = np.maximum(boost_left - dt, 0.0)
            e = np.maximum(y - x1 - x2, 0.0)
            rate_cap = pp["slope_cap"] * np.maximum(e, 1.0)
            gate = pp["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1_rate, -rate_cap, rate_cap)
            dx2 = np.where((e > 0.0) & (np.abs(slope) < gate), dx2, 0.0) * dt
            x2 = x2 + dx2
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, pp["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / pp["tr2"], 0.0))
        D[i] = v - x1 - x2
        X1[i] = x1
        X2[i] = x2
        EE[i] = np.maximum(v - zero - x1 - x2, 0.0)
        YY[i] = v - zero
    return dict(D=D, X1=X1, X2=X2, E=EE, Y=YY, trig=trigs, diag=diag_rows)


# ---------------------------------------------------------------- 事件量测
def find_edges(el, tot, min_step_abs=500.0, min_step_frac=0.05,
               back_s=1.0, min_gap_s=2.0, unload_frac=0.30, min_hold_s=1.0):
    """评估用（非因果）加载沿切分：输入总量出现台阶 ≥ max(min_step_abs, frac·量程)。

    返回 [(i0, i1, step, base, prev_end)]：
      i0 沿起点（回升前最后一帧），i1 保压段末（下一个显著卸载的起点；
      若后续无卸载则为序列末尾/下一沿前），base 沿前基线，step = 段末输入 − base。
    保压段末用「输入自沿后运行最大值回落 > max(300, unload_frac·step)」判定，
    避免把同一段里的卸载动作混进「回落/下冲」量测。
    """
    n = len(el)
    rng = float(tot.max() - tot.min())
    ms = max(min_step_abs, min_step_frac * rng)
    nb = max(5, int(back_s * 100))
    prev = np.empty(n)
    for i in range(n):
        prev[i] = np.median(tot[max(0, i - nb):max(1, i - 2)]) if i > 5 else tot[i]
    rise = tot - prev
    raw = []
    i = 5
    while i < n:
        if rise[i] > ms:
            bbase = prev[i]
            j = i
            while (j > 1 and i - j < 150 and tot[j - 1] > bbase + 0.2 * ms):
                j -= 1
            if raw and el[j] - el[raw[-1]] < min_gap_s:
                raw[-1] = j
            else:
                raw.append(j)
            i = int(np.searchsorted(el, el[j] + min_gap_s))
        else:
            i += 1
    out = []
    for k, j in enumerate(raw):
        base = float(np.median(tot[max(0, j - 100):max(1, j - 2)]))
        limit = raw[k + 1] if k + 1 < len(raw) else n
        seg = tot[j:limit]
        if len(seg) == 0:
            continue
        runmax = np.maximum.accumulate(seg)
        # 台阶幅度用段内高位（p97）估，避免段尾落在卸载后的空载段而误判
        step0 = float(np.percentile(seg, 97)) - base
        if step0 < ms:
            continue
        drop_thr = max(300.0, unload_frac * step0)
        k_un = None
        for q in range(min(30, len(seg) - 1), len(seg)):
            if runmax[q] - seg[q] > drop_thr:
                k_un = j + q
                break
        hold_end = (k_un if k_un is not None else limit)
        hold_end = min(hold_end, n - 1)
        if el[hold_end] - el[j] < min_hold_s:
            continue
        segw = max(15, min(int(0.3 * (hold_end - j)), 200))
        settle = float(np.median(tot[hold_end - segw:hold_end]))
        step = settle - base
        if step < ms:
            continue
        out.append(dict(i=j, j=hold_end, step=step, base=base, settle=settle,
                        t=float(el[j]), unload=(k_un is not None)))
    return out, ms


def unload_edges(el, tot, min_step, min_gap_s=0.8, win=30):
    """卸载沿（下降台阶），用于「卸载沿是否误置位」风险测试。"""
    out = []
    for i in range(win, len(tot)):
        base = float(np.median(tot[max(0, i - win):i - max(2, win // 5)]))
        drop = base - tot[i]
        if drop > min_step:
            if out and el[i] - el[out[-1]] < min_gap_s:
                continue
            out.append(i)
    return out


def post_edge_metrics(el, y, i, j, settle=None):
    """沿后回落量测（y = 显示总量）。

    peak      沿后峰值
    settle    段末稳定电平
    fall      回落幅度 = peak − settle（主诉指标：显示回落多大）
    dip       最大下冲 = max(0, settle − min(沿后显示))（过扣越深越大）
    t_fall    从沿起算到首次进入并保持在 settle±5%·step 带的时刻
    """
    seg = y[i:j]
    if len(seg) < 10:
        return None
    ee = el[i:j]
    peak = float(np.max(seg))
    ip = int(np.argmax(seg))
    if settle is None:
        settle = float(np.median(seg[-min(len(seg), 200):]))
    step = settle - float(np.median(y[max(0, i - 60):max(1, i - 5)]))
    band = 0.05 * abs(step) if step else 1.0
    inside = np.abs(seg - settle) <= band
    k = len(inside) - 1
    while k > 0 and inside[k - 1]:
        k -= 1
    t_fall = float(ee[k] - ee[0]) if inside[k:].all() else None
    return dict(peak=peak, t_peak=float(ee[ip] - ee[0]), settle=settle,
                fall=peak - settle, dip=max(0.0, settle - float(np.min(seg))),
                t_fall=t_fall, step=step)


def band_std(el, y, t0, t1):
    m = (el >= t0) & (el <= t1)
    if m.sum() < 20:
        return None
    return float(np.std(y[m]))


def event_mask(el, ev, pre=0.5, post=20.0):
    """事件窗掩膜：[t0−pre, t0+post]——用于把「事件后的稳态差异」排除在误触发影响之外。"""
    m = np.zeros(len(el), dtype=bool)
    for t0 in ev:
        m |= (el >= t0 - pre) & (el <= t0 + post)
    return m


def idle_mask(din, frac=0.15):
    """空载帧掩膜：输入总量低于 (min + frac·量程)。"""
    return din < (din.min() + frac * (din.max() - din.min()))
