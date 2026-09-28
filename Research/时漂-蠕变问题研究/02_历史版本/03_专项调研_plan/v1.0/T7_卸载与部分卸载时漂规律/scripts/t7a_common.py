# -*- coding: utf-8 -*-
"""T7-A（席位 A：卸载/全卸载的时漂规律 + 零基线漂移）公共层。

只读既有产物，只写本任务目录 `T7_卸载与部分卸载时漂规律/`。不修改 src/、CMake、
progress/ 下任何既有文件。

口径（全部按 `00_共享/指标字典与口径.md`，脚本内不再另立）：
  * 时间轴   = `timestamp` 列（禁用 `elapsed`），np.interp 重采样到 100 Hz 均匀网格（dt=0.01 s）
  * `Z`      = Σ_c X[i,c]（总量）；`Z̄` = median(Z, k)，k = 0.5 s / dt（只用于指标，不做 τ=2 s 平滑）
  * 卸载事件（**冻结定义**，指标字典 §2.1）：`unload` = 减少量 ≥ 80% 当前电平；
    `partial unload` = 20%~80%；两者都要求**事件前后各 2 s 稳定**（用 `Z̄` 判稳）。
    本层的实现：`pre` 窗 = [t0−2, t0)、`post` 窗 = [t0+4, t0+6]（各 2 s，与加载表口径逐字一致）；
    `J = Z̄(post) − Z̄(pre)`（卸载为**负**）；判稳 = 两窗内 (p90−p10) ≤ 0.05·|J|。
  * 方向约定   ：`J < 0`；完成度 `u(τ) = (Z̄(pre) − Z̄(t0+τ)) / |J|`（卸载方向的"镜像 z_at_*"），
    即 `u` 越大＝卸载走得越远。所有时间指标都从 `t0` 起算，`t0` = 卸载沿起点
    （最后一个仍处于旧电平 10% 带内的帧）。报告中必须同时声明这两条。
  * 包结构     ：实录 ~40 ms 一包，卸载沿定位误差 ±1 包 ⇒ 时间指标给 ±1 包敏感性。

算法臂（**复制**原型到本目录并改 import，绝不修改原型）：
  raw   = 无补偿（原始读数，物理规律只能用这条线）
  e3s   = v5.1（现役，免责 3 s：FAST_S=3, EXEMPT_AWIN=1, LEV_ARM_S=3）
  v6    = v6 原型（pin，trim 关）
  v5    = v5（**含"空载自动归零" b_ 链路**，v5.1 已删除；用于 P3 的反事实对照）
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))            # .../T7_*/scripts
TASK = os.path.dirname(HERE)                                 # .../T7_*
PLAN = os.path.dirname(TASK)                                 # plan/v1.0
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))   # 仓库根
TEMP = os.path.join(ROOT, "temp")
PROG = os.path.join(ROOT, "temp", "v4.1flash", "progress")
RES = os.path.join(TASK, "results")
FIG = os.path.join(TASK, "figures")
CACHE = os.path.join(RES, "cache")
for _d in (RES, FIG, CACHE):
    os.makedirs(_d, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from t7a_glm53_v3 import GLM53v3          # noqa: E402
from t7a_glm53_v5 import GLM53v5          # noqa: E402
from t7a_glm53_v51 import GLM53v51        # noqa: E402
from t7a_glm53_v6 import GLM53v6          # noqa: E402

# ─────────────────────────── 数据清单（13 份录制）───────────────────────────
B = os.path.join(TEMP, "变化负载")
HOLD = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
VARY = [
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                       "20260917_133923_single_device_ee20bc",
                                       "device_001_seg000.csv")),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载",
                                "device_001_seg000.csv")),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "device_001_seg000.csv")),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv")),
]
ALL = HOLD + VARY
KIND = {t: ("恒载" if t in [a for a, _ in HOLD] else "实采") for t, _ in ALL}
DOM = {"恒载": "display/force(N)", "实采": "ADC"}
ARMS = ("raw", "e3s", "v6", "v5")
ARM_LABEL = {"raw": "raw (uncompensated)", "e3s": "v5.1 (exempt 3 s, in service)",
             "v6": "v6 (pin, trim off)", "v5": "v5 (with idle auto-zero b_, withdrawn)"}
ARM_COLOR = {"raw": "0.35", "e3s": "#ff7f0e", "v6": "#2ca02c", "v5": "#9467bd"}

# T4-A 冻结事件表（加载/卸载事件列名对齐的参照；只读）
T4A_MORPH = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results", "t4a_morphology.csv")
T4A_SYM = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results", "t4a_unload_symmetry.csv")


# ─────────────────────────────── 日志 ───────────────────────────────
class Log:
    def __init__(self, name):
        self.lines = []
        self.path = os.path.join(RES, f"_t7a_{name}.log")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)

    def close(self, cmd=""):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"# run: {cmd}\n" if cmd else "")
            f.write("\n".join(self.lines) + "\n")
        print(f"-> {self.path}")


def log_reconfigure():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                              # noqa: BLE001
        pass


# ─────────────────────────────── 读取 ───────────────────────────────
def load_rec(path):
    """自动定位 `##Data`（不得写死 skiprows=24），返回 (t, X, cols)。t 用 timestamp 列。"""
    hdr = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("##Data"):
                hdr = i + 1
                break
    if hdr is None:
        raise RuntimeError("no ##Data marker in " + path)
    df = pd.read_csv(path, skiprows=hdr)
    ch = [c for c in df.columns if str(c).startswith("ch")]
    if not ch:
        raise RuntimeError("no ch* columns in " + path)
    t = df["timestamp"].to_numpy(float)
    return t - t[0], df[ch].to_numpy(float), ch


def prep(tag, path=None, fs=100.0):
    """读一份录制 → 100 Hz 均匀网格。返回 dict（含包间隔 pkt_dt 与域声明）。"""
    if path is None:
        path = dict(ALL)[tag]
    t, X, ch = load_rec(path)
    span = float(t[-1] - t[0])
    dtm = 1.0 / fs
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    d = np.diff(t)
    dp = d[d > 0]
    # 包间隔口径：取正差分的 **p90**（= 记录里最典型的"跨包"步长）。
    # 不能用中位：实录每包含 ~4 行、行间时间戳只差 1~2 µs，中位会掉到 1e-5 s 而失去意义。
    pkt = float(np.percentile(dp, 90)) if len(dp) else dtm
    pkt_med = float(np.median(dp)) if len(dp) else dtm
    return dict(tag=tag, path=path, kind=KIND[tag], dom=DOM[KIND[tag]], nch=X.shape[1],
                cols=ch, t=t, X=X, tu=tu, Xu=Xu, dtm=dtm, fs=fs, span=span,
                tot=Xu.sum(axis=1), tot_raw=X.sum(axis=1), pkt_dt=pkt, pkt_dt_med=pkt_med,
                zero_dup_frac=float((d <= 0).mean()),
                xmin=float(X.min()), xzero_frac=float((X <= 0).mean()))


def med_smooth(x, k):
    """居中滑动中值（k 帧）。k = 0.5 s / dt ⇒ 满足"禁止 τ=2 s 平滑"的口径。"""
    k = max(1, int(round(k)))
    return pd.Series(np.asarray(x, float)).rolling(k, center=True, min_periods=1).median().to_numpy()


def zbar(tot, dtm, sec=0.5):
    return med_smooth(tot, max(1, int(round(sec / dtm))))


def wmed(x, i0, i1):
    i0, i1 = int(max(0, i0)), int(min(len(x), i1))
    return float(np.median(x[i0:i1])) if i1 > i0 else float("nan")


def p10p90(x):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    return float(np.percentile(x, 10)), float(np.percentile(x, 90))


# ────────────────────── 卸载事件检测（冻结定义）──────────────────────
def detect_edges(tu, tot, dtm, ds=None, min_frac=0.10, abs_frac=0.02, nms_s=2.0):
    """短滞后电平差的双向边沿库存（只用于"下一个事件"定位与审计，不用于判据）。

    Δ(i) = med(Z̄[i−1.5,i−0.3]) − med(Z̄[i+0.3,i+1.5])；
    门限 = max(min_frac·pre_loc, abs_frac·(p99.5−p5))——**必须带绝对下限**，
    否则空载底附近的 10% 门槛会在噪声里造出成片假边沿（实测 3 份恒载因此被污染）。
    与 `13-v6-assessment/b_common.unload_events` 同法（但本层不使用它的 t_dn：
    指标字典要求用**真实沿**，检测时刻不是真沿）。
    """
    ds = zbar(tot, dtm) if ds is None else ds
    n = len(ds)
    k1 = max(3, int(round(1.5 / dtm)))
    k2 = max(1, int(round(0.3 / dtm)))
    rm = pd.Series(ds).rolling(max(2, k1 - k2), min_periods=1).median().to_numpy()
    prem = np.full(n, np.nan)
    postm = np.full(n, np.nan)
    prem[k2:] = rm[:n - k2]
    postm[:n - k1] = rm[k1:]
    dlt = postm - prem
    amp = float(np.percentile(ds, 99.5) - np.percentile(ds, 5))
    thr = np.maximum(min_frac * np.maximum(prem, 0.0), abs_frac * max(amp, 1e-9))
    cand = [(i, float(dlt[i])) for i in range(n)
            if np.isfinite(dlt[i]) and abs(dlt[i]) >= thr[i]]
    cand.sort(key=lambda z: -abs(z[1]))
    keep = []
    for i, v in cand:
        if all(abs(i - j) > int(nms_s / dtm) or np.sign(v) != np.sign(w) for j, w in keep):
            keep.append((i, v))
    keep.sort()
    return [dict(i_cand=int(i), t_cand=float(tu[i]), dlt=float(v), sign=int(np.sign(v)),
                 pre_loc=float(prem[i])) for i, v in keep], ds


def _adaptive_post(ds, tu, dtm, t0, next_edge_t, strict_len_s=2.0, adapt_min_s=1.0):
    """卸载后窗（严格 = 指标字典 [t0+4, t0+6]；自适应用「下一个检出事件 −0.5 s」截断）。"""
    b0, b1 = t0 + int(4.0 / dtm), t0 + int(6.0 / dtm)
    strict_ok = b1 <= len(ds) - 1
    if next_edge_t is not None and np.isfinite(next_edge_t):
        b1 = min(b1, int(round((next_edge_t - 0.5) / dtm)))
    lens = (b1 - b0) * dtm
    if lens >= adapt_min_s and b1 > b0:
        return b0, b1, float(lens), True
    return b0, b1, float(max(0.0, lens)), False


def detect_unloads(tu, tot, dtm, pkt_dt=0.0, stab_frac=0.05, min_step_frac=0.20,
                   nms_s=2.0):
    """冻结定义实现。返回 (accepted_strict, all_rows, edges, ds)。

    判据（指标字典 §2.1 写死）：
      pre 窗 = [t0−2, t0)、post 窗 = [t0+4, t0+6]（各 2 s）；J = Z̄(post) − Z̄(pre)（卸载为负）；
      step_frac = |J|/Z̄(pre)；unload ≥ 0.80；partial_unload 0.20~0.80；更小记 too_small；
      稳定判据 stab = (p90−p10)/|J| ≤ 0.05，pre 与 post 两窗都要过。
    偏差（已声明，见报告 §2）：
      * 当 post 窗被 6 s 内的下一次变载污染时，严格判据会剔除该事件；本层额外给一组
        `*_adapt` 列（post 窗用「下一个检出事件 − 0.5 s」截断，窗长 ≥1 s），作为
        **敏感性对照**，主结论仍用严格集。理由是固定 [t0+4,t0+6] 窗在"卸载后 5 s 内
        又加载"的场景必然误判（实测 切换负载 @129/182/212 s、再切换负载 @30.4 s）。
    """
    edges, ds = detect_edges(tu, tot, dtm, min_frac=0.10, nms_s=nms_s)
    n = len(ds)
    rows = []
    for e in edges:
        if e["sign"] >= 0:
            continue
        i = e["i_cand"]
        pre_loc = e["pre_loc"]
        post_loc = wmed(ds, i + int(0.3 / dtm), i + int(1.5 / dtm))
        Jloc = post_loc - pre_loc
        if not (Jloc < 0):
            continue
        lo = max(0, i - int(1.0 / dtm))
        hi = min(n - 1, i + int(1.0 / dtm))
        thr_up = pre_loc - 0.10 * abs(Jloc)
        idx = np.where(ds[lo:hi + 1] >= thr_up)[0]
        if len(idx) == 0:
            continue
        t0 = int(lo + idx[-1])
        a0, a1 = t0 - int(2.0 / dtm), t0
        if a0 < 0:
            continue
        pre = float(np.median(ds[a0:a1]))
        nxt = [x["t_cand"] for x in edges if x["t_cand"] > float(tu[t0]) + 2.0]
        nxt_t = min(nxt) if nxt else None
        b0, b1, pdur, post_ok = _adaptive_post(ds, tu, dtm, t0, nxt_t)
        post_win = ds[b0:b1] if b1 > b0 else np.array([np.nan])
        post = float(np.nanmedian(post_win)) if len(post_win) else float("nan")
        J = post - pre
        step_frac = abs(J) / max(pre, 1e-9) if np.isfinite(J) else float("nan")
        st_pre = (np.percentile(ds[a0:a1], 90) - np.percentile(ds[a0:a1], 10)) / abs(J) \
            if np.isfinite(J) and abs(J) > 1e-9 else float("nan")
        st_post = (np.percentile(post_win, 90) - np.percentile(post_win, 10)) / abs(J) \
            if np.isfinite(J) and abs(J) > 1e-9 else float("nan")
        # 严格窗（指标字典原样）另算一份，便于对照
        s0, s1 = t0 + int(4.0 / dtm), t0 + int(6.0 / dtm)
        if s1 <= n - 1:
            pre_s = pre
            post_s = float(np.median(ds[s0:s1]))
            J_s = post_s - pre_s
            st_s = (np.percentile(ds[s0:s1], 90) - np.percentile(ds[s0:s1], 10)) / abs(J_s) \
                if abs(J_s) > 1e-9 else float("nan")
            sf_s = abs(J_s) / max(pre_s, 1e-9)
        else:
            post_s, J_s, st_s, sf_s = float("nan"), float("nan"), float("nan"), float("nan")
        kind_s = ("unload" if sf_s >= 0.80 else "partial_unload" if sf_s >= 0.20
                  else "too_small") if np.isfinite(sf_s) else "post_unavail"
        stable_s = bool(np.isfinite(st_s) and st_s <= stab_frac
                        and np.isfinite(st_pre) and st_pre <= stab_frac
                        and np.isfinite(J_s) and J_s < 0)
        kind_a = ("unload" if step_frac >= 0.80 else "partial_unload" if step_frac >= 0.20
                  else "too_small") if np.isfinite(step_frac) else "post_unavail"
        stable_a = bool(post_ok and np.isfinite(st_post) and st_post <= stab_frac
                        and np.isfinite(st_pre) and st_pre <= stab_frac
                        and np.isfinite(J) and J < 0)
        rows.append(dict(i0=t0, t0=float(tu[t0]), pre=pre,
                         post=post_s, J=J_s, step_frac=sf_s, stab_pre=float(st_pre),
                         stab_post=float(st_s), stable=stable_s, kind=kind_s,
                         post_adapt=post, J_adapt=J, step_frac_adapt=step_frac,
                         stab_post_adapt=float(st_post), stable_adapt=stable_a,
                         kind_adapt=kind_a, post_win_s=pdur, pkt_dt=float(pkt_dt),
                         i_cand=int(i), t_cand=float(tu[i]),
                         t_next_edge=(float(nxt_t) if nxt_t is not None else float("nan"))))
    acc = [r for r in rows if r["kind"] in ("unload", "partial_unload") and r["stable"]]
    return acc, rows, edges, ds


# ───────────────────── 空载平台 / 受载段（用于零点与轮次）─────────────────────
def parse_plateaus(ds, tu, dtm, floor_frac=0.10, min_idle_s=0.6, min_ep_s=2.0):
    """迟滞分段：idle_mask = Z̄ < 底 + floor_frac·(峰−底)。返回 (idle_gaps, episodes, floor, peak)。

    idle_gaps  = 空载平台（长 ≥ min_idle_s）；episodes = 受载段（长 ≥ min_ep_s）。
    两者都是 100 Hz 网格上的 [i0, i1) 半开区间。
    """
    floor = float(np.percentile(ds, 5))
    peak = float(np.percentile(ds, 99.5))
    amp = peak - floor
    if amp <= 1e-9:
        return [], [], floor, peak
    thr = floor + floor_frac * amp
    m = ds < thr
    n = len(m)
    runs = []
    cur, a = bool(m[0]), 0
    for i in range(1, n):
        if bool(m[i]) != cur:
            runs.append((cur, a, i))
            cur, a = bool(m[i]), i
    runs.append((cur, a, n))
    gaps, eps = [], []
    for is_idle, a, b in runs:
        if is_idle and (b - a) * dtm >= min_idle_s:
            gaps.append(dict(i0=int(a), i1=int(b), t0=float(tu[a]),
                             t1=float(tu[min(b, n - 1)]),
                             lvl=float(np.median(ds[a:b]))))
        elif (not is_idle) and (b - a) * dtm >= min_ep_s:
            eps.append(dict(i0=int(a), i1=int(b), t0=float(tu[a]),
                            t1=float(tu[min(b, n - 1)])))
    return gaps, eps, floor, peak


def episode_level(ds, ep, dtm, skip_in_s=2.0, skip_out_s=0.5):
    a = ep["i0"] + int(skip_in_s / dtm)
    b = ep["i1"] - int(skip_out_s / dtm)
    if b <= a:
        a, b = ep["i0"], ep["i1"]
    return float(np.median(ds[a:b])), float(np.max(ds[ep["i0"]:ep["i1"]]))


def zero_level(ds, gap, dtm, skip_in_s=0.5, use_last_s=3.0):
    """空载平台电平：跳过起始瞬态后取平台内最后 use_last_s（取不到就取整段）。"""
    a = gap["i0"] + int(skip_in_s / dtm)
    b = gap["i1"]
    a2 = max(a, b - int(use_last_s / dtm))
    if b <= a2:
        a2 = a
    if b <= a2:
        return float("nan"), 0
    return float(np.median(ds[a2:b])), int(b - a2)


# ─────────────────────────── 算法臂运行 ───────────────────────────
def build_arm(arm, n):
    if arm == "raw":
        return None
    if arm == "e3s":
        c = GLM53v51(n)
        c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = 3.0, 1.0, 3.0
        return c
    if arm == "v6":
        return GLM53v6(n)
    if arm == "v5":
        return GLM53v5(n)
    raise ValueError(arm)


def _internal_ded(c, v):
    """内部蠕变扣除量 Σ（不含输出封顶/hold 冻结）——用于与"实际生效扣除"对照。"""
    try:
        if hasattr(c, "_deduction_vector"):
            return float(np.sum(c._deduction_vector(np.asarray(v, float))))
        if hasattr(c, "_deduction"):
            return float(np.sum(c._deduction()))
    except Exception:                                              # noqa: BLE001
        pass
    return float("nan")


def run_arm(arm, tu, Xu, trace_stride=10):
    """逐帧运行一个臂。返回 dict：
       Y(总量), ded_eff(=原始总量−补偿后总量，实际生效扣除), ded_int(内部 γAg), 
       state, hold, g, A_sum, b_sum, gamma_med；轨迹按 trace_stride 抽稀。
    """
    n, nch = len(tu), Xu.shape[1]
    tot = Xu.sum(axis=1)
    if arm == "raw":
        return dict(arm=arm, Y=tot.copy(), ded_eff=np.zeros(n), ded_int=np.zeros(n),
                    state=np.zeros(n, int), hold=np.zeros(n, bool), g=np.zeros(n),
                    Asum=np.zeros(n), bsum=np.zeros(n), gmed=np.ones(n), c=None)
    c = build_arm(arm, nch)
    Y = np.empty(n)
    ded_eff = np.empty(n)
    ded_int = np.empty(n)
    state = np.zeros(n, int)
    hold = np.zeros(n, bool)
    gs = np.zeros(n)
    asum = np.zeros(n)
    bsum = np.zeros(n)
    gmed = np.ones(n)
    for i in range(n):
        v = Xu[i]
        y = c.process(float(tu[i]), v)
        Y[i] = float(np.sum(y))
        ded_eff[i] = float(np.sum(v)) - Y[i]
        ded_int[i] = _internal_ded(c, v)
        st = getattr(c, "state", None)
        if st is None:
            inload = bool(getattr(c, "in_load", False))
            state[i] = 2 if inload else 0
        else:
            state[i] = {"idle": 0, "event": 1, "slow": 2}.get(st, 0)
        hold[i] = bool(getattr(c, "hold", False)) or (getattr(c, "hold_comp", None) is not None)
        gs[i] = float(getattr(c, "g", 0.0))
        A = np.asarray(getattr(c, "A", np.zeros(nch)), float)
        asum[i] = float(A.sum())
        b = getattr(c, "b", None)
        bsum[i] = float(np.sum(b)) if b is not None else 0.0
        gm = getattr(c, "gamma", None)
        gmed[i] = float(np.median(gm)) if gm is not None else 1.0
    return dict(arm=arm, Y=Y, ded_eff=ded_eff, ded_int=ded_int, state=state, hold=hold,
                g=gs, Asum=asum, bsum=bsum, gmed=gmed, c=c,
                epoch_t=list(getattr(c, "epoch_t", [])),
                kind_log=list(getattr(c, "kind_log", [])))


def run_all_arms(tu, Xu, arms=ARMS):
    return {a: run_arm(a, tu, Xu) for a in arms}


def cache_npz(tag):
    return os.path.join(CACHE, tag.replace("/", "_") + ".npz")


def save_rec_cache(d, arms):
    blob = dict(tu=d["tu"].astype(np.float32), tot=d["tot"].astype(np.float32),
                Xu=d["Xu"].astype(np.float32), dtm=np.array([d["dtm"]]),
                pkt_dt=np.array([d["pkt_dt"]]), nch=np.array([d["nch"]]))
    for a, r in arms.items():
        blob["Y_" + a] = r["Y"].astype(np.float32)
        blob["dedeff_" + a] = r["ded_eff"].astype(np.float32)
        blob["state_" + a] = r["state"].astype(np.int8)
        blob["g_" + a] = r["g"].astype(np.float32)[::10]
        blob["Asum_" + a] = r["Asum"].astype(np.float32)[::10]
        blob["bsum_" + a] = r["bsum"].astype(np.float32)[::10]
        blob["hold_" + a] = r["hold"].astype(np.int8)
    np.savez_compressed(cache_npz(d["tag"]), **blob)


def load_rec_cache(tag):
    z = np.load(cache_npz(tag), allow_pickle=True)
    return z


# ─────────────────────────── 小工具 ───────────────────────────
def first_at(tu, ds, i0, cond):
    """返回满足 cond 的首个 τ（s，从 tu[i0] 起算），无则 NaN。"""
    idx = np.where(cond[i0:])[0]
    return float(tu[i0 + idx[0]] - tu[i0]) if len(idx) else float("nan")


def linfit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    p = np.polyfit(x[m], y[m], 1)
    yh = np.polyval(p, x[m])
    r2 = 1.0 - float(np.sum((y[m] - yh) ** 2)) / max(float(np.sum((y[m] - y[m].mean()) ** 2)), 1e-12)
    return float(p[0]), float(p[1]), r2


def spearman(x, y):
    from scipy.stats import spearmanr, pearsonr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return dict(n=int(m.sum()), rho=float("nan"), p_rho=float("nan"),
                    r=float("nan"), p_r=float("nan"))
    rho, pr = spearmanr(x[m], y[m])
    r, pp = pearsonr(x[m], y[m])
    return dict(n=int(m.sum()), rho=float(rho), p_rho=float(pr), r=float(r), p_r=float(pp))


def save(df, name):
    p = os.path.join(RES, name)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"-> results/{name}  ({len(df)} 行)")
    return p


def load_t4a_morph():
    return pd.read_csv(T4A_MORPH)


def load_t4a_sym():
    return pd.read_csv(T4A_SYM)


def rec_key(tag):
    """T4-A 事件表里的 rec 命名（恒载为 '右拇指指尖/数据1'，与 tag 一致；实录用短名）。"""
    return tag
