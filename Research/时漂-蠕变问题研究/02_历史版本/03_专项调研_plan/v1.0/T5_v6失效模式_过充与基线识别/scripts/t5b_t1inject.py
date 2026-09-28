# -*- coding: utf-8 -*-
"""T1-B 产出：**全项目共享的扰动注入器**（`00-共享/指标字典与口径.md` §5 的参考实现）。

来源与血缘
----------
主体由 `temp/v4.1flash/progress/13-v6-assessment/scripts/a_common.py` **复制并改名**
（复制件，原件未改、不被 import），在其基础上补充：

  * `make_perturb` 增加 `kind="white_common"`（白噪但通道间部分相关）与显式 `fs`；
  * 新增 `add_packet_jitter`（包到达时序抖动 ±k 包）、`add_dropout`（单帧丢包）、
    `add_packet_aggregate`（包聚合：把 k 帧时间戳改成同值）；
  * 新增帧级注入通道：抖动/丢包/聚合**必须在重采样前**作用于原始帧时间戳，
    因此提供 `to_grid(t, X)` 与 `load_uniform` 的姊妹函数，注入顺序在 docstring 里写死；
  * 新增指标实现 `t_stable` / `overshoot` / `settle_time`（口径 = 指标字典 §3，供 T5 直接调用）；
  * 新增 `packets()` 包结构解析（本批 13 份录制的真实包结构：指尖 ~15.4 ms/包、
    变载实录 ~40 ms/包，实测见 `results/t1b_probe.csv`）。

公开接口（T5 可直接复制本文件后调用）
------------------------------------
数据
    `REC`                            : dict[短键] = 绝对路径（4 份变载实录）
    `load_rec(path)`                 : -> (t, X)   自动定位 `##Data`；t 取 `timestamp` 列（减首值）
    `packets(t)`                     : -> dict(n_pkt, pkt_s, starts, ends, ts, frames_per_pkt)
    `to_grid(t, X, fs=100.0, tmax=None)`: -> (tu, Xu)  均匀网格（np.interp）
    `load_uniform(path, fs=100.0, tmax=None)` : -> dict(t,X,tu,Xu,dtm,span)
    `med_smooth(x, k)`               : 居中滚动中值

网格级加性注入（对 (n, n_ch) 的 Xu，总量口径）
    `make_perturb(X, A, kind, rng, f_lo=0.3, f_hi=40.0, corr=0.0, fs=100.0)` -> E
        kind ∈ {"white","band","common"}；`A` = **总量扰动的 RMS（ADC 或显示单位）**
    `add_tap(X, i0, amp, fs=100.0, rise_ms=50, hold_ms=100, fall_ms=None, ch_mask=None)` -> (Y, nr)
    `add_step(X, i0, amp, fs=100.0, rise_ms=50, ch_mask=None)` -> Y

帧级时序注入（对原始 (t, X)，**先注入、后 to_grid**）
    `add_packet_jitter(t, X, k_pkt=1, pkt_s=None, rng=None)` -> (t2, X2)
    `add_dropout(t, X, frac=0.01, rng=None)`                 -> (t2, X2)
    `add_packet_aggregate(t, X, k=2)`                        -> (t2, X2)

算法运行
    `make_traced(cls)`               : 子类化 v6 原型，只记录状态变化（不改行为）
    `run_traced(Cls, tu, X, **kw)`    : -> dict(Y, tot, ev, st, comp, epoch, revoke, handoff, unload, gevent)
    `run_plain(Cls, tu, X, **kw)`     : -> (Y, comp)

指标（口径 = `00-共享/指标字典与口径.md` §3）
    `step_amp(tu, Z, t_on, dtm=0.01)`                      : J = Z̄[t_on+4,t_on+6] − Z̄[t_on−2,t_on)
    `t_stable(tu, Z, t_on, J, ...)`                        : -> dict(tau_v1, tau_v2, ok)
    `overshoot(tu, Z, t_on, J, z_final=None, ...)`         : -> dict(os_pct, us_pct, z_final, zfin_src)
    `settle_time(tu, Z, t_on, J, z_final, eps=0.05)`       : -> 首次进入并保持 |Z−Z_final| ≤ ε·J 的时间

单位与域
    恒载 9 组 = `processed_display`（N，未标定，`force_conversion_active=false`）；
    变载实录 = **ADC**。扰动的 `A` 一律按"**该域总量**"计，跨域不可比（见报告 §2）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
TEMP = os.path.join(ROOT, "temp")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# T5-B 复制件改动（其余逐字复制 T1_*/scripts/t1_common.py）：
#   `from t1b_ad_lib import ...` → `from t5b_ad_lib import ...`（本目录复制件），
#   使本任务 scripts/ 可独立 `python xxx.py` 跑通（复用清单 §3.2）。
#   T1-B 冻结状态：**已就绪**，本任务注入器不是"待回对"状态。
from t5b_ad_lib import load_rec, med_smooth          # noqa: E402

REC = {
    "切换负载-快相无责": os.path.join(
        TEMP, "变化负载", "切换负载-快相无责的测试",
        "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"),
    "零负载-切换负载-零负载-再切换负载": os.path.join(
        TEMP, "变化负载", "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv"),
    "零负载-中途切换负载-零负载-切换负载": os.path.join(
        TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv"),
    "中途切换-最终测试目标": os.path.join(
        TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载", "最终测试目标",
        "device_001_seg000.csv"),
}

HOLD_REC = {
    "RT1": (os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"), "显示域"),
    "RT2": (os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"), "显示域"),
    "RT3": (os.path.join(TEMP, "右拇指指尖", "数据3", "device_001_seg000.csv"), "显示域"),
    "LT1": (os.path.join(TEMP, "左拇指指尖", "数据1", "device_001_seg000.csv"), "显示域"),
    "LT2": (os.path.join(TEMP, "左拇指指尖", "数据2", "device_001_seg000.csv"), "显示域"),
    "LT3": (os.path.join(TEMP, "左拇指指尖", "数据3", "device_001_seg000.csv"), "显示域"),
    "F41": (os.path.join(TEMP, "四指指尖", "数据1", "device_001_seg000.csv"), "显示域"),
    "F42": (os.path.join(TEMP, "四指指尖", "数据2", "device_001_seg000.csv"), "显示域"),
    "F43": (os.path.join(TEMP, "四指指尖", "数据3", "device_001_seg000.csv"), "显示域"),
}


# ───────────────────────── 数据 / 包结构 ─────────────────────────

def packets(t):
    """解析包结构：时间戳（四舍五入到 0.1 ms）相同的连续帧属同一包。"""
    key = np.round(np.asarray(t, float), 4)
    changes = np.where(np.diff(key) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(t)]])
    ts = key[starts]
    d = np.diff(ts)
    return dict(n_pkt=len(starts), starts=starts, ends=ends, ts=ts,
                frames_per_pkt=(ends - starts),
                pkt_s=float(np.median(d)) if len(d) else float("nan"))


def to_grid(t, X, fs=100.0, tmax=None):
    """`np.interp` 重采样到 fs 均匀网格。t 有重复时间戳（包结构）时按 np.interp 语义取值。"""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    if tmax is not None:
        m = t <= tmax
        t, X = t[m], X[m]
    span = float(t[-1] - t[0])
    dtm = 1.0 / fs
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu


def load_uniform(path, fs=100.0, tmax=None):
    t, X = load_rec(path)
    tu, Xu = to_grid(t, X, fs=fs, tmax=tmax)
    return dict(t=t, X=X, tu=tu, Xu=Xu, dtm=1.0 / fs, span=float(t[-1] - t[0]))


# ───────────────────────── 网格级加性注入 ─────────────────────────

def _bandlimited(n, fs, f_lo, f_hi, rng):
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    W[(f < f_lo) | (f > f_hi)] = 0.0
    x = np.fft.irfft(W, n)
    s = x.std()
    return x / s if s > 1e-12 else x


def make_perturb(X, A, kind, rng, f_lo=0.3, f_hi=40.0, corr=0.0, fs=100.0):
    """加性扰动矩阵，**总量扰动的 RMS = A**（单位同 X 所在域）。

    kind: 'white' 每通道独立白噪 | 'band' 带限噪声 [f_lo,f_hi] |
          'common' 同相（通道间完全相关，按通道平均电平分摊，总量 RMS = A）
    corr: 白噪/带限的通道间相关比例（1 = 完全同相），0 = 独立。
    """
    X = np.asarray(X, float)
    n, n_ch = X.shape
    if kind == "common":
        s = rng.standard_normal(n)
        s = s / s.std()
        base = np.abs(X).mean(axis=0)
        w = base / base.sum() if base.sum() > 1e-9 else np.ones(n_ch) / n_ch
        E = np.outer(s, w)
    else:
        E = np.zeros((n, n_ch))
        for c in range(n_ch):
            E[:, c] = (rng.standard_normal(n) if kind == "white"
                       else _bandlimited(n, fs, f_lo, f_hi, rng))
        if corr > 0:
            sc = rng.standard_normal(n)
            sc = sc / sc.std()
            E = (1.0 - corr) * E + corr * np.outer(sc, np.ones(n_ch))
    E = E - E.mean(axis=0, keepdims=True)
    tot = E.sum(axis=1)
    rms = tot.std()
    return np.zeros_like(E) if rms < 1e-12 else E * (A / rms)


def _profile(X, i0, ch_mask):
    prof = np.where(ch_mask > 0, np.maximum(np.abs(X[i0]), 1.0), 0.0)
    return prof / prof.sum() if prof.sum() > 1e-9 else np.ones(X.shape[1]) / X.shape[1]


def add_tap(X, i0, amp, fs=100.0, rise_ms=50.0, hold_ms=100.0, fall_ms=None,
            ch_mask=None, shape="tap"):
    """拍击形态：上升 rise + 保持 hold + 回落 fall（余弦半波）；**总量峰值 = amp**。"""
    Y = np.asarray(X, float).copy()
    if fall_ms is None:
        fall_ms = rise_ms
    nr = max(1, int(round(rise_ms / 1000.0 * fs)))
    nh = max(1, int(round(hold_ms / 1000.0 * fs)))
    nf = max(1, int(round(fall_ms / 1000.0 * fs)))
    up = amp * (1.0 - np.cos(np.linspace(0, np.pi, nr))) / 2.0 if shape == "tap" else np.linspace(0, amp, nr)
    hd = np.full(nh, amp)
    dn = amp * (1.0 + np.cos(np.linspace(0, np.pi, nf))) / 2.0
    w = np.concatenate([up, hd, dn])
    i1 = min(len(Y), i0 + len(w))
    w = w[: i1 - i0]
    if ch_mask is None:
        ch_mask = np.ones(Y.shape[1])
    Y[i0:i1] += np.outer(w, _profile(Y, i0, ch_mask))
    return Y, nr


def add_step(X, i0, amp, fs=100.0, rise_ms=50.0, ch_mask=None):
    """一次真实 restep：同样的上升沿但**不回落**（合成对照用，不用于主结论）。"""
    Y = np.asarray(X, float).copy()
    nr = max(1, int(round(rise_ms / 1000.0 * fs)))
    w = amp * (1.0 - np.cos(np.linspace(0, np.pi, nr))) / 2.0
    n = len(Y)
    ramp = np.concatenate([w, np.full(max(0, n - i0 - nr), amp)])[: n - i0]
    if ch_mask is None:
        ch_mask = np.ones(Y.shape[1])
    Y[i0:] += np.outer(ramp, _profile(Y, i0, ch_mask))
    return Y


# ───────────────────────── 帧级时序注入 ─────────────────────────

def add_packet_jitter(t, X, k_pkt=1, pkt_s=None, rng=None):
    """**包到达时序抖动**：每个包的到达时刻独立均匀偏移 ±k_pkt 个包周期。

    模型（写死，报告 §2 有说明）：
      1. 用 `packets(t)` 解析真实包结构（本批：指尖 15.4 ms/包、实录 40 ms/包）；
      2. 第 j 包的到达时刻 t̂_j = t_j + U(−k·T_pkt, +k·T_pkt)，强制严格递增
         （`np.maximum.accumulate` + 1e-4 s 最小间隔）以免插值退化；
      3. 包内各帧沿用本包的新时间戳（保留"包内同时到达"语义）；
      4. 返回**新的帧级 (t2, X2)**，调用方随后再做 `to_grid`。
    """
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    if rng is None:
        rng = np.random.default_rng(0)
    pk = packets(t)
    T = float(pkt_s) if pkt_s is not None else pk["pkt_s"]
    jit = rng.uniform(-k_pkt * T, k_pkt * T, size=pk["n_pkt"])
    new_ts = pk["ts"] + jit
    new_ts = np.maximum.accumulate(new_ts + np.arange(len(new_ts)) * 1e-4)
    t2 = np.empty_like(t)
    for j in range(pk["n_pkt"]):
        t2[pk["starts"][j]:pk["ends"][j]] = new_ts[j]
    order = np.argsort(t2, kind="stable")
    return t2[order] - t2[order][0], X[order]


def add_dropout(t, X, frac=0.01, rng=None, whole_packet=False):
    """**丢包**：`whole_packet=False` = 单帧丢包（按帧独立概率 frac）；
    `whole_packet=True` = 整包丢失。返回剔除后的 (t2, X2)。"""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    if rng is None:
        rng = np.random.default_rng(0)
    if whole_packet:
        pk = packets(t)
        keep = rng.random(pk["n_pkt"]) >= frac
        if not keep.any():
            keep[0] = True
        idx = np.concatenate([np.arange(pk["starts"][j], pk["ends"][j])
                              for j in range(pk["n_pkt"]) if keep[j]])
    else:
        idx = np.where(rng.random(len(t)) >= frac)[0]
        if len(idx) < 10:
            idx = np.arange(len(t))
    t2 = t[idx]
    return t2 - t2[0], X[idx]


def add_packet_aggregate(t, X, k=2):
    """**包聚合**：每 k 帧合并为 1 个时间戳（模拟主机侧把 k 帧打成同包到达）。"""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    t2 = t.copy()
    n = len(t2)
    for s in range(0, n, k):
        t2[s:min(n, s + k)] = t2[s]
    return t2 - t2[0], X


# ───────────────────────── 算法运行（仪表化） ─────────────────────────

def make_traced(cls):
    """子类化 v6 原型，只记录状态变化，不改变行为（与 a_common.make_traced 同构）。"""
    class T(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.tr_epoch = []
            self.tr_revoke = []
            self.tr_handoff = []
            self.tr_unload = []
            self.tr_gevent = []

        def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
            super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
            self.tr_epoch.append((float(t0), str(kind), float(self.last_ts),
                                  float(base), float(np.sum(v0)), float(np.sum(y0))))

        def _handoff(self, ts, v, ev=None):
            ev = ev if ev is not None else self.ev
            rec = (float(ts), str(ev["kind"]), float(ev["A_hat"]), float(ev["c_applied"]))
            super()._handoff(ts, v, ev)
            self.tr_handoff.append(rec + (float(np.sum(self.A)), float(self.g)))

        def _to_idle(self):
            super()._to_idle()
            self.tr_unload.append(float(self.last_ts))

        def _run_event(self, ts, v, total, dt, eps, idle_now):
            ev = self.ev
            kind = str(ev["kind"]) if ev is not None else None
            n_rev = self.n_revoke
            e0 = float(ev["t0"]) if (ev is not None and kind == "decrease") else None
            out = super()._run_event(ts, v, total, dt, eps, idle_now)
            if kind == "decrease" and self.n_revoke > n_rev:
                self.tr_gevent.append((float(ts), kind, e0))
            return out

        def process(self, ts, v):
            epoch_n = len(self.tr_epoch)
            rev_n = self.n_revoke
            out = super().process(ts, v)
            if self.n_revoke > rev_n and len(self.tr_epoch) == epoch_n:
                self.tr_revoke.append((float(ts), "revoke"))
            return out
    return T


def run_traced(Cls, tu, X, **kw):
    c = Cls(X.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    n = len(tu)
    Y = np.empty_like(X)
    tot = np.empty(n)
    for i in range(n):
        Y[i] = c.process(float(tu[i]), X[i])
        tot[i] = X[i].sum()
    return dict(Y=Y, tot=tot, comp=c, epoch=getattr(c, "tr_epoch", []),
                revoke=getattr(c, "tr_revoke", []), handoff=getattr(c, "tr_handoff", []),
                unload=getattr(c, "tr_unload", []), gevent=getattr(c, "tr_gevent", []))


def run_plain(Cls, tu, X, **kw):
    c = Cls(X.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    Y = np.empty_like(X)
    for i in range(len(tu)):
        Y[i] = c.process(float(tu[i]), X[i])
    return Y, c


# ───────────────────────── 指标（指标字典 §3） ─────────────────────────

def step_amp(tu, Zraw, t_on, dtm=0.01, smooth_s=0.5):
    """J = Z̄(post 窗中位) − Z̄(pre 窗中位)，pre=[t_on−2,t_on)、post=[t_on+4,t_on+6]（字典 §2.1）。"""
    Zb = med_smooth(Zraw, max(3, int(round(smooth_s / dtm))))
    i0 = int(round(t_on / dtm))
    ia, ib = max(0, i0 - int(2 / dtm)), max(1, i0)
    ja, jb = min(len(Zb), i0 + int(4 / dtm)), min(len(Zb), i0 + int(6 / dtm))
    if jb - ja < 3 or ib - ia < 3:
        return float("nan")
    return float(np.median(Zb[ja:jb]) - np.median(Zb[ia:ib]))


def t_stable(tu, Z, t_on, J, dtm=0.01, horizon=30.0, tol=0.05, smooth_s=0.5,
             tail_s=10.0, tail_tol=0.02):
    """T_stable（指标字典 §3 主指标的中性实现）。

    定义：从 t_on 起，**首次**存在 τ 使 t ∈ [t_on+τ, t_on+τ+30 s] 内
          max|Z̄(t) − Z̄(t_on+τ)| ≤ 5%·|J|（此后不再移动）。
    `tau_v1` = 只用上面这条；`tau_v2` = 追加"该窗最后 `tail_s` 秒净漂移 ≤ `tail_tol`·|J|"
    （即字典里"不再出现同向移动"的严格化）。
    `ok=False` 表示记录在 `t_on + τ + horizon` 之前结束或 J 无效（**不可测**，不是 0）。

    返回 dict(tau_v1, tau_v2, ok, i_v1, i_v2)
    """
    Zb = med_smooth(np.asarray(Z, float), max(3, int(round(smooth_s / dtm))))
    nh = int(round(horizon / dtm))
    nt = max(1, int(round(tail_s / dtm)))
    i0 = int(round(t_on / dtm))
    out = dict(tau_v1=float("nan"), tau_v2=float("nan"), ok=False, i_v1=-1, i_v2=-1)
    if not np.isfinite(J) or abs(J) < 1e-9 or i0 < 0 or i0 + nh >= len(Zb):
        return out
    band = tol * abs(J)
    for i in range(i0, len(Zb) - nh):
        seg = Zb[i:i + nh + 1]
        if np.max(np.abs(seg - seg[0])) <= band:
            out["tau_v1"] = float(tu[i] - t_on)
            out["i_v1"] = i
            tail = Zb[i + nh - nt:i + nh + 1]
            if abs(tail[-1] - tail[0]) <= tail_tol * abs(J):
                out["tau_v2"] = float(tu[i] - t_on)
                out["i_v2"] = i
            out["ok"] = True
            break
    if out["ok"] and not np.isfinite(out["tau_v2"]):
        for i in range(out["i_v1"], len(Zb) - nh):
            seg = Zb[i:i + nh + 1]
            if np.max(np.abs(seg - seg[0])) <= band:
                tail = Zb[i + nh - nt:i + nh + 1]
                if abs(tail[-1] - tail[0]) <= tail_tol * abs(J):
                    out["tau_v2"] = float(tu[i] - t_on)
                    out["i_v2"] = i
                    break
    return out


def z_final_level(tu, Z, t_on, dtm=0.01, ref_s=60.0, win_s=10.0, smooth_s=0.5):
    """Z_final：字典口径 = `t_on+60 s` 之后 10 s 的中位。不够长时退回平台中位（标 `plateau`）。"""
    Zb = med_smooth(np.asarray(Z, float), max(3, int(round(smooth_s / dtm))))
    i = int(round((t_on + ref_s) / dtm))
    j = int(round((t_on + ref_s + win_s) / dtm))
    if j < len(Zb):
        return float(np.median(Zb[i:j])), "ref60"
    i2 = int(round((t_on + 10.0) / dtm))
    if len(Zb) - i2 >= int(10.0 / dtm):
        return float(np.median(Zb[i2:])), "plateau"
    return float("nan"), "none"


def overshoot(tu, Z, t_on, J, z_final=None, dtm=0.01, win_s=30.0, smooth_s=0.5):
    """OS% = max_t (Z̄(t) − Z_final)/J（t ∈ [t_on, t_on+30 s]）；US% = |min 下冲|/J。"""
    Zb = med_smooth(np.asarray(Z, float), max(3, int(round(smooth_s / dtm))))
    if z_final is None:
        z_final, src = z_final_level(tu, Z, t_on, dtm=dtm, smooth_s=smooth_s)
    else:
        src = "given"
    i0 = int(round(t_on / dtm))
    i1 = min(len(Zb), i0 + int(round(win_s / dtm)))
    if not np.isfinite(z_final) or not np.isfinite(J) or abs(J) < 1e-9 or i1 <= i0:
        return dict(os_pct=float("nan"), us_pct=float("nan"), z_final=z_final, zfin_src=src)
    dev = Zb[i0:i1] - z_final
    return dict(os_pct=float(100.0 * dev.max() / J),
                us_pct=float(100.0 * max(0.0, -dev.min()) / J),
                z_final=float(z_final), zfin_src=src)


def settle_time(tu, Z, t_on, J, z_final, eps=0.05, dtm=0.01, smooth_s=0.5):
    """T_settle(ε)：首次进入并**保持** |Z̄(t) − Z_final| ≤ ε·|J| 到记录末的时间。"""
    Zb = med_smooth(np.asarray(Z, float), max(3, int(round(smooth_s / dtm))))
    i0 = int(round(t_on / dtm))
    if not np.isfinite(z_final) or not np.isfinite(J) or abs(J) < 1e-9 or i0 >= len(Zb):
        return float("nan")
    band = eps * abs(J)
    inside = np.abs(Zb - z_final) <= band
    for i in range(i0, len(inside)):
        if inside[i] and inside[i:].all():
            return float(tu[i] - t_on)
    return float("nan")
