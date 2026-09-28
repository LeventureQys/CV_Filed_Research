# -*- coding: utf-8 -*-
"""因果（流式）评测框架。

设计
----
* `run_stream` 以 `step(n, y, t)` 逐帧推进，frame n 只能访问 y[n] 及历史。
* `verify_causality` 做前缀一致性校验：把序列截断到前 m 帧重跑，
  前 m-1 帧输出必须逐位一致——抓任何"偷看未来"的实现。
* `prepare` 给出真值：恒定负载的真值取"双时间尺度形态模型"的外推水平
  F̂∞（模型在 log 空间对整段负载拟合，详见 c0_creep_shape.py）。
  同时给出多个"窗口真值"用于敏感性分析。
"""
import os
import sys
import copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, DATASETS


# ---------------------------------------------------------------- 数据准备

def causal_smooth(y, w):
    if w <= 1:
        return y.copy()
    cs = np.concatenate([[0.0], np.cumsum(y)])
    n = len(y)
    idx = np.arange(n)
    lo = np.maximum(0, idx - w + 1)
    return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)


def prepare(name):
    D = load_dataset(name)
    fi = D["frame_index"]
    k, _ = np.polyfit(fi, D["t"], 1)
    t = fi * k
    fs = 1.0 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), t, fs=fs)
    a0, a1 = seg["pre"]; c0, d0 = seg["load"]
    e0, e1 = seg["edge"]
    return dict(name=name, X=X, t=t, fs=fs, ch=D["ch_cols"], n=len(t),
                pre=(int(a0), int(a1)), load=(int(c0), int(d0)),
                post=(int(d0), int(len(t))), seg=seg,
                on_true=int(e0), off_true=int(e1))


def online_load_detect(tot, fs, quiet_frames, k_on=3, factor=0.12, smooth_s=0.15,
                       min_margin_n=0.10):
    """在线加载检测（严格因果，只用历史）。

    本数据的加载沿是 ~0.5s 的陡升（总量从静息 ~0.1N 升到 ~12N），因此用
    "因果 EWMA 平滑 + 相对判决"即可稳健检测：

      1. 在线维护运行最大值 S*（只用历史）
      2. 判决阈值 thr = base + max(factor·(S*-base), min_margin_n)
         base 为开头 quiet_frames 帧的中位数
      3. 平滑总量连续 k_on 帧超过 thr → 判为加载

    返回 (n_on, base, thr)；n_on=None 表示未检测到。
    """
    tot = np.asarray(tot, float)
    m = min(len(tot), max(quiet_frames, 20))
    w = max(1, int(round(smooth_s * fs)))
    a = 1.0 - 1.0 / w
    base = float(np.median(tot[:m]))
    run = 0
    smax = float(np.max(tot[:m]))
    v = base
    for i in range(m, len(tot)):
        v = a * v + (1 - a) * tot[i]
        smax = max(smax, v)
        thr = base + max(factor * (smax - base), min_margin_n)
        if v > thr:
            run += 1
            if run >= k_on:
                return i - k_on + 1, base, thr
        else:
            run = 0
    return None, base, float("nan")


# ---------------------------------------------------------------- 流式执行

def run_stream(algo, X, t, fs, n_on):
    """流式执行；返回 (N, C) 输出。"""
    N, C = X.shape
    algo = copy.deepcopy(algo)
    algo.n0 = n_on
    algo.make(C, fs)
    out = np.empty((N, C))
    for n in range(N):
        out[n] = algo.step(n, X[n], t[n])
    return out


def run_stream_prefix(algo, X, t, fs, n_on, m):
    """只跑前 m 帧（用于因果性校验）。"""
    C = X.shape[1]
    a = copy.deepcopy(algo)
    a.n0 = n_on
    a.make(C, fs)
    out = np.empty((m, C))
    for n in range(m):
        out[n] = a.step(n, X[n], t[n])
    return out


def verify_causality(algo, X, t, fs, n_on, n_check=12, seed=0):
    """前缀一致性：截断到 m 帧的结果必须与完整运行的前 m 帧完全一致。"""
    N = len(t)
    full = run_stream(algo, X, t, fs, n_on)
    rng = np.random.default_rng(seed)
    ms = sorted({int(v) for v in np.concatenate([
        rng.integers(max(20, N // 20), N, size=n_check - 2),
        np.array([max(20, N // 3), N])])})
    worst = 0.0
    for m in ms:
        sub = run_stream_prefix(algo, X, t, fs, n_on, m)
        d = float(np.nanmax(np.abs(sub - full[:m])))
        worst = max(worst, d)
    return worst


# ---------------------------------------------------------------- 评测

def evaluate(algo, D, quiet_frames=None, n_on=None, n_prefix=-1, act=None,
             win_s=1.0, ref_win=(1.0, 3.0), late_win=5.0, plateau_win=10.0,
             noise_bw=(0.01, 1.0)):
    """流式运行 + 指标计算（抗"作弊"口径）。

    为什么不用"相对漂移比"作为主指标
    --------------------------------
    drift_algo/drift_raw 这种比值会被"把信号整体压到 0"的算法刷到负无穷，
    完全没有工程意义。因此改用**绝对量**：

    主指标（全部相对"原始信号 1~3s 的稳定水平 F_ref"表达）：
      F_ref        : 加载后 ref_win（默认 1~3s）原始读数均值，作为该通道的标尺
      drift_raw_mN : 原始信号 (末 late_win 均值 - F_ref)，mN
      drift_algo_mN: 算法输出 (末 late_win 均值 - F_ref)，mN   ← 越小越好；=0 完美
      cv_algo      : 算法输出在"平台期"（ref_win 之后到卸载前）的变异系数
                     CV = std/mean，衡量整个持载过程读数是否恒定（越小越平）
      cv_raw       : 原始信号同口径 CV（对照）
      gain         : 算法输出在 ref_win 的均值 / F_ref
                     （<<1 说明把真实力一起扣掉了；≈1 说明保留）
      gain_plateau : 算法输出在平台期均值 / F_ref，与 gain 配合看是否整体缩水

    辅助指标：
      step_fid   : [on+0.05s, on+ref] 窗口的净上升 / 原始同窗口净上升
      noise_ratio: 平台期【高频带】噪声 σ 之比（带通 0.01~1Hz，避免被慢漂污染）
      noise_lf   : 平台期【低频带 <0.01Hz】起伏 σ 之比（这才是时漂类残留）
      zero_post  : 卸载后 3s 输出均值（相对静息基线，mN）
      rough_ratio: 末段相邻通道差分均值之比
    """
    X, t, fs = D["X"], D["t"], D["fs"]
    N, C = X.shape
    a0, a1 = D["pre"]; c0, d0 = D["load"]; e0, e1 = D["post"]
    w = lambda s: int(round(s * fs))
    if quiet_frames is None:
        quiet_frames = a1
    base = X[a0:a1].mean(axis=0)
    Xn = X - base[None, :]
    tot = Xn.sum(axis=1)
    if n_on is None:
        n_on, _, _ = online_load_detect(tot, fs, quiet_frames)
    keep = Xn if n_prefix < 0 else Xn[:n_prefix]
    tk = t if n_prefix < 0 else t[:n_prefix]
    Y = run_stream(algo, keep, tk, fs, n_on)
    Nk = len(tk)
    if n_on is None:
        n_on = c0
    if act is None:
        act = np.where(Xn[c0:d0].max(axis=0) > 0.05)[0]

    r0, r1 = n_on + w(ref_win[0]), min(n_on + w(ref_win[1]), Nk)
    l1 = min(d0, Nk); l0 = max(l1 - w(late_win), n_on)
    Xr, Xl = keep[r0:r1].mean(axis=0), keep[l0:l1].mean(axis=0)
    Yr, Yl = Y[r0:r1].mean(axis=0), Y[l0:l1].mean(axis=0)
    Fref = Xr.copy()

    d_raw = (Xl - Fref)
    d_algo = (Yl - Fref)

    # 平台期 CV
    p0, p1 = r1, l1
    if p1 - p0 > 20:
        cv_y = Y[p0:p1, act].std(axis=0) / np.maximum(np.abs(Y[p0:p1, act].mean(axis=0)), 1e-9)
        cv_x = keep[p0:p1, act].std(axis=0) / np.maximum(
            np.abs(keep[p0:p1, act].mean(axis=0)), 1e-9)
        cv_algo = float(np.nanmedian(cv_y)); cv_raw = float(np.nanmedian(cv_x))
        Ypl = Y[p0:p1].mean(axis=0)
    else:
        cv_algo = cv_raw = np.nan
        Ypl = Yl

    # 阶跃保真
    s0, s1 = n_on + w(0.05), min(n_on + w(ref_win[1]), Nk)
    fid = []
    for j in act:
        xr = keep[s0:s1, j].mean() - keep[max(s0 - w(0.5), 0):s0, j].mean()
        if abs(xr) > 0.05:
            yr = Y[s0:s1, j].mean() - Y[max(s0 - w(0.5), 0):s0, j].mean()
            fid.append(yr / xr)

    # 噪声（带通分离：高频 vs 低频）
    from scipy import signal as sps
    lo_f, hi_f = noise_bw
    b_bp, a_bp = sps.butter(2, [lo_f / (fs / 2), min(hi_f / (fs / 2), 0.99)], btype="band")
    b_lp, a_lp = sps.butter(2, max(lo_f / (fs / 2), 1e-6), btype="low")
    nz_hi, nz_hi0, nz_lo, nz_lo0 = [], [], [], []
    for j in act:
        yy = Y[p0:p1, j] - Y[p0:p1, j].mean()
        xx = keep[p0:p1, j] - keep[p0:p1, j].mean()
        nz_hi.append(sps.lfilter(b_bp, a_bp, yy).std())
        nz_hi0.append(sps.lfilter(b_bp, a_bp, xx).std())
        nz_lo.append(sps.lfilter(b_lp, a_lp, yy).std())
        nz_lo0.append(sps.lfilter(b_lp, a_lp, xx).std())

    # 零漂
    q0, q1 = max(e1 - w(3.0), e0), e1
    zp = (Y[q0:q1].mean(axis=0) - base) if q1 <= Nk else np.full(C, np.nan)

    rr_a = float(np.mean(np.abs(np.diff(Y[l0:l1], axis=1))))
    rr_x = float(np.mean(np.abs(np.diff(keep[l0:l1], axis=1))))

    out = dict(
        n_on=int(n_on),
        drift_raw_mN=float(np.nanmedian(d_raw[act]) * 1000),
        drift_algo_mN=float(np.nanmedian(d_algo[act]) * 1000),
        drift_algo_maxabs_mN=float(np.nanmax(np.abs(d_algo[act])) * 1000),
        drift_reduction=float(1.0 - np.nanmedian(np.abs(d_algo[act]))
                              / max(np.nanmedian(np.abs(d_raw[act])), 1e-12)),
        cv_algo=float(cv_algo), cv_raw=float(cv_raw),
        gain=float(np.nanmedian(Yr[act] / np.where(np.abs(Fref[act]) < 1e-9, np.nan,
                                                   Fref[act]))),
        gain_plateau=float(np.nanmedian(Ypl[act] / np.where(
            np.abs(Fref[act]) < 1e-9, np.nan, Fref[act]))),
        step_fid=float(np.nanmedian(fid)) if fid else np.nan,
        noise_hi=float(np.nanmedian([n / max(m, 1e-12) * 100
                                     for n, m in zip(nz_hi, nz_hi0)])),
        noise_lo=float(np.nanmedian([n / max(m, 1e-12) * 100
                                     for n, m in zip(nz_lo, nz_lo0)])),
        zero_post=float(np.nanmedian(zp[act]) * 1000),
        rough_ratio=rr_a / rr_x if rr_x > 0 else np.nan,
    )
    out.update(Y=Y, base=base, act=act, n_on=n_on, Fref=Fref)
    return out


def evaluate_multi(algo, D, wins=((0.5, 1.0), (1.0, 2.0), (1.0, 3.0), (2.0, 5.0),
                                 (3.0, 8.0), (5.0, 15.0))):
    """对多个参考窗做敏感性分析。"""
    return {f"{a:g}~{b:g}s": evaluate(algo, D, ref_win=(a, b)) for a, b in wins}
