# -*- coding: utf-8 -*-
"""DSv4 因果（causal/online）算法库。

因果约束：
    任意 n 时刻的输出 y[n] 只允许使用 x[0..n]、检测到的接触/脱离事件，
    以及离线标定好的常数（tau、阈值、滤波系数）。禁止使用 n 之后的样本。

算法一览（全部在线因果）：
  raw            原始数据（基准）
  host_kalman    主机现有 *_kalman_compensated.csv（按在线输出文件参与对比）
  gate_baseline  接触门控自适应基线（修零漂）
  ema_hp         单极点高通（EMA 基线跟踪，演示静态力衰减代价）
  ref_common     门控基线 + 未受载参考通道共模扣除（阵列）
  rls_linear     门控基线 + RLS 线性时漂预测
  rls_exp        门控基线 + RLS 指数时漂预测（tau 由留一法标定）
  rls_log        门控基线 + RLS 对数时漂预测（tau_L 由留一法标定）
  kalman_log     门控基线 + 标量 Kalman 对数时漂状态估计
  pipeline       推荐流程：门控基线 + Kalman 对数去漂 + 因果中值(3)去量化噪声
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

CONTACT_HOLD = 5          # 连续 5 帧超过阈值才确认接触，防毛刺
T_REF = 1.25              # 接触后锁定阶跃参考的锚点时间（s）
GATE_ALPHA = 0.005        # 空载基线 EMA 系数，tau ≈ dt/alpha ≈ 3 s
RLS_R = 1e-4              # RLS/Kalman 测量噪声方差，对应约 0.01 N 噪声
RLS_P0 = 1e3              # 幅度先验方差
KALMAN_Q = 1e-11          # 漂移幅度随机游走过程噪声
DEFAULT_TAU_EXP = 60.0    # 缺省指数时间常数
DEFAULT_TAU_LOG = 2.0     # 缺省对数时间常数（由留一标定支持）


def detect_contact(total, thr):
    """因果接触检测：连续 CONTACT_HOLD 帧超过阈值返回首帧；否则 None。"""
    if len(total) < CONTACT_HOLD:
        return None
    above = total > thr
    for k in range(len(total) - CONTACT_HOLD + 1):
        if np.all(above[k:k + CONTACT_HOLD]):
            return int(k)
    return None


def detect_release(total, thr, start):
    """因果脱离检测：接触后第一帧跌回阈值以下；未脱离返回 len(total)。"""
    if start is None:
        return None
    for k in range(start, len(total)):
        if total[k] <= thr:
            return int(k)
    return len(total)


def _online_gate(d, thr=3.0, alpha=GATE_ALPHA):
    """真正逐样本因果的门控基线。

    状态机：
        total[k] <= thr  -> 更新基线（EMA）
        total[k] >  thr  -> 保持基线；连续 HOLD 帧后确认接触
    返回 (Y, s0_eff, e0)：s0_eff 为接触首帧估计，e0 为脱离首帧。
    """
    X = d["X"]
    total = d["total"]
    n, p = X.shape
    baseline = np.empty_like(X)
    # 严格因果：只用首样本初始化，随后由 EMA 自行暖机
    baseline[0] = X[0].copy()
    contact = False
    above_count = 0
    below_count = 0
    s0_eff = None
    e0 = n
    for k in range(1, n):
        if total[k] > thr:
            above_count += 1
            below_count = 0
            if above_count >= CONTACT_HOLD and not contact:
                contact = True
                s0_eff = k - CONTACT_HOLD + 1
            # 只要当前帧高于阈值就冻结基线，防止把负载阶跃学进基线
            baseline[k] = baseline[k - 1]
        else:
            below_count += 1
            if contact and below_count >= CONTACT_HOLD and e0 == n:
                e0 = k - CONTACT_HOLD + 1
            above_count = 0
            baseline[k] = alpha * X[k] + (1.0 - alpha) * baseline[k - 1]
    if contact and e0 == n:
        e0 = n
    return X - baseline, s0_eff, e0


def correct_gate_baseline(d, thr=3.0, alpha=GATE_ALPHA):
    """接触门控 EMA 基线（因果）：只在空载时更新，接触期间保持。"""
    return _online_gate(d, thr=thr, alpha=alpha)[0]


def _lock_reference(base, s0c, fs):
    """接触后锁定去皮输出中的阶跃参考。返回 pre, on, step。

    参考窗取 [s0c+0.5s, s0c+T_REF]，到 t_ref=1.25s 时全部样本均已是
    过去/当前样本，因此参考锁定严格因果。
    """
    pre = np.median(base[max(0, s0c - int(3 * fs)):s0c], axis=0)
    on = np.median(base[s0c + int(round(0.5 * fs)):s0c + int(round(T_REF * fs))],
                   axis=0)
    return pre, on, on - pre


def _basis_values(kind, tt, tau):
    ref = 0.0
    if kind == "linear":
        basis = tt
        ref = T_REF
    elif kind == "exp":
        basis = 1.0 - np.exp(-tt / max(tau, 1e-9))
        ref = 1.0 - np.exp(-T_REF / max(tau, 1e-9))
    elif kind == "log":
        basis = np.log1p(tt / max(tau, 1e-9))
        ref = np.log1p(T_REF / max(tau, 1e-9))
    else:
        raise ValueError(kind)
    return basis - ref


def _correct_loaded_mask(step):
    return np.where(step > 0.15)[0]


def correct_rls_drift(d, kind, tau=None, thr=3.0, gate_alpha=GATE_ALPHA,
                      R=RLS_R, P0=RLS_P0):
    """门控基线 + 每通道标量 RLS 参数化时漂预测（完全因果）。

    接触确认后，在 t_ref=1.25 s 锁定阶跃参考；之后逐样本用
    y_k = step + a*(basis_k - basis_ref) 更新幅度 a，并立即扣除预测漂移。
    """
    if tau is None:
        tau = DEFAULT_TAU_EXP if kind == "exp" else DEFAULT_TAU_LOG
    base, s0c, e0c = _online_gate(d, thr=thr, alpha=gate_alpha)
    if s0c is None:
        return base
    t = d["t"]
    fs = d["fs"]
    pre, on, step = _lock_reference(base, s0c, fs)
    idx = _correct_loaded_mask(step)
    Y = base.copy()
    for j in idx:
        ahat = 0.0
        P = P0
        y = base[:, j]
        yref = step[j]
        for k in range(s0c, e0c):
            tt = t[k] - t[s0c]
            B = _basis_values(kind, tt, tau)
            z = y[k] - yref
            inn = z - B * ahat
            S = P * B * B + R
            gain = P * B / S
            ahat += gain * inn
            P = (1.0 - gain * B) * P
            if tt >= T_REF:
                Y[k, j] = y[k] - ahat * B
    return Y


def correct_kalman_drift(d, kind="log", tau=None, thr=3.0,
                         gate_alpha=GATE_ALPHA, R=RLS_R, P0=RLS_P0,
                         Q=KALMAN_Q):
    """门控基线 + 标量 Kalman 漂移幅度估计（在线因果）。

    状态 a 服从随机游走，测量方程 y_k = step + a*basis_k + v_k。
    """
    if tau is None:
        tau = DEFAULT_TAU_EXP if kind == "exp" else DEFAULT_TAU_LOG
    base, s0c, e0c = _online_gate(d, thr=thr, alpha=gate_alpha)
    if s0c is None:
        return base
    t = d["t"]
    fs = d["fs"]
    pre, on, step = _lock_reference(base, s0c, fs)
    idx = _correct_loaded_mask(step)
    Y = base.copy()
    for j in idx:
        ahat = 0.0
        P = P0
        y = base[:, j]
        yref = step[j]
        for k in range(s0c, e0c):
            tt = t[k] - t[s0c]
            B = _basis_values(kind, tt, tau)
            z = y[k] - yref
            P = P + Q
            inn = z - B * ahat
            S = P * B * B + R
            gain = P * B / S
            ahat += gain * inn
            P = (1.0 - gain * B) * P
            if tt >= T_REF:
                Y[k, j] = y[k] - ahat * B
    return Y


def _causal_median3(Y, s0c, e0c):
    """3 点因果滑动中值：med(x[k-2], x[k-1], x[k])，只作用于接触段。"""
    Z = Y.copy()
    if s0c is None:
        return Z
    for k in range(s0c, min(e0c, len(Z))):
        a = max(0, k - 2)
        Z[k] = np.median(Y[a:k + 1], axis=0)
    return Z


def correct_pipeline(d, tau_log=None, thr=3.0):
    """推荐因果流程：门控基线 + Kalman 对数去漂 + 3 点因果中值。"""
    if tau_log is None:
        tau_log = DEFAULT_TAU_LOG
    Y = correct_kalman_drift(d, "log", tau=tau_log, thr=thr)
    _, s0c, e0c = _online_gate(d, thr=thr)
    return _causal_median3(Y, s0c, min(e0c, len(Y)))


def correct_ref_common(d, thr=3.0, gate_alpha=GATE_ALPHA):
    """门控基线 + 未受载参考通道共模扣除（阵列，因果）。

    接触后 t_ref 时刻按阶跃响应确定未受载参考组；此后每个当前样本
    扣除参考组相对其接触前基线的共模偏移。
    """
    base, s0c, e0c = _online_gate(d, thr=thr, alpha=gate_alpha)
    if s0c is None:
        return base
    fs = d["fs"]
    pre, on, step = _lock_reference(base, s0c, fs)
    idle = np.where(step < 0.02)[0]
    Y = base.copy()
    if len(idle) == 0:
        return Y
    ref_pre = np.median(base[max(0, s0c - int(3 * fs)):s0c][:, idle])
    start_c = s0c + int(round(T_REF * fs))
    for k in range(start_c, e0c):
        ref_k = np.median(base[k, idle])
        Y[k] = base[k] - (ref_k - ref_pre)
    return Y


def correct_ema_hp(d, alpha=0.001):
    """单极点高通：Y = X - EMA(X)。"""
    X = d["X"]
    n, p = X.shape
    b = np.empty_like(X)
    b[0] = X[0]
    for k in range(1, n):
        b[k] = alpha * X[k] + (1.0 - alpha) * b[k - 1]
    return X - b


def apply_all_causal(d, params):
    """按留一标定参数运行全部因果算法。

    params 应包含 thr, tau_exp, tau_log, tau_kalman。
    """
    thr = float(params.get("thr", 3.0))
    tau_exp = float(params.get("tau_exp", DEFAULT_TAU_EXP))
    tau_log = float(params.get("tau_log", DEFAULT_TAU_LOG))
    tau_kalman = float(params.get("tau_kalman", DEFAULT_TAU_LOG))
    return {
        "raw": d["X"].copy(),
        "host_kalman": d["Xk"].copy(),
        "gate_baseline": correct_gate_baseline(d, thr=thr),
        "ema_hp": correct_ema_hp(d),
        "ref_common": correct_ref_common(d, thr=thr),
        "rls_linear": correct_rls_drift(d, "linear", thr=thr),
        "rls_exp": correct_rls_drift(d, "exp", tau=tau_exp, thr=thr),
        "rls_log": correct_rls_drift(d, "log", tau=tau_log, thr=thr),
        "kalman_log": correct_kalman_drift(d, "log", tau=tau_kalman, thr=thr),
        "pipeline": correct_pipeline(d, tau_log=tau_log, thr=thr),
    }


METHOD_DISPLAY = {
    "raw": ("原始数据", "基准"),
    "host_kalman": ("主机现有Kalman输出", "现有/在线"),
    "gate_baseline": ("接触门控自适应基线", "在线/零漂"),
    "ema_hp": ("一阶高通(EMA)", "在线/单点"),
    "ref_common": ("门控基线+参考共模", "在线/阵列"),
    "rls_linear": ("RLS线性时漂预测", "在线/单点"),
    "rls_exp": ("RLS指数时漂预测", "在线/单点"),
    "rls_log": ("RLS对数时漂预测", "在线/单点"),
    "kalman_log": ("Kalman对数漂移状态", "在线/单点"),
    "pipeline": ("推荐因果流程", "在线/阵列+单点"),
}

METHOD_ORDER = list(METHOD_DISPLAY.keys())


if __name__ == "__main__":
    # 自检：默认参数下全部可运行
    from common import DATASETS, load_dataset, segment_total
    for name in DATASETS:
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        out = apply_all_causal(d, {})
        print(name, {k: v.shape for k, v in out.items()})
