# -*- coding: utf-8 -*-
"""DSv4 算法库：单点 DSP 与阵列信号处理的时漂/零漂校正实现。

所有算法输入/输出均为 (T, 31) 数组。除 raw 与 host_kalman 外，
输出都以“相对前空载基线”的力值为主，便于在同一零点上比较。

算法一览：
  raw            原始数据（基准）
  host_kalman    主机现有 *_kalman_compensated.csv
  tare           静态去皮：减去负载前 3 s 中值
  ref_common     静态去皮 + 未受载参考通道共模扣除（阵列）
  gate_baseline  接触门控自适应基线（在线，只修零漂）
  ema_hp         一阶高通/EMA 基线跟踪（在线，会衰减静态力，作为反例）
  als            Whittaker 非对称最小二乘基线（批处理，步骤/漂移分离困难，作为反例）
  wavelet        小波近似系数去除（批处理，会衰减静态力，作为反例）
  linear         单通道锚定线性去漂（批处理）
  exp            单通道锚定指数去漂（批处理）
  log            单通道锚定对数去漂（批处理）
  shared_log     阵列共享时间核对数去漂（阵列，批处理）
  online_rls     接触触发 RLS 指数漂移预测（在线，仅受载通道）
"""
import os
import sys
import json

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, load_dataset, segment_total

# 接触门控阈值（N，通道总和）。三组数据空载总和均 < 3 N，负载总和 > 5.9 N。
CONTACT_THRESHOLD_TOTAL = 3.0
# 负载起始参考窗：onset 后 0.5~2.0 s；锚点取窗中心
T_REF = 1.25
# RLS 先验时间常数（s），取三组数据单通道指数拟合 tau 的中位数附近
RLS_TAU = 60.0


def _exp_model(t, a, tau, c):
    return a * (1.0 - np.exp(-t / tau)) + c


def _median_pre_onset(d, s0):
    fs = d["fs"]
    X = d["X"]
    pre = np.median(X[max(0, s0 - int(3 * fs)):s0], axis=0)
    i0 = min(s0 + int(round(0.5 * fs)), len(d["t"]) - 1)
    i1 = min(s0 + int(round(2.0 * fs)), len(d["t"]))
    on = np.median(X[i0:i1], axis=0)
    return pre, on


# ----------------------------------------------------------------------
# 各算法
# ----------------------------------------------------------------------
def correct_tare(d, s0, s1):
    pre, _ = _median_pre_onset(d, s0)
    return d["X"] - pre[None, :]


def correct_host_kalman(d, s0, s1):
    return d["Xk"].copy()


def correct_ref_common(d, s0, s1):
    """去皮后，扣除未受载参考通道相对其自身负载前基线的共模变化。"""
    X = d["X"]
    pre, on = _median_pre_onset(d, s0)
    response = on - pre
    idle = response < 0.02
    ref = X[:, idle].mean(axis=1) if idle.sum() else np.zeros(len(X))
    ref_pre = np.median(ref[max(0, s0 - int(3 * d["fs"])):s0])
    Y = X - pre[None, :] - (ref - ref_pre)[:, None]
    return Y


def correct_gate_baseline(d, s0, s1, thr=CONTACT_THRESHOLD_TOTAL, alpha=0.005):
    """接触门控 EMA 基线：只在空载时更新基线，接触期间保持。"""
    X = d["X"]
    total = X.sum(axis=1)
    n, p = X.shape
    init_n = max(1, int(round(d["fs"] * 1.0)))
    baseline = np.empty_like(X)
    baseline[0] = np.median(X[:init_n], axis=0)
    for k in range(1, n):
        if total[k] < thr:
            baseline[k] = alpha * X[k] + (1.0 - alpha) * baseline[k - 1]
        else:
            baseline[k] = baseline[k - 1]
    return X - baseline


def correct_ema_hp(d, s0, s1, alpha=0.001):
    """单极点高通：Y = X - EMA(X)。tau ≈ dt/alpha ≈ 15 s（fs=66.7 Hz）。"""
    X = d["X"]
    n, p = X.shape
    b = np.empty_like(X)
    b[0] = X[0]
    for k in range(1, n):
        b[k] = alpha * X[k] + (1.0 - alpha) * b[k - 1]
    return X - b


def correct_als(d, s0, s1, lam=1e10, p_asym=0.001, iters=8):
    """Whittaker 非对称最小二乘基线估计（每通道）。"""
    X = d["X"]
    n, p = X.shape
    D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n - 2, n), format="csc")
    DTD = lam * (D.T @ D)
    Y = np.empty_like(X)
    for j in range(p):
        y = X[:, j]
        w = np.ones(n)
        z = y.copy()
        for _ in range(iters):
            W = sparse.diags(w, format="csc")
            A = W + DTD
            z = spsolve(A, w * y)
            w = np.where(y > z, p_asym, 1.0 - p_asym)
        Y[:, j] = y - z
    return Y


def correct_wavelet(d, s0, s1, level=6, wavelet="sym8"):
    """DWT 去掉最高层近似系数（低频漂移/直流），保留细节。"""
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError("需要 PyWavelets") from exc
    X = d["X"]
    Y = np.empty_like(X)
    for j in range(X.shape[1]):
        coeffs = pywt.wavedec(X[:, j], wavelet, mode="periodization", level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        rec = pywt.waverec(coeffs, wavelet, mode="periodization")
        Y[:, j] = rec[: X.shape[0]]
    return Y


def _anchor_linear(y, t, t_ref, y_ref):
    """把 y 锚定到 (t_ref, y_ref) 后去线性趋势，返回展平到 y_ref 的序列。"""
    b = t - t_ref
    num = np.sum((y - y_ref) * b)
    den = np.sum(b * b)
    if den <= 1e-12:
        return np.full_like(y, float(y_ref))
    slope = num / den
    return y - slope * b


def _anchor_kernel(y, t, t_ref, y_ref, basis):
    """basis(t) 已知时间核；锚定到 (t_ref, y_ref)，估计幅度并去漂。"""
    b = basis - basis(t_ref) if callable(basis) else basis - np.interp(t_ref, t, basis)
    num = np.sum((y - y_ref) * b)
    den = np.sum(b * b)
    if den <= 1e-12:
        return np.full_like(y, float(y_ref)), 0.0
    amp = num / den
    return y - amp * b, amp


def _fit_tau_exp(y, t):
    try:
        p0 = [max(0.01, y[-1] - y[0]), 60.0, y[0]]
        popt, _ = curve_fit(_exp_model, t, y, p0=p0,
                            bounds=([-5, 2.0, -5], [10, 600.0, 10]),
                            maxfev=30000)
        return float(popt[1])
    except Exception:
        return 60.0


def _model_method(d, s0, s1, kind):
    """线性/指数/对数 单通道锚定去漂。输出：负载段用模型展平，其余段去皮。"""
    X = d["X"]
    t = d["t"]
    pre, on = _median_pre_onset(d, s0)
    step_ref = on - pre  # 去皮后的真实阶跃参考
    Y = X - pre[None, :]
    tl = t[s0:s1] - t[s0]
    yl = X[s0:s1] - pre[None, :]
    for j in range(X.shape[1]):
        y = yl[:, j]
        if kind == "linear":
            Y[s0:s1, j] = _anchor_linear(y, tl, T_REF, step_ref[j])
        elif kind == "exp":
            tau = _fit_tau_exp(y, tl)
            basis = 1.0 - np.exp(-tl / tau)
            Y[s0:s1, j], _ = _anchor_kernel(y, tl, T_REF, step_ref[j], basis)
        elif kind == "log":
            basis = np.log1p(tl)
            Y[s0:s1, j], _ = _anchor_kernel(y, tl, T_REF, step_ref[j], basis)
        else:
            raise ValueError(kind)
    return Y


def _shared_log_residual(tau_l, Yload, tl, loaded_idx, on):
    basis = np.log1p(tl / tau_l)
    total = 0.0
    for j in loaded_idx:
        _, amp = _anchor_kernel(Yload[:, j], tl, T_REF, on[j], basis)
        pred = on[j] + amp * (basis - np.interp(T_REF, tl, basis))
        total += np.sum((Yload[:, j] - pred) ** 2)
    return total


def estimate_shared_tau_log(d, s0, s1):
    """在受载通道上联合估计共享对数时间常数 tau_L。"""
    X = d["X"]
    t = d["t"]
    pre, on = _median_pre_onset(d, s0)
    response = on - pre
    loaded_idx = np.where(response > 0.15)[0]
    tl = t[s0:s1] - t[s0]
    Yload = X[s0:s1] - pre[None, :]
    if len(loaded_idx) == 0:
        return 1.0
    grid = np.logspace(-0.5, 2.5, 25)
    vals = [_shared_log_residual(v, Yload, tl, loaded_idx, response) for v in grid]
    tau0 = grid[int(np.argmin(vals))]
    try:
        res = minimize_scalar(_shared_log_residual,
                              bounds=(max(0.05, tau0 / 4), min(600.0, tau0 * 4)),
                              args=(Yload, tl, loaded_idx, response),
                              method="bounded",
                              options={"xatol": 0.05})
        tau_l = float(res.x) if np.isfinite(res.fun) else tau0
    except Exception:
        tau_l = tau0
    return float(tau_l)


def correct_shared_log_fixed(d, s0, s1, tau_l=1.0):
    """使用给定共享时间常数 tau_L 的对数时间核去漂。"""
    X = d["X"]
    t = d["t"]
    pre, on = _median_pre_onset(d, s0)
    response = on - pre
    Y = X - pre[None, :]
    tl = t[s0:s1] - t[s0]
    Yload = X[s0:s1] - pre[None, :]
    basis = np.log1p(tl / tau_l)
    for j in range(X.shape[1]):
        y = Yload[:, j]
        Y[s0:s1, j], _ = _anchor_kernel(y, tl, T_REF, response[j], basis)
    return Y


def correct_shared_log(d, s0, s1, tau_l=None):
    """阵列共享时间核对数去漂：逐通道估计幅度。

    tau_l 缺省取 2.0 s。这是 04_cross_dataset.py 在 0.2~20 s 扫描中
    三组数据主通道 RMSE 最小且漂移率<7% 的保守固定值；联合估计值
    见 estimate_shared_tau_log()，但跨组不稳定，不建议直接用于未知数据。
    """
    if tau_l is None:
        tau_l = 2.0
    return correct_shared_log_fixed(d, s0, s1, tau_l)


def correct_pipeline(d, s0, s1, tau_l=2.0,
                     thr=CONTACT_THRESHOLD_TOTAL, alpha=0.005):
    """推荐组合流程：接触门控自适应基线(零漂) + 阵列共享对数核(负载时漂)。

    空载/卸载后用门控 EMA 跟踪基线；接触期间保持基线并只做对数时间核去漂。
    """
    X = d["X"]
    t = d["t"]
    # 1) 零漂：接触门控基线
    Y = correct_gate_baseline(d, s0, s1, thr=thr, alpha=alpha)
    # 2) 时漂：在去皮后的负载段上做锚定对数去漂
    pre_g = np.median(Y[max(0, s0 - int(3 * d["fs"])):s0], axis=0)
    on_g = np.median(Y[s0 + int(round(0.5 * d["fs"])):s0 + int(round(2.0 * d["fs"]))],
                     axis=0)
    step_ref = on_g - pre_g
    tl = t[s0:s1] - t[s0]
    Yload = Y[s0:s1] - pre_g[None, :]
    basis = np.log1p(tl / tau_l)
    for j in range(X.shape[1]):
        Y[s0:s1, j] = _anchor_kernel(Yload[:, j], tl, T_REF, step_ref[j], basis)[0]
    return Y


def correct_online_rls(d, s0, s1, tau=RLS_TAU):
    """接触触发 RLS 指数漂移补偿（在线因果）。

    只在受载通道上运行；接触后假设 y(t) ≈ y_ref + a*(K(t)-K(t_ref))，
    K=1-exp(-t/tau)。用标量 RLS 估计 a 并实时扣除。
    """
    X = d["X"]
    t = d["t"]
    pre, on = _median_pre_onset(d, s0)
    response = on - pre
    loaded_idx = np.where(response > 0.15)[0]
    Y = X - pre[None, :]
    R = 1e-4  # 量化噪声约 0.001 N -> var 1e-6，取偏大保证平滑
    P0 = 1e3
    Kref = 1.0 - np.exp(-T_REF / tau)
    for j in loaded_idx:
        y = X[:, j]
        yref = response[j]
        ahat = 0.0
        P = P0
        for k in range(s0, s1):
            tt = t[k] - t[s0]
            basis = (1.0 - np.exp(-tt / tau)) - Kref
            z = y[k] - pre[j] - yref
            inn = z - basis * ahat
            S = P * basis * basis + R
            gain = P * basis / S
            ahat += gain * inn
            P = (1.0 - gain * basis) * P
            # 在阶跃参考窗(0.5~2.0 s)内不修正，避免把起始响应窗本身改掉
            if tt >= T_REF:
                Y[k, j] = y[k] - pre[j] - ahat * basis
    return Y


def apply_all_methods(d, s0, s1):
    """对一组数据依次运行全部算法。"""
    out = {
        "raw": d["X"].copy(),
        "host_kalman": correct_host_kalman(d, s0, s1),
        "tare": correct_tare(d, s0, s1),
        "ref_common": correct_ref_common(d, s0, s1),
        "gate_baseline": correct_gate_baseline(d, s0, s1),
        "ema_hp": correct_ema_hp(d, s0, s1),
        "wavelet": correct_wavelet(d, s0, s1),
        "linear": _model_method(d, s0, s1, "linear"),
        "exp": _model_method(d, s0, s1, "exp"),
        "log": _model_method(d, s0, s1, "log"),
        "shared_log": correct_shared_log(d, s0, s1),
        "online_rls": correct_online_rls(d, s0, s1),
        "pipeline": correct_pipeline(d, s0, s1),
    }
    try:
        out["als"] = correct_als(d, s0, s1)
    except Exception:
        out["als"] = d["X"] - np.median(d["X"][max(0, s0 - int(3 * d["fs"])):s0], axis=0)[None, :]
    return out


METHOD_DISPLAY = {
    "raw": ("原始数据", "基准"),
    "host_kalman": ("主机现有Kalman", "现有/在线"),
    "tare": ("静态去皮", "单点/零漂"),
    "ref_common": ("去皮+参考通道共模", "阵列/零漂"),
    "gate_baseline": ("接触门控自适应基线", "在线/零漂"),
    "ema_hp": ("一阶高通(EMA)", "在线/单点"),
    "als": ("Whittaker ALS基线", "批处理/单点"),
    "wavelet": ("小波去趋势", "批处理/单点"),
    "linear": ("单通道线性去漂", "批处理/单点"),
    "exp": ("单通道指数去漂", "批处理/单点"),
    "log": ("单通道对数去漂", "批处理/单点"),
    "shared_log": ("阵列共享对数去漂", "批处理/阵列"),
    "online_rls": ("在线RLS指数预测", "在线/单点"),
    "pipeline": ("推荐组合流程", "在线+阵列"),
}

METHOD_ORDER = list(METHOD_DISPLAY.keys())


if __name__ == "__main__":
    # 简单自检：全部算法可运行
    for name in DATASETS:
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        out = apply_all_methods(d, s0, s1)
        print(name, {k: v.shape for k, v in out.items()})
