# -*- coding: utf-8 -*-
"""时漂/零漂补偿算法库 —— 统一信号模型版本。

统一信号模型
------------
    y_j(t) = F_j · h(t) + n_j(t)

* F_j      通道 j 的**真实恒定载荷**（待估计量）
* h(t)     归一化漂移形态，h(t) 单调增、且锚定 h(t_ref) = 1
           t 为"负载已施加的持续时间"（秒）
* n_j(t)   噪声（白噪 + 1mN 量化）

即：**时漂是乘性的**（漂移量正比于通道载荷），这一点已由数据验证
（跨通道"漂移量 vs 响应"回归 R²=0.83~0.86，过原点）。

补偿目标：由 y_j(t) 恢复 F_j = y_j(t)/h(t)。

统一接口
--------
    comp = Algorithm(...)
    Y = comp.transform(X, t, fs, load_start_idx)

* X   : (N, C) float64，单位 N，已完成"前空载基线归零"
* t   : (N,)   均匀时间轴（秒）
* fs  : 采样率 Hz
* load_start_idx : 负载阶跃起始帧号（模型类算法需要；其余忽略）
* Y   : (N, C) 补偿后数据；前空载段保持原样

算法分类（group 字段）
---------------------
'0. 参照'           不补偿
'1. 单点时序滤波'    只用本通道时间序列的通用滤波（对慢漂只能压制，不能辨识）
'2. 漂移模型拟合'    显式建模漂移并把观测除以形态（需要已知/估计阶跃时刻）
'3. 阵列信号处理'    利用多通道之间的共模结构估计漂移（本项目推荐路径）
"""
import numpy as np
from scipy import signal as sps

# ================================================================ 基础工具


def _colwise(X, fn):
    return np.column_stack([fn(X[:, j]) for j in range(X.shape[1])])


def _causal_ma(y, w):
    if w <= 1:
        return y.copy()
    cs = np.concatenate([[0.0], np.cumsum(y)])
    n = len(y)
    idx = np.arange(n)
    lo = np.maximum(0, idx - w + 1)
    return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)


def _causal_median(y, w):
    if w <= 2:
        return y.copy()
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(y[max(0, i - w + 1):i + 1])
    return out


def _filtfilt_butter(y, fs, fc, order=2, btype="high"):
    wn = float(np.clip(fc / (fs / 2.0), 1e-6, 0.999))
    b, a = sps.butter(order, wn, btype=btype)
    padlen = min(3 * max(len(a), len(b)), len(y) - 1)
    if padlen <= 0:
        return y.copy()
    return sps.filtfilt(b, a, y, padlen=padlen)


def _lstsq(A, y):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


# ---------------- 稳健形态拟合（网格 + 线性解，避免 curve_fit 病态）

P_GRID_DEFAULT = np.arange(0.02, 1.201, 0.01)


def _pow_shape_fit(t, z, p_grid=P_GRID_DEFAULT):
    """z(t) ≈ c·(1 + a·t^p)，网格搜 p，线性解 a,c（等价于 z ≈ a'·t^p + c'）。"""
    best = None
    for p in p_grid:
        phi = np.power(np.maximum(t, 0.0), p)
        A = np.vstack([phi, np.ones_like(phi)]).T
        a, c = _lstsq(A, z)
        res = z - (a * phi + c)
        ss = float(res @ res)
        if best is None or ss < best[0]:
            best = (ss, float(p), float(a), float(c))
    ss, p, a, c = best
    tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss / tot if tot > 0 else np.nan
    return p, a, c, r2


def _log_shape_fit(t, z):
    phi = np.log1p(np.maximum(t, 0.0))
    A = np.vstack([phi, np.ones_like(phi)]).T
    a, c = _lstsq(A, z)
    r2 = 1.0 - float(np.var(z - (a * phi + c))) / float(np.var(z))
    return a, c, r2


def _exp_shape_fit(t, z, tau_grid=None):
    if tau_grid is None:
        tau_grid = np.concatenate([np.arange(0.5, 20.0, 0.5),
                                   np.arange(20.0, 201.0, 5.0),
                                   np.arange(200.0, 3001.0, 100.0)])
    best = None
    for tau in tau_grid:
        phi = 1.0 - np.exp(-t / tau)
        A = np.vstack([phi, np.ones_like(phi)]).T
        a, c = _lstsq(A, z)
        res = z - (a * phi + c)
        ss = float(res @ res)
        if best is None or ss < best[0]:
            best = (ss, float(tau), float(a), float(c))
    ss, tau, a, c = best
    r2 = 1.0 - ss / float(np.sum((z - z.mean()) ** 2))
    return tau, a, c, r2


class ShapeModel:
    """可复用的一维漂移形态 h(t)，满足 h(0)=1。

    kind : 'power' | 'log' | 'exp' | 'linear'
    拟合目标 z(t) ≈ h(t)，其中 z 已归一化到 z(0)=1。
    拟合时只用 t >= t_floor 的样本（避免 t=0 附近的量化/上升沿污染），
    并对 (a, c) 施加线性最小二乘；形状参数用网格搜索。
    """

    def __init__(self, kind="power", t_floor=0.05):
        self.kind = kind
        self.t_floor = t_floor
        self.params = None
        self.r2 = None

    def fit(self, t, z):
        t = np.asarray(t, float); z = np.asarray(z, float)
        m = t >= self.t_floor
        tm, zm = t[m], z[m]
        if len(tm) < 30:
            return False
        try:
            if self.kind == "power":
                p, a, c, r2 = _pow_shape_fit(tm, zm)
                self._f = lambda x: a * np.power(np.maximum(x, 0.0), p) + c
                self.params = dict(p=p, a=a, c=c)
            elif self.kind == "log":
                a, c, r2 = _log_shape_fit(tm, zm)
                self._f = lambda x: a * np.log1p(np.maximum(x, 0.0)) + c
                self.params = dict(a=a, c=c)
            elif self.kind == "exp":
                tau, a, c, r2 = _exp_shape_fit(tm, zm)
                self._f = lambda x: a * (1.0 - np.exp(-x / tau)) + c
                self.params = dict(tau=tau, a=a, c=c)
            elif self.kind == "linear":
                A = np.vstack([tm, np.ones_like(tm)]).T
                a, c = _lstsq(A, zm)
                r2 = 1.0 - float(np.var(zm - (a * tm + c))) / float(np.var(zm))
                self._f = lambda x: a * np.maximum(x, 0.0) + c
                self.params = dict(a=a, c=c)
            else:
                return False
        except Exception:
            return False
        # 强制 h(0)=1：用 f(0) 归一化
        f0 = float(self._f(np.array([0.0]))[0])
        if not np.isfinite(f0) or abs(f0) < 1e-12:
            return False
        self._norm = f0
        self.r2 = float(r2)
        return True

    def __call__(self, t):
        if self.params is None:
            raise RuntimeError("shape not fitted")
        return np.asarray(self._f(np.asarray(t, float)), float) / self._norm


# ================================================================ 0. 参照


class Raw:
    name = "Raw 原始（不补偿）"
    group = "0. 参照"

    def transform(self, X, t, fs, load_start_idx):
        return X.copy()


# ================================================================ 1. 单点时序滤波
#    仅使用本通道的时间序列；对"持续缓慢增长"的漂移只能压制，无法辨识。


class MovingAverage:
    group = "1. 单点时序滤波"

    def __init__(self, w_s=1.0):
        self.w_s = w_s

    @property
    def name(self):
        return f"MA 因果滑动平均 {self.w_s:g}s"

    def transform(self, X, t, fs, load_start_idx):
        w = max(1, int(round(self.w_s * fs)))
        return _colwise(X, lambda y: _causal_ma(y, w))


class MedianFilter:
    group = "1. 单点时序滤波"

    def __init__(self, w_s=1.0):
        self.w_s = w_s

    @property
    def name(self):
        return f"Median 因果中值 {self.w_s:g}s"

    def transform(self, X, t, fs, load_start_idx):
        w = max(1, int(round(self.w_s * fs)))
        return _colwise(X, lambda y: _causal_median(y, w))


class EWMA:
    group = "1. 单点时序滤波"

    def __init__(self, tau_s=1.0):
        self.tau_s = tau_s

    @property
    def name(self):
        return f"EWMA 一阶低通 τ={self.tau_s:g}s"

    def transform(self, X, t, fs, load_start_idx):
        a = np.exp(-1.0 / (self.tau_s * fs))

        def f(y):
            v = y[0]
            out = np.empty_like(y)
            for i, x in enumerate(y):
                v = a * v + (1 - a) * x
                out[i] = v
            return out
        return _colwise(X, f)


class ButterHighPass:
    group = "1. 单点时序滤波"

    def __init__(self, fc=0.02, order=2):
        self.fc = fc
        self.order = order

    @property
    def name(self):
        return f"HPF 零相位高通 fc={self.fc:g}Hz"

    def transform(self, X, t, fs, load_start_idx):
        return _colwise(X, lambda y: _filtfilt_butter(y, fs, self.fc, self.order, "high"))


class SavitzkyGolay:
    group = "1. 单点时序滤波"

    def __init__(self, w_s=1.0, order=2):
        self.w_s = w_s
        self.order = order

    @property
    def name(self):
        return f"SG 平滑 {self.w_s:g}s o{self.order}"

    def transform(self, X, t, fs, load_start_idx):
        w = max(3, int(round(self.w_s * fs)))
        if w % 2 == 0:
            w += 1
        w = min(w, (len(t) // 2) * 2 - 1)
        return _colwise(X, lambda y: sps.savgol_filter(y, w, min(self.order, w - 2)))


class WaveletDenoise:
    group = "1. 单点时序滤波"

    def __init__(self, wavelet="db4", level=4, mode="soft"):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    @property
    def name(self):
        return f"小波去噪 {self.wavelet} L{self.level}({self.mode})"

    def transform(self, X, t, fs, load_start_idx):
        try:
            import pywt
        except Exception:
            return X.copy()

        def f(y):
            lv = min(self.level, pywt.dwt_max_level(len(y), self.wavelet))
            if lv < 1:
                return y.copy()
            c = pywt.wavedec(y, self.wavelet, level=lv)
            sigma = np.median(np.abs(c[-1])) / 0.6745
            if sigma <= 0:
                return y.copy()
            thr = sigma * np.sqrt(2 * np.log(len(y)))
            c = [c[0]] + [pywt.threshold(cc, thr, self.mode) for cc in c[1:]]
            return pywt.waverec(c, self.wavelet)[:len(y)]
        return _colwise(X, f)


class RLSDetrend:
    """递归最小二乘（遗忘因子）在线跟踪多项式基线；只适用于"基线 + 阶跃"的加性模型，"""
    group = "1. 单点时序滤波"

    def __init__(self, lam=0.9999, order=1, align_s=1.0):
        self.lam = lam
        self.order = order
        self.align_s = align_s

    @property
    def name(self):
        return f"RLS 递推基线(λ={self.lam:g}, o{self.order})"

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        order = self.order
        ia = min(int(self.align_s * fs), n - s - 1)
        for j in range(C):
            y = X[:, j]
            if y[s:].max() - y[s:].min() < 1e-6:
                continue
            P = np.eye(order + 1) * 1e3
            th = np.zeros(order + 1)
            tt = t - t[s]
            base = np.zeros(n)
            for i in range(s, n):
                phi = np.array([tt[i] ** k for k in range(order + 1)])
                pv = phi @ th
                base[i] = pv - 0.0
                Pp = P @ phi
                g = Pp / (self.lam + phi @ Pp)
                th = th + g * (y[i] - pv)
                P = (P - np.outer(g, Pp)) / self.lam
            b = base[s:]
            corr = -b + b[ia]
            Y[s:, j] = y[s:] + corr
        return Y


class KalmanDrift:
    """卡尔曼滤波：状态 [水平, 漂移速率] 的常速模型；离线用 RTS 平滑。"""

    group = "1. 单点时序滤波"

    def __init__(self, q=1e-8, r=1e-5, align_s=1.0, smooth=False):
        self.q = q
        self.r = r
        self.align_s = align_s
        self.smooth = smooth

    @property
    def name(self):
        return f"Kalman 常速模型 q={self.q:g}" + ("+RTS" if self.smooth else "")

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        dt = 1.0 / fs
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        R = np.array([[self.r]])
        ia = min(int(self.align_s * fs), n - s - 1)
        m = n - s
        for j in range(C):
            y = X[s:, j]
            if y.max() - y.min() < 1e-6:
                continue
            xf = np.zeros((m, 2)); Pf = np.zeros((m, 2, 2))
            xp = np.zeros((m, 2)); Pp = np.zeros((m, 2, 2))
            x = np.array([y[0], 0.0]); P = np.eye(2) * 1e-6
            for i in range(m):
                x = F @ x; P = F @ P @ F.T + Q
                xp[i] = x; Pp[i] = P
                S = H @ P @ H.T + R
                K = P @ H.T / S
                x = x + (K * (y[i] - (H @ x)[0])).ravel()
                P = (np.eye(2) - K @ H) @ P
                xf[i] = x; Pf[i] = P
            est = xf[:, 0]
            if self.smooth:
                xs = xf.copy(); Ps = Pf.copy()
                for i in range(m - 2, -1, -1):
                    C = Pf[i] @ F.T @ np.linalg.inv(Pp[i + 1])
                    xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
                    Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
                est = xs[:, 0]
            # 把估计轨迹当作"含漂移的水平"，除以它相对对齐点的比值
            e0 = est[ia]
            if abs(e0) < 1e-9:
                continue
            Y[s:, j] = y / (est / e0)
        return Y


# ================================================================ 2. 漂移模型拟合


class ModelFit:
    """逐通道拟合归一化形态 h(t)，输出 y/h(t)。"""

    group = "2. 漂移模型拟合"

    def __init__(self, kind="power", t_ref=0.5, t_floor=0.05):
        self.kind = kind
        self.t_ref = t_ref
        self.t_floor = t_floor

    @property
    def name(self):
        return f"ModelFit-{self.kind}(归一化形态)"

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        tt = t[s:] - t[s]
        self.fits = {}
        for j in range(C):
            y = X[s:, j]
            if y.max() - y.min() < 1e-6:
                continue
            # 用锚点窗口把序列归一到 h(t_ref)≈1
            ir = int(np.clip(self.t_ref * fs, 0, len(tt) - 1))
            hw = max(1, int(round(0.25 * fs)))
            i0, i1 = max(0, ir - hw), min(len(tt), ir + hw)
            ref = float(y[i0:i1].mean())
            if abs(ref) < 1e-6:
                continue
            sm = ShapeModel(self.kind, t_floor=self.t_floor)
            if not sm.fit(tt - self.t_ref, y / ref):
                continue
            h = sm(tt - self.t_ref)
            if not np.all(np.isfinite(h)) or np.any(np.abs(h) < 1e-6):
                continue
            self.fits[j] = dict(r2=sm.r2, **sm.params)
            Y[s:, j] = y / h
        return Y


class SegBaseline:
    """分段基线（无参数形态）：用负载段内若干窗口的稳健水平作结点，
    PCHIP 插值得到连续形态 h(t)（h(t_ref)=1），输出 y/h(t)。"""

    group = "2. 漂移模型拟合"

    def __init__(self, n_knots=9, win_s=2.0, t_ref=0.5):
        self.n_knots = n_knots
        self.win_s = win_s
        self.t_ref = t_ref

    @property
    def name(self):
        return f"SegBaseline 分段基线({self.n_knots}节点,{self.win_s:g}s)"

    def transform(self, X, t, fs, load_start_idx):
        from scipy.interpolate import PchipInterpolator
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        tt = t[s:] - t[s]
        T = tt[-1]
        centers = np.linspace(0.0, T, self.n_knots)
        w = max(1, int(round(self.win_s * fs)))
        for j in range(C):
            y = X[s:, j]
            if y.max() - y.min() < 1e-6:
                continue
            vals = []
            for cc in centers:
                i0 = int(np.clip(cc * fs, 0, len(tt) - 2))
                vals.append(np.median(y[i0:min(len(tt), i0 + w)]))
            vals = np.asarray(vals, float)
            if np.any(np.abs(vals) < 1e-9):
                continue
            b = PchipInterpolator(centers, vals, extrapolate=True)
            h = np.asarray(b(tt), float)
            hr = float(b(np.array([min(self.t_ref, T)]) )[0])
            if abs(hr) < 1e-9 or np.any(np.abs(h) < 1e-9):
                continue
            Y[s:, j] = y / (h / hr)
        return Y


# ================================================================ 3. 阵列信号处理


class ArraySharedShape:
    """阵列共享漂移形态（本项目的核心推荐算法）。

    数据模型是乘性的：y_j(t) = F_j·h(t)。因此**归一化形态 h(t) 在通道间是
    公共的**，可以从整个阵列一次性稳健地估计：

      1) 每通道归一化：u_j(t) = y_j(t)/m_j，m_j = t_ref±w 窗口均值
      2) 以载荷幅度为权重加权平均：ĥ(t) = Σ w_j·u_j(t) / Σ w_j
         （权重大 = 该通道信噪比高，抗噪能力天然好）
      3) 用参数化形态拟合 ĥ(t) 以进一步压噪（kind=None 则用经验形态）
      4) 每通道输出 ŷ_j(t) = m_j · ĥ(t)
    """

    group = "3. 阵列信号处理"

    def __init__(self, kind="power", t_ref=0.5, weights="late", min_resp=0.05,
                 param_blend=1.0):
        self.kind = kind
        self.t_ref = t_ref
        self.weights = weights
        self.min_resp = min_resp
        self.param_blend = param_blend

    @property
    def name(self):
        return f"ArrayShared 阵列共享形态({self.kind or '经验'})"

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        tt = t[s:] - t[s]
        Xl = X[s:]
        ir = int(np.clip(self.t_ref * fs, 0, len(tt) - 1))
        hw = max(1, int(round(0.25 * fs)))
        i0, i1 = max(0, ir - hw), min(len(tt), ir + hw)
        m = Xl[i0:i1].mean(axis=0)
        alive = (Xl.max(axis=0) - Xl.min(axis=0)) > 1e-6
        resp = Xl[int(0.05 * fs):int(0.5 * fs)].mean(axis=0)
        use = alive & (np.abs(resp) > self.min_resp) & (np.abs(m) > 1e-6)
        if use.sum() < 2:
            return Y
        if self.weights == "late":
            w = np.abs(Xl[-int(2 * fs):].mean(axis=0)) * use
        elif self.weights == "early":
            w = np.abs(resp) * use
        else:
            w = m * 0 + 1.0
            w = w * use
        w = np.where(use, np.clip(w, 0, None), 0.0)
        if w.sum() <= 0:
            w = use.astype(float)
        U = Xl[:, use] / m[use][None, :]
        W = w[use] / w[use].sum()
        h_emp = U @ W
        if abs(h_emp[ir]) < 1e-9:
            return Y
        h_emp = h_emp / h_emp[ir]
        h = h_emp
        self.r2 = None
        self.params = None
        if self.kind:
            sm = ShapeModel(self.kind, t_floor=0.05)
            if sm.fit(tt - self.t_ref, h_emp):
                h_par = sm(tt - self.t_ref)
                h = (1 - self.param_blend) * h_emp + self.param_blend * h_par
                self.r2 = sm.r2
                self.params = sm.params
        self.h = h
        self.m = m
        for j in np.where(alive)[0]:
            Y[s:, j] = m[j] * h
        return Y


class CommonModeRemoval:
    """共模剔除。

    取阵列（受载通道）的共模时间序列 m(t)（均值或中位数），用零相位平滑得到
    其慢变部分，构造相对漂移 rel(t) = m(t)/m(t_ref) - 1，再按通道灵敏度 a_j
    做乘性修正：ŷ_j = y_j / (1 + a_j·rel(t))。

    a_j 由"通道相对漂移 = 共模相对漂移"的一阶最小二乘估计（默认关闭，
    即 a_j = 1）。
    """

    group = "3. 阵列信号处理"

    def __init__(self, use="mean", smooth_s=5.0, t_ref=0.5, per_channel_alpha=False,
                 min_resp=0.05):
        self.use = use
        self.smooth_s = smooth_s
        self.t_ref = t_ref
        self.per_channel_alpha = per_channel_alpha
        self.min_resp = min_resp

    @property
    def name(self):
        return (f"CMR 共模剔除({self.use},{self.smooth_s:g}s"
                f"{',a_j' if self.per_channel_alpha else ''})")

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        Xl = X[s:]
        tt = t[s:] - t[s]
        resp = Xl[int(0.05 * fs):int(0.5 * fs)].mean(axis=0)
        use = np.where(np.abs(resp) > self.min_resp)[0]
        if len(use) < 2:
            return Y
        sub = Xl[:, use]
        m = sub.mean(axis=1) if self.use == "mean" else np.median(sub, axis=1)
        w = max(3, int(round(self.smooth_s * fs)))
        if w % 2 == 0:
            w += 1
        w = min(w, (len(m) // 2) * 2 - 1)
        if w < 3:
            return Y
        mm = sps.savgol_filter(m, w, 1)
        ir = int(np.clip(self.t_ref * fs, 0, len(mm) - 1))
        if abs(mm[ir]) < 1e-9:
            return Y
        rel = mm / mm[ir] - 1.0
        for idx, j in enumerate(use):
            aj = 1.0
            if self.per_channel_alpha:
                # 用通道自身相对漂移对共模相对漂移做一阶回归
                A = np.vstack([rel, np.ones_like(rel)]).T
                aj, bj = _lstsq(A, sub[:, idx] / max(abs(mm[ir]), 1e-9))
                aj = float(np.clip(aj, 0.0, 5.0))
            Y[s:, j] = sub[:, idx] / (1.0 + aj * rel)
        return Y


class DriftSubspace:
    """漂移子空间（截断 SVD）——阵列盲分离版本。

    1) 用锚点剖面与阵列总和的归一化时间模板构造并剔除"阶跃成分"
    2) 残差按锚点剖面归一化 → 得到各通道的相对漂移矩阵 Rrel
    3) 对 Rrel 做 SVD，取前 k 个主成分重构漂移子空间
    4) 乘性修正：ŷ_j = y_j / (1 + D_j(t))
    """

    group = "3. 阵列信号处理"

    def __init__(self, n_comp=1, t_ref=0.5, standardize=False, min_resp=0.05):
        self.n_comp = n_comp
        self.t_ref = t_ref
        self.standardize = standardize
        self.min_resp = min_resp

    @property
    def name(self):
        return (f"DriftSubspace 漂移子空间 k={self.n_comp}"
                + ("(归一化)" if self.standardize else ""))

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        Xl = X[s:]
        tt = t[s:] - t[s]
        resp = Xl[int(0.05 * fs):int(0.5 * fs)].mean(axis=0)
        use = np.where(np.abs(resp) > self.min_resp)[0]
        if len(use) < 3:
            return Y
        re = Xl[:, use]
        ir = int(np.clip(self.t_ref * fs, 0, len(tt) - 1))
        hw = max(1, int(round(0.25 * fs)))
        prof = re[max(0, ir - hw):min(len(re), ir + hw)].mean(axis=0)
        if np.any(np.abs(prof) < 1e-9):
            return Y
        tot = re.sum(axis=1)
        tmpl = tot / tot[ir]
        step = np.outer(tmpl, prof)
        R = (re - step) / np.abs(prof)[None, :]
        Rrel = R - R[ir][None, :]
        mu = Rrel.mean(axis=0, keepdims=True)
        Rc = Rrel - mu
        sc = Rc.std(axis=0, keepdims=True)
        sc[sc == 0] = 1.0
        Z = Rc / sc if self.standardize else Rc
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        k = min(self.n_comp, len(S))
        Zk = (U[:, :k] * S[:k]) @ Vt[:k]
        D = Zk * sc if self.standardize else Zk
        for idx, j in enumerate(use):
            d = np.clip(D[:, idx], -0.95, 20.0)
            Y[s:, j] = re[:, idx] / (1.0 + d)
        return Y


class CommonModeRefFit:
    """参考量回归：参考 r(t) = 阵列均值；对每通道做 y_j ≈ P(r) 的多项式回归，
    输出 y_j - [P(r) - P(r(t_ref))]（加性残差法）。"""

    group = "3. 阵列信号处理"

    def __init__(self, ref="mean", order=1, t_ref=0.5, min_resp=0.05):
        self.ref = ref
        self.order = order
        self.t_ref = t_ref
        self.min_resp = min_resp

    @property
    def name(self):
        return f"CMRefFit 参考量回归({self.ref}, o{self.order})"

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        Xl = X[s:]
        resp = Xl[int(0.05 * fs):int(0.5 * fs)].mean(axis=0)
        use = np.where(np.abs(resp) > self.min_resp)[0]
        if len(use) < 2:
            return Y
        sub = Xl[:, use]
        r = {"mean": sub.mean(axis=1), "median": np.median(sub, axis=1),
             "max": sub.max(axis=1)}[self.ref]
        A = np.vstack([r ** k for k in range(self.order + 1)]).T
        ir = int(np.clip(self.t_ref * fs, 0, len(r) - 1))
        for idx, j in enumerate(use):
            coef = _lstsq(A, sub[:, idx])
            pred = A @ coef
            Y[s:, j] = sub[:, idx] - (pred - pred[ir])
        return Y


class AdaptiveNoiseCanceller:
    """自适应噪声抵消（NLMS）：参考输入 = 阵列均值（去均值后），
    自适应 FIR 学出与参考相关的干扰并从每通道减去。"""

    group = "3. 阵列信号处理"

    def __init__(self, mu=0.5, order=64, t_ref=0.5, min_resp=0.05):
        self.mu = mu
        self.order = order
        self.t_ref = t_ref
        self.min_resp = min_resp

    @property
    def name(self):
        return f"ANC-NLMS 自适应抵消 μ={self.mu:g},M={self.order}"

    def transform(self, X, t, fs, load_start_idx):
        n, C = X.shape
        Y = X.copy()
        s = load_start_idx
        Xl = X[s:]
        resp = Xl[int(0.05 * fs):int(0.5 * fs)].mean(axis=0)
        use = np.where(np.abs(resp) > self.min_resp)[0]
        if len(use) < 2:
            return Y
        sub = Xl[:, use]
        r = sub.mean(axis=1)
        r = r - r.mean()
        m = len(r)
        M = self.order
        R = np.zeros((m, M))
        for k in range(M):
            R[k:, k] = r[:m - k]
        ir = int(np.clip(self.t_ref * fs, 0, m - 1))
        for idx, j in enumerate(use):
            y = sub[:, idx]
            w = np.zeros(M)
            e = np.empty(m)
            for i in range(m):
                ri = R[i]
                ei = y[i] - ri @ w
                e[i] = ei
                w = w + (self.mu / (1e-6 + ri @ ri)) * ei * ri
            Y[s:, j] = e - (e[ir] - sub[ir, idx])
        return Y
