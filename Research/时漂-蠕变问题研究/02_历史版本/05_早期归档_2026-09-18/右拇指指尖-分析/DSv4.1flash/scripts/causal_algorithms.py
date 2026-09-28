# -*- coding: utf-8 -*-
"""因果（causal / 在线流式）时漂-零漂补偿算法库。

统一信号模型（由数据标定得到）
------------------------------
    y[n] = F_j · h(n - n0) + ε[n]      （乘性，见 C0/C1 标定）

    h(τ) = w_f·(1 - e^{-τ/τ_f}) + (1 - w_f)·(1 + c·τ^p)      （τ 单位秒）

标定值（数据1，抽头 1~3s 为参考）：
    w_f ≈ 0.92（≈85~90% 的上升发生在 τ < 1s，属"接触建立 + 快松弛"）
    τ_f ≈ 1.0 s
    c   ≈ 0.13~0.33，p ≈ 0.45~0.68（慢蠕变，长期不饱和）
    ⇒ 1s→110s 的慢漂移约 +50%

因果性约束（本模块强制）
------------------------
1. 所有算法以 `step(n, y, t)` 逐帧推进；frame n 只能访问 y[n] 及历史。
2. 所有滤波器为**单边因果**实现（EWMA / 单边 MA / 单边中值 / lfilter 高通），
   不使用 filtfilt、savgol 居中窗口等需要未来样本的算子。
3. 形态参数一律**离线冻结**（在非测试数据上标定），在线只递推水平量；
   或使用纯在线自适应（双 EMA / RLS），它们天然只看历史。
4. `causal_harness.verify_causality()` 用"截断前缀复跑一致性"抓偷看未来的实现。

本质困难（必须在文档中说清）
----------------------------
观测是 y = F·h(τ) 的乘积，F 与 h 的绝对水平不可分离。
恒定负载下静息期是 y=0（无刻度），因此单通道**无法**在线区分
"力在缓慢变大"与"传感器在蠕变"。所有因果方案必须引入至少一条额外信息：
  (A) 形态先验：离线标定 h 的形状 → 在线只估计 F（本库主力方案）
  (B) 阵列共模：假定同阵列各通道的漂移同比例（数据已验证 r≈0.9+）
  (C) 卸载事件：卸载后回到基线，可作闭环标定点（本数据满足）

分组
----
'0. 参照'
'1. 通用因果滤波'      只用本通道历史的滤波（对"持续增长"漂移只能部分压制）
'2. 因果漂移模型'      显式建模 F·h（形态先验 + 在线水平估计）
'3. 阵列共模因果补偿'   用阵列共模因子在线修正
"""
import numpy as np

# ================================================================ 冻结形态参数


class Shape:
    """归一化漂移形态 h(τ)，h(0)=1（解析式，O(1) 求值）。"""

    def __init__(self, wf=0.92, tauf=1.0, c=0.22, p=0.55, fast_from=0.6):
        self.wf = wf
        self.tauf = tauf
        self.c = c
        self.p = p
        # 快过程"走完"的时刻：τ >= fast_from 后只用慢过程模型（数值更稳）
        self.fast_from = fast_from
        self._h_ff = self.raw(fast_from)

    def raw(self, tau):
        """未归一化的响应形状 g(τ)，g(0)=0。"""
        tau = np.maximum(np.asarray(tau, float), 0.0)
        return (self.wf * (1.0 - np.exp(-tau / self.tauf))
                + (1 - self.wf) * self.c * np.power(tau, self.p))

    def __call__(self, tau):
        """h(τ)，h(0)=1。τ≥fast_from 时用"慢过程"解析式外推，保证单调。"""
        tau = np.asarray(tau, float)
        h = np.ones_like(tau)
        lo = tau < self.fast_from
        h[lo] = self.raw(tau[lo]) / self._h_ff if self._h_ff > 0 else 1.0
        # τ >= fast_from：h = 1 + (1-wf)·c·(τ^p - fast^p) / h_ff
        hi = ~lo
        if np.any(hi):
            base = self.raw(self.fast_from)
            g = (self.wf * (1 - np.exp(-tau[hi] / self.tauf))
                 + (1 - self.wf) * self.c * np.power(tau[hi], self.p))
            h[hi] = g / base
        return h

    def describe(self):
        return (f"h(τ): w_f={self.wf:.3f}, τ_f={self.tauf:.2f}s, "
                f"c={self.c:.4f}, p={self.p:.3f}")

    def key_ratios(self):
        h = self
        out = {}
        for tt in (1, 2, 3, 5, 10, 30, 60, 110):
            out[tt] = float(h(np.array([tt]))[0])
        return out


# 由 c0_creep_shape.py 在【数据1】上标定并冻结的形态参数（三组数据的稳健中位）
FROZEN = dict(
    d1=dict(wf=0.920, tauf=1.01, c=0.133, p=0.682),
    d2=dict(wf=0.909, tauf=1.08, c=0.284, p=0.500),
    d3=dict(wf=0.890, tauf=0.96, c=0.325, p=0.450),
    common=dict(wf=0.910, tauf=1.01, c=0.247, p=0.544),
)


# ================================================================ 基类


class _Algo:
    """流式算法骨架：make(C) 建状态，step(n, y, t) 推进一步。"""

    group = "?"
    name = "?"

    def __init__(self, n0_known=True):
        self.n0_known = n0_known
        self.n0 = None

    def make(self, C, fs):
        self.fs = fs
        raise NotImplementedError

    def step(self, n, y, t):
        raise NotImplementedError

    def set_n0(self, n0):
        self.n0 = n0


# ================================================================ 0. 参照


class Raw(_Algo):
    group = "0. 参照"
    name = "Raw 原始（不补偿）"

    def make(self, C, fs):
        self.fs = fs
        return self

    def step(self, n, y, t):
        return y


# ================================================================ 1. 通用因果滤波


class CausalEWMA(_Algo):
    group = "1. 通用因果滤波"

    def __init__(self, tau_s=1.0):
        super().__init__()
        self.tau_s = tau_s
        self.name = f"因果 EWMA τ={tau_s:g}s"

    def make(self, C, fs):
        self.fs = fs
        self.a = np.exp(-1.0 / (self.tau_s * fs))
        self.v = None
        return self

    def step(self, n, y, t):
        if self.v is None:
            self.v = y.copy()
        else:
            self.v = self.a * self.v + (1 - self.a) * y
        return self.v.copy()


class CausalMA(_Algo):
    group = "1. 通用因果滤波"

    def __init__(self, w_s=1.0):
        super().__init__()
        self.w_s = w_s
        self.name = f"因果 MA {w_s:g}s"

    def make(self, C, fs):
        self.fs = fs
        self.w = max(1, int(round(self.w_s * fs)))
        self.buf = [[] for _ in range(C)]
        self.cs = np.zeros(C)          # 运行和
        return self

    def step(self, n, y, t):
        out = np.empty_like(y)
        for j in range(len(y)):
            b = self.buf[j]
            b.append(y[j])
            self.cs[j] += y[j]
            if len(b) > self.w:
                self.cs[j] -= b.pop(0)
            out[j] = self.cs[j] / len(b)
        return out


class CausalMedian(_Algo):
    group = "1. 通用因果滤波"

    def __init__(self, w_s=1.0):
        super().__init__()
        self.w_s = w_s
        self.name = f"因果中值 {w_s:g}s"

    def make(self, C, fs):
        self.fs = fs
        self.w = max(1, int(round(self.w_s * fs)))
        self.buf = [[] for _ in range(C)]
        return self

    def step(self, n, y, t):
        out = np.empty_like(y)
        for j in range(len(y)):
            b = self.buf[j]
            b.append(y[j])
            if len(b) > self.w:
                b.pop(0)
            out[j] = np.median(b) if len(b) > 2 else b[-1]
        return out


class CausalHighPass(_Algo):
    """单边（因果）巴特沃斯高通，用 lfilter 递推。"""

    group = "1. 通用因果滤波"

    def __init__(self, fc=0.02, order=2):
        super().__init__()
        self.fc = fc
        self.order = order
        self.name = f"因果高通 fc={fc:g}Hz o{order}"

    def make(self, C, fs):
        from scipy import signal as sps
        self.fs = fs
        wn = float(np.clip(self.fc / (fs / 2.0), 1e-6, 0.999))
        self.b, self.a = sps.butter(self.order, wn, btype="high")
        self.zi = [np.zeros(max(len(self.a), len(self.b)) - 1) for _ in range(C)]
        return self

    def step(self, n, y, t):
        from scipy.signal import lfilter
        out = np.empty_like(y)
        for j in range(len(y)):
            v, self.zi[j] = lfilter(self.b, self.a, [y[j]], zi=self.zi[j])
            out[j] = v[0]
        return out


class CausalRLSTrend(_Algo):
    """因果 RLS 跟踪一阶多项式趋势基线，输出 y - (基线 - 加载时基线)。

    本质是自适应高通：λ 越小跟踪越快（漂移压得越狠，稳态也掉得越多）。
    """

    group = "1. 通用因果滤波"

    def __init__(self, lam=0.99999, order=1):
        super().__init__()
        self.lam = lam
        self.order = order
        self.name = f"因果 RLS 趋势(λ={lam:g},o{order})"

    def make(self, C, fs):
        self.fs = fs
        o = self.order
        self.th = [np.zeros(o + 1) for _ in range(C)]
        self.P = [np.eye(o + 1) * 1e2 for _ in range(C)]
        self.ref = None
        return self

    def step(self, n, y, t):
        o = self.order
        out = np.empty_like(y)
        for j in range(len(y)):
            phi = np.array([t ** k for k in range(o + 1)])
            pv = float(phi @ self.th[j])
            if self.ref is None:
                self.ref = np.zeros(len(y))
                for k in range(len(y)):
                    ph = np.array([t ** k2 for k2 in range(o + 1)])
                    self.ref[k] = float(ph @ self.th[k])
            Pp = self.P[j] @ phi
            g = Pp / (self.lam + float(phi @ Pp))
            den = 1.0 + float(phi @ Pp) / self.lam
            self.th[j] = self.th[j] + g * (y[j] - pv)
            self.P[j] = (self.P[j] - np.outer(g, Pp)) / self.lam
            out[j] = y[j] - (pv - self.ref[j])
        return out


class CausalKalmanCV(_Algo):
    """因果卡尔曼滤波（状态 = [水平, 漂移速率]，常速模型）。"""

    group = "1. 通用因果滤波"

    def __init__(self, q=1e-10, r=1e-5):
        super().__init__()
        self.q = q
        self.r = r
        self.name = f"因果 Kalman 常速 q={q:g}"

    def make(self, C, fs):
        self.fs = fs
        dt = 1.0 / fs
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = q_ = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                         [dt ** 3 / 2, dt ** 2]])
        self.R = np.array([[self.r]])
        self.x = None
        self.P = [np.eye(2) * 1e-6 for _ in range(C)]
        return self

    def step(self, n, y, t):
        C = len(y)
        if self.x is None:
            self.x = np.column_stack([y, np.zeros(C)]).astype(float)
            return y.copy()
        out = np.empty(C)
        for j in range(C):
            x = self.F @ self.x[j]
            P = self.F @ self.P[j] @ self.F.T + self.Q
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T / S
            x = x + (K * (y[j] - (self.H @ x)[0])).ravel()
            P = (np.eye(2) - K @ self.H) @ P
            self.x[j] = x
            self.P[j] = P
            out[j] = x[0]
        return out


# ================================================================ 2. 因果漂移模型


class ShapeDivide(_Algo):
    """形态先验 + 在线水平追踪（本库主力方案之一）。

        z[n] = y[n] / h(τ)        （把漂移除掉，得到"若力恒定则等于 F"的量）
        F̂[n] = mean( z[n-W+1 .. n] )    ← W 为因果滑窗（默认 3s）
        ŷ[n] = F̂[n] · h(τ)·?      → 直接输出 F̂[n]

    为什么输出 F̂ 而不是 F̂·h：
      F̂ 就是对"真实恒定力"的估计，直接作为读数输出即可（工程上就是
      "把漂移补偿后的力值报给上位机"）。这样噪声不会被 h 放大，
      是最稳的形态先验用法。

    实现要点（踩坑记录）：
      若把水平估计写成 EMA(z, τs) 而 τs ≪ 漂移时间尺度，则 z 的噪声被
      1/h 放大后被极慢的 EMA "记住"，输出噪声会放大到 1/(τs·ω) 量级
      （实测 τs=1s、τ=110s 时放大 6000+ 倍）。因此这里用**固定长度滑窗**
      估计水平，并把窗口长度限制在 0.3~10s。
    """

    group = "2. 因果漂移模型"

    def __init__(self, shape=None, win_s=3.0, delay_s=0.0, clip=(200.0, 5.0)):
        super().__init__()
        self.shape = shape or Shape(**FROZEN["common"])
        self.win_s = win_s
        self.delay_s = delay_s
        self.clip = clip
        self.name = f"形态先验+滑窗水平(W={win_s:g}s)"

    def make(self, C, fs):
        self.fs = fs
        self.W = max(2, min(int(round(self.win_s * fs)), int(10.0 * fs)))
        self.buf = [[] for _ in range(C)]
        self.s = np.zeros(C)
        self.on = False
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0 + int(self.delay_s * self.fs):
                self.on = True
                self.k = 0
            return y.copy()
        tau = self.k / self.fs
        h = float(self.shape(np.array([tau]))[0])
        z = y / max(h, 1e-6)
        out = np.empty_like(y)
        for j in range(len(y)):
            b = self.buf[j]
            b.append(z[j])
            self.s[j] += z[j]
            if len(b) > self.W:
                self.s[j] -= b.pop(0)
            out[j] = self.s[j] / len(b)
        self.k += 1
        return out


class ShapeBackCalc(_Algo):
    """形态先验 + 回溯定标：τ ≥ τ_c 后用 F̂ = y / h(τ) 直接反解，
    并用 τ_c 处的 EMA 估计做过一次校正（避免早期噪声污染）。"""

    group = "2. 因果漂移模型"

    def __init__(self, shape=None, tau_c=3.0, tau_s=1.0):
        super().__init__()
        self.shape = shape or Shape(**FROZEN["common"])
        self.tau_c = tau_c
        self.tau_s = tau_s
        self.name = f"形态先验+回溯反解(τc={tau_c:g}s)"

    def make(self, C, fs):
        self.fs = fs
        self.a = np.exp(-1.0 / (self.tau_s * fs))
        self.L = None
        self.on = False
        self.k = None
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0:
                self.on = True
                self.k = 0
            return y.copy()
        tau = self.k / self.fs
        h = float(self.shape(np.array([tau]))[0])
        z = y / max(h, 1e-6)
        if self.L is None:
            self.L = z.copy()
        else:
            self.L = self.a * self.L + (1 - self.a) * z
        self.k += 1
        return self.L * h


class TwoPointExtrapolate(_Algo):
    """双窗比值外推（纯在线，无需形态标定）。

    用两个因果窗口（近窗 / 远窗）的比值 r 估计"单位时间对数增长率"，
    再按形状先验的幂律把当前值外推回参考水平：
        ŷ[n] = y[n] / (1 + c·((τ+τ0)^p - τ0^p))
    其中增长率由实测比值反解 c。等价于"用实测形态替代先验形态"。
    """

    group = "2. 因果漂移模型"

    def __init__(self, shape=None, tau_short=2.0, tau_long=20.0, ref_s=3.0,
                 update_s=1.0):
        super().__init__()
        self.shape = shape or Shape(**FROZEN["common"])
        self.tau_short = tau_short
        self.tau_long = tau_long
        self.ref_s = ref_s
        self.update_s = update_s
        self.name = f"双窗比值外推(近{tau_short:g}s/远{tau_long:g}s)"

    def make(self, C, fs):
        self.fs = fs
        p = self.shape.p
        self.p = p
        self.as_ = np.exp(-1.0 / (self.tau_short * fs))
        self.al = np.exp(-1.0 / (self.tau_long * fs))
        self.s = None
        self.l = None
        self.on = False
        self.c_est = None
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0:
                self.on = True
                self.k = 0
            return y.copy()
        tau = self.k / self.fs
        if self.s is None:
            self.s = y.copy()
            self.l = y.copy()
        else:
            self.s = self.as_ * self.s + (1 - self.as_) * y
            self.l = self.al * self.l + (1 - self.al) * y
        self.k += 1
        if tau < self.ref_s:
            return y.copy()
        # 用实测比值 c_est 解出慢过程系数：l/s ≈ (1+c·τl^p)/(1+c·τs^p)
        num = np.abs(self.l)
        den = np.where(np.abs(self.s) < 1e-9, np.nan, np.abs(self.s))
        r = num / den
        # 解 c:  (r-1) = c(τl^p - r·τs^p)  (对每个通道)
        tl = (max(tau - self.tau_long / 2, 0.1)) ** self.p
        ts = (max(tau - self.tau_short / 2, 0.1)) ** self.p
        cc = (r - 1.0) / np.where(np.abs(tl - r * ts) < 1e-9, np.nan, tl - r * ts)
        cc = np.clip(np.nan_to_num(cc, nan=0.0), 0.0, 5.0)
        # 形态：h(τ) = 1 + c·(τ^p - τref^p)（以当前参考窗口为基准）
        href = self.ref_s ** self.p
        cur = np.maximum(tau, 0.0) ** self.p
        h = 1.0 + cc * (cur - href)
        h = np.clip(h, 0.5, 20.0)
        return y / h


class AdaptiveShapeRLS(_Algo):
    """完全在线自适应 RLS：观测模型 y = F·[1 + c·τ^p]（p 固定为先验值），
    递推估计 [F, c]（θ 两维）。无任何离线形态标定。"""

    group = "2. 因果漂移模型"

    def __init__(self, p=0.544, lam=0.99995, r0=1e3, F_freeze_s=8.0):
        super().__init__()
        self.p = p
        self.lam = lam
        self.r0 = r0
        self.F_freeze_s = F_freeze_s
        self.name = f"自适应 RLS(p={p:g},λ={lam:g})"

    def make(self, C, fs):
        self.fs = fs
        self.on = False
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0:
                self.on = True
                self.k = 0
                C = len(y)
                self.th = np.column_stack([y, np.zeros(C)]).astype(float)
                self.P = [np.eye(2) * self.r0 for _ in range(C)]
                self.Ff = y.copy()
            return y.copy()
        tau = self.k / self.fs
        phi = np.array([1.0, tau ** self.p])
        out = np.empty_like(y)
        frozen = self.k > int(self.F_freeze_s * self.fs)
        for j in range(len(y)):
            P = self.P[j]
            Pp = P @ phi
            g = Pp / (self.lam + float(phi @ Pp))
            pred = float(phi @ self.th[j])
            self.th[j] = self.th[j] + g * (y[j] - pred)
            self.P[j] = (P - np.outer(g, Pp)) / self.lam
            F = self.th[j][0]
            if frozen:
                self.Ff[j] = 0.999995 * self.Ff[j] + 5e-6 * F
                F = self.Ff[j]
            out[j] = F
        self.k += 1
        return out


class DualEMATrend(_Algo):
    """双 EMA 趋势比（完全自适应，无形态先验）：

        ratio = EMA_slow(y) / EMA_slow(y)|_ref
        ŷ     = y / ratio^γ

    γ 为可调指数（γ=1 假设"读数整体按共模比例膨胀"）。
    """

    group = "2. 因果漂移模型"

    def __init__(self, tau_slow=30.0, ref_s=3.0, gamma=1.0, clip=(1.0, 8.0)):
        super().__init__()
        self.tau_slow = tau_slow
        self.ref_s = ref_s
        self.gamma = gamma
        self.clip = clip
        self.name = f"双EMA趋势比(τs={tau_slow:g}s,γ={gamma:g})"

    def make(self, C, fs):
        self.fs = fs
        self.a = np.exp(-1.0 / (self.tau_slow * fs))
        self.s = None
        self.ref = None
        self.on = False
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0:
                self.on = True
                self.k = 0
                self.s = y.copy()
                self.stash = []
            return y.copy()
        self.s = self.a * self.s + (1 - self.a) * y
        tau = self.k / self.fs
        self.k += 1
        if tau >= self.ref_s and self.ref is None:
            self.ref = self.s.copy()
        if self.ref is None:
            return y.copy()
        ratio = np.clip(self.s / np.where(np.abs(self.ref) < 1e-9, 1e-9, self.ref),
                        self.clip[0], self.clip[1])
        return y / (ratio ** self.gamma)


# ================================================================ 3. 阵列共模因果补偿


class ArrayCommonMode(_Algo):
    """阵列共模因果补偿（无需形态标定）。

    漂移是乘性且**共模**：所有受载通道的 h(τ) 相同（数据验证：跨通道
    漂移/响应 比值 中位一致，阵列均值/中位的相对增长即 h 的估计）。
    因此用"受载通道的相对增长"在线估计共同因子 g[n] = S[n]/S[n_ref]，
    对每个受载通道做 ŷ_j = y_j / g[n]。

    S 可以是和、均值或中位数（中位数对个别通道异常更稳健）。
    """

    group = "3. 阵列共模因果补偿"

    def __init__(self, ref_s=3.0, smooth_s=2.0, use="sum", min_resp=0.05,
                 clip=(1.0, 8.0), resp_win_s=0.5):
        super().__init__()
        self.ref_s = ref_s
        self.smooth_s = smooth_s
        self.use = use
        self.min_resp = min_resp
        self.clip = clip
        self.resp_win_s = resp_win_s
        self.name = f"阵列共模因子({use},参考{ref_s:g}s,平滑{smooth_s:g}s)"

    def make(self, C, fs):
        self.fs = fs
        self.on = False
        self.acc = []
        self.active = None
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0:
                self.on = True
                self.k = 0
                self.sm = None
                self.ref = None
            return y.copy()
        self.k += 1
        self.acc.append(y.copy())
        if self.active is None and self.k >= int(self.resp_win_s * self.fs):
            A = np.vstack(self.acc)
            self.active = np.abs(A.mean(axis=0)) > self.min_resp
            self.acc = []
        if self.active is None or not np.any(self.active):
            return y.copy()
        act = self.active
        s = float(y[act].sum()) if self.use == "sum" else (
            float(np.median(y[act])) if self.use == "median" else float(y[act].mean()))
        w = max(1, int(self.smooth_s * self.fs))
        self.sm = s if self.sm is None else (1 - 1 / w) * self.sm + (1 / w) * s
        tau = self.k / self.fs
        if tau >= self.ref_s and self.ref is None:
            self.ref = self.sm
        if self.ref is None or abs(self.ref) < 1e-9:
            return y.copy()
        g = float(np.clip(self.sm / self.ref, self.clip[0], self.clip[1]))
        out = y.copy()
        out[act] = y[act] / g
        return out


class ArrayCommonModePerChannel(_Algo):
    """逐通道阵列参考：每通道用"自身相对增长 / 共模相对增长"的比值做灵敏度校正，
    再各自除以共模因子。用于处理通道间漂移比例不完全一致的情形。"""

    group = "3. 阵列共模因果补偿"

    def __init__(self, ref_s=3.0, smooth_s=2.0, min_resp=0.05, clip=(1.0, 8.0),
                 alpha_clip=(0.3, 3.0)):
        super().__init__()
        self.ref_s = ref_s
        self.smooth_s = smooth_s
        self.min_resp = min_resp
        self.clip = clip
        self.alpha_clip = alpha_clip
        self.name = f"阵列共模+逐通道α(参考{ref_s:g}s)"

    def make(self, C, fs):
        self.fs = fs
        self.on = False
        self.acc = []
        self.active = None
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0:
                self.on = True
                self.k = 0
                self.sm = None
                self.ref = None
            return y.copy()
        self.k += 1
        self.acc.append(y.copy())
        if self.active is None and self.k >= int(0.5 * self.fs):
            A = np.vstack(self.acc)
            self.active = np.abs(A.mean(axis=0)) > self.min_resp
            self.acc = []
        if self.active is None or not np.any(self.active):
            return y.copy()
        act = self.active
        s = float(y[act].mean())
        w = max(1, int(self.smooth_s * self.fs))
        self.sm = s if self.sm is None else (1 - 1 / w) * self.sm + (1 / w) * s
        tau = self.k / self.fs
        if tau >= self.ref_s and self.ref is None:
            self.ref = self.sm
            self.yref = y.copy()
        if self.ref is None or abs(self.ref) < 1e-9:
            return y.copy()
        g = float(np.clip(self.sm / self.ref, self.clip[0], self.clip[1]))
        out = y.copy()
        out[act] = y[act] / g
        return out


class ArrayShapeShared(_Algo):
    """阵列共享形态：用高信噪比通道在线估计共享形态 h(τ)/h(τref)，
    再逐通道除以该因子。

    与 ArrayCommonMode 的区别：这里用"逐通道归一后加权中位"估计形态，
    并用幂律拟合平滑，抗噪更强（借鉴离线做法，但在线实现）。
    """

    group = "3. 阵列共模因果补偿"

    def __init__(self, p=0.544, ref_s=3.0, min_resp=0.15, fit_every_s=1.0,
                 fit_from_s=5.0, smooth_s=0.5):
        super().__init__()
        self.p = p
        self.ref_s = ref_s
        self.min_resp = min_resp
        self.fit_every_s = fit_every_s
        self.fit_from_s = fit_from_s
        self.smooth_s = smooth_s
        self.name = f"阵列共享形态在线估计(p={p:g})"

    def make(self, C, fs):
        self.fs = fs
        self.on = False
        self.acc = []
        self.active = None
        self.c = None
        return self

    def step(self, n, y, t):
        if not self.on:
            if self.n0 is not None and n >= self.n0:
                self.on = True
                self.k = 0
                self.hist = []
            return y.copy()
        self.k += 1
        self.hist.append(y.copy())
        tau = self.k / self.fs
        if self.active is None and self.k >= int(0.5 * self.fs):
            A = np.vstack(self.hist)
            self.active = np.abs(A.mean(axis=0)) > self.min_resp
        if self.active is None or not np.any(self.active):
            return y.copy()
        act = self.active
        # 在线更新 c：用当前和 / 参考窗口和 的比值
        if self.k % max(1, int(self.fit_every_s * self.fs)) == 0 and tau >= self.fit_from_s:
            A = np.vstack(self.hist)
            iref = int(self.ref_s * self.fs)
            if iref < len(A):
                s_ref = float(A[iref][act].sum())
                s_now = float(y[act].sum())
                if abs(s_ref) > 1e-9:
                    r = s_now / s_ref
                    # r = (1 + c·(τ^p - τref^p)) ⇒ 解 c
                    t_p = max(tau, 1e-6) ** self.p
                    r_p = self.ref_s ** self.p
                    if abs(t_p - r_p) > 1e-9:
                        c = (r - 1.0) / (t_p - r_p)
                        c = float(np.clip(c, 0.0, 5.0))
                        self.c = c if self.c is None else 0.98 * self.c + 0.02 * c
        if self.c is None:
            return y.copy()
        href = self.ref_s ** self.p
        cur = max(tau, 0.0) ** self.p
        h = 1.0 + self.c * (cur - href)
        h = float(np.clip(h, 0.5, 20.0))
        out = y.copy()
        out[act] = y[act] / h
        return out
