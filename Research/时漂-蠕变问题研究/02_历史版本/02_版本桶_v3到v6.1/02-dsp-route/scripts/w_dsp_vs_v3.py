# -*- coding: utf-8 -*-
"""v4.1flash 步骤2：真实数据上的 DSP（dsp.md）方案 vs 现有 GLM53 v3 方案。

对象：
  raw                 原始显示值（无补偿）
  glm53_v3            现有实现 `src/domain/drift/drift_compensator.{h,cpp}` 的忠实 Python 移植
  dsp_doc             dsp.md §6.1 公式生成的二阶逆 IIR（逐通道），参数按数据拟合
  dsp_magic           dsp.md §6.2 硬编码系数（不做任何标定），同一套系数作用于所有数据
  dsp_parallel        dsp.md 思路的等价并行实现（逐 τ 一阶级联），用于检验 IIR 的数值条件

评估指标与 GLM53 既有分析保持一致，另加 DSP 方案特有的两项：
  drift_main / drift_loaded   负载段(末10%-首10%)/幅度
  step_ratio                  阶跃保真（onset 后 0.5~2.5s 幅度 / 原始同窗幅度）
  noise_ratio                 负载段去线性趋势 std 之比
  flat_main                   负载段平坦度（去趋势 std / 幅度 %）
  zero_resid                  卸载后 5~30s 均值 / 幅度
  level_err                   补偿后稳态电平相对真值的偏差（DSP 方案的关键风险项）
  jump_excess                 事件处单帧 |Δ显示 − Δ原始| 最大值

输出：results/dsp_vs_v3_metrics.csv、dsp_vs_v3_summary.csv、dsp_fit_params.csv、figures/*.png
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, optimize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))          # temp/v4.1flash/scripts
OUT = os.path.dirname(HERE)                                # temp/v4.1flash
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))                                # temp
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "docs", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
assert os.path.isdir(TEMP), TEMP
assert os.path.isdir(os.path.join(TEMP, "右拇指指尖")), os.listdir(TEMP)

DATASETS = [
    ("右拇指指尖", "数据1"), ("右拇指指尖", "数据2"), ("右拇指指尖", "数据3"),
    ("左拇指指尖", "数据1"), ("左拇指指尖", "数据2"), ("左拇指指尖", "数据3"),
    ("四指指尖", "数据1"), ("四指指尖", "数据2"), ("四指指尖", "数据3"),
]


# ======================================================================
# 数据装载
# ======================================================================
def load_recording(path):
    """返回 (t, X, meta)。时间轴用 timestamp（elapsed 量化重复，不可用）"""
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t_raw = df["timestamp"].to_numpy(dtype=float)
    t = t_raw - t_raw[0]
    X = df[ch].to_numpy(dtype=float)
    meta = {}
    with open(os.path.join(os.path.dirname(path), "session.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return t, X, meta


def find_segment(total, frac=0.15):
    thr = frac * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if loaded[0]:
        s = np.r_[0, s]
    if loaded[-1]:
        e = np.r_[e, len(loaded)]
    segs = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)
    return segs[0] if segs else (0, len(loaded))


# ======================================================================
# GLM53 v3：src/domain/drift/drift_compensator.cpp 的忠实移植（参数一字不改）
# ======================================================================
class GLM53v3:
    TAU_TOTAL, TAU_FAST, TAU_SLOW, TAU_LEVEL = 0.3, 0.7, 6.0, 10.0
    TAU_BASE, TAU_G = 2.0, 3.0
    ONSET_REL, STEP_REL, STEP_ABS, STEP_PERSIST = 0.5, 0.18, 0.01, 2.5
    STEP_SUPPRESS, UNLOAD_FAST = 6.0, 3.0
    IDLE_FRAC, UNLOAD_MIN_RATIO, BASE_GATE_FRAC, PENDING_RESET = 0.10, 1.5, 0.20, 0.5
    LOADED_FRAC, GAMMA_MIN, GAMMA_MAX = 0.10, 0.3, 2.0
    CREEP_LO, CREEP_HI, G_ENABLE = -0.5, 1.5, 0.02
    A_W0, A_W1 = 1.0, 3.0

    def __init__(self, n):
        self.n = n
        self.b = np.zeros(n)
        self.first = True
        self.last_ts = 0.0
        self.t0 = 0.0
        self.ts_smooth = self.fast = self.slow = self.level_ref = 0.0
        self.min_ts = self.max_ts = 0.0
        self.pending = False
        self.pending_ts = 0.0
        self.armed = False
        self.in_load = False
        self.onset_ts = 0.0
        self.hold = False
        self.hold_comp = None
        self.a_new_acc = np.zeros(n)
        self.a_new_frames = 0
        self.a_captured = False
        self.a_acc = np.zeros(n)
        self.a_frames = 0
        self.A = np.zeros(n)
        self.loaded = np.zeros(n, dtype=bool)
        self.g = 0.0
        self.g2 = 0.0
        self.g_rel = np.zeros(n)
        self.gamma = np.ones(n)

    def _eps(self):
        return 1e-6 * (1.0 + abs(self.max_ts))

    def _begin_load(self, ts):
        self.in_load = True
        self.onset_ts = ts
        self.a_captured = False
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
        self.g = 0.0
        self.g2 = 0.0
        self.g_rel = np.zeros(self.n)
        self.gamma = np.ones(self.n)

    def _align(self, lv):
        self.level_ref = lv
        self.fast = lv
        self.slow = lv
        self.pending = False
        self.pending_ts = 0.0

    def _restep(self, ts, z_now):
        self.in_load = True
        self.onset_ts = ts
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
        self.A = self.a_new_acc / self.a_new_frames if self.a_new_frames > 0 else z_now.copy()
        amax = self.A.max()
        self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 else np.zeros(self.n, bool)
        self.a_captured = True
        self.g = 0.0
        if self.hold_comp is not None:
            denom = self.gamma * self.A
            m = self.loaded & (self.A > 1e-9) & (np.abs(denom) > 1e-9)
            if m.any():
                self.g = float(np.clip(np.median(self.hold_comp[m] / denom[m]), 0.0, 1.0))
        self.hold = False
        self.hold_comp = None
        self.a_new_acc = np.zeros(self.n)
        self.a_new_frames = 0
        self.pending = False
        self.pending_ts = 0.0

    def process(self, ts, v):
        v = v.copy()
        total = v.sum()
        dt = 0.0
        if self.first:
            self.first = False
            self.last_ts = ts
            self.t0 = ts
            self.ts_smooth = self.fast = self.slow = total
            self.min_ts = self.max_ts = self.level_ref = total
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            if not (dt > 0.0):
                dt = 0.0
            if dt > 0.1:
                dt = 0.1
            if dt > 0.0:
                self.ts_smooth += (dt / self.TAU_TOTAL) * (total - self.ts_smooth)
                self.fast += (dt / self.TAU_FAST) * (total - self.fast)
                self.slow += (dt / self.TAU_SLOW) * (total - self.slow)
                self.level_ref += (dt / self.TAU_LEVEL) * (self.ts_smooth - self.level_ref)
        self.min_ts = min(self.min_ts, self.ts_smooth)
        self.max_ts = max(self.max_ts, self.ts_smooth)
        eps = self._eps()

        div_onset = self.ONSET_REL * max(self.slow, eps)
        div_step = max(self.STEP_REL * max(self.slow, eps), self.STEP_ABS * self.max_ts)
        div = abs(self.fast - self.slow)
        u = ts - self.onset_ts if self.in_load else 0.0
        thr = div_step if self.in_load else max(div_onset, self.STEP_ABS * self.max_ts)
        step_now = (div > thr) if self.in_load else (div > thr and self.fast > self.slow)

        if step_now:
            if not self.pending:
                self.pending = True
                self.pending_ts = ts
        elif self.pending and div < self.PENDING_RESET * thr:
            self.pending = False
        step_conf = self.pending and (ts - self.pending_ts > self.STEP_PERSIST)

        idle = (self.ts_smooth < self.IDLE_FRAC * max(self.level_ref, eps) or
                self.ts_smooth < self.UNLOAD_MIN_RATIO * self.min_ts + eps)

        if not self.in_load:
            if abs(ts - self.t0) <= 1e-9:
                if total > eps:
                    self._begin_load(ts)
                    self._align(self.ts_smooth)
            elif step_conf:
                self._begin_load(ts)
                self._align(self.fast)
            self.hold = False
        elif u > self.UNLOAD_FAST and idle:
            self.in_load = False
            self.armed = True
            self._align(self.ts_smooth)
            self.hold = False
            self.hold_comp = None
            self.pending = False
            self.pending_ts = 0.0
        elif u > self.STEP_SUPPRESS and step_conf:
            if idle:
                self.in_load = False
                self.armed = True
                self._align(self.ts_smooth)
                self.hold = False
                self.hold_comp = None
                self.pending = False
                self.pending_ts = 0.0
            else:
                self._restep(ts, v - self.b)
                self._align(self.fast)
        elif self.pending:
            self.hold = True
            self.a_new_acc = self.a_new_acc + (v - self.b)
            self.a_new_frames += 1
        else:
            self.hold = False
            self.hold_comp = None
            if self.a_new_frames > 0:
                self.a_new_acc = np.zeros(self.n)
                self.a_new_frames = 0

        if (not self.in_load) and self.armed and self.ts_smooth < self.BASE_GATE_FRAC * self.max_ts:
            self.b = self.b + (dt / self.TAU_BASE) * (v - self.b)

        Z = v - self.b
        if not self.in_load:
            return Z

        if not self.a_captured:
            if self.A_W0 <= u <= self.A_W1:
                self.a_acc = self.a_acc + Z
                self.a_frames += 1
            if u > self.A_W1:
                self.A = self.a_acc / self.a_frames if self.a_frames > 0 else Z.copy()
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 else np.zeros(self.n, bool)
                self.a_captured = True
            else:
                return Z

        if self.hold:
            if self.hold_comp is None:
                self.hold_comp = np.zeros(self.n)
                m = self.loaded & (self.A > 1e-9)
                self.hold_comp[m] = np.clip(self.gamma[m] * self.A[m] * self.g,
                                            self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
            out = Z.copy()
            m = self.loaded & (self.A > 1e-9)
            out[m] = Z[m] - self.hold_comp[m]
            return out

        m = self.loaded & (self.A > 1e-9)
        if m.any():
            g_raw = float(np.median((Z[m] - self.A[m]) / self.A[m]))
            self.g += (dt / self.TAU_G) * (g_raw - self.g)
        if self.g > self.G_ENABLE:
            self.g2 += dt * self.g * self.g
            if m.any():
                self.g_rel[m] += dt * self.g * ((Z[m] - self.A[m]) / self.A[m])
            if self.g2 > 1e-8:
                self.gamma = np.where(self.loaded,
                                      np.clip(self.g_rel / self.g2, self.GAMMA_MIN, self.GAMMA_MAX), 1.0)
        out = Z.copy()
        if m.any():
            creep = np.clip(self.gamma[m] * self.A[m] * self.g,
                            self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
            out[m] = Z[m] - creep
        return out


def run_glm53_v3(t, X):
    n = X.shape[1]
    c = GLM53v3(n)
    Y = np.empty_like(X)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y


# ======================================================================
# dsp.md 方案
# ======================================================================
def iir_from_params(a1, t1, a2, t2, fs):
    """dsp.md §6.1：num_s = gain_corr·D(s)（gain_corr=N0/D0），den_s = N(s)，Tustin 离散化"""
    N = [t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2]
    D = [t1 * t2, t1 + t2, 1.0]
    num_s = [D[0] * N[2], D[1] * N[2], D[2] * N[2]]
    b, a = signal.bilinear(num_s, N, fs=fs)
    return b / a[0], a / a[0]


def dsp_iir_filter(X, b, a):
    """逐通道二阶 IIR（因果），时间轴已在外部重采样为等间隔"""
    return signal.lfilter(b, a, X, axis=0)


def dsp_parallel_filter(X, a1, t1, a2, t2, dt):
    """dsp.md 思路的并行等价实现（数值稳健版）。

    1/H_A = N0·[1 + Σ Kᵢ·HPᵢ(z)],  HPᵢ(z) = (1-z⁻¹)/(1-τᵢz⁻¹),  τᵢ = e^{-dt/τᵢ}
            Kᵢ = (aᵢ/N0)/(1+aᵢ)
    与 §6.1 的 IIR 是同一个传递函数，但不需要把 N0·D(s) 与 N(s) 都化成
    二阶多项式，因此不受「极点/零点近似相消」的数值条件影响。"""
    N0 = 1 + a1 + a2
    out = N0 * X
    for ai, ti in ((a1, t1), (a2, t2)):
        if ai <= 0:
            continue
        tau_p = float(np.exp(-dt / ti))
        Ki = (ai / N0) / (1 + ai)
        hp = signal.lfilter([1.0, -1.0], [1.0, -tau_p], X, axis=0)
        out = out + (N0 * Ki) * hp
    return out


def iir_conditioning(b, a):
    """逆滤波器的数值条件数：κ = ‖b‖₁ / |B(1)|。
    κ 越大，说明二阶多项式形式的分子存在越严重的「大幅值项相消」，
    定点或单精度实现会直接失真。"""
    b1 = float(np.sum(np.abs(b)))
    dc = float(np.polyval(b, 1.0))
    return b1 / abs(dc) if abs(dc) > 1e-300 else float("inf")


def fit_one_channel(u, y, dtm):
    """对单通道归一化曲线拟合 (a1,τ1,a2,τ2)。
    y = X(t)/X_elastic，理想为 1 + a1(1-e^-u/τ1) + a2(1-e^-u/τ2)。
    X_elastic 用「加载后 0.3~1.0s 窗」估计（跳过机械建立段）。"""
    k0, k1 = max(1, int(0.3 / dtm)), max(2, int(1.0 / dtm))
    x_el = y[k0:k1].mean()
    if not np.isfinite(x_el) or abs(x_el) < 1e-9:
        return None
    yn = y / x_el
    step = max(1, len(u) // 3000)
    uu, yy = u[::step], yn[::step]

    def resid(p):
        a1, t1, a2, t2 = p
        return 1 + a1 * (1 - np.exp(-uu / t1)) + a2 * (1 - np.exp(-uu / t2)) - yy

    best, rng = None, np.random.default_rng(1)
    for _ in range(30):
        x0 = np.array([rng.uniform(0.02, 0.4), rng.uniform(0.5, 15.0),
                       rng.uniform(0.05, 0.8), rng.uniform(20.0, 600.0)])
        r = optimize.least_squares(resid, x0, method="trf",
                                   bounds=([0.0, 0.1, 0.0, 2.0], [4.0, 120.0, 6.0, 3000.0]))
        if best is None or r.cost < best.cost:
            best = r
    a1, t1, a2, t2 = best.x
    resid_end = float(1 + a1 * (1 - np.exp(-uu[-1] / t1)) + a2 * (1 - np.exp(-uu[-1] / t2)) - yy[-1])
    return dict(a1=float(a1), tau1=float(t1), a2=float(a2), tau2=float(t2),
                rms=float(np.sqrt(np.mean(best.fun ** 2))), x_el=float(x_el),
                y_end_resid=resid_end)


def fit_creep_params(tu, Xu, s0, s1, n_on, dtm, ch_sel):
    """逐通道拟合后取中位数：各通道蠕变强度差异很大，先各自归一化再汇总。
    同时输出主通道单独拟合的结果用于对照。"""
    u = tu[s0:s1] - tu[n_on]
    base = Xu[max(0, n_on - int(2.0 / dtm)):n_on].mean(axis=0)
    fits = []
    for c in ch_sel:
        r = fit_one_channel(u, Xu[s0:s1, c] - base[c], dtm)
        if r is not None:
            fits.append((c, r))
    if not fits:
        return None
    keys = ("a1", "tau1", "a2", "tau2", "rms", "x_el")
    med = {k: float(np.median([r[k] for _, r in fits])) for k in keys}
    med["n_fit"] = len(fits)
    med["i1"] = float(np.median([r["a1"] / (1 + r["a1"] + r["a2"]) for _, r in fits]))
    med["i2"] = float(np.median([r["a2"] / (1 + r["a1"] + r["a2"]) for _, r in fits]))
    med["a1_max"] = float(max(r["a1"] for _, r in fits))
    med["a2_max"] = float(max(r["a2"] for _, r in fits))
    med["rms_max"] = float(max(r["rms"] for _, r in fits))
    return med


DSP_MAGIC_B = np.array([0.724123, -1.412351, 0.690218])
DSP_MAGIC_A = np.array([1.0, -1.421543, 0.423533])


# ======================================================================
# 指标
# ======================================================================
def uidx(tu, tval):
    """把原时间轴上的时刻映射到等间隔网格 tu 上的下标"""
    return int(np.clip(np.searchsorted(tu, tval), 0, len(tu) - 1))


def evaluate(Y, Xg, tt, s0, s1, amp, loaded, main, dtm):
    """Y / Xg 都在同一时间网格 tt 上（等间隔 dtm）"""
    nL = s1 - s0
    L = Y[s0:s1]
    base_y = Y[:s0, main].mean()
    base_x = Xg[:s0, main].mean()
    drift = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
    drift_main = 100 * drift[main] / amp
    drift_loaded = 100 * np.median(drift[loaded] / amp)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    step_x = Xg[i1:i2, main].mean() - base_x
    step_ratio = ((Y[i1:i2, main].mean() - base_y) / step_x) if abs(step_x) > 1e-9 else np.nan
    tts = tt[s0:s1] - tt[s0]

    def dstd(sig, tq):
        k, b0 = np.polyfit(tq, sig, 1)
        return (sig - (k * tq + b0)).std()

    n_y = dstd(L[10:, main], tts[10:])
    n_x = dstd(Xg[s0 + 10:s1, main], tts[10:])
    noise_ratio = n_y / n_x if n_x > 1e-12 else np.nan
    j1 = s1 + int(5.0 / dtm)
    j2 = min(len(tt), s1 + int(30.0 / dtm))
    zero_resid = (100 * (Y[j1:j2, main].mean() - base_y) / amp) if j2 > j1 else np.nan
    level_comp = Y[s0 + int(0.8 * nL):s1, main].mean()
    level_raw = Xg[s0 + int(0.1 * nL):s0 + int(0.3 * nL), main].mean()
    level_err = 100 * (level_comp - level_raw) / amp
    dY, dX = np.diff(Y[:, main]), np.diff(Xg[:, main])
    exc = np.abs(dY - dX)
    jump_excess = float(exc[s0:s1].max()) if nL > 2 else np.nan
    return dict(drift_main=drift_main, drift_loaded=drift_loaded, step_ratio=step_ratio,
                noise_ratio=noise_ratio, flat_main=100 * n_y / amp, zero_resid=zero_resid,
                level_err=level_err, jump_excess=jump_excess)


# ======================================================================
# 主流程
# ======================================================================
rows, fit_rows, store = [], [], {}
for loc, name in DATASETS:
    ddir = os.path.join(TEMP, loc, name)
    t, X, meta = load_recording(os.path.join(ddir, "device_001_seg000.csv"))
    # 采样周期：用「时间跨度 / 帧数」估计（timestamp 有大量重复帧，
    # 取非零间隔的中位数会偏大；elapsed 列同理且更粗，两者都不能直接用）
    tspan = t[-1] - t[0]
    dtm = tspan / max(1, len(t) - 1)
    fs = 1.0 / dtm
    # 统一时间网格（DSP 的 IIR 要求等间隔；所有算法与指标都在这个网格上算，口径一致）
    tu = np.arange(0.0, tspan, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T

    s0r, s1r = find_segment(X.sum(axis=1))
    s0, s1 = uidx(tu, t[s0r]), uidx(tu, t[s1r])
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    main = int(np.argmax(amp))
    A_main = amp[main]
    loaded = amp > 0.10 * amp.max()
    # 台阶沿：负载段起点后总量首次超过「空载 + 半幅」的时刻
    tot = Xu.sum(axis=1)
    base_tot = tot[:s0].mean()
    half = 0.5 * (tot[s0:s1].mean() - base_tot)
    n_on = s0
    for i in range(s0, min(s1, s0 + int(3.0 / dtm))):
        if tot[i] > base_tot + half:
            n_on = i
            break

    Y_raw = Xu.copy()
    Y_glm = run_glm53_v3(tu, Xu)

    # 参数拟合：只用受载最重的通道（SNR 最好；逐通道蠕变强度差异大，均值会被稀释）
    strong = np.where(amp > 0.4 * amp.max())[0]
    if len(strong) < 3:
        strong = np.where(loaded)[0]
    fit = fit_creep_params(tu, Xu, s0, s1, n_on, dtm, strong)
    if fit is None:
        print(f"[{loc}/{name}] 拟合失败，跳过")
        continue
    fit_rows.append(dict(location=loc, dataset=name, fs=fs, n_channels=Xu.shape[1],
                         main_ch=main, n_on=n_on,
                         **{k: fit[k] for k in ("a1", "tau1", "a2", "tau2", "rms",
                                                "n_fit", "i1", "i2", "a1_max", "a2_max", "rms_max")}))
    b_fit, a_fit = iir_from_params(fit["a1"], fit["tau1"], fit["a2"], fit["tau2"], fs)
    Y_dsp = np.vstack([dsp_iir_filter(Xu[:, c], b_fit, a_fit) for c in range(Xu.shape[1])]).T

    Y_magic = np.vstack([dsp_iir_filter(Xu[:, c], DSP_MAGIC_B, DSP_MAGIC_A)
                         for c in range(Xu.shape[1])]).T
    Y_par = dsp_parallel_filter(Xu, fit["a1"], fit["tau1"], fit["a2"], fit["tau2"], dtm)

    # 诊断：IIR 与并行实现的一致性（衡量 §6.1 的数值条件）
    par_diff = float(np.abs(Y_dsp - Y_par).max())
    g0_iir = float(np.polyval(b_fit, 1.0) / np.polyval(a_fit, 1.0))
    kappa = iir_conditioning(b_fit, a_fit)
    rel = 100 * par_diff / max(1e-9, np.abs(Y_par).max())
    print(f"[{loc}/{name}] main=ch{main} fs={fs:.1f}Hz nfit={fit['n_fit']} "
          f"a1={fit['a1']:.3f} T1={fit['tau1']:.2f}s a2={fit['a2']:.3f} T2={fit['tau2']:.1f}s "
          f"rms={fit['rms']:.4f}(max {fit['rms_max']:.4f}) N0={1+fit['a1']+fit['a2']:.2f} "
          f"G0={g0_iir:.4f} kappa={kappa:.3g} IIR-vs-parallel={par_diff:.3f}({rel:.2f}%)")
    fit["kappa"] = kappa
    fit["iir_parallel_rel_pct"] = rel

    store[(loc, name)] = dict(t=tu, X=Xu, s0=s0, s1=s1, main=main, amp=A_main, loaded=loaded,
                              dtm=dtm, fit=fit, Ys=dict(raw=Y_raw, glm53_v3=Y_glm,
                                                        dsp_doc=Y_dsp, dsp_magic=Y_magic,
                                                        dsp_parallel=Y_par),
                              tu=tu)
    for algo, Y in store[(loc, name)]["Ys"].items():
        met = evaluate(Y, Xu, tu, s0, s1, A_main, loaded, main, dtm)
        met.update(location=loc, dataset=name, algo=algo)
        rows.append(met)

mdf = pd.DataFrame(rows)
mdf.to_csv(os.path.join(RES, "dsp_vs_v3_metrics.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(fit_rows).to_csv(os.path.join(RES, "dsp_fit_params.csv"), index=False, encoding="utf-8-sig")

LAB = {"raw": "原始(无补偿)", "glm53_v3": "GLM53 v3(现有实现)", "dsp_doc": "DSP §6.1(按数据标定)",
       "dsp_magic": "DSP §6.2(硬编码魔数)", "dsp_parallel": "DSP 并行等价实现(参考)"}
agg = mdf.groupby("algo").agg(
    时漂残余_主通道=("drift_main", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    阶跃保真=("step_ratio", "mean"),
    噪声比=("noise_ratio", "mean"),
    平坦度=("flat_main", "mean"),
    零漂残余=("zero_resid", lambda s: s.abs().mean()),
    稳态电平误差=("level_err", lambda s: s.abs().mean()),
    事件跳变超额=("jump_excess", "max"),
).reindex(["raw", "glm53_v3", "dsp_doc", "dsp_magic", "dsp_parallel"])
agg.index = [LAB[i] for i in agg.index]
agg.to_csv(os.path.join(RES, "dsp_vs_v3_summary.csv"), encoding="utf-8-sig")
print("\n===== 9 组恒载数据汇总（均值；幅值越小越好，阶跃保真越接近 1 越好）=====")
print(agg.round(2).to_string())

# ---------------- 图 ----------------
COL = {"raw": "0.6", "glm53_v3": "tab:blue", "dsp_doc": "tab:red", "dsp_magic": "tab:orange"}
fig, axes = plt.subplots(3, 3, figsize=(18, 11), constrained_layout=True)
for ax, (loc, name) in zip(axes.flat, DATASETS):
    d = store[(loc, name)]
    m = d["main"]
    for k in ["raw", "glm53_v3", "dsp_doc", "dsp_magic"]:
        ax.plot(d["t"], d["Ys"][k][:, m], lw=0.6, alpha=0.9, color=COL[k], label=LAB[k])
    ax.axvspan(d["t"][d["s0"]], d["t"][d["s1"]], color="orange", alpha=0.07)
    ax.set_title(f"{loc}/{name} ch{m}（{d['t'][d['s1']]-d['t'][d['s0']]:.0f}s 恒载）", fontsize=9)
    ax.set_xlabel("t (s)", fontsize=8)
    ax.tick_params(labelsize=7)
axes.flat[0].legend(fontsize=7)
fig.suptitle("恒载 9 组：主通道时序对比（DSP 逆滤波 vs GLM53 v3）", fontsize=12)
fig.savefig(os.path.join(FIG, "f1_timeseries_grid.png"), dpi=130)
plt.close(fig)

# 负载段放大（数据2 三处）
fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
for ax, loc in zip(axes, ["右拇指指尖", "左拇指指尖", "四指指尖"]):
    d = store[(loc, "数据2")]
    m = d["main"]
    s0, s1 = d["s0"], d["s1"]
    tt = d["t"][s0:s1] - d["t"][s0]
    for k in ["raw", "glm53_v3", "dsp_doc", "dsp_magic"]:
        Y = d["Ys"][k][s0:s1, m]
        ax.plot(tt, Y - Y[0], lw=0.8, alpha=0.9, color=COL[k], label=LAB[k])
    ax.set_title(f"{loc}/数据2 ch{m} 负载段（相对各自起点）", fontsize=10)
    ax.set_xlabel("负载持续时间 (s)")
    ax.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "f2_load_zoom.png"), dpi=130)
plt.close(fig)

# 阶跃沿放大
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
for ax, loc in zip(axes, ["右拇指指尖", "左拇指指尖", "四指指尖"]):
    d = store[(loc, "数据2")]
    m, s0 = d["main"], d["s0"]
    n = int(3.0 / d["dtm"])
    sl = slice(s0 - int(1.0 / d["dtm"]), s0 + n)
    tt = d["t"][sl] - d["t"][s0]
    for k in ["raw", "glm53_v3", "dsp_doc", "dsp_magic"]:
        Y = d["Ys"][k][sl, m]
        ax.plot(tt, Y - d["Ys"][k][:s0, m].mean(), lw=1.0, alpha=0.9, color=COL[k], label=LAB[k])
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_title(f"{loc}/数据2 ch{m} 阶跃沿 ±3s", fontsize=10)
    ax.set_xlabel("相对 onset (s)")
axes[0].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "f3_step_edge.png"), dpi=130)
plt.close(fig)

print("\nsaved: results/dsp_vs_v3_metrics.csv, dsp_vs_v3_summary.csv, dsp_fit_params.csv")
print("saved: figures/f1_timeseries_grid.png, f2_load_zoom.png, f3_step_edge.png")
