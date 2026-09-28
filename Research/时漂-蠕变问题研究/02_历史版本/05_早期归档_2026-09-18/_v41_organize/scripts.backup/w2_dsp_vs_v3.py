# -*- coding: utf-8 -*-
"""v4.1flash 步骤2（定稿）：dsp.md 方案 vs 现有 GLM53 v3，在 9 组恒载数据上对比。

对 dsp.md 方案的**正确实现口径**（这一点很关键，先说明清楚）：

  §2.1 的实测蠕变曲线 = 弹性阶跃 E 作用于物理系统 H_A(s)=N(s)/(N0·D(s)) 的阶跃响应：
      X(t) = E·h_A(t),  h_A(0⁺)=1, h_A(∞)=N0=1+a1+a2
  加载瞬间读数 X(0⁺)=E 就是真值（此时蠕变尚未发生），故补偿目标是**把 X 变回恒定 E**。
  §6.1 的逆滤波器 G=1/H_A 正好做到这一点（合成验证：输出恒 = E，误差 0.000%）。
  ⇒ 实现等价于：先减去空载基线，再过 §6.1 的逐通道 IIR。

  常见的误实现是把 §2.1 的时域公式当成「已经含归一化的 X/E」直接反演，
  那会得到 Y = N0·E（系统性高估 N0 倍）——本脚本用 model_identification.csv 的
  M1（正确的卷积口径）与 M2（误用 §2.1 公式做逆滤波器标定）两种都跑，量化差异。

对照算法：
  raw        原始显示值
  glm53_v3   现有 src/domain/drift/drift_compensator.{h,cpp} 的忠实移植
  dsp_M1     dsp.md §6.1 逆滤波器（正确口径）
  dsp_M2     dsp.md §6.1 逆滤波器（把 §2.1 公式当 X/E 直接反演，即误用口径）
  dsp_magic  §6.2 的硬编码系数
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]


# ======================================================================
# 数据
# ======================================================================
def load_recording(path):
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(dtype=float)
    return tr - tr[0], df[ch].to_numpy(dtype=float)


def find_segment(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)[0]


# ======================================================================
# GLM53 v3（src/domain/drift/drift_compensator.cpp 的忠实移植，参数一字未改）
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
        self.last_ts = self.t0 = 0.0
        self.ts_smooth = self.fast = self.slow = self.level_ref = 0.0
        self.min_ts = self.max_ts = 0.0
        self.pending = False
        self.pending_ts = 0.0
        self.armed = self.in_load = False
        self.onset_ts = 0.0
        self.hold = False
        self.hold_comp = None
        self.a_new_acc = np.zeros(n)
        self.a_new_frames = 0
        self.a_captured = False
        self.a_acc = np.zeros(n)
        self.a_frames = 0
        self.A = np.zeros(n)
        self.loaded = np.zeros(n, bool)
        self.g = self.g2 = 0.0
        self.g_rel = np.zeros(n)
        self.gamma = np.ones(n)

    def _eps(self):
        return 1e-6 * (1.0 + abs(self.max_ts))

    def _begin(self, ts):
        self.in_load, self.onset_ts, self.a_captured = True, ts, False
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
        self.g = self.g2 = 0.0
        self.g_rel = np.zeros(self.n)
        self.gamma = np.ones(self.n)

    def _align(self, lv):
        self.level_ref = self.fast = self.slow = lv
        self.pending = False
        self.pending_ts = 0.0

    def process(self, ts, v):
        v = v.copy()
        total = v.sum()
        dt = 0.0
        if self.first:
            self.first = False
            self.last_ts = self.t0 = ts
            self.ts_smooth = self.fast = self.slow = total
            self.min_ts = self.max_ts = self.level_ref = total
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            dt = 0.0 if not (dt > 0.0) else min(dt, 0.1)
            if dt > 0.0:
                self.ts_smooth += (dt / self.TAU_TOTAL) * (total - self.ts_smooth)
                self.fast += (dt / self.TAU_FAST) * (total - self.fast)
                self.slow += (dt / self.TAU_SLOW) * (total - self.slow)
                self.level_ref += (dt / self.TAU_LEVEL) * (self.ts_smooth - self.level_ref)
        self.min_ts = min(self.min_ts, self.ts_smooth)
        self.max_ts = max(self.max_ts, self.ts_smooth)
        eps = self._eps()
        div = abs(self.fast - self.slow)
        thr = (max(self.STEP_REL * max(self.slow, eps), self.STEP_ABS * self.max_ts)
               if self.in_load else
               max(self.ONSET_REL * max(self.slow, eps), self.STEP_ABS * self.max_ts))
        step_now = (div > thr) if self.in_load else (div > thr and self.fast > self.slow)
        if step_now:
            if not self.pending:
                self.pending, self.pending_ts = True, ts
        elif self.pending and div < self.PENDING_RESET * thr:
            self.pending = False
        step_conf = self.pending and (ts - self.pending_ts > self.STEP_PERSIST)
        u = ts - self.onset_ts if self.in_load else 0.0
        idle = (self.ts_smooth < self.IDLE_FRAC * max(self.level_ref, eps) or
                self.ts_smooth < self.UNLOAD_MIN_RATIO * self.min_ts + eps)
        if not self.in_load:
            if abs(ts - self.t0) <= 1e-9:
                if total > eps:
                    self._begin(ts)
                    self._align(self.ts_smooth)
            elif step_conf:
                self._begin(ts)
                self._align(self.fast)
            self.hold = False
        elif u > self.UNLOAD_FAST and idle:
            self.in_load, self.armed = False, True
            self._align(self.ts_smooth)
            self.hold, self.hold_comp = False, None
            self.pending, self.pending_ts = False, 0.0
        elif u > self.STEP_SUPPRESS and step_conf:
            if idle:
                self.in_load, self.armed = False, True
                self._align(self.ts_smooth)
                self.hold, self.hold_comp = False, None
                self.pending, self.pending_ts = False, 0.0
            else:
                self._restep(ts, v - self.b)
                self._align(self.fast)
        elif self.pending:
            self.hold = True
            self.a_new_acc = self.a_new_acc + (v - self.b)
            self.a_new_frames += 1
        else:
            self.hold, self.hold_comp = False, None
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
            out[m] = Z[m] - np.clip(self.gamma[m] * self.A[m] * self.g,
                                    self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
        return out

    def _restep(self, ts, z_now):
        self.in_load, self.onset_ts = True, ts
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
        self.A = self.a_new_acc / self.a_new_frames if self.a_new_frames > 0 else z_now.copy()
        amax = self.A.max()
        self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 else np.zeros(self.n, bool)
        self.a_captured = True
        self.g = 0.0
        if self.hold_comp is not None:
            den = self.gamma * self.A
            m = self.loaded & (self.A > 1e-9) & (np.abs(den) > 1e-9)
            if m.any():
                self.g = float(np.clip(np.median(self.hold_comp[m] / den[m]), 0.0, 1.0))
        self.hold, self.hold_comp = False, None
        self.a_new_acc = np.zeros(self.n)
        self.a_new_frames = 0
        self.pending, self.pending_ts = False, 0.0


def run_glm53_v3(t, X):
    c = GLM53v3(X.shape[1])
    Y = np.empty_like(X)
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y


# ======================================================================
# dsp.md
# ======================================================================
def iir_from_params(a1, t1, a2, t2, fs):
    """§6.1 逐字复刻"""
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0], N[2]


def dsp_filter(X, b, a, base):
    """逐通道：先减空载基线，再过 IIR，再加回基线（基线必须去掉，否则
    录制起点从 0 跳到空载电平会激发逆滤波器的长瞬态）"""
    Y = np.empty_like(X)
    for c in range(X.shape[1]):
        Y[:, c] = signal.lfilter(b, a, X[:, c] - base[c]) + base[c]
    return Y


def fit_model(tu, Xu, s0, s1, n_on, dtm, ch_sel):
    """拟合 M1（卷积口径）：X(τ)=sc·h_A(τ) 与 M2（§2.1 公式口径）：X=sc·[1+Σaᵢ(1-e^-τ/τᵢ)]。
    逐通道拟合后取中位数。返回两种口径各自的 (a1,τ1,a2,τ2) 与拟合质量。"""
    u = tu[s0:s1] - tu[n_on]
    base = Xu[max(0, n_on - int(2.0 / dtm)):n_on].mean(axis=0)
    out = {"M1": [], "M2": []}
    for c in ch_sel:
        y = Xu[s0:s1, c] - base[c]
        if y[:int(0.1 / dtm)].mean() <= 0:
            continue
        stride = max(1, len(y) // 900)
        y = y[::stride]
        uu = u[::stride]

        def resid1(p):
            a1, t1, a2, t2, sc = p
            N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
            D = np.array([t1 * t2, t1 + t2, 1.0])
            b, a = signal.bilinear(N / N[2], D, fs=1 / dtm)
            b, a = b / a[0], a / a[0]
            return sc * signal.lfilter(b, a, np.ones(len(uu))) - y

        def resid2(p):
            a1, t1, a2, t2, sc = p
            return sc * (1 + a1 * (1 - np.exp(-uu / t1)) + a2 * (1 - np.exp(-uu / t2))) - y

        rng = np.random.default_rng(7)
        for tag, resid, ntry in (("M1", resid1, 8), ("M2", resid2, 14)):
            best = None
            for _ in range(ntry):
                p0 = [rng.uniform(0.01, 0.4), rng.uniform(0.5, 10), rng.uniform(0.01, 0.5),
                      rng.uniform(20, 400), rng.uniform(0.5, 1.5) * y[:10].mean()]
                r = optimize.least_squares(resid, p0, bounds=([0, 0.1, 0, 2, 1e-9], [4, 120, 6, 3000, 1e6]))
                if best is None or r.cost < best.cost:
                    best = r
            out[tag].append(dict(a1=best.x[0], t1=best.x[1], a2=best.x[2], t2=best.x[3],
                                 sc=best.x[4], rms=float(np.sqrt(np.mean(best.fun ** 2)))))
    res = {}
    for tag in ("M1", "M2"):
        if not out[tag]:
            continue
        res[tag] = {k: float(np.median([d[k] for d in out[tag]]))
                    for k in ("a1", "t1", "a2", "t2", "sc", "rms")}
        res[tag]["n"] = len(out[tag])
    return res


DSP_MAGIC_B = np.array([0.724123, -1.412351, 0.690218])
DSP_MAGIC_A = np.array([1.0, -1.421543, 0.423533])


# ======================================================================
# 指标
# ======================================================================
def evaluate(Y, Xg, tt, s0, s1, amp, loaded, main, dtm):
    nL = s1 - s0
    L = Y[s0:s1]
    base_y = Y[:s0, main].mean()
    base_x = Xg[:s0, main].mean()
    drift = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    step_x = Xg[i1:i2, main].mean() - base_x
    step_ratio = ((Y[i1:i2, main].mean() - base_y) / step_x) if abs(step_x) > 1e-9 else np.nan
    tts = tt[s0:s1] - tt[s0]

    def dstd(sig, tq):
        k, b0 = np.polyfit(tq, sig, 1)
        return (sig - (k * tq + b0)).std()

    n_y, n_x = dstd(L[10:, main], tts[10:]), dstd(Xg[s0 + 10:s1, main], tts[10:])
    j1, j2 = s1 + int(5.0 / dtm), min(len(tt), s1 + int(30.0 / dtm))
    dY, dX = np.diff(Y[:, main]), np.diff(Xg[:, main])
    return dict(
        drift_main=100 * drift[main] / amp,
        drift_loaded=100 * np.median(drift[loaded] / amp),
        step_ratio=step_ratio,
        noise_ratio=(n_y / n_x) if n_x > 1e-12 else np.nan,
        flat_main=100 * n_y / amp,
        zero_resid=(100 * (Y[j1:j2, main].mean() - base_y) / amp) if j2 > j1 else np.nan,
        jump_excess=float(np.abs(dY - dX)[s0:s1].max()),
    )


# ======================================================================
# 主流程
# ======================================================================
rows, fit_rows, store = [], [], {}
for loc, name in DATASETS:
    t, X = load_recording(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dtm = span / (len(t) - 1)
    fs = 1.0 / dtm
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    s0r, s1r = find_segment(X.sum(axis=1))
    s0, s1 = int(np.searchsorted(tu, t[s0r])), int(np.searchsorted(tu, t[s1r]))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    main = int(np.argmax(amp))
    loaded = amp > 0.10 * amp.max()
    tot = Xu.sum(axis=1)
    base_tot, half = tot[:s0].mean(), 0.5 * (tot[s0:s1].mean() - tot[:s0].mean())
    n_on = next((i for i in range(s0, min(s1, s0 + int(3.0 / dtm)))
                 if tot[i] > base_tot + half), s0)
    base = Xu[:s0].mean(axis=0)

    strong = np.where(amp > 0.4 * amp.max())[0]
    if len(strong) < 3:
        strong = np.where(loaded)[0]
    # M1（卷积口径）每次残差都要跑一遍 lfilter，代价高：只取受载最重的 5 个通道
    strong_fit = strong[np.argsort(amp[strong])[::-1][:5]] if len(strong) > 5 else strong
    fits = fit_model(tu, Xu, s0, s1, n_on, dtm, strong_fit)

    Ys = {"raw": Xu.copy(), "glm53_v3": run_glm53_v3(tu, Xu)}
    for tag in ("M1", "M2"):
        if tag not in fits:
            continue
        f = fits[tag]
        b, a, _ = iir_from_params(f["a1"], f["t1"], f["a2"], f["t2"], fs)
        Ys[f"dsp_{tag}"] = dsp_filter(Xu, b, a, base)
    Ys["dsp_magic"] = dsp_filter(Xu, DSP_MAGIC_B, DSP_MAGIC_A, base)

    f1 = fits.get("M1", {})
    f2 = fits.get("M2", {})
    print(f"[{loc}/{name}] main=ch{main} fs={fs:.1f}Hz  "
          f"M1: a1={f1.get('a1', np.nan):.3f} T1={f1.get('t1', np.nan):.2f} "
          f"a2={f1.get('a2', np.nan):.3f} T2={f1.get('t2', np.nan):.0f} rms={f1.get('rms', np.nan):.4f} | "
          f"M2: a1={f2.get('a1', np.nan):.3f} T1={f2.get('t1', np.nan):.2f} "
          f"a2={f2.get('a2', np.nan):.3f} T2={f2.get('t2', np.nan):.0f} rms={f2.get('rms', np.nan):.4f}")
    fit_rows.append(dict(location=loc, dataset=name, main_ch=main, n_fit=f1.get("n", 0),
                         **{f"M1_{k}": f1.get(k, np.nan) for k in ("a1", "t1", "a2", "t2", "sc", "rms")},
                         **{f"M2_{k}": f2.get(k, np.nan) for k in ("a1", "t1", "a2", "t2", "sc", "rms")},
                         N0_M1=1 + f1.get("a1", 0) + f1.get("a2", 0),
                         N0_M2=1 + f2.get("a1", 0) + f2.get("a2", 0)))
    store[(loc, name)] = dict(t=tu, X=Xu, s0=s0, s1=s1, n_on=n_on, main=main, amp=amp[main],
                              dtm=dtm, Ys=Ys, fits=fits)
    for algo, Y in Ys.items():
        m = evaluate(Y, Xu, tu, s0, s1, amp[main], loaded, main, dtm)
        m.update(location=loc, dataset=name, algo=algo)
        rows.append(m)

mdf = pd.DataFrame(rows)
mdf.to_csv(os.path.join(RES, "dsp_vs_v3_metrics.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(fit_rows).to_csv(os.path.join(RES, "dsp_fit_params.csv"), index=False, encoding="utf-8-sig")

LAB = {"raw": "原始(无补偿)", "glm53_v3": "GLM53 v3(现有实现)",
       "dsp_M1": "DSP §6.1 正确口径", "dsp_M2": "DSP §6.1 误用口径",
       "dsp_magic": "DSP §6.2 硬编码魔数"}
agg = mdf.groupby("algo").agg(
    时漂残余_主通道=("drift_main", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    阶跃保真=("step_ratio", "mean"),
    噪声比=("noise_ratio", "mean"),
    平坦度=("flat_main", "mean"),
    零漂残余=("zero_resid", lambda s: s.abs().mean()),
    事件跳变超额=("jump_excess", "max"),
).reindex([a for a in ["raw", "glm53_v3", "dsp_M1", "dsp_M2", "dsp_magic"] if a in set(mdf.algo)])
agg.index = [LAB[i] for i in agg.index]
agg.to_csv(os.path.join(RES, "dsp_vs_v3_summary.csv"), encoding="utf-8-sig")
print("\n===== 9 组恒载数据汇总（各指标为 9 组均值；越小越好，阶跃保真越接近 1 越好）=====")
print(agg.round(2).to_string())
print("\nsaved: results/dsp_vs_v3_metrics.csv / dsp_vs_v3_summary.csv / dsp_fit_params.csv")
