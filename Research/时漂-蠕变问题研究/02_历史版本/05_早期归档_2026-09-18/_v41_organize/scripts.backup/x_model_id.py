# -*- coding: utf-8 -*-
"""定论实验：真实传感器的实测曲线到底符合哪个模型？

模型甲（物理卷积，也是 dsp.md §6.1 的逆滤波器所反演的对象）：
    X(τ) = E + (X∞ − E)·(1 − e^{−τ/λ*)   ← 一阶近似；严格形式为
    X(t) = E·h_A(t)，h_A 的阶跃响应，极点为 τ1、τ2 的某种组合。
模型乙（dsp.md §2.1 时域公式）：
    X(t) = E·[1 + a1(1−e^{−t/τ1}) + a2(1−e^{−t/τ2})]

两者在 t=0⁺ 都等于 E，但 X(∞)/E 都等于 N0 = 1+a1+a2；
差别在于**中间形状**与「逆滤波器把它反演成什么」。

本脚本做两件事：
 A. 合成数据闭环：分别用甲/乙生成信号，喂给 §6.1 的逆滤波器，看输出是否回到 E；
 B. 实测判定：把 9 组恒载数据的负载段归一化后，比较甲/乙的拟合残差，
    并统计实测的 X(∞)/X(0⁺) 与拟合出的 N0 是否一致。
"""
import os
import numpy as np
import pandas as pd
from scipy import signal, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
FS = 100.5
DT = 1.0 / FS


def coeffs(a1, t1, a2, t2, fs=FS):
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0], N


def step_response_of(a1, t1, a2, t2, n, fs=FS):
    """H_A(s)=N(s)/(N0 D(s)) 的离散阶跃响应（= 模型甲的归一化形状）"""
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(N / N[2], D, fs=fs)
    b, a = b / a[0], a / a[0]
    return signal.lfilter(b, a, np.ones(n))


# ======================================================================
# A. 合成数据闭环
# ======================================================================
print("=" * 100)
print("[A] 合成数据闭环：把同一组 (a1,τ1,a2,τ2) 的两种模型分别喂给 §6.1 的逆滤波器")
print("=" * 100)
A1, T1, A2, T2 = 0.15, 1.2, 0.20, 45.0
N0 = 1 + A1 + A2
b, a, _ = coeffs(A1, T1, A2, T2)
E = 1000.0
n = int(900 * FS)
t = np.arange(n) * DT
i_on = int(60 * FS)
u = np.clip(t - t[i_on], 0, None)

hA = step_response_of(A1, T1, A2, T2, n - i_on)
X_jia = np.zeros(n)
X_jia[i_on:] = E * hA
X_yi = np.zeros(n)
X_yi[i_on:] = E * (1 + A1 * (1 - np.exp(-u[i_on:] / T1)) + A2 * (1 - np.exp(-u[i_on:] / T2)))

print(f"参数 a1={A1} τ1={T1} a2={A2} τ2={T2}  ⇒ N0={N0}")
print(f"{'模型':<12}|{'X 在 t=0⁺':>12}{'X 在 60s':>12}{'X 在 300s':>12}{'X∞':>10}|"
      f"{'逆滤波 Y@1s':>12}{'Y@60s':>12}{'Y@300s':>12}{'Y@900s':>12}|{'偏差%':>8}")
print("-" * 132)
for tag, X in [("甲 卷积 E·h_A", X_jia), ("乙 §2.1公式", X_yi)]:
    Y = signal.lfilter(b, a, X)
    xs = [X[i_on], X[i_on + int(1 * FS)], X[i_on + int(60 * FS)], X[i_on + int(300 * FS)], X[n - 1]]
    ys = [Y[i_on + int(1 * FS)], Y[i_on + int(60 * FS)], Y[i_on + int(300 * FS)], Y[n - 1]]
    print(f"{tag:<12}|{xs[0]:12.1f}{xs[2]:12.1f}{xs[3]:12.1f}{xs[4]:10.1f}|"
          + "".join(f"{v:12.1f}" for v in ys) + f"|{100*(ys[-1]-E)/E:8.1f}")
print("\n理想补偿结果都应回到 E = 1000（因为加载瞬时读数就是真值）。")

# ======================================================================
# B. 实测判定
# ======================================================================
print("\n" + "=" * 100)
print("[B] 实测判定：9 组恒载数据的负载段归一化曲线更像哪个模型？")
print("=" * 100)


def load_csv(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


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


rows = []
for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"]:
    for name in ["数据1", "数据2", "数据3"]:
        t, X = load_csv(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
        span = t[-1] - t[0]
        dt = span / (len(t) - 1)
        tu = np.arange(0.0, span, dt)
        Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
        s0r, s1r = find_segment(X.sum(axis=1))
        s0 = int(np.searchsorted(tu, t[s0r]))
        s1 = int(np.searchsorted(tu, t[s1r]))
        amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
        main = int(np.argmax(amp))
        base = Xu[max(0, s0 - int(2 / dt)):s0, main].mean()
        y = Xu[s0:s1, main] - base
        uu = tu[s0:s1] - tu[s0]
        x0 = y[:max(1, int(0.1 / dt))].mean()      # t≈0⁺
        xinf = y[-int(10 / dt):].mean()            # 末端（近似稳态）
        ratio = xinf / x0

        def fit(model):
            def resid(p):
                a1, t1, a2, t2, sc = p
                if model == "jia":
                    n2 = len(uu)
                    h = step_response_of(a1, t1, a2, t2, n2, fs=1 / dt)
                else:
                    h = 1 + a1 * (1 - np.exp(-uu / t1)) + a2 * (1 - np.exp(-uu / t2))
                return sc * h - y
            best = None
            rng = np.random.default_rng(3)
            for _ in range(25):
                p0 = [rng.uniform(0.01, 0.4), rng.uniform(0.5, 10), rng.uniform(0.01, 0.5),
                      rng.uniform(20, 400), rng.uniform(0.5, 1.5) * x0]
                r = optimize.least_squares(resid, p0, bounds=(
                    [0, 0.1, 0, 2, 1e-6], [4, 120, 6, 3000, 1e4]))
                if best is None or r.cost < best.cost:
                    best = r
            return best

        rj, ry = fit("jia"), fit("yi")
        rows.append(dict(location=loc, dataset=name, main_ch=main,
                         onset_amp=x0, inf_amp=xinf, inf_over_onset=ratio,
                         rms_jia=np.sqrt(np.mean(rj.fun ** 2)) / x0,
                         rms_yi=np.sqrt(np.mean(ry.fun ** 2)) / x0,
                         a1_jia=rj.x[0], T1_jia=rj.x[1], a2_jia=rj.x[2], T2_jia=rj.x[3],
                         N0_jia=1 + rj.x[0] + rj.x[2], sc_jia=rj.x[4]))

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "model_identification.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
print(dfm[["location", "dataset", "onset_amp", "inf_amp", "inf_over_onset",
           "rms_jia", "rms_yi", "N0_jia", "sc_jia", "T1_jia", "T2_jia"]].round(4).to_string(index=False))
print(f"\n平均：模型甲(卷积) 归一化 RMS = {dfm.rms_jia.mean():.5f}  "
      f"模型乙(§2.1公式) 归一化 RMS = {dfm.rms_yi.mean():.5f}")
print(f"实测 X∞/X(0⁺) 中位 = {dfm.inf_over_onset.median():.3f}，拟合 N0 中位 = {dfm.N0_jia.median():.3f}")
print(f"拟合幅度 sc（应为弹性幅度 E）中位 = {dfm.sc_jia.median():.4f} vs 实测首值 {dfm.onset_amp.median():.4f}")
print("\nsaved: results/model_identification.csv")
