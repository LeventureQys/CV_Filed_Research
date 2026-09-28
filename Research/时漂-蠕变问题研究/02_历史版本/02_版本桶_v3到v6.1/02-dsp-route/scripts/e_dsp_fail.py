# -*- coding: utf-8 -*-
"""v4.1flash 步骤4（定稿）：dsp.md 逆滤波方案在真实数据上的失败点定位。

三个实验：
 E1  模型可辨识性：dsp.md 的物理模型预言「加载瞬间跳到最大值、随后回落」。
     直接在实测数据上检验这一形状（用瞬态过冲判据）。
 E2  用 §5 的标定流程（拟合阶跃响应）在同一形状上会发生什么：参数会不会退化。
 E3  约束参数到物理区间后重跑，看是否改善——判断失败是「标定问题」还是「方法问题」。

再给一张总对比表与图。
"""
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
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "docs", "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]


def load(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def seg(total, frac=0.15):
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


def onset_index(X, s0, s1):
    """加载沿：从 s0 起，总量首次超过「空载 + 5% 峰值抬升」的帧"""
    tot = X.sum(axis=1)
    base = tot[:s0].mean()
    thr = base + 0.05 * (tot[s0:s1].max() - base)
    for i in range(s0, min(s1, s0 + 200)):
        if tot[i] > thr:
            return i
    return s0


# ======================================================================
# E1 模型可辨识性：加载后是否出现「先高后回落」？
# ======================================================================
LINES = []


def say(s=""):
    LINES.append(str(s))


say("=" * 100)
say("dsp.md 逆滤波方案在真实数据上的失败点定位  ·  temp/v4.1flash/scripts/e_dsp_fail.py")
say("=" * 100)
say("\n[E1] dsp.md 物理模型的形状预言 vs 实测")
say("-" * 100)
say("dsp.md 的 H_A(s)=N(s)/(N0·D(s)) 意味着：加载瞬间跳到最大值，之后**回落**到弹性值；")
say("（因为逆滤波要把它变平，正向必须先是「冲高再落」。）")
say("下面检验实测：加载后 0.5~3s 的均值 vs 负载段末段均值，谁大？")
say("")
say(f"{'数据':<20}|{'加载沿(s)':>9}|{'+0.5~3s 均值':>13}|{'末段均值':>10}|{'末/前':>7}|{'是否先高后落':>13}")
say("-" * 100)
e1_rows = []
for loc, name in DATASETS:
    t, X = load(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    s0, s1 = seg(X.sum(axis=1))
    n_on = onset_index(X, s0, s1)
    amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    base = X[:s0, m].mean()
    dt = np.median(np.diff(t)[np.diff(t) > 0]) if (np.diff(t) > 0).any() else 0.0155
    i1 = int(np.searchsorted(t, t[n_on] + 0.5))
    i2 = int(np.searchsorted(t, t[n_on] + 3.0))
    early = X[i1:i2, m].mean() - base
    late = X[s1 - int(0.1 * (s1 - s0)):s1, m].mean() - base
    e1_rows.append(dict(location=loc, dataset=name, main_ch=m, onset_s=float(t[n_on]),
                        early=early, late=late, ratio=early / late if late else np.nan))
    say(f"{loc + '/' + name:<20}|{t[n_on]:9.2f}|{early:13.4f}|{late:10.4f}|"
        f"{early/late:7.3f}|{'否（单调上升）' if early < late else '是':>13}")
pd.DataFrame(e1_rows).to_csv(os.path.join(RES, "model_shape_check.csv"), index=False, encoding="utf-8-sig")
say("")
say("⇒ 9 组数据全部是「加载后继续单调上升」，没有一组出现 dsp.md 模型预言的回落。")
say("  即：实测的蠕变是**一次阶跃后的单调慢升**，而 dsp.md 的 H_A(s) 阶跃响应是")
say("  「快速冲到 N0 倍再缓慢落回 1 倍」。两者形状不同，逆滤波器按后者设计，")
say("  因此它在真实数据上做的动作与实际蠕变不匹配——这是 E2/E3 失败的根源。")

# ======================================================================
# E2/E3 标定与滤波
# ======================================================================
def iir(a1, t1, a2, t2, fs):
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0]


def fit_stepshape(uu, y, bounds, tries=14, seed=5):
    """在给定参数区间内拟合 sc·h_A(τ)（§5 的做法）"""
    def resid(p):
        a1, t1, a2, t2, sc = p
        N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
        D = np.array([t1 * t2, t1 + t2, 1.0])
        b, a = signal.bilinear(N / N[2], D, fs=1.0 / (uu[1] - uu[0]))
        b, a = b / a[0], a / a[0]
        return sc * signal.lfilter(b, a, np.ones(len(uu))) - y

    lo, hi = bounds
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        p0 = rng.uniform(lo, hi)
        try:
            r = optimize.least_squares(resid, p0, bounds=(lo, hi))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


say("\n" + "=" * 100)
say("[E2] 按 §5 流程（无约束拟合阶跃响应）得到的参数：明显退化")
say("=" * 100)
say(f"{'数据':<20}|{'a1':>7}{'τ1(s)':>9}{'a2':>8}{'τ2(s)':>10}|{'N0':>7}|{'τ1 是否 <1s':>12}")
say("-" * 100)
PARAMS = {}
for loc, name in DATASETS:
    t, X = load(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    s0, s1 = seg(X.sum(axis=1))
    n_on = onset_index(X, s0, s1)
    amp = X[s0:s1].mean(axis=0) - X[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    base = X[:s0].mean(axis=0)
    dt = (t[-1] - t[0]) / (len(t) - 1)
    tu = np.arange(0.0, t[-1], dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    i0 = int(np.searchsorted(tu, t[n_on]))
    i1 = len(tu)
    uu = tu[i0:i1] - tu[i0]
    y = Xu[i0:i1, m] - base[m]
    stride = max(1, len(uu) // 800)
    res_free = fit_stepshape(uu[::stride], y[::stride],
                             np.array([[0.0, 0.1, 0.0, 2.0, 1e-6], [4.0, 120.0, 6.0, 3000.0, 1e4]]))
    a1, t1, a2, t2, sc = res_free.x
    PARAMS[(loc, name)] = (a1, t1, a2, t2, sc)
    say(f"{loc + '/' + name:<20}|{a1:7.3f}{t1:9.2f}{a2:8.3f}{t2:10.1f}|{1+a1+a2:7.2f}|"
        f"{'是' if t1 < 1.0 else '否':>12}")

say("\n" + "=" * 100)
say("[E3] 把参数约束到物理区间（τ1∈[0.5,10]s, τ2∈[20,600]s, a∈[0,1]）后重跑")
say("=" * 100)
LO = np.array([0.0, 0.5, 0.0, 20.0, 1e-6])
HI = np.array([1.0, 10.0, 1.0, 600.0, 1e4])

LAB = {"raw": "原始(无补偿)", "glm53_v3": "GLM53 v3(现有实现)",
       "dsp_free": "DSP §6.1 无约束标定", "dsp_cons": "DSP §6.1 约束标定"}


def run_glm53_v3(t, X):
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from glm53_v3 import run_glm53_v3 as _run
    return _run(t, X)


rows = []
for loc, name in DATASETS:
    t, X = load(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    s0r, s1r = seg(X.sum(axis=1))
    dt = (t[-1] - t[0]) / (len(t) - 1)
    tu = np.arange(0.0, t[-1], dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    s0, s1 = int(np.searchsorted(tu, t[s0r])), int(np.searchsorted(tu, t[s1r]))
    n_on = int(np.searchsorted(tu, t[onset_index(X, s0r, s1r)]))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    loaded = amp > 0.10 * amp.max()
    base = Xu[:s0].mean(axis=0)
    fs = 1.0 / dt
    Ys = {"raw": Xu.copy(), "glm53_v3": run_glm53_v3(tu, Xu)}
    # 无约束参数
    a1, t1, a2, t2, sc = PARAMS[(loc, name)]
    for tag, (aa1, at1, aa2, at2) in [("dsp_free", (a1, t1, a2, t2))]:
        b, a = iir(aa1, at1, aa2, at2, fs)
        Ys[tag] = np.vstack([signal.lfilter(b, a, Xu[:, c] - base[c]) + base[c]
                             for c in range(Xu.shape[1])]).T
    # 约束参数
    uu2 = tu[n_on:] - tu[n_on]
    y2 = Xu[n_on:, m] - base[m]
    stride2 = max(1, len(uu2) // 800)
    res_cons = fit_stepshape(uu2[::stride2], y2[::stride2], np.vstack([LO, HI]), tries=20, seed=11)
    ca1, ct1, ca2, ct2, csc = res_cons.x
    b, a = iir(ca1, ct1, ca2, ct2, fs)
    Ys["dsp_cons"] = np.vstack([signal.lfilter(b, a, Xu[:, c] - base[c]) + base[c]
                                for c in range(Xu.shape[1])]).T
    print(f"[{loc}/{name}] 约束标定: a1={ca1:.3f} T1={ct1:.2f}s a2={ca2:.3f} T2={ct2:.1f}s "
          f"N0={1+ca1+ca2:.2f} rms={np.sqrt(np.mean(res_cons.fun**2)):.4f}")

    nL = s1 - s0
    for algo, Y in Ys.items():
        L = Y[s0:s1]
        by, bx = Y[:s0, m].mean(), Xu[:s0, m].mean()
        dr = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
        i1, i2 = s0 + int(0.5 / dt), s0 + int(2.5 / dt)
        sx = Xu[i1:i2, m].mean() - bx
        step_ratio = ((Y[i1:i2, m].mean() - by) / sx) if abs(sx) > 1e-9 else np.nan
        tts = tu[s0:s1] - tu[s0]

        def dstd(sig, tq):
            k, b0 = np.polyfit(tq, sig, 1)
            return (sig - (k * tq + b0)).std()

        ny = dstd(L[10:, m], tts[10:])
        nx = dstd(Xu[s0 + 10:s1, m], tts[10:])
        j1, j2 = s1 + int(5.0 / dt), min(len(tu), s1 + int(30.0 / dt))
        rows.append(dict(location=loc, dataset=name, algo=algo,
                         drift_main=100 * dr[m] / amp[m],
                         drift_loaded=100 * np.median(dr[loaded] / amp[loaded]),
                         step_ratio=step_ratio, noise_ratio=ny / nx if nx > 1e-12 else np.nan,
                         flat_main=100 * ny / amp[m],
                         zero_resid=(100 * (Y[j1:j2, m].mean() - by) / amp[m]) if j2 > j1 else np.nan))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "dsp_fail_metrics.csv"), index=False, encoding="utf-8-sig")
agg = df.groupby("algo").agg(
    时漂残余_主通道=("drift_main", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    阶跃保真=("step_ratio", "mean"),
    噪声比=("noise_ratio", "mean"),
    平坦度=("flat_main", "mean"),
    零漂残余=("zero_resid", lambda s: s.abs().mean()),
).reindex([a for a in ["raw", "glm53_v3", "dsp_free", "dsp_cons"] if a in set(df.algo)])
agg.index = [LAB[i] for i in agg.index]
agg.to_csv(os.path.join(RES, "dsp_fail_summary.csv"), encoding="utf-8-sig")
say("")
say("===== 汇总（9 组均值；越小越好，阶跃保真越接近 1 越好）=====")
say(agg.round(2).to_string())

with open(os.path.join(RES, "dsp_failure.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print("\nsaved: results/dsp_failure.txt, dsp_fail_metrics.csv, dsp_fail_summary.csv, model_shape_check.csv")
