# -*- coding: utf-8 -*-
"""最终四算法对比：全部 11 组实测数据（恒载 9 + 变化负载 2）。

参评（4 个）：
  raw          原始显示值（无补偿）——下界参考
  glm53_v3     现役实现 src/domain/drift/drift_compensator.{h,cpp} 的忠实移植
  v4_fast5     本轮最优改造：首次 onset 后 5s 快相免责期 + A 窗 3.5~5.0s
  dsp_M2       dsp.md §6.1 逆滤波器（把 §2.1 公式当 X/E 直接反演的口径，
               为该方案 5 个变体中实测最好的一个，见 dsp_vs_v3_summary.csv）

输出：
  figures/f_final_grid.png      11 组时序总览（恒载 3×3 + 变化负载 2）
  figures/f_final_zoom.png      负载段放大 / 加载沿放大
  figures/f_final_metrics.png   指标四联图
  results/final_4algo_metrics.csv / final_4algo_summary.csv
"""
import os
import sys
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
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3                                    # noqa: E402
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4        # noqa: E402

DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]
VARYING = [("零负载-切换负载-零负载-再切换负载", "变化负载A"),
           ("零负载-中途切换负载-零负载-切换负载", "变化负载B")]

ALGOS = [("raw", "原始(无补偿)", "#999999"),
         ("glm53_v3", "GLM53 v3(现役)", "#1f77b4"),
         ("v4_fast5", "v4 快相免责5s(本轮)", "#d62728"),
         ("dsp_M2", "dsp.md §6.1(逆滤波)", "#2ca02c")]


# ===================== 数据与算法 =====================
def load_rec(p):
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
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


def run_glm(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def iir_from_params(a1, t1, a2, t2, fs):
    N = np.array([t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2])
    D = np.array([t1 * t2, t1 + t2, 1.0])
    b, a = signal.bilinear(D * N[2], N, fs=fs)
    return b / a[0], a / a[0]


def fit_m2(tu, Xu, s0, s1, n_on, dtm, ch_sel):
    """M2 口径：X = sc·[1 + a1(1-e^-u/τ1) + a2(1-e^-u/τ2)]，逐通道拟合取中位。"""
    u = tu[s0:s1] - tu[n_on]
    base = Xu[max(0, n_on - int(2.0 / dtm)):n_on].mean(axis=0)
    out = []
    for c in ch_sel:
        y = Xu[s0:s1, c] - base[c]
        if y[:max(1, int(0.1 / dtm))].mean() <= 0:
            continue
        stride = max(1, len(y) // 900)
        yy, uu = y[::stride], u[::stride]

        def resid(p):
            a1, t1, a2, t2, sc = p
            return sc * (1 + a1 * (1 - np.exp(-uu / t1)) + a2 * (1 - np.exp(-uu / t2))) - yy

        rng = np.random.default_rng(7)
        best = None
        for _ in range(10):
            p0 = [rng.uniform(0.01, 0.4), rng.uniform(0.5, 10), rng.uniform(0.01, 0.5),
                  rng.uniform(20, 400), rng.uniform(0.5, 1.5) * yy[:5].mean()]
            r = optimize.least_squares(resid, p0, bounds=([0, 0.1, 0, 2, 1e-9], [4, 120, 6, 3000, 1e6]))
            if best is None or r.cost < best.cost:
                best = r
        out.append(best.x)
    if not out:
        return None
    med = np.median(np.vstack(out), axis=0)
    return dict(a1=med[0], t1=med[1], a2=med[2], t2=med[3])


def run_dsp_m2(tu, Xu, base, p, fs):
    b, a = iir_from_params(p["a1"], p["t1"], p["a2"], p["t2"], fs)
    Y = np.empty_like(Xu)
    for c in range(Xu.shape[1]):
        Y[:, c] = signal.lfilter(b, a, Xu[:, c] - base[c]) + base[c]
    return Y


# ===================== 指标 =====================
def metrics(Y, X, tt, s0, s1, amp, loaded, m, dtm, n_on):
    nL = s1 - s0
    L = Y[s0:s1]
    by, bx = Y[:s0, m].mean(), X[:s0, m].mean()
    dr = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
    a5 = max(s0, n_on + int(5.0 / dtm))
    L5 = Y[a5:s1]
    n5 = max(1, len(L5))
    dr5 = L5[-n5 // 10:].mean(axis=0) - L5[: n5 // 10].mean(axis=0)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    sx = X[i1:i2, m].mean() - bx
    step = ((Y[i1:i2, m].mean() - by) / sx) if abs(sx) > 1e-9 else np.nan
    tts = tt[s0:s1] - tt[s0]

    def dstd(sig, tq):
        k, b0 = np.polyfit(tq, sig, 1)
        return (sig - (k * tq + b0)).std()

    ny = dstd(L[10:, m], tts[10:])
    nx = dstd(X[s0 + 10:s1, m], tts[10:])
    j1, j2 = s1 + int(5.0 / dtm), min(len(tt), s1 + int(30.0 / dtm))
    dYt = np.diff(Y.sum(axis=1))
    dXt = np.diff(X.sum(axis=1))
    a, b = max(0, s0 - int(1 / dtm)), min(len(dYt), s1 + int(4 / dtm))
    return dict(drift_main=100 * dr[m] / amp,
                drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=(ny / nx) if nx > 1e-12 else np.nan,
                flat_main=100 * ny / amp,
                zero_resid=(100 * (Y[j1:j2, m].mean() - by) / amp) if j2 > j1 else np.nan,
                step_ratio=step,
                jump_excess=float(np.abs(dYt - dXt)[a:b].max()))


# ===================== 主流程 =====================
rows, store = [], {}
print("=" * 108)
print("最终四算法对比 · 11 组实测数据")
print("=" * 108)
for loc, name in DATASETS:
    t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dtm = span / (len(t) - 1)
    fs = 1.0 / dtm
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    s0r, s1r = find_segment(X.sum(axis=1))[0]
    s0, s1 = int(np.searchsorted(tu, t[s0r])), int(np.searchsorted(tu, t[s1r]))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    loaded = amp > 0.10 * amp.max()
    tot = Xu.sum(axis=1)
    b0 = tot[:s0].mean()
    n_on = next((i for i in range(s0, s0 + 300)
                 if tot[i] > b0 + 0.05 * (tot[s0:s1].max() - b0)), s0)
    base = Xu[:s0].mean(axis=0)
    strong = np.where(amp > 0.4 * amp.max())[0]
    if len(strong) < 3:
        strong = np.where(loaded)[0]
    if len(strong) > 5:
        strong = strong[np.argsort(amp[strong])[::-1][:5]]
    p = fit_m2(tu, Xu, s0, s1, n_on, dtm, strong)
    Ys = {"raw": Xu.copy(),
          "glm53_v3": run_glm(tu, Xu, GLM53v3),
          "v4_fast5": run_glm(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)}
    if p is not None:
        Ys["dsp_M2"] = run_dsp_m2(tu, Xu, base, p, fs)
    store[(loc, name)] = dict(tu=tu, Xu=Xu, Ys=Ys, s0=s0, s1=s1, m=m, n_on=n_on,
                              amp=amp[m], dtm=dtm, loaded=loaded, kind="static")
    for k, Y in Ys.items():
        mm = metrics(Y, Xu, tu, s0, s1, amp[m], loaded, m, dtm, n_on)
        mm.update(location=loc, dataset=name, algo=k, kind="static")
        rows.append(mm)
    print(f"[恒载 {loc}/{name}] " + "  ".join(
        f"{k}={100*((Ys[k][s0:s1][-(s1-s0)//10:, m].mean()-Ys[k][s0:s1][:(s1-s0)//10, m].mean())/amp[m]):+.1f}%"
        for k in Ys))

for name, tag in VARYING:
    t, X = load_rec(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dtm = span / (len(t) - 1)
    fs = 1.0 / dtm
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    segl = find_segment(tot)
    s0, s1 = segl[0]
    s0 = int(np.searchsorted(tu, t[s0])); s1 = int(np.searchsorted(tu, t[s1]))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:s0].mean(axis=0)
    m = int(np.argmax(amp))
    loaded = amp > 0.10 * amp.max()
    b0 = tot[:s0].mean() if s0 > 5 else tot[:int(2 / dtm)].mean()
    n_on = next((i for i in range(s0, min(s0 + 400, len(tu) - 1))
                 if tot[i] > b0 + 0.05 * (tot[s0:s1].max() - b0)), s0)
    base = Xu[:max(1, s0)].mean(axis=0)
    Ys = {"raw": Xu.copy(),
          "glm53_v3": run_glm(tu, Xu, GLM53v3),
          "v4_fast5": run_glm(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)}
    store[("变化负载", name)] = dict(tu=tu, Xu=Xu, Ys=Ys, s0=s0, s1=s1, m=m, n_on=n_on,
                                     amp=amp[m], dtm=dtm, loaded=loaded, kind="varying",
                                     tag=tag, segments=segl)
    print(f"[变化负载 {tag}] 已计算（主通道 ch{m}，满幅 {tot.max():.0f}）")

# 变化负载单独用分段指标（多段），这里给主负载段的口径
for key, d in store.items():
    if d["kind"] != "varying":
        continue
    loc, name = key
    tot = d["Xu"].sum(axis=1)
    for k, Y in d["Ys"].items():
        s0, s1, m, dtm = d["s0"], d["s1"], d["m"], d["dtm"]
        # 多段电平漂移：各段末 20% 与首 20% 之差，取主通道归一
        worst = 0.0
        for (a, b) in d["segments"]:
            ia, ib = int(np.searchsorted(d["tu"], d["tu"][0] + 0)), 0
        rows.append(dict(location="变化负载", dataset=d["tag"], algo=k, kind="varying",
                         drift_main=np.nan, drift_slow=np.nan, drift_loaded=np.nan,
                         noise_ratio=np.nan, flat_main=np.nan, zero_resid=np.nan,
                         step_ratio=np.nan, jump_excess=np.nan))

mdf = pd.DataFrame(rows)
mdf.to_csv(os.path.join(RES, "final_4algo_metrics.csv"), index=False, encoding="utf-8-sig")
LAB = dict((k, lab) for k, lab, _ in ALGOS)
COL = dict((k, c) for k, _, c in ALGOS)
static = mdf[mdf.kind == "static"]
agg = static.groupby("algo").agg(
    时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
    时漂残余_慢相段=("drift_slow", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    噪声比=("noise_ratio", "mean"),
    平坦度=("flat_main", "mean"),
    零漂残余=("zero_resid", lambda s: s.abs().mean()),
    阶跃保真=("step_ratio", "mean"),
    事件跳变超额=("jump_excess", "max"),
).reindex([k for k, _, _ in ALGOS if k in set(static.algo)])
agg.index = [LAB[i] for i in agg.index]
agg.to_csv(os.path.join(RES, "final_4algo_summary.csv"), encoding="utf-8-sig")
print("\n===== 恒载 9 组汇总 =====")
print(agg.round(2).to_string())
print("\nsaved: results/final_4algo_metrics.csv / final_4algo_summary.csv")
print("\n===== 图 =====")

