# -*- coding: utf-8 -*-
"""最终对比（5 算法）：raw / glm53_v3 / v4_fast3 / v4_fast5 / dsp_M2。

比上一版新增 v4_fast3（免责期 3s）。
另外把 dsp_M2 的逐通道拟合种子固定，避免图与表不一致。

产出：
  figures/G1_overview.png    恒载 9 组总览（3×3）
  figures/G2_varying.png     变化负载单独长图（拉长时间轴，分段放大）
  figures/G3_zoom.png        负载段放大 + 加载沿放大（含免责期对比）
  figures/G4_metrics.png     指标四联（5 算法）
  results/final5_*.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal, optimize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3                                    # noqa: E402
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4        # noqa: E402

DATASETS = [(loc, f"数据{i}") for loc in ["右拇指指尖", "左拇指指尖", "四指指尖"] for i in (1, 2, 3)]
VARYING = [("零负载-切换负载-零负载-再切换负载", "变化负载A"),
           ("零负载-中途切换负载-零负载-切换负载", "变化负载B")]

ORDER = ["raw", "glm53_v3", "v4_fast3", "v4_fast5", "dsp_M2"]
LAB = {"raw": "原始(无补偿)", "glm53_v3": "GLM53 v3(现役)",
       "v4_fast3": "v4 免责3s", "v4_fast5": "v4 免责5s", "dsp_M2": "dsp.md §6.1(逆滤波)"}
COL = {"raw": "#9e9e9e", "glm53_v3": "#1f77b4", "v4_fast3": "#ff7f0e",
       "v4_fast5": "#d62728", "dsp_M2": "#2ca02c"}
LW = {"raw": 0.7, "glm53_v3": 1.0, "v4_fast3": 1.0, "v4_fast5": 1.1, "dsp_M2": 0.9}
AL = {"raw": 0.85, "glm53_v3": 0.95, "v4_fast3": 0.9, "v4_fast5": 1.0, "dsp_M2": 0.75}
KW = {"v4_fast3": dict(FAST_S=3.0, A_W0_V4=2.0, A_W1_V4=3.0),
      "v4_fast5": dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)}


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
    dY, dX = np.diff(Y[:, m]), np.diff(X[:, m])
    jump = float(np.abs(dY - dX)[s0:s1].max()) if s1 > s0 else np.nan
    return dict(drift_main=100 * dr[m] / amp, drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=(ny / nx) if nx > 1e-12 else np.nan,
                flat_main=100 * ny / amp,
                zero_resid=(100 * (Y[j1:j2, m].mean() - by) / amp) if j2 > j1 else np.nan,
                step_ratio=step, jump_excess=jump)


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


def fit_m2(tu, Xu, s0, s1, n_on, dtm, ch_sel, seed=7):
    """固定种子的逐通道拟合（保证图与表一致）"""
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

        rng = np.random.default_rng(seed)
        best = None
        for _ in range(10):
            p0 = [rng.uniform(0.01, 0.4), rng.uniform(0.5, 10), rng.uniform(0.01, 0.5),
                  rng.uniform(20, 400), rng.uniform(0.5, 1.5) * yy[:5].mean()]
            r = optimize.least_squares(resid, p0, bounds=([0, 0.1, 0, 2, 1e-9], [4, 120, 6, 3000, 1e6]))
            if best is None or r.cost < best.cost:
                best = r
        out.append(best.x)
    return None if not out else dict(zip(("a1", "t1", "a2", "t2"),
                                         np.median(np.vstack(out), axis=0)[:4]))


def prep(loc, name):
    t, X = load_rec(os.path.join(TEMP, loc, name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dtm = span / (len(t) - 1)
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    segs = find_segment(Xu.sum(axis=1))
    s0r, s1r = segs[0]
    s0 = int(np.searchsorted(tu, tu[min(s0r, len(tu) - 1)]))
    s1 = int(np.searchsorted(tu, min(t[s1r], span)))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp))
    tot = Xu.sum(axis=1)
    b0 = tot[:max(1, s0)].mean()
    n_on = next((i for i in range(s0, min(s0 + 400, len(tu) - 1))
                 if tot[i] > b0 + 0.05 * (tot[max(0, s0):s1].max() - b0)), s0)
    base = Xu[:max(1, s0)].mean(axis=0)
    loaded = amp > 0.10 * amp.max()
    Ys = {"raw": Xu.copy(), "glm53_v3": run_glm(tu, Xu, GLM53v3),
          "v4_fast3": run_glm(tu, Xu, GLM53v4, **KW["v4_fast3"]),
          "v4_fast5": run_glm(tu, Xu, GLM53v4, **KW["v4_fast5"])}
    strong = np.where(amp > 0.4 * amp.max())[0]
    if len(strong) < 3:
        strong = np.where(loaded)[0]
    if len(strong) > 5:
        strong = strong[np.argsort(amp[strong])[::-1][:5]]
    prm = fit_m2(tu, Xu, s0, s1, n_on, dtm, strong)
    if prm is not None:
        b, a = iir_from_params(prm["a1"], prm["t1"], prm["a2"], prm["t2"], 1 / dtm)
        Yd = np.empty_like(Xu)
        for c in range(Xu.shape[1]):
            Yd[:, c] = signal.lfilter(b, a, Xu[:, c] - base[c]) + base[c]
        Ys["dsp_M2"] = Yd
    return dict(tu=tu, Xu=Xu, Ys=Ys, s0=s0, s1=s1, m=m, n_on=n_on, dtm=dtm,
                amp=amp[m], segs=segs, loaded=loaded)


# ============================== 计算 ==============================
print("计算中 ...")
S = {}
for loc, name in DATASETS:
    S[(loc, name)] = prep(loc, name)
    print(f"  {loc}/{name} ok")
for name, tag in VARYING:
    S[("变化负载", name)] = prep("变化负载", name)
    print(f"  {tag} ok")

# ============================== G1 恒载总览 ==============================
fig, axes = plt.subplots(3, 3, figsize=(19, 11.5), constrained_layout=True)
for ax, (loc, name) in zip(axes.flat, DATASETS):
    d = S[(loc, name)]
    m = d["m"]
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        ax.plot(d["tu"], d["Ys"][k][:, m], color=COL[k], lw=LW[k], alpha=AL[k], label=LAB[k])
    ax.axvspan(d["tu"][d["s0"]], d["tu"][min(d["s1"], len(d["tu"]) - 1)], color="orange", alpha=0.08)
    raw = d["Ys"]["raw"]
    ax.set_title(f"{loc}/{name}  ch{m}   负载段 {d['tu'][d['s1']-1]-d['tu'][d['s0']]:.0f}s   "
                 f"原始时漂 {100*(raw[d['s1']-1,m]-raw[d['s0'],m])/d['amp']:+.1f}%", fontsize=10)
    ax.set_xlabel("时间 (s)", fontsize=8.5)
    ax.set_ylabel("显示读数", fontsize=8.5)
    ax.tick_params(labelsize=8)
axes.flat[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("恒载 9 组：5 个算法主通道时序对比（橙色带 = 负载段）", fontsize=14)
fig.savefig(os.path.join(FIG, "G1_overview.png"), dpi=125)
plt.close(fig)
print("saved: figures/G1_overview.png")

# ============================== G2 变化负载长图 ==============================
fig = plt.figure(figsize=(20, 13.5), constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.30)

for row, (name, tag) in enumerate(VARYING):
    d = S[("变化负载", name)]
    ax = fig.add_subplot(gs[row, 0])
    tot = d["Xu"].sum(axis=1)
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        ax.plot(d["tu"], d["Ys"][k].sum(axis=1), color=COL[k], lw=1.35, alpha=AL[k], label=LAB[k])
    ax.plot(d["tu"], tot, color="k", lw=0.6, alpha=0.35, ls=":", label="原始（细点线，参考）")
    for (a, b) in d["segs"]:
        xa = d["tu"][min(a, len(d["tu"]) - 1)]
        xb = d["tu"][min(b, len(d["tu"]) - 1)]
        ax.axvspan(xa, xb, color="orange", alpha=0.07)
    ax.set_title(f"{tag}（整阵总量 21ch，ADC 域）  全长 {d['tu'][-1]:.0f}s",
                 fontsize=12, loc="left")
    ax.set_ylabel("显示总量 (ADC)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.18)
    if row == 0:
        ax.legend(fontsize=9, ncol=6, loc="upper left")
        ax.set_xlabel("时间 (s)", fontsize=10)
    else:
        ax.set_xlabel("时间 (s)", fontsize=10)

# 第三格：数据B 的「中途变载」局部放大（拉长看关键段）
d = S[("变化负载", "零负载-中途切换负载-零负载-切换负载")]
ax = fig.add_subplot(gs[2, 0])
tot = d["Xu"].sum(axis=1)
lo, hi = 16.0, 34.0
sl = slice(int(lo / d["dtm"]), int(hi / d["dtm"]))
for k in ORDER:
    if k not in d["Ys"]:
        continue
    ax.plot(d["tu"][sl], d["Ys"][k].sum(axis=1)[sl], color=COL[k], lw=1.6, alpha=AL[k], label=LAB[k])
ax.plot(d["tu"][sl], tot[sl], color="k", lw=0.7, alpha=0.4, ls=":", label="原始（参考）")
ax.axvline(20.95, color="k", ls="--", lw=1.1)
ax.annotate("真实变载 5N→10N\n@20.95s", xy=(20.95, tot[sl].max() * 0.55),
            xytext=(22.5, tot[sl].max() * 0.42), fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1.0))
ax.set_xlim(lo, hi)
ax.set_title("数据B「中途变载」局部放大（16~34s，时间轴拉长）——"
             "v3 在变载后继续按旧幅度扣、欠报更大；v4 免责期清零补偿后严格跟随原始",
             fontsize=11.5, loc="left")
ax.set_xlabel("时间 (s)", fontsize=10)
ax.set_ylabel("显示总量 (ADC)", fontsize=10)
ax.tick_params(labelsize=9)
ax.grid(alpha=0.18)
ax.legend(fontsize=9, ncol=3, loc="lower right")

fig.suptitle("变化负载专项：整阵总量时序（单独长图，便于拉长观察）", fontsize=14)
fig.savefig(os.path.join(FIG, "G2_varying.png"), dpi=125)
plt.close(fig)
print("saved: figures/G2_varying.png")

# ============================== G3 放大 ==============================
fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
for j, loc in enumerate(["右拇指指尖", "左拇指指尖", "四指指尖"]):
    d = S[(loc, "数据2")]
    m, s0, s1 = d["m"], d["s0"], d["s1"]
    tt = d["tu"][s0:s1] - d["tu"][s0]
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        Y = d["Ys"][k][s0:s1, m]
        axes[0, j].plot(tt, Y - Y[0], color=COL[k], lw=LW[k] + 0.3, alpha=AL[k], label=LAB[k])
    axes[0, j].set_title(f"{loc}/数据2  ch{m}  负载段内（相对各自起点）", fontsize=10.5)
    axes[0, j].set_xlabel("负载持续时间 (s)", fontsize=9)
    axes[0, j].set_ylabel("读数增量", fontsize=9)
    axes[0, j].legend(fontsize=7.5)
    nn = int(9.0 / d["dtm"])
    sl = slice(max(0, s0 - int(1.0 / d["dtm"])), min(len(d["tu"]), s0 + nn))
    tt2 = d["tu"][sl] - d["tu"][s0]
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        Y = d["Ys"][k][sl, m]
        axes[1, j].plot(tt2, Y - d["Ys"][k][:s0, m].mean(), color=COL[k],
                        lw=LW[k] + 0.3, alpha=AL[k], label=LAB[k])
    axes[1, j].axhline(0, color="k", lw=0.7, ls="--")
    axes[1, j].axvline(3, color="#ff7f0e", lw=1.0, ls="--", alpha=0.8)
    axes[1, j].axvline(5, color="#d62728", lw=1.0, ls="--", alpha=0.8)
    axes[1, j].axvspan(0, 3, color="#ff7f0e", alpha=0.07)
    axes[1, j].axvspan(3, 5, color="#d62728", alpha=0.07)
    axes[1, j].set_title(f"{loc}/数据2  ch{m}  加载沿 0~9s（橙=免责3s，红=免责5s）", fontsize=10.5)
    axes[1, j].set_xlabel("相对 onset (s)", fontsize=9)
    axes[1, j].set_ylabel("读数增量", fontsize=9)
    axes[1, j].legend(fontsize=7.5)
fig.suptitle("放大视图：负载段内漂移抑制 / 加载沿（三个免责期变体的差异）", fontsize=13)
fig.savefig(os.path.join(FIG, "G3_zoom.png"), dpi=130)
plt.close(fig)
print("saved: figures/G3_zoom.png")

# ============================== 指标 + G4 ==============================
rows = []
for (loc, name), d in S.items():
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        mm = metrics(d["Ys"][k], d["Xu"], d["tu"], d["s0"], d["s1"], d["amp"],
                     d["loaded"], d["m"], d["dtm"], d["n_on"])
        mm.update(location=loc, dataset=name, algo=k,
                  kind=("varying" if loc == "变化负载" else "static"))
        rows.append(mm)
mdf = pd.DataFrame(rows)
mdf.to_csv(os.path.join(RES, "final5_metrics.csv"), index=False, encoding="utf-8-sig")
static = mdf[mdf.kind == "static"]
agg = static.groupby("algo").agg(
    时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
    时漂残余_慢相段=("drift_slow", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    噪声比=("noise_ratio", "mean"), 平坦度=("flat_main", "mean"),
    零漂残余=("zero_resid", lambda s: s.abs().mean()),
    阶跃保真=("step_ratio", "mean"), 事件跳变超额=("jump_excess", "max"),
).reindex([k for k in ORDER if k in set(static.algo)])
agg.index = [LAB[i] for i in agg.index]
agg.to_csv(os.path.join(RES, "final5_summary.csv"), encoding="utf-8-sig")
print("\n===== 恒载 9 组汇总（5 算法）=====")
print(agg.round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
met = [("drift_main", "时漂残余（全负载段，占幅 %）", True),
       ("drift_slow", "时漂残余（慢相段 onset+5s 起，占幅 %）", True),
       ("noise_ratio", "噪声比（<1 = 噪声被抑制）", False),
       ("step_ratio", "阶跃保真（1.0 = 完美）", False)]
xs = np.arange(len(ORDER))
for ax, (key, title, use_abs) in zip(axes.flat, met):
    vals, errs = [], []
    for k in ORDER:
        v = static[static.algo == k][key].astype(float)
        vals.append(abs(v).mean() if use_abs else v.mean())
        errs.append(v.abs().max() if use_abs else v.std())
    ax.bar(xs, vals, yerr=errs, capsize=5, color=[COL[k] for k in ORDER], alpha=0.92)
    ax.set_xticks(xs)
    ax.set_xticklabels([LAB[k] for k in ORDER], rotation=12, ha="center", fontsize=9)
    ax.set_title(title + "（柱=9 组均值，误差棒=最大/标准差）", fontsize=10.5)
    if key in ("noise_ratio", "step_ratio"):
        ax.axhline(1.0, color="k", ls="--", lw=1.0)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("恒载 9 组指标对比（主通道口径，5 算法）", fontsize=13)
fig.savefig(os.path.join(FIG, "G4_metrics.png"), dpi=130)
plt.close(fig)
print("saved: figures/G4_metrics.png")

piv = static.pivot_table(index=["location", "dataset"], columns="algo",
                         values="drift_main").reindex(columns=ORDER)
piv.to_csv(os.path.join(RES, "final5_drift_by_dataset.csv"), encoding="utf-8-sig")
print("\n===== 各组时漂残余（全负载段，占幅 %）=====")
print(piv.round(2).to_string())
print("\nsaved: results/final5_metrics.csv / final5_summary.csv / final5_drift_by_dataset.csv")
