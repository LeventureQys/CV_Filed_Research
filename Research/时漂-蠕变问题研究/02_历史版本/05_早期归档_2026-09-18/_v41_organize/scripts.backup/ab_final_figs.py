# -*- coding: utf-8 -*-
"""最终对比图：4 个算法 × 11 组实测数据。

产出：
  figures/F1_overview.png   11 组时序总览（恒载 3×3 + 变化负载 1×2）
  figures/F2_zoom.png       负载段放大（各位置数据2）+ 加载沿放大
  figures/F3_metrics.png    指标四联图 + 分位置明细
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

ORDER = ["raw", "glm53_v3", "v4_fast5", "dsp_M2"]
LAB = {"raw": "原始(无补偿)", "glm53_v3": "GLM53 v3(现役)",
       "v4_fast5": "v4 快相免责5s(本轮)", "dsp_M2": "dsp.md §6.1(逆滤波)"}
COL = {"raw": "#9e9e9e", "glm53_v3": "#1f77b4", "v4_fast5": "#d62728", "dsp_M2": "#2ca02c"}
LW = {"raw": 0.7, "glm53_v3": 1.0, "v4_fast5": 1.1, "dsp_M2": 0.9}
AL = {"raw": 0.85, "glm53_v3": 0.95, "v4_fast5": 1.0, "dsp_M2": 0.75}


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


# ===================== 指标（口径与 GLM53 既有分析一致，只在主通道上算）=====
def metrics(Y, X, tt, s0, s1, amp, loaded, m, dtm, n_on):
    """drift_main  负载段(末10%均值 − 首10%均值)/幅度
       drift_slow  同上但区间从 onset+5s 起（本轮改造的目标量）
       noise_ratio 负载段去线性趋势 std 之比（相对原始同窗）
       step_ratio  onset 后 0.5~2.5s 均值 / 原始同窗均值
       zero_resid  卸载后 5~30s 均值 − 前空载基线，再/幅度
       jump_excess 负载段内（**不含卸载沿**）单帧 |Δ显示−Δ原始| 最大值"""
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
    return dict(drift_main=100 * dr[m] / amp,
                drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=(ny / nx) if nx > 1e-12 else np.nan,
                flat_main=100 * ny / amp,
                zero_resid=(100 * (Y[j1:j2, m].mean() - by) / amp) if j2 > j1 else np.nan,
                step_ratio=step,
                jump_excess=jump)


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
    return None if not out else dict(zip(("a1", "t1", "a2", "t2"),
                                         np.median(np.vstack(out), axis=0)[:4]))


def prep(loc, name, varying=False):
    p = os.path.join(TEMP, loc, name, "device_001_seg000.csv")
    t, X = load_rec(p)
    span = t[-1] - t[0]
    dtm = span / (len(t) - 1)
    tu = np.arange(0.0, span, dtm)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    segs = find_segment(Xu.sum(axis=1))
    s0r, s1r = segs[0]
    s0, s1 = int(np.searchsorted(tu, tu[s0r] if s0r < len(tu) else 0)), \
        int(np.searchsorted(tu, min(t[s1r], span)))
    amp = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp))
    tot = Xu.sum(axis=1)
    b0 = tot[:max(1, s0)].mean()
    n_on = next((i for i in range(s0, min(s0 + 400, len(tu) - 1))
                 if tot[i] > b0 + 0.05 * (tot[max(0, s0):s1].max() - b0)), s0)
    base = Xu[:max(1, s0)].mean(axis=0)
    Ys = {"raw": Xu.copy(),
          "glm53_v3": run_glm(tu, Xu, GLM53v3),
          "v4_fast5": run_glm(tu, Xu, GLM53v4, FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)}
    loaded = amp > 0.10 * amp.max()
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
                amp=amp[m], segs=segs, varying=varying, loaded=loaded)


# ============================ 计算 ============================
print("计算中 ...")
S = {}
for loc, name in DATASETS:
    S[(loc, name)] = prep(loc, name)
    print(f"  {loc}/{name} ok")
for name, tag in VARYING:
    S[("变化负载", name)] = prep("变化负载", name, varying=True)
    print(f"  {tag} ok")

# ============================ F1 总览 ============================
fig = plt.figure(figsize=(19, 15.5), constrained_layout=True)
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.15])

for idx, (loc, name) in enumerate(DATASETS):
    r, c = divmod(idx, 3)
    ax = fig.add_subplot(gs[r, c])
    d = S[(loc, name)]
    m = d["m"]
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        ax.plot(d["tu"], d["Ys"][k][:, m], color=COL[k], lw=LW[k], alpha=AL[k], label=LAB[k])
    ax.axvspan(d["tu"][d["s0"]], d["tu"][min(d["s1"], len(d["tu"]) - 1)],
               color="orange", alpha=0.08)
    end = d["Ys"]["raw"][d["s1"] - 1, m]
    ax.set_title(f"{loc}/{name}  ch{m}  负载段 {d['tu'][d['s1']-1]-d['tu'][d['s0']]:.0f}s  "
                 f"原始时漂 {100*(d['Ys']['raw'][d['s1']-1,m]-d['Ys']['raw'][d['s0'],m])/d['amp']:+.1f}%",
                 fontsize=9.5)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("显示读数", fontsize=8)
    ax.tick_params(labelsize=7.5)

# 变化负载两图（整阵总量）
for j, (name, tag) in enumerate(VARYING):
    ax = fig.add_subplot(gs[3, j])
    d = S[("变化负载", name)]
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        ax.plot(d["tu"], d["Ys"][k].sum(axis=1), color=COL[k], lw=LW[k], alpha=AL[k], label=LAB[k])
    ax.set_title(f"{tag}（整阵总量，8×5/21ch，ADC 域）", fontsize=9.5)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("显示总量 (ADC)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    if j == 0:
        ax.legend(fontsize=8, loc="upper left")

# 第 4 行第 3 格：图例 + 说明
ax = fig.add_subplot(gs[3, 2])
ax.axis("off")
handles = [Line2D([0], [0], color=COL[k], lw=2.0, label=LAB[k]) for k in ORDER]
ax.legend(handles=handles, loc="upper left", fontsize=11, frameon=False, title="四个参评算法")
ax.text(0.02, 0.52,
        "口径说明\n"
        "· 恒载 9 组：主通道（受载最重）原始读数，橙色带 = 负载段\n"
        "· 变化负载 2 组：整阵总量（ADC 域，单位与指尖力值不可比）\n"
        "· raw 为不补偿下界；glm53_v3 为现役实现\n"
        "· v4 = 首次 onset 后 5s 冻结补偿（快相免责）+ A 窗 3.5~5.0s\n"
        "· dsp_M2 = dsp.md §6.1 逆滤波（按 §2.1 公式直接反演的口径，\n"
        "  为该方案 5 个变体中实测最好的一个）\n"
        "· 三处指尖为力值(N)，变化负载为 ADC，跨位置不可直接比数值",
        fontsize=8.6, va="top", family="Microsoft YaHei")

fig.suptitle("四个抗蠕变算法在 11 组实测数据上的总览对比", fontsize=14)
fig.savefig(os.path.join(FIG, "F1_overview.png"), dpi=125)
plt.close(fig)
print("saved: figures/F1_overview.png")

# ============================ F2 放大 ============================
fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
for j, loc in enumerate(["右拇指指尖", "左拇指指尖", "四指指尖"]):
    d = S[(loc, "数据2")]
    m, s0, s1 = d["m"], d["s0"], d["s1"]
    tt = d["tu"][s0:s1] - d["tu"][s0]
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        Y = d["Ys"][k][s0:s1, m]
        axes[0, j].plot(tt, Y - Y[0], color=COL[k], lw=LW[k] + 0.2, alpha=AL[k], label=LAB[k])
    axes[0, j].set_title(f"{loc}/数据2  ch{m}  负载段内（相对各自起点）", fontsize=10)
    axes[0, j].set_xlabel("负载持续时间 (s)", fontsize=9)
    axes[0, j].set_ylabel("读数增量", fontsize=9)
    axes[0, j].legend(fontsize=7.5)
    # 加载沿 ±4s
    nn = int(4.0 / d["dtm"])
    sl = slice(max(0, s0 - int(1.0 / d["dtm"])), min(len(d["tu"]), s0 + nn))
    tt2 = d["tu"][sl] - d["tu"][s0]
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        Y = d["Ys"][k][sl, m]
        axes[1, j].plot(tt2, Y - d["Ys"][k][:s0, m].mean(), color=COL[k], lw=LW[k] + 0.2,
                        alpha=AL[k], label=LAB[k])
    axes[1, j].axhline(0, color="k", lw=0.7, ls="--")
    axes[1, j].axvspan(0, 5, color="orange", alpha=0.10)
    axes[1, j].set_title(f"{loc}/数据2  ch{m}  加载沿（橙色 = v4 免责期 0~5s）", fontsize=10)
    axes[1, j].set_xlabel("相对 onset (s)", fontsize=9)
    axes[1, j].set_ylabel("读数增量", fontsize=9)
fig.suptitle("放大视图：负载段内漂移抑制 / 加载沿保真", fontsize=13)
fig.savefig(os.path.join(FIG, "F2_zoom.png"), dpi=130)
plt.close(fig)
print("saved: figures/F2_zoom.png")

# ============================ 指标汇总 + F3 ============================
rows = []
for (loc, name), d in S.items():
    for k in ORDER:
        if k not in d["Ys"]:
            continue
        mm = metrics(d["Ys"][k], d["Xu"], d["tu"], d["s0"], d["s1"], d["amp"],
                     d["loaded"] if "loaded" in d else (np.ones(d["Xu"].shape[1], bool)),
                     d["m"], d["dtm"], d["n_on"])
        mm.update(location=loc, dataset=name, algo=k,
                  kind=("varying" if d["varying"] else "static"))
        rows.append(mm)
mdf = pd.DataFrame(rows)
mdf.to_csv(os.path.join(RES, "final_4algo_metrics.csv"), index=False, encoding="utf-8-sig")
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
).reindex([k for k in ORDER if k in set(static.algo)])
agg.index = [LAB[i] for i in agg.index]
agg.to_csv(os.path.join(RES, "final_4algo_summary.csv"), encoding="utf-8-sig")
print("\n===== 恒载 9 组汇总（越小越好；阶跃保真越接近 1 越好）=====")
print(agg.round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
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
    ax.bar(xs, vals, yerr=errs, capsize=5, color=[COL[k] for k in ORDER], alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([LAB[k] for k in ORDER], rotation=10, ha="center", fontsize=9)
    ax.set_title(title + "（柱=9 组均值，误差棒=最大/标准差）", fontsize=10)
    if key in ("noise_ratio", "step_ratio"):
        ax.axhline(1.0, color="k", ls="--", lw=1.0)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("恒载 9 组指标对比（主通道口径）", fontsize=13)
fig.savefig(os.path.join(FIG, "F3_metrics.png"), dpi=130)
plt.close(fig)
print("saved: figures/F3_metrics.png")

piv = static.pivot_table(index=["location", "dataset"], columns="algo",
                         values="drift_main").reindex(columns=ORDER)
piv.to_csv(os.path.join(RES, "final_4algo_drift_by_dataset.csv"), encoding="utf-8-sig")
print("\n===== 各组时漂残余（全负载段，占幅 %）=====")
print(piv.round(2).to_string())
print("\nsaved: results/final_4algo_metrics.csv / final_4algo_summary.csv / "
      "final_4algo_drift_by_dataset.csv")
print("saved: figures/F1_overview.png / F2_zoom.png / F3_metrics.png")
