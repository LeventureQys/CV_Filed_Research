# -*- coding: utf-8 -*-
"""**全量数据**上的免责期对比：1 s / 3 s（默认）/ 5 s，外加 raw 下界。

数据 = `temp/` 下全部 13 份录制：
  恒载 9 组（右拇指指尖 / 左拇指指尖 / 四指指尖 各 3 次；display 域）
  变化负载 4 份（切换负载-快相无责、再切换负载、中途切换-1d9493、中途切换-13ffca；ADC 域）

参数口径同 `DriftCompensator::SetFastPhase`：
  FAST_S = 1.0 / 3.0 / 5.0，EXEMPT_AWIN = FAST_S/3，LEV_ARM_S = FAST_S；其余参数不动。

产出：
  results/all_datasets_1s.csv          每份数据 × 每档的全部指标
  results/_all_datasets_1s.log         控制台全文
  figures/H6_1s_overview_all.png       小倍数总览（每份一格：raw / 1s / 3s）
  figures/H7_1s_metrics_all.png        指标对比（恒载时漂 / 实录偏差 / 变载跟踪）
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

KS = ["1s", "3s", "5s"]
FASTOF = {"1s": 1.0, "3s": 3.0, "5s": 5.0}
COL = {"raw": "0.6", "1s": "#1f77b4", "3s": "#ff7f0e", "5s": "#d62728"}
B = os.path.join(TEMP, "变化负载")
HOLD = [(f"{loc}/{f'数据{i}'}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
VARY = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]


def run(fast, tu, Xu):
    class _T(GLM53v51):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    c = _T(Xu.shape[1])
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = fast, fast / 3.0, fast
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def find_segment(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    dd = np.diff(ld.astype(int))
    s = list(np.where(dd == 1)[0] + 1)
    e = list(np.where(dd == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


def hold_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d=None):
    """恒载口径（同 Document/06 §10.3）。"""
    s0r, s1r = find_segment(Xu.sum(axis=1))[0]
    s0 = int(np.searchsorted(tu, tu[min(s0r, len(tu) - 1)]))
    s1 = int(np.searchsorted(tu, min(tu[-1], tu[min(s1r, len(tu) - 1)])))
    amp_v = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp_v))
    amp = amp_v[m]
    loaded = amp_v > 0.10 * amp_v.max()
    base = float(np.median(tot_s[max(0, s0 - int(2 / dtm)):s0]))
    n_on = next((i for i in range(s0, min(s0 + int(5 / dtm), s1))
                 if tot_s[i] > base + 0.05 * (tot_s[s0:s1].max() - base)), s0)
    nL = s1 - s0
    seg = Y[s0:s1]
    by, bx = Y[:s0, m].mean(), Xu[:s0, m].mean()
    dr = seg[-nL // 10:].mean(axis=0) - seg[:nL // 10].mean(axis=0)
    a5 = max(s0, n_on + int(5.0 / dtm))
    n5 = max(1, s1 - a5)
    dr5 = Y[a5:s1][-n5 // 10:].mean(axis=0) - Y[a5:s1][:n5 // 10].mean(axis=0)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    sx = Xu[i1:i2, m].mean() - bx
    step = (Y[i1:i2, m].mean() - by) / sx if abs(sx) > 1e-9 else np.nan
    tq = tu[s0:s1] - tu[s0]

    def dstd(sig, t):
        k, b0 = np.polyfit(t, sig, 1)
        return (sig - (k * t + b0)).std()

    ny = dstd(seg[10:, m], tq[10:])
    nx = dstd(Xu[s0 + 10:s1, m], tq[10:])
    ded = (Xu - Y).sum(axis=1)
    hit = np.where(ded[s0:s1] > 0.005 * abs(tot_s[s1] - base))[0]
    return dict(drift_main=100 * dr[m] / amp, drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=ny / nx if nx > 1e-12 else np.nan, flat=100 * ny / amp,
                step_ratio=step,
                ded_delay=float(hit[0] * dtm) if len(hit) else np.nan,
                a_max=float(c.A.max()), g_end=float(c.g), epoch=len(c.epoch_t),
                max_gap=np.nan, pct=np.nan, cap_med=np.nan, cap_min=np.nan,
                n_event=0, gap_med=np.nan, gap_max=np.nan)


def vary_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d):
    d["Ys"] = {"a": Y}
    d["periods"] = L.find_periods(Xu.sum(axis=1), dtm)
    d["events"] = [e for e, _ in L.detect_events(Xu.sum(axis=1), dtm)]
    y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    gap = np.abs(y - tot_s)
    ev = L.event_table(d, d["events"], {"a": Y}, [], gain_s=6.0, algos=["a"])
    ev["big"] = ev["jump"].abs() >= 2000.0
    thr = 0.30 * ev["pre"].max() if len(ev) else 0
    ev["mid"] = (ev["pre"] > thr) & (ev["post"] > thr) if len(ev) else False
    ml = ev[ev.mid & ev.big] if len(ev) else ev
    cap = (ml["gain_a"] / ml["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna() \
        if len(ml) else pd.Series(dtype=float)
    return dict(drift_main=np.nan, drift_slow=np.nan, drift_loaded=np.nan, noise_ratio=np.nan,
                flat=np.nan, step_ratio=np.nan, ded_delay=np.nan, a_max=float(c.A.max()),
                g_end=float(c.g), epoch=len(c.epoch_t), max_gap=float(gap.max()),
                pct=100 * float(gap.max()) / peak,
                cap_med=float(cap.median()) if len(cap) else np.nan,
                cap_min=float(cap.min()) if len(cap) else np.nan, n_event=int(len(ml)),
                gap_med=float(ml["gap_a"].median()) if len(ml) else np.nan,
                gap_max=float(ml["gap_a"].max()) if len(ml) else np.nan)


rows, curves = [], {}
print("=" * 122)
print("全量数据 · 免责期 1 s / 3 s（默认）/ 5 s 对比（raw 为下界）")
print("=" * 122)
for tag, path in HOLD + VARY:
    if not os.path.exists(path):
        print(f"[skip] 缺文件 {tag}")
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(Xu.sum(axis=1), 0.5 / dtm)
    peak = float(Xu.sum(axis=1).max())
    kind = "恒载" if tag in [t for t, _ in HOLD] else "实采"
    line = f"{tag:>18} [{kind}] {d['span']:5.1f}s ch={Xu.shape[1]:2d}"
    curves[tag] = dict(tu=tu, tot=tot_s, Y={})
    for k in KS:
        Y, c = run(FASTOF[k], tu, Xu)
        mm = (hold_metrics if kind == "恒载" else vary_metrics)(Y, c, tu, Xu, dtm, tot_s, peak, d)
        mm.update(dataset=tag, kind=kind, algo=k, span=d["span"], nch=Xu.shape[1])
        rows.append(mm)
        curves[tag]["Y"][k] = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    if kind == "恒载":
        line += ("  | 时漂 全段/慢相段: " + "  ".join(
            f"{k} {rows[-3 + KS.index(k)]['drift_main']:+.2f}/{rows[-3 + KS.index(k)]['drift_slow']:+.2f}%"
            for k in KS))
    else:
        line += ("  | 全程偏差(ADC): " + "  ".join(
            f"{k} {rows[-3 + KS.index(k)]['max_gap']:.0f}" for k in KS))
        line += f"  | 台阶捕获中位: " + "  ".join(
            f"{k} {rows[-3 + KS.index(k)]['cap_med']:.2f}" for k in KS)
    print(line)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "all_datasets_1s.csv"), index=False, encoding="utf-8-sig")

# ── 汇总 ──────────────────────────────────────────────────────
hold = df[df.kind == "恒载"]
vary = df[df.kind == "实采"]
print("\n----- 恒载 9 组（9 份均值 / |·|）-----")
print(hold.groupby("algo").agg(时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
                              时漂残余_慢相段=("drift_slow", lambda s: s.abs().mean()),
                              受载中位=("drift_loaded", lambda s: s.abs().mean()),
                              噪声比=("noise_ratio", "mean"), 平坦度=("flat", "mean"),
                              阶跃保真=("step_ratio", "mean"), 首扣时延=("ded_delay", "mean"),
                              A占幅度=("a_max", "mean"), g末端=("g_end", "mean"),
                              epoch数=("epoch", "sum")).reindex(KS).round(2).to_string())
print("\n----- 实采 4 份 -----")
print(vary.groupby("algo").agg(全程偏差中位=("max_gap", "median"), 占峰值中位=("pct", "median"),
                              变载窗偏差中位=("gap_med", "median"), 捕获比中位=("cap_med", "mean"),
                              捕获比最小=("cap_min", "min"), 有效事件=("n_event", "sum"),
                              epoch数=("epoch", "sum")).reindex(KS).round(2).to_string())
print("\n每份实录的全程最大偏差（ADC）")
print(vary.pivot_table(index="dataset", columns="algo", values="max_gap").reindex(columns=KS).round(0).to_string())
print("\n每份实录的台阶捕获比（中位；空 = 无有效负载内变载事件）")
print(vary.pivot_table(index="dataset", columns="algo", values="cap_med").reindex(columns=KS).round(2).to_string())

# ═══════════════════ 图 1：小倍数总览 ═══════════════════
order = [t for t, _ in HOLD] + [t for t, _ in VARY if t in curves]
n = len(order)
ncol = 4
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(21, 2.6 * nrow))
fig.suptitle("全量 13 份数据 · 免责期 1 s / 3 s / 5 s 效果总览（灰=原始，蓝=1s，橙=3s 默认，红=5s）",
             fontsize=15)
for ax, tag in zip(axes.ravel(), order):
    cu = curves[tag]
    ax.plot(cu["tu"], cu["tot"], color=COL["raw"], lw=0.9, label="原始")
    for k in KS:
        ax.plot(cu["tu"], cu["Y"][k], color=COL[k], lw=1.1, label=f"免责 {k}")
    r = df[(df.dataset == tag) & (df.algo == "3s")].iloc[0]
    r1 = df[(df.dataset == tag) & (df.algo == "1s")].iloc[0]
    if r.kind == "恒载":
        ttl = (f"{tag}｜时漂(全段) 1s {r1.drift_main:+.2f}% / 3s {r.drift_main:+.2f}%"
               f"｜慢相段 1s {r1.drift_slow:+.2f}% / 3s {r.drift_slow:+.2f}%")
    else:
        ttl = (f"{tag}｜全程偏差 1s {r1.max_gap:.0f} / 3s {r.max_gap:.0f} ADC"
               f"（占峰值 {r1.pct:.1f}% / {r.pct:.1f}%）")
    ax.set_title(ttl, fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
for ax in axes.ravel()[n:]:
    ax.axis("off")
axes.ravel()[0].legend(fontsize=8, loc="lower left")
fig.subplots_adjust(left=0.045, right=0.99, top=0.925, bottom=0.035, hspace=0.42, wspace=0.18)
f1 = os.path.join(FIG, "H6_1s_overview_all.png")
fig.savefig(f1, dpi=105)
plt.close(fig)
print(f"\n图已保存：{f1}")

# ═══════════════════ 图 2：指标对比 ═══════════════════
fig2, axs = plt.subplots(1, 3, figsize=(19.5, 5.6))
fig2.suptitle("全量数据 · 1 s / 3 s / 5 s 指标对比", fontsize=14)

ax = axs[0]
g = hold.groupby("algo").agg(full=("drift_main", lambda s: s.abs().mean()),
                             slow=("drift_slow", lambda s: s.abs().mean())).reindex(KS)
xs = np.arange(2)
for j, k in enumerate(KS):
    b = ax.bar(xs + (j - 1) * 0.26, [g.loc[k, "full"], g.loc[k, "slow"]], 0.25,
               color=COL[k], label=f"免责 {k}")
    ax.bar_label(b, fmt="%.2f", fontsize=9)
rawfull = 15.46
rawslow = 11.74
for i, rv in enumerate((rawfull, rawslow)):
    ax.plot([i - 0.42, i + 0.42], [rv, rv], color="0.45", ls=":", lw=1.3)
    ax.text(i + 0.44, rv, f"原始 {rv:.1f}%", fontsize=9, color="0.35", va="center")
ax.set_xticks(xs); ax.set_xticklabels(["时漂残余 全段", "时漂残余 慢相段"])
ax.set_ylabel("恒载 9 组 均值 (%)"); ax.set_ylim(0, 18)
ax.set_title("(1) 恒载 9 组：时漂残余（越小越好）")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = axs[1]
pv = vary.pivot_table(index="dataset", columns="algo", values="max_gap").reindex(columns=KS)
xs = np.arange(len(pv))
for j, k in enumerate(KS):
    b = ax.bar(xs + (j - 1) * 0.26, pv[k].values, 0.25, color=COL[k], label=f"免责 {k}")
    ax.bar_label(b, fmt="%.0f", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels([s.replace("-", "-\n", 1) for s in pv.index], fontsize=9)
ax.set_ylabel("全程 max|显示−原始| (ADC)")
ax.set_title("(2) 实采 4 份：全程最大偏差（越小越好）")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = axs[2]
pv2 = vary.pivot_table(index="dataset", columns="algo", values="cap_med").reindex(columns=KS)
pv3 = vary.pivot_table(index="dataset", columns="algo", values="gap_med").reindex(columns=KS)
xs = np.arange(len(pv2))
for j, k in enumerate(KS):
    b = ax.bar(xs + (j - 1) * 0.26, pv2[k].values, 0.25, color=COL[k], label=f"捕获比 {k}")
    ax.bar_label(b, fmt="%.2f", fontsize=8)
ax.axhline(1.0, color="0.4", ls=":", lw=1.2)
ax.text(len(pv2) - 0.5, 1.01, "1.0 = 台阶完整透传", fontsize=8.5, color="0.35", ha="right")
ax2 = ax.twinx()
for j, k in enumerate(KS):
    ax2.plot(xs + (j - 1) * 0.26, pv3[k].values, marker="v", ms=6, color=COL[k],
             ls="none", alpha=0.85)
ax2.set_ylabel("变载窗最大偏差 中位 (ADC)（▽）")
ax.set_xticks(xs); ax.set_xticklabels([s.replace("-", "-\n", 1) for s in pv2.index], fontsize=9)
ax.set_ylabel("台阶捕获比（柱）"); ax.set_ylim(0, 1.35)
ax.set_title("(3) 实采：变载台阶捕获比（柱）与变载窗偏差（▽）")
ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=0.3)

fig2.subplots_adjust(left=0.05, right=0.94, top=0.87, bottom=0.14, wspace=0.30)
f2 = os.path.join(FIG, "H7_1s_metrics_all.png")
fig2.savefig(f2, dpi=115)
plt.close(fig2)
print(f"图已保存：{f2}")
