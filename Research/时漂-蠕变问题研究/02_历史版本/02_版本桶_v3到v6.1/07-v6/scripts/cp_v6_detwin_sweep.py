# -*- coding: utf-8 -*-
"""v6 检测窗长度扫描：只改「判断台阶有没有/什么时候起」的窗，其余参数不动。

判据形式（与 v6 一致）：
    d = mean(total, [t−FAST, t]) − mean(total, [t−FAST−GAP−LAG, t−FAST−GAP])
    hit = |d| > max(5σ_d, 5%·|lv_ref|, 1%·max_tot)，连续 3 帧确认
跨度 span = FAST + GAP + LAG（当前值 0.10+0.15+0.20 = 0.45 s）。

产出：results/v6_detwin_sweep.csv、results/_v6_detwin_sweep.log、
      figures/J1_v6_detwin_sweep.png（速度 vs 稳健性的权衡曲线，含 v5-3s 参照线）
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
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402
from glm53_v6 import GLM53v6                           # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

B = os.path.join(TEMP, "变化负载")
HOLD = [(f"{loc}/{f'数据{i}'}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
VARY = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]

CFG = [("S1 现值 0.10/0.15/0.20", dict(DET_FAST=0.10, DET_GAP=0.15, DET_LAG=0.20)),
       ("S2 0.15/0.15/0.25", dict(DET_FAST=0.15, DET_GAP=0.15, DET_LAG=0.25)),
       ("S3 0.20/0.15/0.30", dict(DET_FAST=0.20, DET_GAP=0.15, DET_LAG=0.30)),
       ("S4 0.25/0.20/0.35", dict(DET_FAST=0.25, DET_GAP=0.20, DET_LAG=0.35)),
       ("S5 0.30/0.20/0.45", dict(DET_FAST=0.30, DET_GAP=0.20, DET_LAG=0.45))]


def run_v6(tu, Xu, **kw):
    c = GLM53v6(Xu.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    c.A = np.full(Xu.shape[1], c.A_peak)
    return Y, c


def run_v5(tu, Xu, fast=3.0):
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


def hold_metrics(Y, c, tu, Xu, dtm, tot_s):
    s0r, s1r = find_segment(Xu.sum(axis=1))[0]
    s0, s1 = int(s0r), min(int(s1r), len(tu) - 1)
    amp_v = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp_v))
    amp = amp_v[m]
    n_on = next((i for i in range(s0, min(s0 + int(5 / dtm), s1))
                 if tot_s[i] > np.median(tot_s[max(0, s0 - int(2 / dtm)):s0])
                 + 0.05 * (tot_s[s0:s1].max() - np.median(tot_s[max(0, s0 - int(2 / dtm)):s0]))), s0)
    nL = max(1, s1 - s0)
    dr = Y[s0:s1][-nL // 10:].mean(axis=0) - Y[s0:s1][:nL // 10].mean(axis=0)
    a5 = max(s0, n_on + int(5.0 / dtm))
    n5 = max(1, s1 - a5)
    dr5 = Y[a5:s1][-n5 // 10:].mean(axis=0) - Y[a5:s1][:n5 // 10].mean(axis=0)
    return dict(drift_main=100 * dr[m] / amp, drift_slow=100 * dr5[m] / amp, epoch=len(c.epoch_t))


def vary_metrics(Y, c, d, tu, Xu, dtm, tot_s):
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
    return dict(max_gap=float(gap.max()), gap_med=float(ml["gap_a"].median()) if len(ml) else np.nan,
                cap_med=float(cap.median()) if len(cap) else np.nan,
                cap_min=float(cap.min()) if len(cap) else np.nan, epoch=len(c.epoch_t))


def first_onset(tu, tot, dtm):
    peak = float(np.percentile(tot, 99.5))
    idx = np.where(tot > 0.5 * peak)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    pre = float(np.median(tot[max(0, i - int(1.5 / dtm)):max(1, i - int(0.3 / dtm))]))
    j = i
    while j > 0 and tot[j] > pre + 0.05 * (tot[i] - pre):
        j -= 1
    return j + 1, pre


def stable_time(tu, Ytot, i0, step, dtm, hold_s=30.0, tol_frac=0.05):
    if step <= 0:
        return np.nan
    n = len(Ytot)
    H = int(hold_s / dtm)
    tol = tol_frac * step
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            break
        if np.max(np.abs(Ytot[k:e] - Ytot[k])) <= tol:
            return float(tu[k] - tu[i0])
    return np.nan


DATA = {}
print("载入 13 份数据 ...")
for tag, path in HOLD + VARY:
    d = L.prep(path)
    DATA[tag] = d
    print(f"  {tag:>18}  {d['span']:6.1f}s  ch={d['Xu'].shape[1]:2d}")

rows = []
for label, kw in [("v5-3s（参照）", None)] + CFG:
    span = None if kw is None else kw["DET_FAST"] + kw["DET_GAP"] + kw["DET_LAG"]
    print(f"\n=== {label}  span={span if span else '-'} ===")
    for tag, d in DATA.items():
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        tot_s = L.med_smooth(Xu.sum(axis=1), 0.5 / dtm)
        kind = "恒载" if tag in [t for t, _ in HOLD] else "实采"
        Y, c = (run_v5(tu, Xu) if kw is None else run_v6(tu, Xu, **kw))
        mm = (hold_metrics if kind == "恒载" else vary_metrics)(
            Y, c, d, tu, Xu, dtm, tot_s) if kind != "恒载" else hold_metrics(
            Y, c, tu, Xu, dtm, tot_s)
        fo = first_onset(tu, Xu.sum(axis=1), dtm)
        ts_ = np.nan
        if fo:
            i0 = fo[0]
            tgt = float(np.median(Xu.sum(axis=1)[i0 + int(4.6 / dtm):i0 + int(5.4 / dtm)]))
            ts_ = stable_time(tu, Y.sum(axis=1), i0, tgt - fo[1], dtm)
        mm.update(dataset=tag, kind=kind, label=label, span=span if span else np.nan,
                  t_stable=ts_)
        rows.append(mm)
    print("  完成")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "v6_detwin_sweep.csv"), index=False, encoding="utf-8-sig")

names = ["v5-3s（参照）"] + [c[0] for c in CFG]
spans = [np.nan] + [c[1]["DET_FAST"] + c[1]["DET_GAP"] + c[1]["DET_LAG"] for c in CFG]


def agg(label):
    s = df[df.label == label]
    h, v = s[s.kind == "恒载"], s[s.kind == "实采"]
    return dict(span=spans[names.index(label)],
                恒载全段时漂=abs(h.drift_main).mean(), 恒载慢相段时漂=abs(h.drift_slow).mean(),
                T_stable中位=h.t_stable.median(), T_stable达标组数=int((h.t_stable <= 1.0).sum()),
                epoch恒载=int(h.epoch.sum()),
                实录全程偏差中位=v.max_gap.median(), 实录变载窗偏差中位=v.gap_med.median(),
                捕获比中位=v.cap_med.mean(), 捕获比最小=v.cap_min.min(),
                epoch实录=int(v.epoch.sum()))


tab = pd.DataFrame([agg(n) for n in names], index=names)
print("\n" + "=" * 130)
print("检测窗长度扫描汇总（只改检测窗，其余参数不动）")
print("=" * 130)
with pd.option_context("display.width", 200, "display.max_columns", 30):
    print(tab.round(2).to_string())
tab.to_csv(os.path.join(RES, "v6_detwin_sweep_summary.csv"), encoding="utf-8-sig")

# ── 权衡曲线 ──
fig, axs = plt.subplots(2, 2, figsize=(15, 9.5))
fig.suptitle("v6 检测窗长度扫描：判\"有没有台阶\"的窗拉长后，速度与稳健性的权衡", fontsize=14)
xs = [c[1]["DET_FAST"] + c[1]["DET_GAP"] + c[1]["DET_LAG"] for c in CFG]
t = tab.loc[[c[0] for c in CFG]]

ax = axs[0, 0]
b = ax.bar(range(len(xs)), t["T_stable中位"].values, 0.55, color="#2ca02c")
ax.bar_label(b, fmt="%.2f", fontsize=10)
ax.axhline(tab.loc["v5-3s（参照）", "T_stable中位"], color="0.45", ls="--", lw=1.3)
ax.text(len(xs) - 0.5, tab.loc["v5-3s（参照）", "T_stable中位"] + 0.08,
        f"v5-3s {tab.loc['v5-3s（参照）','T_stable中位']:.2f}s", fontsize=9, color="0.35", ha="right")
ax.axhline(1.0, color="#d62728", ls=":", lw=1.3)
ax.text(0, 1.05, "1 s 目标", fontsize=9, color="#d62728")
ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{s:.2f}s" for s in xs])
ax.set_xlabel("检测窗跨度（FAST+GAP+LAG）"); ax.set_ylabel("T_stable 中位 (s)")
ax.set_title("(1) 平稳时刻：越短越快（越小越好）"); ax.grid(axis="y", alpha=0.3)

ax = axs[0, 1]
b = ax.bar(np.arange(len(xs)) - 0.2, t["epoch恒载"].values, 0.36, color="#1f77b4", label="恒载 9 组")
ax.bar_label(b, fontsize=10)
b2 = ax.bar(np.arange(len(xs)) + 0.2, t["epoch实录"].values, 0.36, color="#ff7f0e", label="实录 4 份")
ax.bar_label(b2, fontsize=10)
ax.axhline(16, color="#1f77b4", ls="--", lw=1.2)
ax.axhline(29, color="#ff7f0e", ls="--", lw=1.2)
ax.text(len(xs) - 0.55, 16.6, "v5-3s 恒载 16", fontsize=8.5, color="#1f77b4", ha="right")
ax.text(len(xs) - 0.55, 29.6, "v5-3s 实录 29", fontsize=8.5, color="#ff7f0e", ha="right")
ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{s:.2f}s" for s in xs])
ax.set_xlabel("检测窗跨度"); ax.set_ylabel("epoch 数（越少越稳）")
ax.set_title("(2) 开 epoch 次数：越短越容易误判"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

ax = axs[1, 0]
b = ax.bar(range(len(xs)), t["实录全程偏差中位"].values, 0.55, color="#d62728")
ax.bar_label(b, fmt="%.0f", fontsize=10)
ax.axhline(tab.loc["v5-3s（参照）", "实录全程偏差中位"], color="0.45", ls="--", lw=1.3)
ax.text(len(xs) - 0.5, tab.loc["v5-3s（参照）", "实录全程偏差中位"] + 60,
        f"v5-3s {tab.loc['v5-3s（参照）','实录全程偏差中位']:.0f}", fontsize=9, color="0.35", ha="right")
ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{s:.2f}s" for s in xs])
ax.set_xlabel("检测窗跨度"); ax.set_ylabel("全程 max|显示−原始| 中位 (ADC)")
ax.set_title("(3) 实录跟踪偏差（越小越好）"); ax.grid(axis="y", alpha=0.3)

ax = axs[1, 1]
b = ax.bar(range(len(xs)), t["恒载慢相段时漂"].values, 0.55, color="#9467bd")
ax.bar_label(b, fmt="%.2f", fontsize=10)
ax.axhline(tab.loc["v5-3s（参照）", "恒载慢相段时漂"], color="0.45", ls="--", lw=1.3)
ax.text(len(xs) - 0.5, tab.loc["v5-3s（参照）", "恒载慢相段时漂"] + 0.03,
        f"v5-3s {tab.loc['v5-3s（参照）','恒载慢相段时漂']:.2f}%", fontsize=9, color="0.35", ha="right")
ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{s:.2f}s" for s in xs])
ax.set_xlabel("检测窗跨度"); ax.set_ylabel("慢相段时漂残余 (%)")
ax.set_title("(4) 恒载慢相段时漂（越小越好）"); ax.grid(axis="y", alpha=0.3)

fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.07, wspace=0.22, hspace=0.30)
f = os.path.join(FIG, "J1_v6_detwin_sweep.png")
fig.savefig(f, dpi=115)
plt.close(fig)
print(f"\n图已保存：{f}")
