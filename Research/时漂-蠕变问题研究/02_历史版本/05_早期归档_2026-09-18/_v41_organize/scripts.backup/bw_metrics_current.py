# -*- coding: utf-8 -*-
"""**当前实现**（`glm53_v51.py`，即 src/domain/drift 的对照原型）的统一指标复算。

用途：给 `Document/06-抗蠕变漂移补偿算法说明.md` 的「验证结果」提供单一来源、单一口径的数字，
口径与该文档 §10/§12 完全一致（不再引用任何历史版本的结果）。

A. 恒载 9 组：时漂残余(全段/慢相段/受载中位)、噪声比、平坦度、阶跃保真、
   **事件跳变超额（不含卸载沿/加载沿）**、首扣时延、自检(A/loaded/g)
B. 4 份实采录制：台阶捕获比(中位/最小，仅 |台阶|≥2000 ADC 的有效负载内变载)、
   变载窗最大偏差(中位/最大)、全程最大偏差(绝对/占峰值)、慢相稳定窗时漂残余、epoch 数、自检

产出：results/metrics_current.csv（明细）、results/_metrics_current.log
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

FAST, AWIN, ARM = 3.0, 1.0, 3.0                          # 默认档（无责 3s）
STATIC = [(loc, f"数据{i}") for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
B = os.path.join(TEMP, "变化负载")
RECS = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]


def run(tu, Xu):
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
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = FAST, AWIN, ARM
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


# ── A. 恒载 9 组 ────────────────────────────────────────────────
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


print("=" * 112)
print("A. 恒载 9 组（当前实现 · 默认档 免责 3 s）")
print("=" * 112)
print(f"{'数据集':>16} {'全段%':>7} {'慢相段%':>8} {'受载中位%':>9} {'噪声比':>7} {'平坦度%':>8} "
      f"{'阶跃保真':>8} {'跳变超额':>8} {'首扣s':>6} {'A_max':>7} {'loaded':>6}")
print("-" * 112)
srows = []
for loc, name in STATIC:
    p = os.path.join(TEMP, loc, name, "device_001_seg000.csv")
    if not os.path.exists(p):
        continue
    d = L.prep(p)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    s0r, s1r = find_segment(d["tot"])[0]
    s0 = int(np.searchsorted(tu, tu[min(s0r, len(tu) - 1)]))
    s1 = int(np.searchsorted(tu, min(d["t"][s1r], d["span"])))
    amp_v = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp_v))
    amp = amp_v[m]
    loaded = amp_v > 0.10 * amp_v.max()
    base = float(np.median(tot_s[max(0, s0 - int(2 / dtm)):s0]))
    n_on = next((i for i in range(s0, min(s0 + int(5 / dtm), s1))
                 if tot_s[i] > base + 0.05 * (tot_s[s0:s1].max() - base)), s0)
    Y, c = run(tu, Xu)
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
    dY, dX = np.diff(Y[:, m]), np.diff(Xu[:, m])
    ex = np.abs(dY - dX)
    # 事件跳变超额：排除加载/卸载沿（原始单帧变化 > 20% 幅度）与段首 0.3 s
    keep = np.ones(len(ex), bool)
    keep[:s0 + int(0.3 / dtm)] = False
    keep[s1:] = False
    keep &= np.abs(dX) <= 0.20 * amp
    jump = float(ex[keep].max())
    ded = (Xu - Y).sum(axis=1)
    hit = np.where(ded[s0:s1] > 0.005 * abs(tot_s[s1] - base))[0]
    fd = float(hit[0] * dtm) if len(hit) else np.nan
    row = dict(scope="static", ds=f"{loc}/{name}", drift_main=100 * dr[m] / amp,
               drift_slow=100 * dr5[m] / amp, drift_loaded=100 * np.median(dr[loaded] / amp),
               noise_ratio=ny / nx, flat=100 * ny / amp, step_ratio=step, jump_excess=jump,
               ded_delay=fd, A_max=float(c.A.max()), loaded_n=int(c.loaded.sum()),
               g_end=float(c.g))
    srows.append(row)
    print(f"{loc + '/' + name:>16} {row['drift_main']:7.2f} {row['drift_slow']:8.2f} "
          f"{row['drift_loaded']:9.2f} {row['noise_ratio']:7.2f} {row['flat']:8.2f} "
          f"{row['step_ratio']:8.2f} {row['jump_excess']:8.3f} {row['ded_delay']:6.1f} "
          f"{row['A_max']:7.0f} {row['loaded_n']:6d}")
dfs = pd.DataFrame(srows)
print("-" * 112)
print(f"{'9 组均值(|·|)':>16} {dfs.drift_main.abs().mean():7.2f} {dfs.drift_slow.abs().mean():8.2f} "
      f"{dfs.drift_loaded.abs().mean():9.2f} {dfs.noise_ratio.mean():7.2f} {dfs.flat.mean():8.2f} "
      f"{dfs.step_ratio.mean():8.2f} {dfs.jump_excess.max():8.3f} {dfs.ded_delay.mean():6.1f}")
print("（跳变超额列打印的是 9 组最大值；原始时漂参考：全段 15.46% / 慢相段 11.74%）")

# ── B. 4 份实采录制 ─────────────────────────────────────────────
print("\n" + "=" * 112)
print("B. 4 份实采录制（当前实现 · 免责 3 s）")
print("=" * 112)
rrows, erows, wrows = [], [], []
for tag, path in RECS:
    if not os.path.exists(path):
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    peak = float(d["tot"].max())
    d["periods"] = L.find_periods(d["tot"], dtm)
    d["events"] = [e for e, _ in L.detect_events(d["tot"], dtm)]
    Y, c = run(tu, Xu)
    d["Ys"] = {"a": Y, "raw": Xu}
    y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    gap = np.abs(y - tot_s)
    ev = L.event_table(d, d["events"], {"a": Y}, [], gain_s=6.0, algos=["a"])
    ev["big"] = ev["jump"].abs() >= 2000.0
    ev["mid"] = False
    if len(ev):
        thr = 0.30 * ev["pre"].max()
        ev["mid"] = (ev["pre"] > thr) & (ev["post"] > thr)
    ml = ev[ev.mid & ev.big] if len(ev) else ev
    cap = (ml["gain_a"] / ml["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna() \
        if len(ml) else pd.Series(dtype=float)
    # 慢相稳定窗（窗内无真实变载）
    st = L.slow_windows(d, algos=["a"])
    st_w = []
    for w0, w1 in st[["window_s", "end_s"]].drop_duplicates().itertuples(index=False):
        i0, i1 = int(w0 / dtm), int(w1 / dtm)
        segt = tot_s[i0:i1]
        n = len(segt)
        step, lag = max(1, int(0.3 / dtm)), max(1, int(0.5 / dtm))
        ok = False
        if n > step + lag:
            dm = max(abs(segt[i - step:i].mean() - segt[i - step - lag:i - step].mean())
                     for i in range(step + lag, n))
            ok = dm < 0.03 * max(segt.mean(), 1e-9)
        if ok:
            lvl = float(np.median(segt))
            yl = y[i0:i1]
            nq = max(1, len(yl) // 5)
            st_w.append(dict(rec=tag, w0=float(w0), w1=float(w1), lvl=lvl,
                             raw=100 * (segt[-nq:].mean() - segt[:nq].mean()) / max(lvl, 1e-9),
                             algo=100 * (yl[-nq:].mean() - yl[:nq].mean()) / max(lvl, 1e-9)))
            wrows.append(st_w[-1])
    row = dict(scope="rec", ds=tag, max_abs_gap=float(gap.max()), pct=100 * float(gap.max()) / peak,
               events=int(len(ml)), cap_med=float(cap.median()) if len(cap) else np.nan,
               cap_min=float(cap.min()) if len(cap) else np.nan,
               gap_med=float(ml["gap_a"].median()) if len(ml) else np.nan,
               gap_max=float(ml["gap_a"].max()) if len(ml) else np.nan,
               epochs=len(c.epoch_t), A_max=float(c.A.max()), loaded_n=int(c.loaded.sum()),
               g_end=float(c.g), gamma_lo=float(c.gamma.min()), gamma_hi=float(c.gamma.max()))
    rrows.append(row)
    print(f"[{tag}] 全程最大偏差 {row['max_abs_gap']:7.0f} ADC（占峰值 {row['pct']:4.1f}%）｜"
          f"有效负载内变载 {row['events']:2d} 个｜捕获比中位/最小 "
          f"{row['cap_med']:.2f}/{row['cap_min']:.2f}｜变载窗偏差中位/最大 "
          f"{row['gap_med']:.0f}/{row['gap_max']:.0f}｜epoch {row['epochs']}｜"
          f"A_max {row['A_max']:.0f}｜loaded {row['loaded_n']}｜γ∈[{row['gamma_lo']:.2f},{row['gamma_hi']:.2f}]")
    print(f"        慢相稳定窗：原始 {np.mean([w['raw'] for w in st_w]):.2f}% → "
          f"本算法 {np.mean([w['algo'] for w in st_w]):.2f}%（{len(st_w)} 个窗）")
dfr = pd.DataFrame(rrows)
print("-" * 112)
print(f"4 份合计：全程最大偏差中位 {dfr.max_abs_gap.median():.0f} ADC（占峰值中位 {dfr.pct.median():.1f}%）｜"
      f"捕获比中位均值 {dfr.cap_med.mean():.2f}｜捕获比最小 {dfr.cap_min.min():.2f}｜"
      f"变载窗偏差中位 {dfr.gap_med.median():.0f} ADC")

pd.concat([dfs.assign(scope="static"), dfr], ignore_index=True).to_csv(
    os.path.join(RES, "metrics_current.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(wrows).to_csv(os.path.join(RES, "metrics_current_slowwin.csv"),
                           index=False, encoding="utf-8-sig")
print(f"\n慢相稳定窗合计：原始 {np.mean([w['raw'] for w in wrows]):.2f}% → "
      f"本算法 {np.mean([w['algo'] for w in wrows]):.2f}%（{len(wrows)} 个窗，占本窗电平 %）")
