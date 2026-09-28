# -*- coding: utf-8 -*-
"""v4 快相免责期算法在新实采「切换负载」数据上的单组运行与指标。

数据: temp/变化负载/切换负载-快相无责的测试/20260917_133923_single_device_ee20bc
算法: raw / glm53_v3(现役) / glm53_v4(快相免责 5s, Document/02-v4算法说明.md)
额外: glm53_v4r —— 把免责期同时挂到「负载内变载(restep)」的变体
      (Document/02 §3.3 与 04 §3.3 描述的写法)。原型 GLM53v4 的 _restep 不经过 _begin，
      故其免责期实际只在「空载→负载」时启动；两者在本数据上不同，故并排给出。

产出: results/new_switch_load_{events,metrics}.csv + results/new_switch_load.npz
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3                                        # noqa: E402
from ad_v4 import GLM53v4, GLM53v4r                                 # noqa: E402

REC_DIR = (r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
           r"\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc")
REC = os.path.join(REC_DIR, "device_001_seg000.csv")

FAST_S, AW0, AW1 = 5.0, 3.5, 5.0
ALGOS = [("raw", None, {}),
         ("v3", GLM53v3, {}),
         ("v4_fast5", GLM53v4, dict(FAST_S=FAST_S, A_W0_V4=AW0, A_W1_V4=AW1)),
         ("v4r_fast5", GLM53v4r, dict(FAST_S=FAST_S, A_W0_V4=AW0, A_W1_V4=AW1))]
LBL = {"raw": "原始(无补偿)", "v3": "GLM53 v3(现役)", "v4_fast5": "v4 免责5s(最新)",
       "v4r_fast5": "v4r 免责5s+变载也免责"}


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def med_smooth(x, k):
    return pd.Series(x).rolling(k, center=True, min_periods=1).median().to_numpy()


def make_traced(cls):
    """记录每次蠕变 epoch 起点（onset / restep），用于界定慢相窗。"""

    class Traced(cls):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    return Traced


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def find_periods(tot, dt, frac=0.05):
    thr = frac * tot.max()
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    return sorted([(a, b) for a, b in zip(s, e) if (b - a) * dt >= 3.0])


def detect_events(tot, dt, rel=0.15, absfrac=0.08):
    """沿用 temp/v4.1flash/scripts/z8_gap.py 的事件检测口径"""
    n = len(tot)
    pn = int(2.0 / dt)
    cand = []
    for i in range(pn, n - pn, max(1, int(0.1 / dt))):
        pre = np.median(tot[i - pn:i])
        post = np.median(tot[i:i + pn])
        if abs(post - pre) > max(rel * abs(pre), absfrac * tot.max()):
            cand.append((i, post - pre))
    ev = []
    for i, dl in cand:
        if ev and i - ev[-1][0] <= int(1.5 / dt):
            if abs(dl) > abs(ev[-1][1]):
                ev[-1] = (i, dl)
        else:
            ev.append((i, dl))
    ref = []
    for i, dl in ev:
        a, b = max(0, i - int(2 / dt)), min(n - 1, i + int(2 / dt))
        sm = med_smooth(tot[a:b], max(3, int(0.15 / dt)))
        k = int(np.argmax(np.abs(np.diff(sm))))
        ref.append((a + k, dl))
    return ref


# ============================== 载入 ==============================
t, X = load_rec(REC)
span = t[-1] - t[0]
dtm = span / (len(t) - 1)
tu = np.arange(0.0, span, dtm)
Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
tot = Xu.sum(axis=1)
tot_s = med_smooth(tot, max(3, int(0.5 / dtm)))
print("=" * 112)
print("新实采数据 · v4 快相免责期算法单组运行")
print(f"  录制: {os.path.basename(REC_DIR)}")
print(f"  帧数={len(t)}  时长={span:.1f}s  采样={1/dtm:.2f}Hz  通道={X.shape[1]}  "
      f"重复时间戳={int((np.diff(t) <= 0).sum())}")
print(f"  显示域: ADC (display_mode=adc / force_conversion_active=false)；录制列=补偿前原始显示值")
print(f"  v4 参数: FAST_S={FAST_S}s  幅度窗=[{AW0},{AW1}]s")
print("=" * 112)

periods = find_periods(tot, dtm)
events = detect_events(tot, dtm)
print(f"\n负载段 {len(periods)} 个 / 变载事件 {len(events)} 个:")
for a, b in periods:
    print(f"  负载段 [{tu[a]:7.2f}, {tu[b]:7.2f}]  时长 {tu[b]-tu[a]:6.2f}s  电平 {np.median(tot_s[a:b]):8.0f}")

# ============================== 运行算法 ==============================
print("\n运行算法 ...")
Ys, checks, traces = {}, {}, {}
for lab, cls, kw in ALGOS:
    if cls is None:
        Ys[lab] = Xu.copy()
        checks[lab] = traces[lab] = None
        continue
    Ys[lab], c = run(tu, Xu, make_traced(cls), **kw)
    checks[lab] = c
    ep = c.epoch_t
    traces[lab] = ep
    print(f"  [{lab}] 自检: A_max={c.A.max():.3f}(非零={int(c.A.max() > 1e-9)})  "
          f"loaded={int(c.loaded.sum())}  g_end={c.g:+.4f}  γ∈[{c.gamma.min():.3f},{c.gamma.max():.3f}]")
    print(f"        蠕变 epoch 起点 {len(ep)} 个: {[round(x, 2) for x in ep]}")

print("\nv3 与各变体的逐帧分歧区间（整阵总量口径, |Δ|>1e-6）:")
for lab in ("v4_fast5", "v4r_fast5"):
    err = np.abs(Ys[lab] - Ys["v3"]).max(axis=1)
    idx = np.where(err > 1e-6)[0]
    segs = []
    if len(idx):
        a = idx[0]
        for k in range(1, len(idx)):
            if idx[k] != idx[k - 1] + 1:
                segs.append((a, idx[k - 1]))
                a = idx[k]
        segs.append((a, idx[-1]))
    print(f"  {lab}: {len(segs)} 段  最大|Δ|={err.max():.1f} ADC  " +
          ", ".join(f"[{tu[a]:.2f},{tu[b]:.2f}]" for a, b in segs))

# ============================== 事件指标 ==============================
rows = []
for e, _ in events:
    xp = np.median(tot_s[max(0, e - int(2 / dtm)):e])
    xq = np.median(tot_s[min(len(tu) - 1, e + int(4 / dtm)):min(len(tu), e + int(6 / dtm))])
    jump = xq - xp
    if abs(jump) < 1e-6:
        continue
    kind = "加载" if xp < 0.5 * tot_s.max() and xq > 0.5 * tot_s.max() else \
           ("卸载" if xq < 0.5 * tot_s.max() and xp > 0.5 * tot_s.max() else "变载")
    for lab, _, _ in ALGOS:
        if lab == "raw":
            continue
        y = med_smooth(Ys[lab].sum(axis=1), max(3, int(0.5 / dtm)))
        a, b = max(0, e - int(1 / dtm)), min(len(tu), e + int(12 / dtm))
        gap = y[a:b] - tot_s[a:b]
        worst = float(np.max(np.abs(gap)) / abs(jump))
        k6, k8 = min(len(tu) - 1, e + int(6 / dtm)), min(len(tu) - 1, e + int(8 / dtm))
        y_stable = np.median(y[k6:k8 + 1]) if k8 > k6 else y[k6]
        tol = 0.10 * abs(jump)
        dur = np.nan
        for k in range(e, min(len(tu), e + int(12 / dtm))):
            if abs(y[k] - y_stable) <= tol:
                dur = (k - e) * dtm
                break
        rows.append(dict(event_s=float(tu[e]), kind=kind, jump=float(jump), algo=lab,
                         worst_gap_pct=100 * worst, dur_s=dur,
                         max_abs_gap_adc=float(np.max(np.abs(gap)))))
ev_df = pd.DataFrame(rows)
ev_df.to_csv(os.path.join(RES, "new_switch_load_events.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 112)
print("变载事件「最坏欠报 / 恢复时间」(最坏欠报 = 事件窗内 |显示−原始| 峰值 ÷ |本次跳变|)")
print("=" * 112)
heads = [lab for lab, _, _ in ALGOS if lab != "raw"]
print(f"  {'t(s)':>8}{'类型':>5}{'原始跳变':>10}" + "".join(f"{LBL[l]:>22}" for l in heads))
print(f"  {'':>8}{'':>5}{'':>10}" + "".join(f"{'欠报%':>12}{'恢复s':>10}" for _ in heads))
for e, _ in events:
    sub = ev_df[ev_df.event_s == float(tu[e])]
    if sub.empty:
        continue
    line = f"  {tu[e]:8.2f}{sub.iloc[0]['kind']:>5}{sub.iloc[0]['jump']:10.0f}"
    for l in heads:
        r = sub[sub.algo == l]
        line += f"{r.iloc[0]['worst_gap_pct']:12.1f}{r.iloc[0]['dur_s']:10.2f}"
    print(line)
agg = ev_df.groupby("algo").agg(事件数=("event_s", "count"),
                                欠报_均值pct=("worst_gap_pct", "mean"),
                                欠报_最大pct=("worst_gap_pct", "max"),
                                最大绝对偏差ADC=("max_abs_gap_adc", "max"),
                                恢复_均值s=("dur_s", "mean")).round(2)
print("\n" + agg.to_string())

# ============================== 稳定窗（慢相）时漂 ==============================
# 窗口 = [原始变载事件 + 8s 保护带, 下一个原始事件]：8s 覆盖「2.5s 阶跃确认滞后 + 5s 免责期/重建期」，
# 保证 v3 与 v4 都已在稳态，比较才公平。归一化用该窗自身的电平（切换负载下电平逐段不同）。
GUARD_S, MIN_WIN_S = 8.0, 6.0
cuts_all = sorted({round(tu[a], 3) for a, _ in periods} |
                  {round(tu[b], 3) for _, b in periods} |
                  {round(float(tu[e]), 3) for e, _ in events})
stab = []
for i in range(len(cuts_all) - 1):
    w0 = int(np.searchsorted(tu, cuts_all[i] + GUARD_S))
    w1 = int(np.searchsorted(tu, cuts_all[i + 1]))
    if (w1 - w0) * dtm >= MIN_WIN_S:
        stab.append((w0, w1))
print(f"\n慢相稳定窗（事件后 {GUARD_S:.0f}s 起、到下一事件止、≥{MIN_WIN_S:.0f}s）共 {len(stab)} 个：")
rows2 = []
for w0, w1 in stab:
    lvl = float(np.median(tot_s[w0:w1]))
    nL = w1 - w0
    for lab, _, _ in ALGOS:
        L = Ys[lab][w0:w1].sum(axis=1)
        sd = L[-max(1, nL // 5):].mean() - L[:max(1, nL // 5)].mean()
        rows2.append(dict(window_s=float(tu[w0]), end_s=float(tu[w1]), dur_s=(w1 - w0) * dtm,
                          algo=lab, level=lvl, drift_pct=100 * sd / lvl))
st_df = pd.DataFrame(rows2)
st_df.to_csv(os.path.join(RES, "new_switch_load_metrics.csv"), index=False, encoding="utf-8-sig")
piv = st_df.pivot_table(index=["window_s", "end_s", "dur_s"], columns="algo",
                        values="drift_pct").reindex(columns=[a for a, _, _ in ALGOS])
piv.columns = [LBL[c] for c in piv.columns]
print(piv.round(2).to_string())
print("\n慢相时漂残余 |%|:  均值 / 最大")
print(pd.DataFrame({"均值": st_df.groupby("algo")["drift_pct"].apply(lambda s: s.abs().mean()),
                    "最大": st_df.groupby("algo")["drift_pct"].apply(lambda s: s.abs().max())}
                   ).round(2).rename(index=LBL).to_string())

np.savez_compressed(os.path.join(RES, "new_switch_load.npz"),
                    tu=tu, Xu=Xu, periods=np.array(periods),
                    events=np.array([e for e, _ in events]),
                    **{f"Y_{k}": v for k, v in Ys.items()})
print("\nsaved: results/new_switch_load_events.csv / new_switch_load_metrics.csv / new_switch_load.npz")
