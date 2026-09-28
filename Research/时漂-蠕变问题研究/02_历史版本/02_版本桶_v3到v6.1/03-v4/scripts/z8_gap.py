# -*- coding: utf-8 -*-
"""变化负载「变载可辨识性」的最终指标。

前几版指标的问题：用「事件后 2~3s / 4~6s 的显示−原始偏移」推算阶跃比，
把「补偿在事件后重新长起来」也算成了阶跃丢失 —— 那其实是蠕变补偿的正常重建。

正确做法（直接回答"负载变化会不会被抹平"）：
  取原始总量 Ẋ(t)（1s 中值平滑）与显示总量 Ẏ(t)，
  事件后原始电平稳定在 L_raw，显示电平稳定在 L_disp：
    最坏欠报 worst_gap  = max_t |Ẏ(t) − Ẋ(t)| / |Δ原始跳变|   （变载窗内，相对本次跳变）
    欠报持续 dur       = 显示进入「自身稳定值 ±10%×本次跳变」所需时间（s）
  这两个量直接说明：变载是否看得见（worst_gap 小）、看得快不快（dur 小）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
from glm53_v3 import GLM53v3  # noqa: E402
from glm53_v7 import GLM53v7  # noqa: E402
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4

CASES = [("零负载-切换负载-零负载-再切换负载", "数据A"),
         ("零负载-中途切换负载-零负载-切换负载", "数据B")]


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def med_smooth(x, k):
    return pd.Series(x).rolling(k, center=True, min_periods=1).median().to_numpy()


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def detect_events(tot, dt, rel=0.15, absfrac=0.08):
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


ALGOS = [("v3", GLM53v3, {}),
         ("v4_fast5", GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
         ("v7", GLM53v7, {})]

rows = []
print("=" * 112)
print("变载可辨识性最终指标  ·  temp/v4.1flash/scripts/z8_gap.py")
print("=" * 112)
for name, tag in CASES:
    t, X = load_rec(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"))
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    tot = Xu.sum(axis=1)
    tot_s = med_smooth(tot, max(3, int(0.5 / dt)))
    ev = detect_events(tot, dt)
    Ys = {lab: run(tu, Xu, cls, **kw) for lab, cls, kw in ALGOS}
    print(f"\n=== {tag} ===")
    print(f"  {'事件':>8}{'原始跳变':>10}|" + "".join(f"{lab:>30}" for lab, _, _ in ALGOS))
    print(f"  {'':>8}{'':>10}|" + "".join(f"{'最坏欠报%':>11}{'持续时间s':>10}{'超调%':>9}" for _ in ALGOS))
    for e, dl in ev:
        xp = np.median(tot_s[max(0, e - int(2 / dt)):e])
        xq = np.median(tot_s[e + int(4 / dt):e + int(6 / dt)])
        jump = xq - xp
        if abs(jump) < 1e-6:
            continue
        line = f"  {tu[e]:8.2f}{jump:10.0f}|"
        for lab, _, _ in ALGOS:
            y = med_smooth(Ys[lab].sum(axis=1), max(3, int(0.5 / dt)))
            a, b = max(0, e - int(1 / dt)), min(len(tu), e + int(12 / dt))
            gap = (y[a:b] - tot_s[a:b])
            worst = np.max(np.abs(gap)) / abs(jump)
            k6 = min(len(tu) - 1, e + int(6 / dt))
            k8 = min(len(tu) - 1, e + int(8 / dt))
            dq = y[k6] - xq
            over = float(np.max((y[a:b] - tot_s[a:b]) * np.sign(jump) - dq * np.sign(jump))) / abs(jump)
            y_stable = np.median(y[k6:k8 + 1]) if k8 > k6 else y[k6]
            tol = 0.10 * abs(jump)
            dur = np.nan
            for k in range(e, min(len(tu), e + int(12 / dt))):
                if abs(y[k] - y_stable) <= tol:
                    dur = (k - e) * dt
                    break
            line += f"{100*worst:11.1f}{dur:10.2f}{100*over:9.1f}"
            rows.append(dict(dataset=tag, event_s=float(tu[e]), jump=float(jump), algo=lab,
                             worst_gap_pct=100 * worst, dur_s=dur, overshoot_pct=100 * over))
        print(line)

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "varying_identifiability.csv"), index=False, encoding="utf-8-sig")
agg = dfm.groupby("algo").agg(事件数=("event_s", "count"),
                              最坏欠报_pct=("worst_gap_pct", "mean"),
                              最坏欠报_最大pct=("worst_gap_pct", "max"),
                              持续时间_s=("dur_s", "mean"),
                              持续时间_最大s=("dur_s", "max"),
                              超调_pct=("overshoot_pct", "mean")).round(2)
print("\n" + "=" * 112)
print(agg.to_string())
print("\n【读法】最坏欠报 = 变载窗内 |显示−原始| 的峰值 / |本次跳变| —— 越小说明变载越不会被抹平；")
print("        持续时间 = 显示回到自身稳定值 ±10%×跳变 所需时间 —— 越小说明看得越快。")
print("saved: results/varying_identifiability.csv")
