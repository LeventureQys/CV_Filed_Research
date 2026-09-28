# -*- coding: utf-8 -*-
"""自检 + 变化负载验证：v7（正确实现的快相免责期）。

先跑自检（恒载 1 组）：确认 v7 真的在补偿（A 已捕获、creep 不为 0），
再跑变化负载的阶跃保真对比。
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
from glm53_v3 import GLM53v3  # noqa: E402
from glm53_v7 import GLM53v7  # noqa: E402
import importlib
GLM53v4 = importlib.import_module("r_fastphase").GLM53v4
_z2 = importlib.import_module("z2_v5")
GLM53v5 = _z2.GLM53v5


def load_rec(p):
    df = pd.read_csv(p, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def grid(t, X):
    span = t[-1] - t[0]
    dt = span / (len(t) - 1)
    tu = np.arange(0.0, span, dt)
    Xu = np.vstack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])]).T
    return tu, Xu, dt


def run(tu, Xu, cls, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


print("=" * 100)
print("[自检] v7 是否真的在补偿？（恒载 右拇指/数据1）")
print("=" * 100)
t, X = load_rec(os.path.join(TEMP, "右拇指指尖", "数据1", "device_001_seg000.csv"))
tu, Xu, dt = grid(t, X)
for lab, cls, kw in [("v3", GLM53v3, {}), ("v4_fast5", GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
                     ("v7", GLM53v7, {})]:
    Y, c = run(tu, Xu, cls, **kw)
    comp = (Xu - Y).sum(axis=1)
    print(f"  {lab:<10} 末端补偿总量={comp[-1]:+9.2f}  峰值补偿={comp.max():+9.2f}  "
          f"A.max={float(c.A.max()):8.4f}  loaded={int(c.loaded.sum())}/{c.n}  "
          f"g末端={c.g:.4f}")

print()
print("=" * 100)
print("[主检验] 变化负载：阶跃保真 + 变载后蠕变恢复")
print("=" * 100)


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
        sm = pd.Series(tot[a:b]).rolling(max(3, int(0.15 / dt)), min_periods=1).median().to_numpy()
        k = int(np.argmax(np.abs(np.diff(sm))))
        ref.append((a + k, dl))
    return ref


CASES = [("零负载-切换负载-零负载-再切换负载", "数据A"),
         ("零负载-中途切换负载-零负载-切换负载", "数据B")]
ALGOS = [("v3", GLM53v3, {}), ("v4_fast5", GLM53v4, dict(FAST_S=5.0, A_W0_V4=3.5, A_W1_V4=5.0)),
         ("v7", GLM53v7, {})]
rows = []
for name, tag in CASES:
    t, X = load_rec(os.path.join(TEMP, "变化负载", name, "device_001_seg000.csv"))
    tu, Xu, dt = grid(t, X)
    tot = Xu.sum(axis=1)
    ev = detect_events(tot, dt)
    Ys = {}
    for lab, cls, kw in ALGOS:
        Ys[lab], _ = run(tu, Xu, cls, **kw)
    print(f"\n=== {tag} ===")
    print(f"  {'事件':>8}{'原始跳变':>10}|" + "".join(f"{lab:>24}" for lab, _, _ in ALGOS))
    print(f"  {'':>8}{'':>10}|" + "".join(f"{'增益':>8}{'延迟':>8}{'超额':>8}" for _ in ALGOS))
    for e, dl in ev:
        xj = np.median(tot[e:e + int(2 / dt)]) - np.median(tot[max(0, e - int(2 / dt)):e])
        line = f"  {tu[e]:8.2f}{xj:10.0f}|"
        for lab, _, _ in ALGOS:
            Y = Ys[lab]
            # 增益：用「事件后显示相对原始的稳定电平差」推算显示跳变
            pre_off = np.median(Y[max(0, e - int(2 / dt)):e - int(0.3 / dt)].sum(axis=1)
                                - tot[max(0, e - int(2 / dt)):e - int(0.3 / dt)])
            post_off = np.median(Y[e + int(4 / dt):e + int(6 / dt)].sum(axis=1)
                                 - tot[e + int(4 / dt):e + int(6 / dt)])
            ratio = (xj + post_off - pre_off) / xj if abs(xj) > 1e-9 else np.nan
            lag = np.nan
            if abs(post_off - pre_off) > 1e-9:
                for k in range(e - int(0.5 / dt), min(len(tu), e + int(10 / dt))):
                    if (Y[k].sum() - tot[k] - pre_off) / (post_off - pre_off) >= 0.9:
                        lag = (k - e) * dt
                        break
            a, b = max(0, e - int(1 / dt)), min(len(tu) - 1, e + int(8 / dt))
            exc = float(np.abs(np.diff(Y.sum(axis=1))[a:b] - np.diff(tot)[a:b]).max())
            line += f"{ratio:8.3f}{lag:8.2f}{exc:8.0f}"
            rows.append(dict(dataset=tag, event_s=float(tu[e]), raw_jump=float(xj), algo=lab,
                             step_ratio=ratio, lag_s=lag, excess=exc,
                             level_err=float(post_off)))
        print(line)

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(RES, "varying_steps_v7.csv"), index=False, encoding="utf-8-sig")
print("\n" + "=" * 100)
agg = dfm.groupby("algo").agg(事件数=("event_s", "count"), 阶跃比=("step_ratio", "mean"),
                              阶跃比_最差=("step_ratio", lambda s: (s - 1).abs().max() + 1),
                              延迟_s=("lag_s", "mean"), 跳变超额_整阵=("excess", "mean"),
                              跳变超额_最大=("excess", "max"),
                              事件后电平误差=("level_err", "mean")).round(3)
print(agg.to_string())
print("\n【读法】阶跃比越接近 1 越好（1.0 = 负载变化完全保留）；延迟越小越好；超额是整阵单帧 ADC。")
print("saved: results/varying_steps_v7.csv")
