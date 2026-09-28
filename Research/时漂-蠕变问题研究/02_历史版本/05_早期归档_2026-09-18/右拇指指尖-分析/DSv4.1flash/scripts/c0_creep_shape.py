# -*- coding: utf-8 -*-
"""步骤 C0（重做）：蠕变形态精查，先做因果平滑再拟合，只在负载段内。

模型：F(t) = F∞ · [w_f·(1-e^{-t/τf}) + (1-w_f)·c·t^p]
  - 快过程 τf ≈ 0.7s，权重 w_f ≈ 0.85（接触建立 + 弹性体快松弛）
  - 慢过程 幂律 c·t^p（黏弹性蠕变，长时间不饱和）
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, dump_json, DATASETS, RES
from scipy.optimize import least_squares

np.set_printoptions(precision=5, suppress=True)
lines = []


def P(s=""):
    print(s)
    lines.append(s)


def model(t, F, wf, tauf, c, p):
    fast = 1.0 - np.exp(-t / tauf)
    slow = np.power(np.maximum(t, 0.0), p)
    return F * (wf * fast + (1 - wf) * c * slow)


def causmooth(y, w):
    if w <= 1:
        return y.copy()
    cs = np.concatenate([[0.0], np.cumsum(y)])
    n = len(y)
    idx = np.arange(n)
    lo = np.maximum(0, idx - w + 1)
    return (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)


RES_ALL = {}
for name in DATASETS:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, _ = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1 / k
    X = D["X"]
    seg = detect_segments(X.sum(axis=1), t, fs=fs)
    a0, a1 = seg["pre"]; c0, d0 = seg["load"]
    Xn = X - X[a0:a1].mean(axis=0)[None, :]
    w = lambda s: int(round(s * fs))
    tau = t[c0:d0] - t[c0]
    resp = Xn[c0:d0].max(axis=0)
    act = np.where(resp > 0.15)[0]          # 只用高信噪比通道
    P("=" * 100)
    P(f"### {name}  负载段 {tau[-1]:.2f}s  fs={fs:.3f}Hz  高信噪比通道 {len(act)} 个")
    P(f"    {'通道':>6s} {'F∞[N]':>8s} {'w_f':>7s} {'τf[s]':>7s} {'c':>9s} {'p':>7s} "
      f"{'R²':>7s} {'h(1s)':>7s} {'h(5s)':>7s} {'h(110s)':>8s}")
    fits = {}
    for j in act:
        y = causmooth(Xn[c0:d0, j], w(0.5))
        F0 = y[w(1.0):w(3.0)].mean()

        def resid(th):
            return model(tau, *th) - y
        best = None
        for wf0 in (0.7, 0.85, 0.95):
            for tf0 in (0.3, 0.7, 1.5):
                for p0 in (0.2, 0.35, 0.6):
                    try:
                        r = least_squares(
                            resid, [F0 * 1.3, wf0, tf0, 0.4 / max(F0, 1e-6), p0],
                            bounds=([F0 * 0.8, 0.0, 0.05, 1e-4, 0.02],
                                    [F0 * 3.0, 1.0, 5.0, 1e3, 1.5]),
                            max_nfev=6000)
                    except Exception:
                        continue
                    if best is None or r.cost < best.cost:
                        best = r
        if best is None:
            continue
        F, wf, tauf, c, p = best.x
        yh = model(tau, *best.x)
        r2 = 1 - np.var(y - yh) / np.var(y)
        hs = model(tau, 1.0, wf, tauf, c, p)
        fits[int(j)] = dict(F=float(F), wf=float(wf), tauf=float(tauf), c=float(c),
                            p=float(p), r2=float(r2))
        P(f"    {D['ch_cols'][j]:>6s} {F:8.4f} {wf:7.3f} {tauf:7.3f} {c:9.5f} {p:7.3f} "
          f"{r2:7.4f} {hs[w(1)]:7.4f} {hs[w(5)]:7.4f} {hs[w(110)] if w(110) < len(hs) else hs[-1]:8.4f}")

    if fits:
        arr = {kk: np.array([v[kk] for v in fits.values()])
               for kk in ("F", "wf", "tauf", "c", "p", "r2")}
        P(f"    汇总({len(fits)}通道): w_f={np.median(arr['wf']):.3f} "
          f"τf={np.median(arr['tauf']):.3f}s  c={np.median(arr['c']):.5f} "
          f"p={np.median(arr['p']):.3f}")
        P(f"      w_f 范围 {arr['wf'].min():.3f}~{arr['wf'].max():.3f}  "
          f"τf 范围 {arr['tauf'].min():.2f}~{arr['tauf'].max():.2f}  "
          f"p 范围 {arr['p'].min():.2f}~{arr['p'].max():.2f}  "
          f"R² 中位={np.median(arr['r2']):.4f}")
        # 共享形态（按 F∞ 加权）
        wf_m = float(np.median(arr["wf"])); tf_m = float(np.median(arr["tauf"]))
        c_m = float(np.median(arr["c"])); p_m = float(np.median(arr["p"]))
        RES_ALL[name] = dict(fits=fits, fs=float(fs), t_load=float(tau[-1]),
                             shared=dict(wf=wf_m, tauf=tf_m, c=c_m, p=p_m),
                             taus=tau.tolist(),
                             n_act=int(len(act)),
                             drift_1_110=float(model(110, 1, wf_m, tf_m, c_m, p_m)
                                               / model(1.0, 1, wf_m, tf_m, c_m, p_m)))
    # 实测形态表（平滑后，高 SNR 通道加权）
    wts = resp[act] / resp[act].sum()
    ref = Xn[c0 + w(1.0):c0 + w(3.0), act].mean(axis=0)
    ok = np.abs(ref) > 1e-6
    hs = (Xn[c0:d0, act[ok]] / ref[ok][None, :]) @ (wts[ok] / wts[ok].sum())
    hs = causmooth(hs, w(0.5))
    P(f"    实测共享形态 h(τ)（平滑 0.5s，以 1~3s 为 1.0）:")
    row = []
    for tt in (0.2, 0.5, 1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 120):
        if tt * fs < len(hs) - 1:
            row.append(f"τ={tt:5.1f}s:{hs[int(tt*fs)]:.3f}")
    P("      " + "  ".join(row))

dump_json(RES_ALL, "C0_creep_shape.json")
with open(os.path.join(RES, "C0_creep_shape.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\n[ok] C0 完成")
