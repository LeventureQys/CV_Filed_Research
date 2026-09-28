# -*- coding: utf-8 -*-
"""快速调参：kalman2 的 τc、creep_field 的聚合方式"""
import os
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "右拇指指尖")  # temp/右拇指指尖
MAIN = 17


def seg(total):
    thr = 0.15 * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if loaded[0]:
        s = np.r_[0, s]
    if loaded[-1]:
        e = np.r_[e, len(loaded)]
    segs = sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)
    return segs[0]


def kalman2_c(X, dt, R, tau_f, tau_c):
    n, m = X.shape
    f = X[0].copy()
    c = np.zeros(m)
    Pf = Pc = R
    c_hist = np.empty((n, m))
    c_hist[0] = c
    for i in range(n):
        if i:
            d_ = dt[i]
            qf = 4 * R * d_ / tau_f
            qc = 4 * R * d_ / tau_c
            S = (Pf + qf) + (Pc + qc) + R
            Kf = (Pf + qf) / S
            Kc = (Pc + qc) / S
            inn = X[i] - (f + c)
            f = f + Kf * inn
            c = c + Kc * inn
            Pf = (Pf + qf) * (1 - Kf)
            Pc = (Pc + qc) * (1 - Kc)
        c_hist[i] = c
    return c_hist


for name in ["数据1", "数据2", "数据3"]:
    df = pd.read_csv(os.path.join(BASE, name, "device_001_seg000.csv"), skiprows=24)
    tc = [c for c in df.columns if c.startswith("ch")]
    X = df[tc].to_numpy()
    t = df["elapsed"].to_numpy()
    s0, s1 = seg(X.sum(1))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.1)
    dtm = np.median(np.diff(t))
    R = max(X[:s0].std(axis=0).mean() ** 2, 1e-6)
    # 蠕变场公共量
    total = X.sum(1)
    unloaded = total < 0.05 * total[s0:s1].max()
    b0 = np.zeros(X.shape[1])
    a_b = np.clip(dt / 2.0, 0, 1)
    baseline = np.empty(X.shape)
    for i in range(len(X)):
        if unloaded[i]:
            b0 = b0 + a_b[i] * (X[i] - b0)
        baseline[i] = b0
    A = X[s0 + int(1 / dtm): s0 + int(3 / dtm)].mean(axis=0) - baseline[s0]
    loaded_ch = A > 0.10 * A.max()
    print(f"\n=== {name} ===")
    # kalman2 扫参
    for tau_c in [20, 40, 80, 150, 300, 600]:
        for tau_f in [0.3]:
            c_hat = kalman2_c(X, dt, R, tau_f, tau_c)
            Y = X - c_hat
            L = Y[s0:s1]
            nL = len(L)
            drift = 100 * (L[-nL // 10:, MAIN].mean() - L[: nL // 10, MAIN].mean()) / A[MAIN]
            i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
            step = (Y[i1:i2, MAIN].mean() - baseline[s0, MAIN]) / (X[i1:i2, MAIN].mean() - baseline[s0, MAIN])
            print(f"  kalman2 tau_c={tau_c:3.0f}: drift={drift:+6.1f}%  step={step:.2f}")
    # creep_field 聚合方式
    for mode in ["median", "mean", "wmean"]:
        Y = X.copy()
        g_smooth = 0.0
        a_g = dtm / 3.0
        for i in range(s0, s1):
            rel = (X[i, loaded_ch] - baseline[s0, loaded_ch] - A[loaded_ch]) / A[loaded_ch]
            if mode == "median":
                g_raw = np.median(rel)
            elif mode == "mean":
                g_raw = rel.mean()
            else:
                w = A[loaded_ch]
                g_raw = np.sum(w * rel) / np.sum(w)
            g_smooth = g_smooth + a_g * (g_raw - g_smooth)
            creep = np.minimum(np.maximum(A * g_smooth, -0.5 * A), 1.5 * A)
            Y[i] = X[i] - creep
        L = Y[s0:s1]
        nL = len(L)
        drift_main = 100 * (L[-nL // 10:, MAIN].mean() - L[: nL // 10, MAIN].mean()) / A[MAIN]
        dr = L[-nL // 10:].mean(axis=0) - L[: nL // 10].mean(axis=0)
        drift_loaded = 100 * np.median(dr[loaded_ch] / A[loaded_ch])
        i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
        step = (Y[i1:i2, MAIN].mean() - baseline[s0, MAIN]) / (X[i1:i2, MAIN].mean() - baseline[s0, MAIN])
        print(f"  creep_field {mode}: drift_main={drift_main:+6.1f}%  drift_loaded={drift_loaded:+5.1f}%  step={step:.2f}")
