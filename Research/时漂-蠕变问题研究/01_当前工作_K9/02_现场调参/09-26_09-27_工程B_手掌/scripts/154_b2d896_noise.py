# -*- coding: utf-8 -*-
"""154_b2d896_noise：验证「小 cap 清不掉尾漂」的真因是否是"速率估计被传感器噪声淹没"。

做法：拿同一段输入，逐通道做 3 s 滑动平均去噪后重跑 cap 扫描。
若去噪后小 cap 也能跟住 → 噪声是主因（cap 的作用是"信噪比门限"而非"速率上限"）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm10" / "out"
E, WIN = 16502.0, 1.30


def evalx(out_tot, t, fps):
    y = s93.medfilt1s(out_tot, fps)
    m = t >= WIN
    yy, tt = y[m], t[m]
    end = float(yy[-1])
    tail = tt >= tt[-1] - 60.0
    i10 = int(np.searchsorted(tt, 10.0))
    return {"dev": end - E, "over": max(0.0, E - float(yy.min())),
            "drop": float(yy.max()) - end,
            "slope_tail": float(np.polyfit(tt[tail], yy[tail], 1)[0]),
            "d10": float(yy[-1] - yy[i10])}


def smooth(V, fps, win_s):
    k = max(int(round(win_s * fps)) | 1, 3)
    pad = np.pad(V, ((k // 2, k // 2), (0, 0)), mode="edge")
    c = np.cumsum(pad, axis=0)
    c = np.vstack([np.zeros((1, V.shape[1])), c])
    return (c[k:] - c[:-k]) / k


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "b2d896.npz")
    t, V = z["t"], z["V"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    # 尾端噪声测量：末 60 s 逐通道去线性趋势后的 std / slope 估计噪声
    m0 = int(np.searchsorted(t, t[-1] - 60.0))
    tail = V[m0:]
    coef = np.polyfit(t[m0:], tail, 1)          # (2, m)
    detr = tail - (coef[0][None, :] * t[m0:][:, None] + coef[1][None, :])
    print(f"尾端 60 s 逐通道噪声：std 中位={np.median(detr.std(0)):.2f} ADC、"
          f"p90={np.percentile(detr.std(0),90):.2f} ADC")
    # 算法内部的 slope 估计噪声 = (v − v_lp)/1s 的 std（尾端）
    tau = 1.0
    v_lp = V[0].astype(float).copy()
    est = np.zeros_like(V)
    last = float(t[0])
    for i in range(1, len(t)):
        dt = min(max(float(t[i]) - last, 0.0), 0.1); last = float(t[i])
        if dt <= 0:
            est[i] = est[i - 1]
            continue
        est[i] = (V[i] - v_lp) / tau
        v_lp = v_lp + (dt / tau) * (V[i] - v_lp)
    te = est[m0:]
    print(f"算法内部 slope 估计（尾端）逐通道：|slope| 中位={np.median(np.abs(te)):.3f} ADC/s、"
          f"p90={np.percentile(np.abs(te),90):.3f}、std(中位通道)={np.median(te.std(0)):.3f} ADC/s")
    print(f"真实尾端速率：总值 0.12 ADC/s ÷ 50 通道 = 0.0024 ADC/s·通道 "
          f"⇒ 噪声/真实 = {np.median(np.abs(te))/0.0024:.0f} 倍")

    for win in (0.0, 3.0):
        Vs = V if win == 0 else smooth(V, fps, win)
        print(f"\n===== {'原始输入' if win == 0 else f'去噪输入（逐通道 {win:g} s 滑动平均）'} =====")
        print(f"{'cap':>8s} | {'落点':>7s} {'过扣':>6s} {'下坠':>6s} {'尾端斜率':>9s} {'10s→末':>8s}")
        for cap in (0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012):
            p = obs.default_with({"slope_cap_frac": cap, "r_slow_max": 0.15})
            e = evalx(obs.run(t, Vs, p)["out_tot"], t, fps)
            print(f"{cap:8g} | {e['dev']:+7.0f} {e['over']:6.0f} {e['drop']:6.0f} "
                  f"{e['slope_tail']:+9.3f} {e['d10']:+8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
