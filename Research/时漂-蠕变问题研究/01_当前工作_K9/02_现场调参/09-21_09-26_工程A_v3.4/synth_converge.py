# -*- coding: utf-8 -*-
"""这份传感器（x1 快分量小、x2 慢分量主导）在当前参数下到底怎么收敛：
把 18 s 录制按**逐通道双指数模型**外推到 300 s，再看扣除量/显示量的渐近行为。

方法：
  1) 对 52 个通道各自拟合 v_k(t) = E_k + c1_k(1−e^(−t/τ1)) + c2_k(1−e^(−t/τ2))（τ 全通道共享，幅度逐通道），
     阶跃用实测的 0.35 s 线性上升沿；
  2) 合成 300 s @100 Hz 的输入（整数化 + 叠加实测残差噪声，模拟真实 ADC 量化）；
  3) 用 creep_observer_k9 逐帧跑各参数组，报告：
       creep(t)  = 模型给出的真实累计蠕变
       ded(t)    = 算法扣除量（in − out）
       err(t)    = 显示 − E（正 = 欠扣/偏高，负 = 过扣/下漂）
       k30/末段  = 最后 30 s 的显示斜率（ADC/s）

用法：python synth_converge.py [--t-end 300] [--out out]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402

CSV = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法"
           r"\20260924_155543_single_device_1201c1\device_001_seg000.csv")
T0, RISE = 1.10, 0.35


def shape(t, tau):
    """0.35 s 线性上升沿 + 之后的一阶蠕变形状（幅值归一）。"""
    x = np.maximum(t - T0, 0.0)
    ramp = np.minimum(x / RISE, 1.0)
    rel = np.maximum(x - RISE, 0.0)
    return ramp * (1.0 - np.exp(-rel / tau))


def fit_channel(t, y, tau1, tau2):
    m = t >= T0 + RISE
    f = lambda tt, E, c1, c2: E + c1 * shape(tt, tau1) + c2 * shape(tt, tau2)
    E0 = float(np.mean(y[(t >= T0) & (t <= T0 + 0.2)]))
    try:
        popt, _ = curve_fit(f, t[m], y[m], p0=[E0, 50.0, 50.0],
                            bounds=([0.0, 0.0, 0.0], [2000.0, 2000.0, 2000.0]), maxfev=8000)
    except Exception:
        popt = np.array([E0, 0.0, 0.0])
    return popt


ARMS = [
    ("现役默认 τc1=12 cap=0.010", LIVE),
    ("τc1=2", replace(LIVE, tau_c_fast_s=2.0)),
    ("τc1=2 + cap=0.002", replace(LIVE, tau_c_fast_s=2.0, slope_cap_frac=0.002)),
    ("τc1=2 + cap=0.001", replace(LIVE, tau_c_fast_s=2.0, slope_cap_frac=0.001)),
    ("τc1=2 + r_slow_max=0.06", replace(LIVE, tau_c_fast_s=2.0, r_slow_max=0.06)),
    ("τc1=2 + r_slow_max=0.11", replace(LIVE, tau_c_fast_s=2.0, r_slow_max=0.11)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-end", type=float, default=300.0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    d = read_any_session_csv(CSV)
    t, V = d["t"], d["V"]
    n_ch = V.shape[1]

    # 1) 逐通道拟合（τ 用整段拟合出来的 1.63 / 11.21）
    TAU1, TAU2 = 1.63, 11.21
    Es, C1s, C2s = [], [], []
    for k in range(n_ch):
        E, c1, c2 = fit_channel(t, V[:, k], TAU1, TAU2)
        Es.append(E)
        C1s.append(c1)
        C2s.append(c2)
    Es, C1s, C2s = np.array(Es), np.array(C1s), np.array(C2s)
    print(f"逐通道拟合：ΣE={Es.sum():.0f}（载荷）  Σc1={C1s.sum():.0f}（快分量，"
          f"τ={TAU1}s）  Σc2={C2s.sum():.0f}（慢分量，τ={TAU2}s）  "
          f"总蠕变 Σ(c1+c2)={C1s.sum() + C2s.sum():.0f} ADC")

    # 2) 合成 300 s 输入：模型 + 实测残差噪声（重采样）
    dt = 0.01
    n = int(args.t_end / dt)
    tt = np.arange(n) * dt          # 0 起，阶跃前保留 1.1 s 空载基线（与实机一致）
    # 实测残差（模型 − 实测），按通道重采样，保证噪声/量化形态贴近真实
    res = V[(t >= T0 + 0.4)] - (Es + np.outer(shape(t[(t >= T0 + 0.4)], TAU1), C1s)
                                + np.outer(shape(t[(t >= T0 + 0.4)], TAU2), C2s))
    rng = np.random.default_rng(20260924)
    idx = rng.integers(0, len(res), size=n)
    Vsyn = np.empty((n, n_ch))
    sh1, sh2 = shape(tt, TAU1), shape(tt, TAU2)
    for k in range(n_ch):
        Vsyn[:, k] = Es[k] + C1s[k] * sh1 + C2s[k] * sh2 + res[idx, k]
    # 阶跃前保持基线（E_k 之前是 0/基线）
    pre = tt < T0
    Vsyn[pre, :] = np.maximum(Es * 0 + 0.0, 0.0)
    Vsyn = np.rint(np.maximum(Vsyn, 0.0))
    creep_t = (C1s[:, None] * sh1[None, :] + C2s[:, None] * sh2[None, :]).sum(axis=0)
    tin = Vsyn.sum(axis=1)
    print(f"合成输入：{n} 帧 {args.t_end:.0f} s，阶跃后载荷 {tin[-1]:.0f} ADC，"
          f"总蠕变 {creep_t[-1]:.0f} ADC")

    marks = [5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0, args.t_end]
    idx_m = [min(int(np.searchsorted(tt, x)), n - 1) for x in marks]
    print(f"\n{'方案':28s}" + "".join(f"{'t=' + format(x, '.0f') + 's':>10s}" for x in marks))

    results = {}
    for tag, p in ARMS:
        c = CreepObserverK9(p)
        c._trace_frame = lambda *a, **k: None
        out = np.empty(n)
        x1 = np.empty(n)
        x2 = np.empty(n)
        for i in range(n):
            out[i] = c.process(float(tt[i]), Vsyn[i]).sum()
            x1[i] = c.x_fast.sum()
            x2[i] = c.x_slow.sum()
        ded = tin - out
        err = out - Es.sum()          # 显示 − 弹性电平（正 = 偏高/欠扣）
        tail = slice(int(n - 3000), n)
        k30 = float(np.polyfit(tt[tail], out[tail], 1)[0])
        print(f"{tag:28s}" + "".join(f"{err[i]:+10.0f}" for i in idx_m)
              + f"   末段显示斜率 {k30:+.2f} ADC/s   x1末 {x1[-1]:.0f}  x2末 {x2[-1]:.0f}  "
                f"末扣除 {ded[-1]:.0f}/{creep_t[-1]:.0f}")
        results[tag] = {
            "err_at": {format(m, ".0f"): float(err[i]) for m, i in zip(marks, idx_m)},
            "k_tail": k30, "ded_end": float(ded[-1]), "creep_end": float(creep_t[-1]),
            "x1_end": float(x1[-1]), "x2_end": float(x2[-1]),
            "err_series": err[::100].tolist(), "t_series": tt[::100].tolist(),
        }

    out_dir = HERE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synth_converge.json").write_text(json.dumps({
        "csv": str(CSV), "tau1": TAU1, "tau2": TAU2,
        "E_total": float(Es.sum()), "c1_total": float(C1s.sum()), "c2_total": float(C2s.sum()),
        "t_end": args.t_end, "arms": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(out_dir / "synth_converge.npz", t=tt, tin=tin, creep=creep_t, E=float(Es.sum()))
    print(f"\n写出 {out_dir / 'synth_converge.json'}")


if __name__ == "__main__":
    main()
