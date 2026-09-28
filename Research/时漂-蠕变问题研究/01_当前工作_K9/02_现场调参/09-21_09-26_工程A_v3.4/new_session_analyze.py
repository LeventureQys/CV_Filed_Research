# -*- coding: utf-8 -*-
"""会话 20260924_155543_single_device_1201c1（ADC 模式 / 无快漂 / 慢漂为主）参数分析。

传感器指纹（把输入总 ADC 拟合成「弹性电平 E + 两个蠕变分量」）：
    v(t) = E + c1·(1 − e^(−(t−t0)/τ1)) + c2·(1 − e^(−(t−t0)/τ2))
拟合结果：E ≈ 29278 ADC，c1 ≈ 1412（τ1 ≈ 1.6 s，4.8% 载荷）、c2 ≈ 1856（τ2 ≈ 11.2 s，6.3%），
总蠕变 ≈ 2857 ADC = 载荷的 9.8%，18 s 处仍以 ≈ +37 ADC/s 在线。

评价口径（都以拟合出的 E 为基准，E = 该阶跃下"无蠕变时本该显示的读数"）：
    末值−E  ：末帧显示与 E 的差（0 = 恰好扣掉全部蠕变；>0 欠扣，<0 过扣）
    ride    ：2~18 s 内显示高于 E 的最大幅度（慢态跟不上 → 读数偏高）
    dip     ：8~18 s 内显示低于 E 的最大幅度（过扣深度）
    settle  ：显示首次进入 E±100 且此后不再离开的时刻（相对 t0=1.1 s，秒）
    |偏差|均 ：受载期显示与 E 的平均绝对偏差（越小越"贴"）
    sink    ：末 3 s 显示斜率（理想 ≈ 0；负 = 仍在缓慢下漂）

用法：python new_session_analyze.py [--csv 路径]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402

CSV_DEFAULT = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法"
                   r"\20260924_155543_single_device_1201c1\device_001_seg000.csv")
T0 = 1.10        # 阶跃起点（总 ADC 首次非零的上一帧）


def creep_model(t, E, c1, tau1, c2, tau2):
    x = np.maximum(t - T0, 0.0)
    return E + c1 * (1.0 - np.exp(-x / tau1)) + c2 * (1.0 - np.exp(-x / tau2))


def fit_elastic(t, tin, fit_from=1.6, fit_to=None):
    fit_to = fit_to if fit_to is not None else float(t[-1]) + 0.1
    m = (t >= fit_from) & (t <= fit_to)
    p0 = [29000.0, 1500.0, 3.0, 1200.0, 20.0]
    bounds = ([20000.0, 0.0, 0.3, 0.0, 1.0], [35000.0, 8000.0, 60.0, 8000.0, 400.0])
    popt, pcov = curve_fit(creep_model, t[m], tin[m], p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    resid = tin[m] - creep_model(t[m], *popt)
    return popt, perr, float(np.std(resid))


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def metrics(t, out, E):
    m_end = t >= t[-1] - 3.0
    sink = float(np.polyfit(t[m_end], out[m_end], 1)[0])
    ride = float(np.max(out[t >= 2.0] - E))
    dip = float(np.min(out[t >= 8.0] - E))
    integ = float(np.mean(np.abs(out[t >= 2.0] - E)))
    ok = np.abs(out - E) <= 100.0
    idx = np.nonzero(ok)[0]
    settle = float("nan")
    if idx.size:
        lb = np.nonzero(~ok[idx[0]:])[0]
        j = min(idx[0] + (lb[-1] + 1 if lb.size else 0), len(t) - 1)
        settle = float(t[j] - T0)
    return out[-1] - E, sink, ride, dip, settle, integ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    d = read_any_session_csv(Path(args.csv))
    t, V = d["t"], d["V"]
    tin = V.sum(axis=1)
    (E, c1, tau1, c2, tau2), perr, rstd = fit_elastic(t, tin)
    print(f"数据：{Path(args.csv).name}  {V.shape[0]} 帧 × {V.shape[1]} 通道  "
          f"{t[-1] - t[0]:.2f} s  显示模式={d['header'].get('显示模式')}")
    print(f"拟合 E = {E:.0f} ± {perr[0]:.0f} ADC（残差 std {rstd:.1f}）")
    print(f"  快分量 c1={c1:.0f}（{c1 / E * 100:.1f}% 载荷，τ1={tau1:.2f} s）")
    print(f"  慢分量 c2={c2:.0f}（{c2 / E * 100:.1f}% 载荷，τ2={tau2:.2f} s）")
    tot = creep_model(t[-1], E, c1, tau1, c2, tau2) - E
    print(f"  18 s 处蠕变累计 {tot:.0f} ADC（载荷的 {tot / E * 100:.1f}%），"
          f"仍在线速率 {c1 / tau1 * np.exp(-(t[-1] - T0) / tau1) + c2 / tau2 * np.exp(-(t[-1] - T0) / tau2):.1f} ADC/s")
    print(f"  总蠕变（t→∞）{(c1 + c2) / E * 100:.1f}% 载荷\n")

    print(f"{'方案':34s} {'末值−E':>9s} {'sink':>7s} {'ride':>8s} {'dip':>8s} {'settle':>7s} "
          f"{'|偏差|均':>9s}")

    def show(tag, p):
        out = run(p, t, V)
        a, b, c, e, f, g = metrics(t, out, E)
        print(f"{tag:34s} {a:+9.1f} {b:+7.2f} {c:+8.1f} {e:+8.1f} {f:7.1f} {g:9.1f}")

    show("现役默认（τc1=12）", LIVE)
    for tc in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        show(f"τc1={tc}", replace(LIVE, tau_c_fast_s=tc))
    show("τc1=2 + confirm=4", replace(LIVE, tau_c_fast_s=2.0, slow_confirm_s=4.0))
    show("τc1=2 + confirm=1", replace(LIVE, tau_c_fast_s=2.0, slow_confirm_s=1.0))
    show("τc1=2 + cap=0.02", replace(LIVE, tau_c_fast_s=2.0, slope_cap_frac=0.02))
    show("τc1=2 + τslope=2", replace(LIVE, tau_c_fast_s=2.0, tau_slope_s=2.0))
    show("τc1=2 + r_slow_max=0.10", replace(LIVE, tau_c_fast_s=2.0, r_slow_max=0.10))
    show("τc1=2 + r_fast=0.08", replace(LIVE, tau_c_fast_s=2.0, r_fast=0.08))


if __name__ == "__main__":
    main()
