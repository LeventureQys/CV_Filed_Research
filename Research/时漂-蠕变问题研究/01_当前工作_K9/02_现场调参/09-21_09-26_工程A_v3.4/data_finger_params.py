# -*- coding: utf-8 -*-
"""data\\四指指腹（device1 / device2，四份长时恒载，ADC 模式）参数评估与选型。

方法：逐份会话先把总读数拟合成「弹性电平 E + 两个蠕变分量」得到 τ1/τ2 与真实总蠕变，
再逐通道拟合出 E_k（各通道弹性电平），以此判定算法的：
    err_end   末帧显示 − ΣE（+ = 欠扣偏高，− = 过扣下漂）
    err_min   后 60 % 里显示相对 ΣE 的最低点（下漂最深；用户不可接受的是这一项很负）
    tail30    末 30 s 显示斜率（负 = 还在往下漂）
    ride_max  加载初期显示相对 ΣE 的最高点（欠扣峰值，可接受）
    ded/creep 末帧扣除量 / 真实累计蠕变（>100 % = 过扣）

用法：python data_finger_params.py [--sessions all|d1|d2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402

ROOT = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\四指指腹")
SESSIONS = {
    "d1/6a679f(277s)": ROOT / "device1/20260926_091946_single_device_6a679f/device_001_seg000.csv",
    "d1/5ce2c4(198s)": ROOT / "device1/20260926_092426_single_device_5ce2c4/device_001_seg000.csv",
    "d2/857759(192s)": ROOT / "device2/20260926_092950_single_device_857759/device_001_seg000.csv",
    "d2/8e127e(98s)": ROOT / "device2/20260926_093310_single_device_8e127e/device_001_seg000.csv",
}


def creep_model(t, E, c1, tau1, c2, tau2, t0, rise):
    x = np.maximum(t - t0, 0.0)
    ramp = np.minimum(x / rise, 1.0)
    rel = np.maximum(x - rise, 0.0)
    return E + (c1 * (1.0 - np.exp(-rel / tau1)) + c2 * (1.0 - np.exp(-rel / tau2))) * ramp


def fit_total(t, tin):
    """先粗略定位阶跃（总读数首次超过峰值的 50 %）。"""
    peak = float(tin.max())
    idx = int(np.argmax(tin > 0.5 * peak))
    t0 = max(float(t[idx - 1]), 0.0)
    m = t >= t0 + 0.6
    p0 = [peak * 0.9, peak * 0.03, 3.0, peak * 0.05, 20.0]
    bounds = ([0.0, 0.0, 0.3, 0.0, 1.0],
              [peak * 1.5, peak, 60.0, peak, 400.0])
    popt, _ = curve_fit(lambda tt, E, c1, tau1, c2, tau2: creep_model(
        tt, E, c1, tau1, c2, tau2, t0, 0.6), t[m], tin[m], p0=p0, bounds=bounds, maxfev=40000)
    return t0, popt


def fit_channels(t, V, tau1, tau2):
    peak = float(V.sum(axis=1).max())
    idx = int(np.argmax(V.sum(axis=1) > 0.5 * peak))
    t0 = max(float(t[idx - 1]), 0.0)
    m = t >= t0 + 0.6
    Es = np.zeros(V.shape[1])
    for k in range(V.shape[1]):
        f = lambda tt, E, c1, c2: creep_model(tt, E, c1, c2, tau1, tau2, t0, 0.6)
        try:
            popt, _ = curve_fit(f, t[m], V[m, k],
                                p0=[float(V[m][:, k].mean()) * 0.9, 20.0, 20.0],
                                bounds=([0.0, 0.0, 0.0], [peak, peak, peak]), maxfev=20000)
            Es[k] = popt[0]
        except Exception:
            Es[k] = float(V[m][:, k].mean() * 0.9)
    return t0, Es


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


ARMS = [
    ("现役默认 τc1=12 rslow=0.35", LIVE),
    ("τc1=2", replace(LIVE, tau_c_fast_s=2.0)),
    ("τc1=2 + rslow=0.08", replace(LIVE, tau_c_fast_s=2.0, r_slow_max=0.08)),
    ("τc1=2 + rslow=0.06", replace(LIVE, tau_c_fast_s=2.0, r_slow_max=0.06)),
    ("τc1=2 + rslow=0.04", replace(LIVE, tau_c_fast_s=2.0, r_slow_max=0.04)),
    ("τc1=2 + rslow=0.06 + cap=0.005",
     replace(LIVE, tau_c_fast_s=2.0, r_slow_max=0.06, slope_cap_frac=0.005)),
]


def metrics(t, out, Esum):
    n = len(t)
    err = out - Esum
    m60 = slice(int(n * 0.4), n)
    m30 = t >= t[-1] - 30.0
    m_early = t >= (t[0] + 5.0)
    return (float(err[-1]), float(err[m60].min()), float(err[m_early].max()),
            float(np.polyfit(t[m30], out[m30], 1)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="all")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    keys = list(SESSIONS)
    if args.sessions == "d1":
        keys = [k for k in keys if k.startswith("d1")]
    elif args.sessions == "d2":
        keys = [k for k in keys if k.startswith("d2")]

    res = {}
    for key in keys:
        d = read_any_session_csv(SESSIONS[key])
        t, V = d["t"], d["V"]
        tin = V.sum(axis=1)
        t0, (E, c1, tau1, c2, tau2) = fit_total(t, tin)
        _, Es = fit_channels(t, V, tau1, tau2)
        Esum = float(Es.sum())
        creep_end = float(tin[t >= t[-1]].sum() if False else tin[-1] - Esum)
        print(f"\n===== {key} =====", flush=True)
        print(f"  总拟合：E={E:.0f} 快分量={c1:.0f}(τ{tau1:.1f}s) 慢分量={c2:.0f}(τ{tau2:.1f}s) "
              f"总蠕变={c1 + c2:.0f}（载荷的 {(c1 + c2) / E * 100:.1f}%）")
        print(f"  逐通道 ΣE={Esum:.0f}（与总拟合差 {Esum - E:+.0f}）")
        print(f"  {'方案':30s} {'err_end':>8s} {'err_min':>8s} {'ride_max':>9s} {'tail30':>8s} "
              f"{'末扣除':>8s} {'/蠕变':>7s}")
        for tag, p in ARMS:
            t1 = time.time()
            out = run(p, t, V)
            err_end, err_min, ride, tail = metrics(t, out, Esum)
            ded = float((tin - out)[-1])
            print(f"  {tag:30s} {err_end:+8.0f} {err_min:+8.0f} {ride:+9.0f} {tail:+8.2f} "
                  f"{ded:8.0f} {ded / creep_end * 100:6.0f}%   [{time.time() - t1:.0f}s]", flush=True)
            res.setdefault(key, {})[tag] = {
                "err_end": err_end, "err_min": err_min, "ride_max": ride,
                "tail30": tail, "ded_end": ded, "creep_end": creep_end,
            }
    out_dir = HERE / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data4_params.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {out_dir / 'data4_params.json'}")


if __name__ == "__main__":
    main()
