# -*- coding: utf-8 -*-
"""T2 步骤2：基线逐帧一致性校验 + x1 与 e 的「形状」实测诊断。

(a) parity：t2_observer.observe(variant='base') 必须与 v34_observer_core3.observe3
    逐帧一致（证明变体骨架没跑偏）。
(b) 形状诊断：跑基线，记录 y/e/x1 的通道和轨迹，对首个加载事件做
      - x1 与「一阶模型 r1(1-e^{-t/τ})」的对比；
      - 实测 α(t) = x1/y 的演变；
      - e 的双指数拟合（快分量幅度/时间常数 + 慢分量幅度/时间常数）。
"""
import os
import sys

import numpy as np

import t2_lib as T
import t2_observer as O

sys.path.insert(0, T.SCRIPTS)
from v34_observer_core3 import observe3  # noqa: E402


def parity():
    ok = 0
    tot = 0
    worst = 0.0
    for label, d in T.all_sessions():
        if "archived" not in label and "反复" not in label and "同一" not in label:
            continue
        s = T.load_pre(d)
        D0, X0 = observe3(s["el"].copy(), s["V"])
        D0 = D0.sum(axis=1)
        X0 = X0.sum(axis=1)
        r = O.observe(s["el"].copy(), s["V"], "base")
        e1 = float(np.max(np.abs(r["D"] - D0)))
        e2 = float(np.max(np.abs(r["X1"] + r["X2"] - X0)))
        worst = max(worst, e1, e2)
        tot += 1
        good = e1 < 1.0 and e2 < 1.0
        ok += good
        print("  %-62s maxDiff 显示=%.3e x=%.3e %s"
              % (label[:62], e1, e2, "OK" if good else "**不一致**"))
    print("parity: %d/%d 会话一致（容差 1 ADC，仅浮点求和顺序差；量程 1e4～3e4 ADC），最大偏差 %.3e\n"
          % (ok, tot, worst))
    return ok == tot


def fit_double(t, s):
    """s(t) = A(1-e^{-t/τf}) + B(1-e^{-t/τs}) 的粗网格拟合（有界，τf<τs）。"""
    best = None
    for tf in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        for ts_ in (8.0, 15.0, 25.0, 40.0, 60.0, 90.0):
            if ts_ <= tf:
                continue
            F = np.column_stack([1.0 - np.exp(-t / tf), 1.0 - np.exp(-t / ts_)])
            coef, res, *_ = np.linalg.lstsq(F, s, rcond=None)
            if coef[0] < -1e-9 or coef[1] < -1e-9:
                continue
            r = float(np.sum((F @ coef - s) ** 2))
            if best is None or r < best[0]:
                best = (r, tf, ts_, float(coef[0]), float(coef[1]))
    if best is None:
        return None
    r, tf, ts_, A, B = best
    rms = float(np.sqrt(r / len(s)))
    return dict(tau_f=tf, tau_s=ts_, A=A, B=B, rms=rms,
                slow_frac=B / max(A + B, 1e-9))


def diagnose():
    print("=" * 108)
    print("x1 形状诊断（基线参数，通道和视角；事件=t_up 起 45s 窗）")
    print("%-52s %7s %7s %7s %7s %7s %7s %7s %7s"
          % ("会话", "台阶", "x1@45s", "α@5s", "α@20s", "α@45s",
             "τx1拟合", "e快τ", "e慢τ/占比"))
    for label, d in T.all_sessions():
        s = T.load_pre(d)
        el, V = s["el"], s["V"]
        tot = T.total(s)
        evs = T.load_events(el, tot)
        if not evs:
            continue
        r = O.observe(el.copy(), V, "base")
        ev = evs[0]
        i = ev["i_up"]
        # 以沿前 1s 均值为基线，看 y/e/x1 的相对增量
        b0 = max(0, i - 100)
        y0 = float(np.mean(r["Y"][b0:i]))
        e0 = float(np.mean(r["E"][b0:i]))
        x0 = float(np.mean(r["X1"][b0:i]))
        m = (el >= ev["t_up"]) & (el <= ev["t_up"] + 45.0)
        t = el[m] - ev["t_up"]
        y = r["Y"][m] - y0
        e = r["E"][m] - e0
        x = r["X1"][m] - x0
        if len(t) < 100:
            continue

        def at(sec, arr):
            k = int(np.argmin(np.abs(t - sec)))
            return float(arr[k])

        al5 = at(5.0, x) / max(at(5.0, y), 1.0)
        al20 = at(20.0, x) / max(at(20.0, y), 1.0)
        al45 = at(45.0, x) / max(at(45.0, y), 1.0)
        fx = fit_double(t, x)
        fe = fit_double(t, e)
        print("%-52s %7.0f %7.0f %7.3f %7.3f %7.3f %7.1f %7.1f %5.1f/%.0f%%"
              % (label[-52:], ev["step"], at(45.0, x), al5, al20, al45,
                 fx["tau_s"] if fx else float("nan"),
                 fe["tau_f"] if fe else float("nan"),
                 fe["tau_s"] if fe else float("nan"),
                 100 * fe["slow_frac"] if fe else float("nan")))
    print()


if __name__ == "__main__":
    ok = parity()
    if not ok:
        print("parity 失败，停止后续诊断")
        sys.exit(1)
    diagnose()
