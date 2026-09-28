# -*- coding: utf-8 -*-
"""四指指腹（device1/device2）参数搜索：目标 = 先保证"不过扣/不下漂"，再尽量多扣掉真实蠕变。

背景（上一轮实测）：这四份的**快分量只占载荷 2.2~3.3 %**，而 `r_fast` 默认 0.06 会站着扣掉
6 % 载荷 ⇒ 一开始就过扣 3~4 % 载荷（四份实测 err_min 全部为负，最差 −1 543 ADC），
再叠加 x2 的积分过扣。故本脚本以「归一化过扣深度」为硬约束，再比「蠕变扣除率」。

输出列：
  errmin%  后 60 % 显示相对弹性电平的最低点 / 载荷（负 = 过扣深度；用户不可接受的是它很负）
  errend%  末帧显示相对弹性电平 / 载荷（正 = 欠扣偏高）
  tail30   末 30 s 显示斜率（ADC/s，负 = 还在下漂）
  ded%     末帧扣除量 / 真实累计蠕变（100 % 为理想）
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv  # noqa: E402
from data_finger_params import SESSIONS, fit_total, fit_channels  # noqa: E402


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=["d1/6a679f(277s)", "d2/857759(192s)"])
    ap.add_argument("--fast", nargs="*", type=float, default=[0.0, 0.02, 0.04, 0.06])
    ap.add_argument("--slowmax", nargs="*", type=float, default=[0.02, 0.03, 0.05, 0.35])
    ap.add_argument("--tauc1", type=float, default=2.0)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    prepared = {}
    for key in args.sessions:
        d = read_any_session_csv(SESSIONS[key])
        t, V = d["t"], d["V"]
        tin = V.sum(axis=1)
        t0, (E, c1, tau1, c2, tau2) = fit_total(t, tin)
        _, Es = fit_channels(t, V, tau1, tau2)
        prepared[key] = dict(t=t, V=V, tin=tin, Esum=float(Es.sum()),
                             creep=float(tin[-1] - Es.sum()), tau1=tau1, tau2=tau2,
                             E=E, c1=c1, c2=c2)
        print(f"{key}: 载荷 ΣE={prepared[key]['Esum']:.0f}  真实蠕变（末帧−E）="
              f"{prepared[key]['creep']:.0f} = 载荷的 {prepared[key]['creep'] / prepared[key]['Esum'] * 100:.1f}%  "
              f"（拟合：快 {c1:.0f}/τ{tau1:.1f}s，慢 {c2:.0f}/τ{tau2:.1f}s）")
    print(f"\nτc1 = {args.tauc1}（固定）；行 = r_fast / r_slow_max")
    hdr = "  {:>12s} {:>10s}".format("r_slow_max", "r_fast")
    for key in args.sessions:
        hdr += f" | {key:>26s}"
    print(hdr + "   （errmin% / errend% / ded%）")

    for rsm in args.slowmax:
        for rf in args.fast:
            p = replace(LIVE, tau_c_fast_s=args.tauc1, r_fast=rf, r_slow_max=rsm)
            line = "  {:>12.3f} {:>10.2f}".format(rsm, rf)
            for key in args.sessions:
                q = prepared[key]
                t, V, tin, Es = q["t"], q["V"], q["tin"], q["Esum"]
                out = run(p, t, V)
                n = len(t)
                m60 = slice(int(n * 0.4), n)
                err = out - Es
                err_min = float(err[m60].min()) / Es * 100
                err_end = float(err[-1]) / Es * 100
                ded = float((tin - out)[-1]) / q["creep"] * 100
                line += f" | {err_min:+8.2f} {err_end:+7.2f} {ded:6.0f}  "
            print(line, flush=True)


if __name__ == "__main__":
    main()
