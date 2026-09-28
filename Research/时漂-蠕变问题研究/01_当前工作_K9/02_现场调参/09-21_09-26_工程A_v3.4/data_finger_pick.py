# -*- coding: utf-8 -*-
"""四指指腹：细化扫描 + 在全部四份会话上验证最终候选参数组。

约束（用户）：过扣（向下漂）不可接受；欠扣（向上漂一点）可接受。
选取准则：errmin% ≥ −1 %（后 60 % 显示不低于弹性电平 1 % 载荷）前提下，尽量提高 ded%（蠕变扣除率）。
"""

from __future__ import annotations

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


PREP = {}


def prep(key):
    if key not in PREP:
        d = read_any_session_csv(SESSIONS[key])
        t, V = d["t"], d["V"]
        tin = V.sum(axis=1)
        t0, (E, c1, tau1, c2, tau2) = fit_total(t, tin)
        _, Es = fit_channels(t, V, tau1, tau2)
        PREP[key] = dict(t=t, V=V, tin=tin, Es=float(Es.sum()), creep=float(tin[-1] - Es.sum()),
                         tau1=tau1, tau2=tau2, c1=c1, c2=c2, E=E)
    return PREP[key]


def evaluate(p, key):
    q = prep(key)
    t, V, tin, Es, creep = q["t"], q["V"], q["tin"], q["Es"], q["creep"]
    out = run(p, t, V)
    n = len(t)
    err = out - Es
    m60 = slice(int(n * 0.4), n)
    m30 = t >= t[-1] - 30.0
    return dict(
        errmin=float(err[m60].min()) / Es * 100,
        errend=float(err[-1]) / Es * 100,
        ride=float(err[t >= 5.0].max()) / Es * 100,
        tail=float(np.polyfit(t[m30], out[m30], 1)[0]),
        ded=float((tin - out)[-1]) / creep * 100,
    )


CANDS = [
    ("★A r_fast.02 rslow.03 τc1=2", dict(r_fast=0.02, r_slow_max=0.03, tau_c_fast_s=2.0)),
    ("B r_fast.02 rslow.04 τc1=2", dict(r_fast=0.02, r_slow_max=0.04, tau_c_fast_s=2.0)),
    ("C r_fast.02 rslow.05 τc1=2", dict(r_fast=0.02, r_slow_max=0.05, tau_c_fast_s=2.0)),
    ("D r_fast.03 rslow.04 τc1=2", dict(r_fast=0.03, r_slow_max=0.04, tau_c_fast_s=2.0)),
    ("E r_fast.02 rslow.04 τc1=2 conf5", dict(r_fast=0.02, r_slow_max=0.04, tau_c_fast_s=2.0,
                                             slow_confirm_s=5.0)),
    ("F r_fast.02 rslow.05 τc1=6", dict(r_fast=0.02, r_slow_max=0.05, tau_c_fast_s=6.0)),
    ("现役默认", {}),
]


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    keys = list(SESSIONS)
    print("四份会话指纹：")
    for k in keys:
        q = prep(k)
        print(f"  {k:18s} 载荷 ΣE={q['Es']:7.0f}  真实蠕变 {q['creep']:6.0f} = 载荷 "
              f"{q['creep'] / q['Es'] * 100:4.1f}%  快 {q['c1']:5.0f}({q['c1'] / q['Es'] * 100:4.1f}%,"
              f"τ{q['tau1']:4.1f}s)  慢 {q['c2']:5.0f}({q['c2'] / q['Es'] * 100:4.1f}%,τ{q['tau2']:5.1f}s)")

    hdr = f"\n{'候选':32s}" + "".join(f"| {k[:16]:>16s} " for k in keys)
    print(hdr)
    print(f"{'':32s}" + "".join(f"| {'min% end% ded%':>16s} " for k in keys))
    for tag, kw in CANDS:
        p = replace(LIVE, **kw)
        line = f"{tag:32s}"
        for k in keys:
            r = evaluate(p, k)
            line += f"| {r['errmin']:+6.2f} {r['errend']:+5.1f} {r['ded']:4.0f} "
        print(line, flush=True)

    # 逐份详细指标（最优候选 A）
    print("\n候选 A 逐份明细（tail = 末 30 s 显示斜率 ADC/s，负 = 仍在下漂；ride = 欠扣峰值 %）：")
    p = replace(LIVE, **CANDS[0][1])
    for k in keys:
        r = evaluate(p, k)
        print(f"  {k:18s} errmin {r['errmin']:+6.2f}%  errend {r['errend']:+6.2f}%  "
              f"ride {r['ride']:+6.2f}%  tail {r['tail']:+6.2f}  ded {r['ded']:4.0f}%")


if __name__ == "__main__":
    main()
