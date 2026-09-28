# -*- coding: utf-8 -*-
"""slow_confirm_s 4→2 复核（仅长期数据 4 段）：值不值得继续往前挪。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_v34_on_csv import read_session_csv  # noqa: E402
from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

P0 = Params()
P5 = replace(P0, r_fast=0.06, slow_confirm_s=4.0, soft_unfreeze_s=2.0,
             tau_r_slow_idle_s=2.0, tau_r_fast_s=0.5)
P6 = replace(P5, slow_confirm_s=2.0)
CKPTS = (1.0, 3.0, 5.0, 15.0)
SESSIONS = sorted((HERE / "长期数据").glob("*/device_001_seg000.csv"))


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    out = np.empty(n); x2 = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x2[i] = c.x_slow.sum()
    return out, x2


def at(t, a, tq):
    return float(a[min(int(np.searchsorted(t, tq)), len(a) - 1)])


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"{'会话':14s} {'方案':10s} {'x2起效':>8s} " + " ".join(f"Δ{c_:>3.0f}s" for c_ in CKPTS)
          + f" {'显示最低':>9s} {'尾部std':>8s} {'x2末值':>7s} {'x2峰值':>7s}")
    for csv in SESSIONS:
        d = read_session_csv(csv)
        t, V = d["t"], d["values"]
        tin = V.sum(axis=1)
        step = tin.max() - np.percentile(tin, 5)
        base5 = float(np.percentile(tin, 5))
        i_on = int(np.argmax(tin > 0.2 * step + base5))
        i_pk = next((i for i in range(i_on, len(tin)) if tin[i] > 0.8 * step + base5), i_on)
        i_un = next((i for i in range(i_pk + 1, len(tin)) if tin[i] < 0.3 * step + base5),
                    len(tin) - 1)
        t_on, t_end = t[i_on], t[i_un]
        seg = (t >= t_on) & (t <= t_end)
        tail = (t >= t_on + 0.5 * (t_end - t_on)) & (t <= t_end)
        for tag, p in (("现役", P0), ("最终5参(sc=4)", P5), ("sc=2", P6)):
            out, x2 = run(p, t, V)
            m_plat = tail & (tin > 0.5 * step + base5)
            plat = float(np.mean(out[m_plat]))
            above = x2 > 0.005 * step
            tx2 = f"{t[int(np.argmax(above))]-t_on:+5.1f}s" if above.any() else " 未起效"
            ck = " ".join(f"{plat - at(t, out, min(t_on + c_, t_end)):+5.0f}" for c_ in CKPTS)
            print(f"{csv.parent.name[-6:]:14s} {tag:12s} {tx2:>8s} {ck} "
                  f"{out[seg].min():9.0f} {np.std(out[tail]):8.0f} "
                  f"{x2[-1]:7.0f} {x2.max():7.0f}")
        print()


if __name__ == "__main__":
    main()
