# -*- coding: utf-8 -*-
"""T1-B / 07b：旋钮生效性自检（防止"改了但没生效"的假消融）。

对同一事件（RT2 onset）分别跑 clean / +30%L 拍击 / +2%L 白噪，
逐个旋钮打印 epoch 数、拍击窗 epoch 数、T_stable、建事件延迟。
判据：**至少有一个旋钮在某个场景下产生可测差异**，否则说明旋钮未生效。

产物：results/_t1b_07b_knobcheck.log
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

import t1b_lib as L                                     # noqa: E402
import t1b_07_cost_benefit as C                         # noqa: E402
from t1_common import make_perturb, add_tap             # noqa: E402

KNOBS = ["base", "dwell0.10", "dwell0.15", "dwell0.25", "dwell0.40", "rel0.08",
         "rel0.12", "k7", "k8", "revoke0.80", "idle0.25", "capf1.0", "capf2.0",
         "rescue0.40"]


def main():
    logp = os.path.join(RES, "_t1b_07b_knobcheck.log")
    f = open(logp, "a", encoding="utf-8")

    def w(s):
        print(s)
        f.write(s + "\n")

    w("=== T1-B / 07b 旋钮生效性自检 ===")
    ev = L.event_table()
    e = ev[ev.ev == "RT2@12.32"].iloc[0]
    key, t_on, Lh = e["key"], float(e["t_on"]), float(e["L_hold"])
    d = L.get_grid(key)
    tu, Xu = L.window_of(d, dict(kind="hold", t_on=t_on, span=C.CROP_S))
    i_tap = int(round((t_on + 20) / 0.01))
    Xs, _ = add_tap(Xu, i_tap, 0.30 * Lh, rise_ms=50, hold_ms=100, fall_ms=50)
    Xn = Xu + make_perturb(Xu, 0.02 * Lh, "white", np.random.default_rng(0))
    Xn2 = Xu + make_perturb(Xu, 0.10 * Lh, "white", np.random.default_rng(1))
    scen = (("clean", Xu), ("tap30", Xs), ("noise2%", Xn), ("noise10%", Xn2))
    w(f"{'knob':11s} " + " ".join(f"{s[0]:>18s}" for s in scen))
    for kn in KNOBS:
        cells = []
        for tag, X in scen:
            r = C.run_knob(kn, tu, X)
            ep = np.array([x[0] for x in r["epoch"]], float)
            ts = L.eval_event(tu, r["Z"], t_on, float(e["J"]), float(e["t_next"]))
            near = int(np.sum(np.abs(ep - (t_on + 20)) <= 1.0)) if ep.size else 0
            cells.append(f"ep{len(ep):2d} tap{near} Ts{ts['t_stable_v1']:5.2f}")
        w(f"{kn:11s} " + " ".join(f"{c:>18s}" for c in cells))
    w("说明：'ep' = 全窗 epoch 数；'tap' = 拍击 ±1 s 内 epoch 数（误触发）；"
      "Ts = 总通道 T_stable（40 s 窗口径）。若某旋钮四列与 base 全同，则该旋钮在本事件上未生效。")
    f.close()


if __name__ == "__main__":
    main()
