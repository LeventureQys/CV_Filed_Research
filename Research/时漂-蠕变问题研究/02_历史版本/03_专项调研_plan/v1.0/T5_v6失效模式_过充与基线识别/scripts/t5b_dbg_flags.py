# -*- coding: utf-8 -*-
"""T5-B 临时诊断：低幅值下到底哪个 M 分量在触发（不属于交付脚本，仅留痕）。"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from t5b_core import (WIN, load_window, load_gt, gt_in_window, run_full,  # noqa: E402
                      perturb_window, TraceV6)
from t5b_02_sweep import eval_trial2                                      # noqa: E402


def main():
    gt = load_gt()
    for wname in ("W1_1w", "W3_1p5w"):
        W = WIN[wname]
        tu, Xu, W = load_window(W)
        gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
        ref = run_full(tu, Xu, TraceV6)
        lvl = float(np.median(Xu.sum(axis=1)))
        print("==", wname, "lvl", round(lvl), "clean epochs",
              [(round(e[0], 2), e[1]) for e in ref["epoch"]])
        print("GT:", [(round(float(r.t_on), 2), r.kind, round(float(r.jump)))
                      for r in gt_w.itertuples()])
        for amp in (0.0, 200.0, 500.0):
            Xp = Xu if amp == 0 else perturb_window(tu, Xu, "white", amp, 7)[0]
            res = eval_trial2(tu, Xp, ref, gt_w, W["t0"])
            print("  amp=%s fail=%s flags=%s n_ep=%s excess_rel=%.3f dA_end=%.0f" % (
                amp, res["fail"], res["flags"], res["n_epoch"],
                res["excess_rel"], res["dA_end"]))
            print("   epochs:", [(round(e[0], 2), e[1]) for e in res["r"]["epoch"]])
            for d in res["det_rows"]:
                print("   ", d)


if __name__ == "__main__":
    main()
