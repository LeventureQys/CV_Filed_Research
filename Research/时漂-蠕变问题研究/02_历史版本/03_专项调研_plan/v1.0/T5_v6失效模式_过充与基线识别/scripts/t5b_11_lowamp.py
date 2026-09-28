# -*- coding: utf-8 -*-
"""T5-B / 11：低幅值补测（把 50% 失效边界钉到实测，而不是外推）。

主扫描的最低幅值是 200 ADC，而 W1/W2/W3/W4 在 200 ADC 就已经饱和（p_fail=1.0），
所以 logistic A50 只能外推。本脚本补测 20 / 50 / 100 / 150 ADC（白噪 + 同相共模），
6 个种子/档，覆盖 W1（1.2w 平台）与 W3/W4（1.5w/1.8w 平台），并把结果**追加**进
`results/t5b_trials_raw.csv`（并在日志里注明是追加，不是覆盖）。

产物
    results/_t5b_11.log
    results/t5b_lowamp.csv（本补测的逐档汇总）
    results/t5b_trials_raw.csv（追加）
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_core import (TraceV6, load_window, run_full, gt_in_window, load_gt, WIN)  # noqa: E402
from t5b_02_sweep import eval_trial2, pct_level, wilson                            # noqa: E402
from t5b_core import perturb_window                                                # noqa: E402

WINS = ["W1_1w", "W3_1p5w", "W4_1p9w"]
AMPS = [20.0, 50.0, 100.0, 150.0]
SEEDS = list(range(6))


def main():
    t00 = time.time()
    gt = load_gt()
    out, allrows = [], []
    for wname in WINS:
        W = WIN[wname]
        tu, Xu, W = load_window(W)
        gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
        ref = run_full(tu, Xu, TraceV6)
        lvl = float(np.median(Xu.sum(axis=1)))
        print(f"### {wname} 电平={lvl:.0f} 干净 epoch={len(ref['epoch'])} 真值={len(gt_w)}",
              flush=True)
        for cls in ("white", "common"):
            for amp in AMPS:
                rows = []
                for sd in SEEDS:
                    Xp, _ = perturb_window(tu, Xu, cls, amp, sd * 1000 + 7)
                    res = eval_trial2(tu, Xp, ref, gt_w, W["t0"])
                    rows.append(dict(win=wname, rec=W["rec"], level=lvl, cls=cls, kind="noise",
                                     amp=amp, amp_pct=np.nan, dur_ms=np.nan, f_lo=np.nan,
                                     f_hi=np.nan, seed=sd, site=np.nan, fail=res["fail"],
                                     fail_det=res["fail_det"], fail_anchor=res["fail_anchor"],
                                     **res["flags"], n_epoch=res["n_epoch"],
                                     n_epoch_clean=res["n_epoch_clean"],
                                     n_revoke=res["n_revoke"], n_handoff=res["n_handoff"],
                                     excess_rel=round(res["excess_rel"], 4),
                                     maxdev=round(res["maxdev"], 1),
                                     dA_end=round(res["dA_end"], 1),
                                     lowconf_frac=round(res["n_lowconf_frac"], 4)))
                d = pd.DataFrame(rows)
                allrows.append(d)
                k, n = int(d["fail"].sum()), len(d)
                lo, hi = wilson(k, n)
                out.append(dict(win=wname, cls=cls, amp=amp, level=lvl, n=n, n_fail=k,
                                p_fail=round(k / n, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                                p_M1=round(float(d["M1_miss"].gt(0).mean()), 3),
                                p_M2=round(float(d["M2_wrong_anchor"].gt(0).mean()), 3),
                                p_M3=round(float(d["M3_false_capture"].gt(0).mean()), 3),
                                p_M4=round(float(d["M4_disp"].gt(0).mean()), 3),
                                excess_rel_med=round(float(d["excess_rel"].median()), 4),
                                amp_pct_level=round(amp / lvl, 5)))
                print(f"  {cls} {amp:.0f} ADC ({amp/lvl*100:.2f}% 电平): 失败 {k}/{n} "
                      f"M1={out[-1]['p_M1']} M2={out[-1]['p_M2']} M4={out[-1]['p_M4']} "
                      f"excess中位={out[-1]['excess_rel_med']:.3f}", flush=True)
    sm = pd.DataFrame(out)
    sm.to_csv(os.path.join(RES, "t5b_lowamp.csv"), index=False, encoding="utf-8-sig")
    pd.concat(allrows, ignore_index=True).to_csv(
        os.path.join(RES, "t5b_lowamp_trials.csv"), index=False, encoding="utf-8-sig")
    print("[说明] 低幅补测结果**单独**存 t5b_lowamp.csv；为免与主表 200 ADC 档重复计数，"
          "不并入 t5b_trials_raw.csv。边界拟合时把本表与主表同口径合并。")
    print(f"\n总耗时 {time.time()-t00:.0f}s")
    print(sm.to_string())


if __name__ == "__main__":
    main()
