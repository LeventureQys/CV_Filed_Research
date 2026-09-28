# -*- coding: utf-8 -*-
"""a7 汇总：把 a2b/a3/a6 的关键数字整理成报告用的表 + 驻留时长-代价曲线。

同时给出「抖动幅度 vs 误触发率」曲线所需的分母（每档抖动下的事件次数）与失守电平。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import REC, load_uniform, make_perturb, TRACED_V6, run_traced, ROOT  # noqa: E402
from a6_fix2 import V6fix, VARIANTS, GT, perturb_on_mask, _plain                # noqa: E402
from a_common import make_traced                                                # noqa: E402

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
FIG = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "figures")


def sweep_dwell():
    """驻留时长 DWELL ∈ {0.10,0.15,0.25,0.40} 的代价收益：干净阶跃延迟 + 拍击/抖动下的误触发。"""
    rows = []
    for name in ["零负载-切换负载-零负载-再切换负载", "中途切换-最终测试目标"]:
        d = load_uniform(REC[name])
        tu, X = d["tu"], d["Xu"]
        i0 = int(round((21.5 if "零负载-" in name else 53.1) / 0.01))
        scenes = {"干净基线": X,
                  "拍击4000": add_tap_local(X, i0, 4000),
                  "抖动白噪2000": X + perturb_on_mask(X, 2000, "white",
                                                       np.random.default_rng(7), dict(f_hi=40.0))}
        for dw in [0.0, 0.10, 0.15, 0.25, 0.40]:
            T = make_traced(type(f"Vd{int(dw*100)}", (V6fix,), dict(F_DWELL=dw > 0, DWELL_S=dw)))
            base = run_traced(T, tu, X)
            ep0 = sorted(e[0] for e in base["epoch"])
            # 干净阶跃检测延迟：注入一次 +4000 restep，取首个新 epoch
            Xs = add_step_local(X, i0, 4000)
            rs = run_traced(T, tu, Xs)
            cand = [e[0] - tu[i0] for e in rs["epoch"] if 0 <= e[0] - tu[i0] <= 3]
            lat = (min(cand) * 1000) if cand else np.nan
            for scene, Xp in scenes.items():
                r = run_traced(T, tu, Xp)
                ep = sorted(e[0] for e in r["epoch"])
                extra = [e for e in ep if not any(abs(e - x) <= 0.8 for x in ep0)]
                miss = [e for e in ep0 if not any(abs(e - x) <= 0.8 for x in ep)]
                dev = np.abs(r["Y"].sum(axis=1) - base["Y"].sum(axis=1))
                rows.append(dict(rec=name, dwell=dw, scene=scene, n_epoch=len(ep),
                                 n_extra=len(extra), n_miss=len(miss),
                                 maxdev=round(float(dev.max()), 1), lat_ms=lat))
                print(f"  [{name}] dwell={dw:.2f} {scene:12s} ep={len(ep):3d} "
                      f"误{len(extra):2d} 漏{len(miss):2d} maxdev={dev.max():7.0f} "
                      f"阶跃延迟={lat if lat == lat else float('nan'):.0f}ms", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a7_dwell_sweep.csv"), index=False, encoding="utf-8-sig")
    print("\n-- 驻留时长 vs 代价（跨记录平均）--")
    print(df.groupby(["dwell", "scene"])[["n_extra", "n_miss", "maxdev"]].mean().round(1).to_string())
    print(df.groupby("dwell")["lat_ms"].mean().round(0).to_string())


def add_tap_local(X, i0, amp):
    from a_common import add_tap
    return add_tap(X, i0, amp, rise_ms=50, hold_ms=100)[0]


def add_step_local(X, i0, amp):
    from a_common import add_step
    return add_step(X, i0, amp, rise_ms=50)


def table_summary():
    """把 a2b 的关键行整理成报告表。"""
    df = pd.read_csv(os.path.join(RES, "a2b_jitter_ext.csv"))
    keep = df[df.kind.isin(["独立白噪", "带限0.3-5Hz", "同相白噪"])]
    piv = keep.pivot_table(index=["rec", "kind"], columns="amp",
                           values=["sig_med", "thr_med", "n_miss", "n_extra",
                                   "maxdev", "slow_dev"])
    piv.to_csv(os.path.join(RES, "a7_jitter_table.csv"), encoding="utf-8-sig")
    print(piv.round(1).to_string())


if __name__ == "__main__":
    sweep_dwell()
    table_summary()
