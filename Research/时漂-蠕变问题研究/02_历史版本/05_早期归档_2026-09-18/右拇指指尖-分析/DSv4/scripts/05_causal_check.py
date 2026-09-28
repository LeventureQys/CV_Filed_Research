# -*- coding: utf-8 -*-
"""05 因果性数值校验。

标准做法：在多个截断点 n_cut 处截断输入，只运行到 n_cut；
若输出与完整数据运行的前 n_cut 完全一致，则算法在该实现中没有使用
未来样本。raw 与主机提供的 host_kalman 文件为预存数据，不做该测试。
"""
import os
import sys
import json
import importlib.util

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, RES, load_dataset, segment_total
import algorithms_causal as alg


def truncate(d, n_cut):
    return {
        "t": d["t"][:n_cut],
        "X": d["X"][:n_cut],
        "Xk": d["Xk"][:n_cut],
        "total": d["total"][:n_cut],
        "fs": d["fs"],
        "name": d["name"],
    }


def load_calibration():
    with open(RES / "causal_calibration.json", encoding="utf-8") as f:
        return json.load(f)


def check():
    calib = load_calibration()
    methods = {
        "gate_baseline": lambda d, p: alg.correct_gate_baseline(d, thr=float(p["thr"])),
        "ema_hp": lambda d, p: alg.correct_ema_hp(d),
        "ref_common": lambda d, p: alg.correct_ref_common(d, thr=float(p["thr"])),
        "rls_linear": lambda d, p: alg.correct_rls_drift(d, "linear", thr=float(p["thr"])),
        "rls_exp": lambda d, p: alg.correct_rls_drift(d, "exp", tau=float(p["tau_exp"]), thr=float(p["thr"])),
        "rls_log": lambda d, p: alg.correct_rls_drift(d, "log", tau=float(p["tau_log"]), thr=float(p["thr"])),
        "kalman_log": lambda d, p: alg.correct_kalman_drift(d, "log", tau=float(p["tau_kalman"]), thr=float(p["thr"])),
        "pipeline": lambda d, p: alg.correct_pipeline(d, tau_log=float(p["tau_log"]), thr=float(p["thr"])),
    }
    report = {"note": "max_abs_diff 应等于 0（或仅浮点舍入级别）"}
    for name in DATASETS:
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        n = len(d["t"])
        cuts = sorted(set([max(1000, int(n * 0.25)), s0 + 50, s0 + 200,
                           int(n * 0.6), s0 + 5000, int(n * 0.85)]))
        cuts = [c for c in cuts if 1000 < c < n]
        p = calib[name]
        report[name] = {}
        for method, fn in methods.items():
            full = fn(d, p)
            worst = 0.0
            for nc in cuts:
                dt = truncate(d, nc)
                part = fn(dt, p)
                L = min(nc, len(full), len(part))
                diff = np.max(np.abs(full[:L] - part[:L])) if L else 0.0
                worst = max(worst, float(diff))
            report[name][method] = {"max_abs_diff": worst, "pass": worst < 1e-12}
            print(f"{name} {method:15s} max_abs_diff={worst:.3e} "
                  f"pass={worst < 1e-12}")
    with open(RES / "causality_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    check()
