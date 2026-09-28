# -*- coding: utf-8 -*-
"""04 因果算法超参数留一标定（leave-one-session-out）。

只使用训练组（另两段记录）选择：
    thr      接触门控阈值
    tau_exp  RLS 指数时间核
    tau_log  RLS 对数时间核
    tau_kal  Kalman 对数时间核
测试段完全不参与选参；在线算法运行时也不使用任何未来样本。

输出：
    DSv4/results/causal_calibration.json
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, RES, load_dataset, segment_total
import algorithms_causal as alg

TAU_EXP_GRID = [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0]
TAU_LOG_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]


def quick_main_metrics(d, s0, s1, Y, raw_pre, raw_on, raw_step, main_ch):
    fs = d["fs"]
    pre = np.median(Y[max(0, s0 - int(3 * fs)):s0, main_ch])
    on = np.median(Y[s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs)), main_ch])
    tail = np.median(Y[s1 - int(3 * fs):s1, main_ch])
    step = on - pre
    drift_abs = abs(tail - on) / max(raw_step, 1e-6) * 100.0
    rmse = np.sqrt(np.mean((Y[s0:s1, main_ch] - (pre + step)) ** 2))
    rmse_pct = rmse / max(raw_step, 1e-6) * 100.0
    step_ret = step / max(raw_step, 1e-6) * 100.0
    return drift_abs, rmse_pct, step_ret


def load_training_info(names):
    info = {}
    for name in names:
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        fs = d["fs"]
        X = d["X"]
        raw_pre = np.median(X[max(0, s0 - int(3 * fs)):s0], axis=0)
        raw_on = np.median(X[s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs))],
                           axis=0)
        raw_step = raw_on - raw_pre
        main_ch = int(np.argmax(X[s0:s1].mean(axis=0)))
        info[name] = dict(d=d, s0=s0, s1=s1, raw_pre=raw_pre, raw_on=raw_on,
                          raw_step=raw_step, main_ch=main_ch,
                          pre_total_max=float(np.max(d["total"][:s0])))
    return info


def calibrate():
    all_info = load_training_info(DATASETS)
    calibration = {}
    for test in DATASETS:
        train = [n for n in DATASETS if n != test]
        # 阈值：训练组最大空载总和的 1.5 倍，且不低于 3 N
        thr = max(3.0, 1.5 * max(all_info[n]["pre_total_max"] for n in train))

        # 指数 tau 网格
        exp_scores = {}
        for tau in TAU_EXP_GRID:
            vals = []
            for n in train:
                inf = all_info[n]
                Y = alg.correct_rls_drift(inf["d"], "exp", tau=tau, thr=thr)
                d_, r_ = quick_main_metrics(inf["d"], inf["s0"], inf["s1"], Y,
                                            inf["raw_pre"], inf["raw_on"],
                                            inf["raw_step"][inf["main_ch"]],
                                            inf["main_ch"])[:2]
                vals.append(d_ + 0.5 * r_)
            exp_scores[str(tau)] = float(np.mean(vals))
        tau_exp = min(exp_scores, key=exp_scores.get)

        # 对数 tau 网格
        log_scores = {}
        for tau in TAU_LOG_GRID:
            vals = []
            for n in train:
                inf = all_info[n]
                Y = alg.correct_rls_drift(inf["d"], "log", tau=tau, thr=thr)
                d_, r_ = quick_main_metrics(inf["d"], inf["s0"], inf["s1"], Y,
                                            inf["raw_pre"], inf["raw_on"],
                                            inf["raw_step"][inf["main_ch"]],
                                            inf["main_ch"])[:2]
                vals.append(d_ + 0.5 * r_)
            log_scores[str(tau)] = float(np.mean(vals))
        tau_log = min(log_scores, key=log_scores.get)

        calibration[test] = {
            "train": train,
            "thr": round(thr, 3),
            "tau_exp": float(tau_exp),
            "tau_log": float(tau_log),
            "tau_kalman": float(tau_log),
            "exp_scores": exp_scores,
            "log_scores": log_scores,
        }
        print(f"{test}: 训练={train} thr={thr:.2f}N tau_exp={tau_exp}s "
              f"tau_log={tau_log}s")
    with open(RES / "causal_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    return calibration


if __name__ == "__main__":
    calibrate()
