# -*- coding: utf-8 -*-
"""02 因果算法评测。

所有算法输出均只使用当前及过去样本；本脚本只负责用完整记录离线评分
（评分窗口使用未来数据，但算法本身不接触未来数据）。

输出：
    DSv4/results/causal_algorithm_metrics.json
    DSv4/results/causal_algorithm_summary.csv
"""
import os
import sys
import json
import importlib.util

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, RES, load_dataset, segment_total
import algorithms_causal as alg


def load_calibration():
    path = RES / "causal_calibration.json"
    if not path.exists():
        spec = importlib.util.spec_from_file_location(
            "calib", os.path.join(os.path.dirname(__file__), "04_causal_calibration.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.calibrate()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def method_metrics(d, s0, s1, Y, raw_pre, raw_on, raw_response, main_ch,
                   loaded_idx):
    """逐通道指标；百分比以原始阶跃为分母。算法本身无需任何未来样本。"""
    fs = d["fs"]
    t = d["t"]
    pre_i = slice(max(0, s0 - int(3 * fs)), s0)
    on_i = slice(s0 + int(round(0.5 * fs)), s0 + int(round(2.0 * fs)))
    tail_i = slice(s1 - int(3 * fs), s1)
    post_i = slice(s1 + int(5 * fs), len(t))

    pre = np.median(Y[pre_i], axis=0)
    on = np.median(Y[on_i], axis=0)
    tail = np.median(Y[tail_i], axis=0)
    post = np.median(Y[post_i], axis=0)

    step = on - pre
    drift = tail - on
    post_resid = post - pre

    denom = np.where(np.abs(raw_response) > 0.05, raw_response, np.nan)
    drift_pct = drift / denom * 100.0
    abs_drift_pct = np.abs(drift_pct)
    post_pct = post_resid / denom * 100.0
    step_retention = step / denom * 100.0

    load_ref = pre + step
    rmse = np.sqrt(np.mean((Y[s0:s1] - load_ref[None, :]) ** 2, axis=0))
    rmse_pct = rmse / denom * 100.0

    li = loaded_idx
    absd = abs_drift_pct[li]
    rm = rmse_pct[li]
    posta = np.abs(post_pct[li])
    w = raw_response[li] / np.sum(raw_response[li])
    weighted_abs_drift = float(np.average(absd, weights=w))
    weighted_rmse = float(np.average(rm, weights=w))

    return dict(
        pre=pre.tolist(), on=on.tolist(), tail=tail.tolist(), post=post.tolist(),
        step=step.tolist(), drift=drift.tolist(), post_resid=post_resid.tolist(),
        drift_pct=drift_pct.tolist(), abs_drift_pct=abs_drift_pct.tolist(),
        post_pct=post_pct.tolist(), step_retention=step_retention.tolist(),
        rmse=rmse.tolist(), rmse_pct=rmse_pct.tolist(),
        main=dict(
            ch="ch17",
            step=float(step[main_ch]),
            drift=float(drift[main_ch]),
            drift_pct=float(drift_pct[main_ch]),
            abs_drift_pct=float(abs_drift_pct[main_ch]),
            post_resid=float(post_resid[main_ch]),
            post_pct=float(post_pct[main_ch]),
            step_retention=float(step_retention[main_ch]),
            rmse=float(rmse[main_ch]),
            rmse_pct=float(rmse_pct[main_ch]),
            pre_abs=float(pre[main_ch]),
        ),
        loaded=dict(
            n=int(len(li)),
            abs_drift_pct_median=float(np.median(absd)),
            abs_drift_pct_weighted=weighted_abs_drift,
            abs_drift_pct_max=float(np.max(absd)),
            rmse_pct_median=float(np.median(rm)),
            rmse_pct_weighted=weighted_rmse,
            rmse_pct_max=float(np.max(rm)),
            post_abs_pct_median=float(np.median(posta)),
            post_abs_pct_max=float(np.max(posta)),
            step_retention_median=float(np.median(step_retention[li])),
        ),
    )


def evaluate():
    calibration = load_calibration()
    allres = {}
    rows = []
    for name in DATASETS:
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        fs = d["fs"]
        X = d["X"]
        raw_pre = np.median(X[max(0, s0 - int(3 * fs)):s0], axis=0)
        raw_on = np.median(X[s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs))],
                           axis=0)
        raw_response = raw_on - raw_pre
        main_ch = int(np.argmax(X[s0:s1].mean(axis=0)))
        loaded_idx = np.where(raw_response > 0.15)[0]
        params = calibration[name]

        outs = alg.apply_all_causal(d, params)
        allres[name] = {"meta": {
            "s0": int(s0), "s1": int(s1), "main_ch": int(main_ch),
            "raw_step_main": float(raw_response[main_ch]),
            "causal_s0": alg.detect_contact(d["total"], float(params["thr"])),
            "params": params,
        }}
        for method in alg.METHOD_ORDER:
            Y = outs[method]
            m = method_metrics(d, s0, s1, Y, raw_pre, raw_on, raw_response,
                               main_ch, loaded_idx)
            allres[name][method] = m
            main = m["main"]
            ld = m["loaded"]
            rows.append({
                "数据": name,
                "方法": method,
                "方法名": alg.METHOD_DISPLAY[method][0],
                "类型": alg.METHOD_DISPLAY[method][1],
                "阶跃_N": round(main["step"], 4),
                "阶跃保持率_pct": round(main["step_retention"], 1),
                "末段漂移_N": round(main["drift"], 4),
                "漂移率_pct": round(main["drift_pct"], 1),
                "绝对漂移率_pct": round(main["abs_drift_pct"], 1),
                "负载RMSE_N": round(main["rmse"], 4),
                "负载RMSE_pct": round(main["rmse_pct"], 1),
                "卸载后残差_N": round(main["post_resid"], 4),
                "卸载后残差_pct": round(main["post_pct"], 1),
                "前基线绝对偏移_N": round(main["pre_abs"], 4),
                "受载组绝对漂移中位_pct": round(ld["abs_drift_pct_median"], 1),
                "受载组RMSE中位_pct": round(ld["rmse_pct_median"], 1),
                "受载组卸载残差中位_pct": round(ld["post_abs_pct_median"], 1),
            })

    with open(RES / "causal_algorithm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(allres, f, ensure_ascii=False, indent=2, allow_nan=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "causal_algorithm_summary.csv", index=False,
              encoding="utf-8-sig")
    return allres, df


def print_table(df):
    keys = ["方法", "阶跃_N", "阶跃保持率_pct", "末段漂移_N", "绝对漂移率_pct",
            "负载RMSE_pct", "卸载后残差_N", "卸载后残差_pct",
            "受载组绝对漂移中位_pct", "受载组RMSE中位_pct"]
    for name in DATASETS:
        sub = df[df["数据"] == name][keys]
        print(f"\n===== {name} =====")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    allres, df = evaluate()
    print_table(df)
    print(f"\nsaved {RES / 'causal_algorithm_metrics.json'}")
    print(f"saved {RES / 'causal_algorithm_summary.csv'}")
