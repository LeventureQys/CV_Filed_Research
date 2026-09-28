# -*- coding: utf-8 -*-
"""02 运行全部算法并计算横向对比指标。

输出：
    DSv4/results/algorithm_metrics.json      方法 x 数据 x 通道完整指标
    DSv4/results/algorithm_summary.csv       主通道/受载组汇总
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, RES, load_dataset, segment_total
from algorithms import apply_all_methods, METHOD_ORDER, METHOD_DISPLAY


def method_metrics(d, s0, s1, Y, raw_pre, raw_on, raw_response, main_ch, loaded_idx):
    """逐通道指标。所有百分比以对应通道的原始阶跃为分母。"""
    fs = d["fs"]
    t = d["t"]
    X = d["X"]
    pre_i = slice(max(0, s0 - int(3 * fs)), s0)
    on_i = slice(s0 + int(round(0.5 * fs)), s0 + int(round(2.0 * fs)))
    tail_i = slice(s1 - int(3 * fs), s1)
    post_i = slice(max(s1 + int(5 * fs), s1 + int(3 * fs)), len(t))  # 卸载恢复 5 s 后

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
    load_diff = Y[s0:s1] - load_ref[None, :]
    rmse = np.sqrt(np.mean(load_diff ** 2, axis=0))
    rmse_pct = rmse / denom * 100.0

    # 受载组汇总
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
    allres = {}
    rows = []
    for name in DATASETS:
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        fs = d["fs"]
        X = d["X"]
        pre_i = slice(max(0, s0 - int(3 * fs)), s0)
        on_i = slice(s0 + int(round(0.5 * fs)), s0 + int(round(2.0 * fs)))
        raw_pre = np.median(X[pre_i], axis=0)
        raw_on = np.median(X[on_i], axis=0)
        raw_response = raw_on - raw_pre
        main_ch = int(np.argmax(X[s0:s1].mean(axis=0)))
        loaded_idx = np.where(raw_response > 0.15)[0]

        outs = apply_all_methods(d, s0, s1)
        allres[name] = {"meta": {"s0": int(s0), "s1": int(s1),
                                 "main_ch": int(main_ch),
                                 "raw_step_main": float(raw_response[main_ch])}}
        for method in METHOD_ORDER:
            Y = outs[method]
            m = method_metrics(d, s0, s1, Y, raw_pre, raw_on, raw_response,
                               main_ch, loaded_idx)
            allres[name][method] = m
            main = m["main"]
            ld = m["loaded"]
            rows.append({
                "数据": name,
                "方法": method,
                "方法名": METHOD_DISPLAY[method][0],
                "类型": METHOD_DISPLAY[method][1],
                "主通道": "ch17",
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

    with open(RES / "algorithm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(allres, f, ensure_ascii=False, indent=2, allow_nan=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "algorithm_summary.csv", index=False, encoding="utf-8-sig")
    return allres, df


def print_table(df):
    import io
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
    print(f"\nsaved {RES / 'algorithm_metrics.json'}")
    print(f"saved {RES / 'algorithm_summary.csv'}")
