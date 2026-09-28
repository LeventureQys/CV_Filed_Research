# -*- coding: utf-8 -*-
"""04 阵列共享时间核的跨组泛化检验与固定参数选择。

用数据 i 拟合的 tau_L 去校正数据 j（i≠j），检验共享核是否可迁移；
并扫描固定 tau_L 在三组数据上的平均表现，给出推荐保守值。
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, RES, FIG, load_dataset, segment_total, setup_mpl
plt = setup_mpl()
import importlib.util
import algorithms as alg
from algorithms import correct_shared_log_fixed, estimate_shared_tau_log

_spec = importlib.util.spec_from_file_location(
    "evaluate02", os.path.join(os.path.dirname(__file__), "02_evaluate.py"))
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)
method_metrics = _eval.method_metrics


def prep(name):
    d = load_dataset(name)
    s0, s1 = segment_total(d["total"])
    fs = d["fs"]
    X = d["X"]
    raw_pre = np.median(X[max(0, s0 - int(3 * fs)):s0], axis=0)
    raw_on = np.median(X[s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs))], axis=0)
    raw_response = raw_on - raw_pre
    main_ch = int(np.argmax(X[s0:s1].mean(axis=0)))
    loaded_idx = np.where(raw_response > 0.15)[0]
    return d, s0, s1, raw_pre, raw_on, raw_response, main_ch, loaded_idx


def evaluate_shared_log():
    preps = {n: prep(n) for n in DATASETS}
    taus = {n: estimate_shared_tau_log(preps[n][0], preps[n][1], preps[n][2])
            for n in DATASETS}
    rows = []
    for test in DATASETS:
        d, s0, s1, raw_pre, raw_on, raw_response, main_ch, loaded_idx = preps[test]
        for train in ["self", "数据1", "数据2", "数据3"]:
            if train == "self":
                tau = taus[test]
            else:
                tau = taus[train]
            Y = correct_shared_log_fixed(d, s0, s1, tau_l=tau)
            m = method_metrics(d, s0, s1, Y, raw_pre, raw_on, raw_response,
                               main_ch, loaded_idx)
            rows.append({
                "测试数据": test,
                "tau来源": train,
                "tau_L_s": round(tau, 3),
                "主通道漂移率_pct": round(m["main"]["abs_drift_pct"], 2),
                "主通道RMSE_pct": round(m["main"]["rmse_pct"], 2),
                "主通道阶跃保持率_pct": round(m["main"]["step_retention"], 2),
                "受载组漂移中位_pct": round(m["loaded"]["abs_drift_pct_median"], 2),
                "受载组RMSE中位_pct": round(m["loaded"]["rmse_pct_median"], 2),
            })
    df_cross = pd.DataFrame(rows)

    # 固定 tau_L 扫描
    fixed_rows = []
    for tau in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]:
        vals_d = []
        vals_r = []
        for test in DATASETS:
            d, s0, s1, raw_pre, raw_on, raw_response, main_ch, loaded_idx = preps[test]
            Y = correct_shared_log_fixed(d, s0, s1, tau_l=tau)
            m = method_metrics(d, s0, s1, Y, raw_pre, raw_on, raw_response,
                               main_ch, loaded_idx)
            vals_d.append(m["main"]["abs_drift_pct"])
            vals_r.append(m["main"]["rmse_pct"])
        fixed_rows.append({
            "tau_L_s": tau,
            "三组主通道平均漂移率_pct": round(float(np.mean(vals_d)), 2),
            "三组主通道最大漂移率_pct": round(float(np.max(vals_d)), 2),
            "三组主通道平均RMSE_pct": round(float(np.mean(vals_r)), 2),
            "三组主通道最大RMSE_pct": round(float(np.max(vals_r)), 2),
        })
    df_fixed = pd.DataFrame(fixed_rows)

    df_cross.to_csv(RES / "cross_dataset_shared_log.csv",
                    index=False, encoding="utf-8-sig")
    df_fixed.to_csv(RES / "fixed_tau_scan.csv", index=False, encoding="utf-8-sig")
    with open(RES / "shared_tau.json", "w", encoding="utf-8") as f:
        json.dump({"per_dataset_tau": taus,
                   "cross": df_cross.to_dict(orient="records"),
                   "fixed_scan": df_fixed.to_dict(orient="records")},
                  f, ensure_ascii=False, indent=2)
    print("每组联合拟合的 tau_L(s):", taus)
    print("\n跨组泛化表:")
    print(df_cross.to_string(index=False))
    print("\n固定 tau_L 扫描表:")
    print(df_fixed.to_string(index=False))

    # ---- fig12: 跨组热图 + 固定 tau 扫描曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax = axes[0]
    M = np.empty((len(DATASETS), len(DATASETS)))
    for i, test in enumerate(DATASETS):
        for k, train in enumerate(DATASETS):
            row = df_cross[(df_cross["测试数据"] == test) &
                           (df_cross["tau来源"] == train)]
            M[i, k] = row["受载组漂移中位_pct"].iloc[0]
    im = ax.imshow(M, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(DATASETS)), [f"tau来自 {x}" for x in DATASETS], fontsize=8)
    ax.set_yticks(range(len(DATASETS)), [f"测试 {x}" for x in DATASETS], fontsize=8)
    for i in range(len(DATASETS)):
        for k in range(len(DATASETS)):
            ax.text(k, i, f"{M[i, k]:.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title("共享对数核跨组泛化：受载组漂移中位数 (%阶跃)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    ax.plot(df_fixed["tau_L_s"], df_fixed["三组主通道平均漂移率_pct"],
            "o-", color="tab:red", label="平均漂移率")
    ax.fill_between(df_fixed["tau_L_s"],
                    df_fixed["三组主通道平均漂移率_pct"],
                    df_fixed["三组主通道最大漂移率_pct"], color="tab:red", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("固定 tau_L (s)"); ax.set_ylabel("漂移率 (%阶跃)", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(df_fixed["tau_L_s"], df_fixed["三组主通道平均RMSE_pct"],
             "s--", color="tab:blue", label="平均负载RMSE")
    ax2.set_ylabel("负载RMSE (%阶跃)", color="tab:blue")
    ax.axvline(2.0, color="green", ls=":", lw=1.2, label="推荐 tau_L=2s")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("固定 tau_L 扫描（推荐 tau_L=2s：RMSE低且跨组稳定）")
    fig.suptitle("图12  阵列共享时间核的泛化性与参数选择", fontsize=13)
    fig.savefig(FIG / "fig12_shared_tau_cv.png", dpi=150)
    plt.close(fig)
    return taus, df_cross, df_fixed


if __name__ == "__main__":
    evaluate_shared_log()
