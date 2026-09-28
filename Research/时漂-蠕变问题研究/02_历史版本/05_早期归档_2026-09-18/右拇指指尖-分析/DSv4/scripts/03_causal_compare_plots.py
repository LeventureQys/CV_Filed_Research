# -*- coding: utf-8 -*-
"""03 因果算法横向对比图件。

输出（DSv4/figures）：
    fig06_causal_main_traces.png
    fig07_causal_metrics_bars.png
    fig08_causal_metrics_heatmap.png
    fig09_causal_array_drift_maps.png
    fig10_causal_online_tradeoff.png
    fig11_causal_zero_drift_methods.png
    fig12_causal_calibration.png
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATASETS, FIG, RES, load_dataset, segment_total,
                    channels_to_map, setup_mpl)
import algorithms_causal as alg

plt = setup_mpl()


def load_summary():
    return pd.read_csv(RES / "causal_algorithm_summary.csv", encoding="utf-8-sig")


def load_metrics_json():
    with open(RES / "causal_algorithm_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def load_calibration():
    with open(RES / "causal_calibration.json", encoding="utf-8") as f:
        return json.load(f)


def get_outputs(d, params):
    return alg.apply_all_causal(d, params)


def fig_main_traces():
    calib = load_calibration()
    methods = ["raw", "host_kalman", "rls_log", "kalman_log", "pipeline"]
    titles = ["原始数据", "主机现有Kalman输出", "RLS对数时漂预测",
              "Kalman对数漂移状态", "推荐因果流程"]
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), constrained_layout=True)
    for r, name in enumerate(DATASETS):
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        t = d["t"]
        fs = d["fs"]
        pre = np.median(d["X"][max(0, s0 - int(3 * fs)):s0], axis=0)
        j = int(np.argmax(d["X"][s0:s1].mean(axis=0)))
        outs = get_outputs(d, calib[name])
        for c, method in enumerate(methods):
            ax = axes[r, c]
            Y = outs[method] - pre[None, :]
            ax.plot(t, Y[:, j], lw=0.55, color="tab:red")
            ax.axvspan(t[s0], t[s1], color="orange", alpha=0.16)
            ax.axhline(0, color="green", ls="--", lw=0.8)
            pre_m = np.median(Y[max(0, s0 - int(3 * fs)):s0, j])
            on = np.median(Y[s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs)), j])
            tail = np.median(Y[s1 - int(3 * fs):s1, j])
            drift_pct = 100.0 * abs(tail - on) / max(abs(on - pre_m), 1e-6)
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{name}\nForce-前基线 (N)")
            ax.set_xlabel("时间 (s)")
            ax.text(0.03, 0.93, f"漂移|Δ|={drift_pct:.1f}%",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            ax.grid(alpha=0.3)
    fig.suptitle("图6  ch17 全时序因果校正效果（全部算法只用当前/过去样本）", fontsize=13)
    fig.savefig(FIG / "fig06_causal_main_traces.png", dpi=150)
    plt.close(fig)


def fig_metrics_bars():
    df = load_summary()
    methods = alg.METHOD_ORDER
    labels = [alg.METHOD_DISPLAY[m][0] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), constrained_layout=True)
    metrics = [
        ("受载组绝对漂移中位_pct", "受载组漂移|Δ|中位数 (% 原始阶跃)", "tab:red"),
        ("受载组RMSE中位_pct", "负载段RMSE中位数 (% 原始阶跃)", "tab:blue"),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        vals = np.array([df[df["方法"] == m][col].median() for m in methods])
        mins = np.array([df[df["方法"] == m][col].min() for m in methods])
        maxs = np.array([df[df["方法"] == m][col].max() for m in methods])
        y = np.arange(len(methods))[::-1]
        ax.barh(y, vals, xerr=[vals - mins, maxs - vals],
                color=color, alpha=0.85, capsize=3)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
        for yi, v in zip(y, vals):
            ax.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=8)
    fig.suptitle("图7  因果算法在三组数据上的漂移与平坦度（误差线=三组数据范围）", fontsize=13)
    fig.savefig(FIG / "fig07_causal_metrics_bars.png", dpi=150)
    plt.close(fig)


def fig_metrics_heatmap():
    df = load_summary()
    methods = alg.METHOD_ORDER
    labels = [alg.METHOD_DISPLAY[m][0] for m in methods]
    metrics = [
        ("绝对漂移率_pct", "主通道漂移|Δ| (%阶跃)", "Reds"),
        ("负载RMSE_pct", "主通道负载RMSE (%阶跃)", "Blues"),
        ("卸载后残差_pct", "主通道卸载残差 (%阶跃)", "PiYG"),
        ("阶跃保持率_pct", "主通道阶跃保持率 (%)", "Greens"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (col, title, cmap) in zip(axes.ravel(), metrics):
        M = np.empty((len(methods), len(DATASETS)))
        for i, m in enumerate(methods):
            for k, ds in enumerate(DATASETS):
                row = df[(df["方法"] == m) & (df["数据"] == ds)]
                M[i, k] = row[col].iloc[0] if len(row) else np.nan
        if col == "卸载后残差_pct":
            vmax = np.nanmax(np.abs(M)) * 1.05
            im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        else:
            im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(DATASETS)), DATASETS)
        ax.set_yticks(range(len(methods)), labels, fontsize=8)
        ax.set_title(title)
        for i in range(len(methods)):
            for k in range(len(DATASETS)):
                ax.text(k, i, f"{M[i, k]:.1f}", ha="center", va="center",
                        fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("图8  因果方法 × 数据 主通道指标热图", fontsize=13)
    fig.savefig(FIG / "fig08_causal_metrics_heatmap.png", dpi=150)
    plt.close(fig)


def fig_array_drift_maps():
    allres = load_metrics_json()
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), constrained_layout=True)
    methods = ["raw", "kalman_log", "pipeline"]
    titles = ["原始负载段漂移", "Kalman对数状态后残余", "推荐流程后残余"]
    for r, name in enumerate(DATASETS):
        raw_drift = np.array(allres[name]["raw"]["drift"])
        vmax = max(0.1, float(np.nanmax(np.abs(raw_drift))) * 1.1)
        for c, method in enumerate(methods):
            ax = axes[r, c]
            vals = np.array(allres[name][method]["drift"])
            grid = channels_to_map(vals)
            im = ax.imshow(grid, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{name} {titles[c]}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("图9  因果校正后的漂移空间分布（色标=末3s-初0.5~2s, N）", fontsize=13)
    fig.savefig(FIG / "fig09_causal_array_drift_maps.png", dpi=150)
    plt.close(fig)


def quick_step_drift(d, s0, s1, Y, j):
    fs = d["fs"]
    pre = np.median(Y[max(0, s0 - int(3 * fs)):s0, j])
    on = np.median(Y[s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs)), j])
    tail = np.median(Y[s1 - int(3 * fs):s1, j])
    step = on - pre
    return step, tail - on, np.sqrt(np.mean((Y[s0:s1, j] - (pre + step)) ** 2))


def fig_online_tradeoff():
    name = "数据2"
    d = load_dataset(name)
    s0, s1 = segment_total(d["total"])
    j = int(np.argmax(d["X"][s0:s1].mean(axis=0)))
    tl = d["t"][s0:s1] - d["t"][s0]
    fs = d["fs"]
    pre_raw = np.median(d["X"][max(0, s0 - int(3 * fs)):s0, j])
    raw_step = (np.median(d["X"][s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs)), j])
                - pre_raw)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(tl, d["X"][s0:s1, j] - pre_raw, lw=0.8, color="k", label="原始")
    for alpha in [0.0003, 0.001, 0.003]:
        Y = alg.correct_ema_hp(d, alpha=alpha)
        step, drift, rmse = quick_step_drift(d, s0, s1, Y, j)
        ax.plot(tl, Y[s0:s1, j], lw=0.9,
                label=f"α={alpha} (τ≈{0.015/alpha:.0f}s, 阶跃保留{100*step/raw_step:.0f}%)")
    ax.set_title(f"{name} ch17  一阶高通(EMA)：漂移换静态衰减")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force-前基线 (N)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(tl, d["X"][s0:s1, j] - pre_raw, lw=0.8, color="k", label="原始")
    for tau in [0.5, 1.0, 2.0, 5.0, 10.0]:
        Y = alg.correct_rls_drift(d, "log", tau=tau, thr=3.0)
        _, drift, rmse = quick_step_drift(d, s0, s1, Y, j)
        ax.plot(tl, Y[s0:s1, j], lw=0.9,
                label=f"τ_L={tau}s (末漂移{100*abs(drift)/raw_step:.1f}%, RMSE={rmse:.3f}N)")
    ax.set_title(f"{name} ch17  RLS对数预测：时间常数敏感性（因果）")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force-前基线 (N)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("图10  因果在线算法的权衡与参数敏感性", fontsize=13)
    fig.savefig(FIG / "fig10_causal_online_tradeoff.png", dpi=150)
    plt.close(fig)


def fig_zero_methods():
    df = load_summary()
    methods = ["raw", "host_kalman", "gate_baseline", "rls_log", "kalman_log", "pipeline"]
    labels = [alg.METHOD_DISPLAY[m][0] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), constrained_layout=True)
    x = np.arange(len(methods))
    width = 0.25
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for ax, col, title in zip(axes, ["前基线绝对偏移_N", "卸载后残差_N"],
                              ["主通道前空载基线（绝对值越小越好）",
                               "主通道卸载后残差（越接近0越好）"]):
        for k, (ds, c) in enumerate(zip(DATASETS, colors)):
            vals = [df[(df["数据"] == ds) & (df["方法"] == m)][col].iloc[0]
                    for m in methods]
            ax.bar(x + (k - 1) * width, vals, width, color=c, alpha=0.85, label=ds)
        ax.set_xticks(x, labels, rotation=22, ha="right", fontsize=8)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("图11  因果零漂校正方法对比", fontsize=13)
    fig.savefig(FIG / "fig11_causal_zero_drift_methods.png", dpi=150)
    plt.close(fig)


def fig_calibration():
    calib = load_calibration()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, name in zip(axes, DATASETS):
        scores = calib[name]["log_scores"]
        taus = [float(k) for k in scores.keys()]
        vals = [scores[k] for k in scores.keys()]
        ax.semilogx(taus, vals, "o-", color="tab:red", label="RLS对数核评分")
        es = calib[name]["exp_scores"]
        etaus = [float(k) for k in es.keys()]
        evals = [es[k] for k in es.keys()]
        ax.semilogx(etaus, evals, "s--", color="tab:blue", label="RLS指数核评分")
        ax.axvline(float(calib[name]["tau_log"]), color="tab:red", ls=":", lw=1)
        ax.axvline(float(calib[name]["tau_exp"]), color="tab:blue", ls=":", lw=1)
        ax.set_title(f"{name}\n训练={calib[name]['train']}  thr={calib[name]['thr']:.2f}N\n"
                     f"选 τ_log={calib[name]['tau_log']}s, τ_exp={calib[name]['tau_exp']}s")
        ax.set_xlabel("τ (s)"); ax.set_ylabel("训练组评分 = 漂移% + 0.5×RMSE%")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("图12  留一法超参数标定（测试段不参与选参）", fontsize=13)
    fig.savefig(FIG / "fig12_causal_calibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_main_traces()
    fig_metrics_bars()
    fig_metrics_heatmap()
    fig_array_drift_maps()
    fig_online_tradeoff()
    fig_zero_methods()
    fig_calibration()
    print("saved fig06~fig12 (causal)")
