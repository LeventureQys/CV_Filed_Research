# -*- coding: utf-8 -*-
"""03 算法横向对比图件。

输出：
    DSv4/figures/fig06_main_traces.png
    DSv4/figures/fig07_metrics_bars.png
    DSv4/figures/fig08_metrics_heatmap.png
    DSv4/figures/fig09_array_drift_maps.png
    DSv4/figures/fig10_online_tradeoff.png
    DSv4/figures/fig11_zero_drift_methods.png
"""
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATASETS, FIG, RES, load_dataset, segment_total,
                    channels_to_map, setup_mpl)
import algorithms as alg

plt = setup_mpl()


def load_summary():
    return pd.read_csv(RES / "algorithm_summary.csv", encoding="utf-8-sig")


def load_metrics_json():
    with open(RES / "algorithm_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def get_selected_outputs(d, s0, s1):
    """只重算用于画图的快速算法。"""
    return {
        "raw": d["X"].copy(),
        "host_kalman": alg.correct_host_kalman(d, s0, s1),
        "gate_baseline": alg.correct_gate_baseline(d, s0, s1),
        "ema_hp": alg.correct_ema_hp(d, s0, s1),
        "shared_log": alg.correct_shared_log(d, s0, s1),
        "online_rls": alg.correct_online_rls(d, s0, s1),
        "pipeline": alg.correct_pipeline(d, s0, s1),
    }


def fig_main_traces():
    methods = ["raw", "host_kalman", "shared_log", "online_rls", "pipeline"]
    titles = ["原始数据", "主机现有Kalman", "阵列共享对数去漂(批处理)",
              "在线RLS指数预测", "推荐组合流程"]
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), constrained_layout=True)
    for r, name in enumerate(DATASETS):
        d = load_dataset(name)
        s0, s1 = segment_total(d["total"])
        t = d["t"]
        pre = np.median(d["X"][max(0, s0 - int(3 * d["fs"])):s0], axis=0)
        j = int(np.argmax(d["X"][s0:s1].mean(axis=0)))
        outs = get_selected_outputs(d, s0, s1)
        for c, method in enumerate(methods):
            ax = axes[r, c]
            Y = outs[method] - pre[None, :]
            ax.plot(t, Y[:, j], lw=0.6, color="tab:red")
            ax.axvspan(t[s0], t[s1], color="orange", alpha=0.16)
            ax.axhline(0, color="green", ls="--", lw=0.8)
            pre_m = np.median(Y[max(0, s0 - int(3 * d["fs"])):s0, j])
            on = np.median(Y[s0 + int(round(0.5 * d["fs"])):s0 + int(round(2.0 * d["fs"])), j])
            tail = np.median(Y[s1 - int(3 * d["fs"]):s1, j])
            drift_pct = 100.0 * abs(tail - on) / max(abs(on - pre_m), 1e-6)
            if r == 0:
                ax.set_title(f"{titles[c]}\n({method})", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{name}\nForce-前基线 (N)")
            ax.set_xlabel("时间 (s)")
            ax.text(0.03, 0.93, f"漂移|Δ|={drift_pct:.0f}%",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            ax.grid(alpha=0.3)
    fig.suptitle("图6  ch17 全时序校正效果对比（所有曲线已相对各自前空载基线归零）", fontsize=13)
    fig.savefig(FIG / "fig06_main_traces.png", dpi=150)
    plt.close(fig)


def fig_metrics_bars():
    df = load_summary()
    methods = alg.METHOD_ORDER
    labels = [alg.METHOD_DISPLAY[m][0] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), constrained_layout=True)
    metrics = [
        ("受载组绝对漂移中位_pct", "受载组漂移|Δ|中位数 (% 原始阶跃)", "tab:red"),
        ("受载组RMSE中位_pct", "负载段RMSE中位数 (% 原始阶跃)", "tab:blue"),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        vals = [df[df["方法"] == m][col].median() for m in methods]
        mins = [df[df["方法"] == m][col].min() for m in methods]
        maxs = [df[df["方法"] == m][col].max() for m in methods]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, vals, xerr=[np.array(vals) - np.array(mins),
                               np.array(maxs) - np.array(vals)],
                color=color, alpha=0.85, capsize=3)
        ax.set_yticks(y, labels)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
        for yi, v in zip(y, vals):
            ax.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=8)
    fig.suptitle("图7  各算法在三组数据上的漂移与平坦度（误差线=三组数据范围）", fontsize=13)
    fig.savefig(FIG / "fig07_metrics_bars.png", dpi=150)
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
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
    fig.suptitle("图8  方法 × 数据 主通道指标热图", fontsize=13)
    fig.savefig(FIG / "fig08_metrics_heatmap.png", dpi=150)
    plt.close(fig)


def fig_array_drift_maps():
    allres = load_metrics_json()
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5), constrained_layout=True)
    methods = ["raw", "shared_log", "pipeline"]
    titles = ["原始负载段漂移", "阵列共享对数去漂后残余", "推荐组合流程后残余"]
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
    fig.suptitle("图9  漂移空间分布：原始 vs 两种校正算法（色标=末3s-初0.5~2s, N）", fontsize=13)
    fig.savefig(FIG / "fig09_array_drift_maps.png", dpi=150)
    plt.close(fig)


def quick_step_drift(d, s0, s1, Y, j):
    fs = d["fs"]
    pre = np.median(Y[max(0, s0 - int(3 * fs)):s0, j])
    on = np.median(Y[s0 + int(round(0.5 * fs)):s0 + int(round(2.0 * fs)), j])
    tail = np.median(Y[s1 - int(3 * fs):s1, j])
    step = on - pre
    return step, tail - on, np.sqrt(np.mean((Y[s0:s1, j] - (pre + step)) ** 2))


def fig_online_tradeoff():
    name = "数据2"  # 负载时间最长，最能暴露时漂
    d = load_dataset(name)
    s0, s1 = segment_total(d["total"])
    j = int(np.argmax(d["X"][s0:s1].mean(axis=0)))
    t = d["t"]
    tl = t[s0:s1] - t[s0]
    pre = np.median(d["X"][max(0, s0 - int(3 * d["fs"])):s0, j])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(tl, d["X"][s0:s1, j] - pre, lw=0.8, color="k", label="原始")
    for alpha in [0.0003, 0.001, 0.003]:
        Y = alg.correct_ema_hp(d, s0, s1, alpha=alpha)
        y = Y[s0:s1, j]
        step, drift, rmse = quick_step_drift(d, s0, s1, Y, j)
        raw_step = np.median(d["X"][s0 + int(round(0.5 * d["fs"])):s0 + int(round(2.0 * d["fs"])), j]) - pre
        ax.plot(tl, y, lw=0.9,
                label=f"α={alpha} (τ≈{0.015/alpha:.0f}s, 阶跃保留{100*step/raw_step:.0f}%)")
    ax.set_title(f"{name} ch17  一阶高通(EMA)：漂移换静态衰减")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force-前基线 (N)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(tl, d["X"][s0:s1, j] - pre, lw=0.8, color="k", label="原始")
    for tau in [20, 40, 60, 100, 200]:
        Y = alg.correct_online_rls(d, s0, s1, tau=tau)
        y = Y[s0:s1, j]
        _, drift, rmse = quick_step_drift(d, s0, s1, Y, j)
        ax.plot(tl, y, lw=0.9,
                label=f"τ={tau}s (末漂移{100*abs(drift)/max(1e-9, (np.median(d['X'][s0+33:s0+133,j])-pre)):.0f}%, RMSE={rmse:.3f}N)")
    ax.set_title(f"{name} ch17  在线RLS指数预测：时间常数敏感性")
    ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force-前基线 (N)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("图10  在线算法的本质权衡与参数敏感性", fontsize=13)
    fig.savefig(FIG / "fig10_online_tradeoff.png", dpi=150)
    plt.close(fig)


def fig_zero_methods():
    df = load_summary()
    methods = ["raw", "host_kalman", "tare", "gate_baseline", "ref_common"]
    labels = [alg.METHOD_DISPLAY[m][0] for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
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
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("图11  零漂校正方法对比：前基线偏移与卸载残余", fontsize=13)
    fig.savefig(FIG / "fig11_zero_drift_methods.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_main_traces()
    fig_metrics_bars()
    fig_metrics_heatmap()
    fig_array_drift_maps()
    fig_online_tradeoff()
    fig_zero_methods()
    print("saved fig06 ~ fig11")
