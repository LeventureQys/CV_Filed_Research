from __future__ import annotations

import csv
import sys
from pathlib import Path
from pprint import pformat

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT.parent  # v1.0.10 - research/
sys.path.insert(0, str(RESEARCH_DIR / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader_v2 import load_recording
from filters import (
    ema, lowpass_iir, scalar_kalman, causal_median3,
    adaptive_baseline, dual_state_kalman, huber_ema,
)

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FINAL_METHODS = ["raw", "ema_0.8s", "iir2_0.5hz", "huber_ema"]
FINAL_COLORS = ["#777777", "#e6194b", "#3cb44b", "#4363d8"]

SHORT_LABELS = {
    "raw": "原始",
    "ema_0.35s": "EMA 0.35s",
    "ema_0.8s": "EMA 0.8s",
    "median3_ema": "M3+EMA",
    "iir1_0.5hz": "IIR1 0.5",
    "iir2_0.5hz": "IIR2 0.5",
    "kalman_fast": "KF快",
    "kalman_stable": "KF稳",
    "cell_median3_ema": "Cell M3+EMA",
    "adaptive_baseline": "自基线",
    "dual_state_kalman": "双状态KF",
    "huber_ema": "Huber EMA",
}

METHOD_LABELS = {
    "raw": "原始",
    "ema_0.35s": "EMA 0.35 s",
    "ema_0.8s": "EMA 0.8 s",
    "median3_ema": "Median-3 + EMA",
    "iir1_0.5hz": "一阶 IIR 0.5 Hz",
    "iir2_0.5hz": "二阶 IIR 0.5 Hz",
    "kalman_fast": "卡尔曼（快响应）",
    "kalman_stable": "卡尔曼（稳态型）",
    "cell_median3_ema": "逐 Cell Median-3 + EMA",
    "adaptive_baseline": "自适应基线 + EMA",
    "dual_state_kalman": "双状态卡尔曼",
    "huber_ema": "鲁棒 EMA（Huber）",
}

RECORDINGS = [
    # Old format (砝码压力)
    {
        "device": "fabric",
        "label": "织物垫 500 g (旧格式)",
        "background": ROOT.parent / "DataSet" / "织物垫" / "背景噪声" / "录制数据_20260710143202_part0.csv",
        "stress": ROOT.parent / "DataSet" / "织物垫" / "500g砝码压力" / "录制数据_20260710143312_part0.csv",
    },
    {
        "device": "fabric",
        "label": "织物垫 1 kg (旧格式)",
        "background": ROOT.parent / "DataSet" / "织物垫" / "背景噪声" / "录制数据_20260710143202_part0.csv",
        "stress": ROOT.parent / "DataSet" / "织物垫" / "1kg砝码压力" / "录制数据_20260710143635_part0.csv",
    },
    {
        "device": "fabric",
        "label": "织物垫 500 g (新格式)",
        "background": ROOT.parent / "DataSet" / "织物垫" / "背景噪声" / "录制数据_20260710143202_part0.csv",
        "stress": ROOT.parent / "DataSet" / "织物垫" / "空载30s-500g 3min-空载30s" / "20260713_160223_single_device_2f5f7f",
    },
    {
        "device": "fabric",
        "label": "织物垫 1 kg (新格式)",
        "background": ROOT.parent / "DataSet" / "织物垫" / "背景噪声" / "录制数据_20260710143202_part0.csv",
        "stress": ROOT.parent / "DataSet" / "织物垫" / "空载30s-1kg 3min-空载30s" / "20260713_161027_single_device_1b77d5",
    },
    {
        "device": "fabric",
        "label": "织物垫 1.5 kg (新格式)",
        "background": ROOT.parent / "DataSet" / "织物垫" / "背景噪声" / "录制数据_20260710143202_part0.csv",
        "stress": ROOT.parent / "DataSet" / "织物垫" / "空载30s-1.5kg 3min-空载30s" / "20260713_161628_single_device_46359f",
    },
    {
        "device": "fabric",
        "label": "织物垫 1 kg 长时保持 (20 min)",
        "background": ROOT.parent / "DataSet" / "织物垫" / "背景噪声" / "录制数据_20260710143202_part0.csv",
        "stress": ROOT.parent / "DataSet" / "织物垫" / "空载1min-1kg 20min-空载1min" / "20260713_162110_single_device_c56f37",
    },
    {
        "device": "fabric",
        "label": "织物垫 空载 20 min",
        "background": ROOT.parent / "DataSet" / "织物垫" / "背景噪声" / "录制数据_20260710143202_part0.csv",
        "stress": ROOT.parent / "DataSet" / "织物垫" / "空载20分钟" / "20260713_151358_single_device_77d029",
    },
]


def apply_algorithms(total: np.ndarray, cells: np.ndarray, sample_rate: float) -> dict[str, np.ndarray]:
    meas_var = max(float(np.var(np.diff(total))) / 2.0, 1e-6)

    results = {
        "raw": total,
        "ema_0.35s": ema(total, sample_rate, 0.35),
        "ema_0.8s": ema(total, sample_rate, 0.8),
        "median3_ema": ema(causal_median3(total), sample_rate, 0.5),
        "iir1_0.5hz": lowpass_iir(total, sample_rate, 0.5, 1),
        "iir2_0.5hz": lowpass_iir(total, sample_rate, 0.5, 2),
        "kalman_fast": scalar_kalman(total, meas_var * 0.012, meas_var),
        "kalman_stable": scalar_kalman(total, meas_var * 0.002, meas_var),
        "cell_median3_ema": ema(causal_median3(cells), sample_rate, 0.5).sum(axis=1),
        "dual_state_kalman": dual_state_kalman(total, sample_rate, q_force=0.01, q_drift=1e-4, r_measure=meas_var),
        "huber_ema": huber_ema(total, sample_rate, 0.8, 3.0),
    }

    ab_cells = adaptive_baseline(cells, sample_rate)
    results["adaptive_baseline"] = ema(ab_cells.sum(axis=1), sample_rate, 0.8)

    return results


def calculate_metrics(device: str, label: str, sample_rate: float,
                      channels: int, methods: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    guard = min(max(round(sample_rate * 5.0), 1), len(methods["raw"]) // 4)
    evaluation = slice(guard, len(methods["raw"]) - guard)
    raw_roughness = max(float(np.std(np.diff(methods["raw"][evaluation]))), 1e-12)

    for method, values in methods.items():
        segment = values[evaluation]
        roughness = float(np.std(np.diff(segment)))
        rows.append({
            "device": device,
            "recording": label,
            "sample_rate_hz": round(sample_rate, 1),
            "channels": channels,
            "frames": len(values),
            "method": method,
            "mean_total_adc": float(np.mean(segment)),
            "std_total_adc": float(np.std(segment)),
            "peak_to_peak_adc": float(np.ptp(segment)),
            "first_difference_std": roughness,
            "roughness_retention": roughness / raw_roughness,
        })
    return rows


def pick_final_methods(all_methods: dict) -> list[str]:
    return ["raw", "ema_0.8s", "iir2_0.5hz", "huber_ema"]


COLORS = ["#777777", "#e6194b", "#3cb44b", "#4363d8", "#f58231",
          "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#469990",
          "#9a6324", "#800000"]


def plot_all_algorithms(cases: list[dict]) -> None:
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n + 1), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        time = case["time"]
        methods = case["methods"]
        for idx, (method, values) in enumerate(methods.items()):
            if method == "raw":
                ax.plot(time, values, color="#777777", linewidth=0.7, alpha=0.7, label=METHOD_LABELS[method])
            else:
                ax.plot(time, values, color=COLORS[(idx - 1) % len(COLORS)],
                        linewidth=1.0, label=METHOD_LABELS[method])
        ax.set_title(f"{case['label']}, {case['sample_rate']:.1f} Hz, {case['channels']} Cell")
        ax.set_ylabel("总 ADC")
        ax.grid(alpha=0.2)

    axes[0].legend(ncol=4, fontsize=7, loc="best")
    axes[-1].set_xlabel("时间（秒）")
    fig.suptitle("beta4 织物垫实物数据：全部 12 组算法比对", fontsize=14)
    fig.savefig(str(OUTPUT_DIR / "fabric_all_algorithms.png"), dpi=160)
    plt.close(fig)
    print(f"  [ok] Saved fabric_all_algorithms.png")


def plot_final_comparison(cases: list[dict]) -> None:
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n + 1), constrained_layout=True)
    if n == 1:
        axes = [axes]

    final_methods = pick_final_methods({})

    for ax, case in zip(axes, cases):
        time = case["time"]
        methods = case["methods"]
        for method, values in methods.items():
            if method not in final_methods:
                continue
            idx = list(methods.keys()).index(method)
            if method == "raw":
                ax.plot(time, values, color="#777777", linewidth=0.7, alpha=0.7, label=METHOD_LABELS[method])
            else:
                ax.plot(time, values, color=COLORS[(idx - 1) % len(COLORS)],
                        linewidth=1.3, label=METHOD_LABELS[method])
        ax.set_title(f"{case['label']}, {case['sample_rate']:.1f} Hz")
        ax.set_ylabel("总 ADC")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="best")

    axes[-1].set_xlabel("时间（秒）")
    fig.suptitle("beta4 织物垫：精炼比对（3-4 组推荐方案）", fontsize=14)
    fig.savefig(str(OUTPUT_DIR / "fabric_final_comparison.png"), dpi=160)
    plt.close(fig)
    print(f"  [ok] Saved fabric_final_comparison.png")


def plot_metrics_summary(all_rows: list[dict]) -> None:
    groups = {
        "旧格式 (10 Hz)": ["织物垫 500 g (旧格式)", "织物垫 1 kg (旧格式)"],
        "新格式 (55 Hz)": ["织物垫 500 g (新格式)", "织物垫 1 kg (新格式)", "织物垫 1.5 kg (新格式)"],
        "长时 / 空载": ["织物垫 1 kg 长时保持 (20 min)", "织物垫 空载 20 min"],
    }
    group_keys = list(groups.keys())
    n_groups = len(group_keys)
    _, axes = plt.subplots(2, n_groups, figsize=(5 * n_groups + 1, 8), constrained_layout=True)

    exclude = {"raw", "adaptive_baseline"}
    method_order = [m for m in SHORT_LABELS if m not in exclude]

    for col, (gname, rec_names) in enumerate(groups.items()):
        sub = [r for r in all_rows if r["recording"] in rec_names]
        if not sub:
            continue
        raw_rows = [r for r in sub if r["method"] == "raw"]
        raw_by_rec = {r["recording"]: r["first_difference_std"] for r in raw_rows}

        methods_used = []
        roughness_vals = []
        mean_ratios = []
        for method in method_order:
            vals = [r for r in sub if r["method"] == method]
            if not vals:
                continue
            r_rough = np.mean([v["first_difference_std"] for v in vals])
            raw_rough = np.mean([raw_by_rec.get(v["recording"], 1.0) for v in vals])
            reduction = 1.0 - (r_rough / max(raw_rough, 1e-12))
            mean_ratio = np.mean([v["mean_total_adc"] for v in vals]) / max(
                np.mean([raw_by_rec.get(v["recording"], 1.0) for v in vals]), 1e-12)
            methods_used.append(method)
            roughness_vals.append(reduction * 100)
            mean_ratios.append(mean_ratio)

        if len(methods_used) == 0:
            continue

        ax_rough = axes[0, col]
        idx = np.arange(len(methods_used))
        colors = [FINAL_COLORS[FINAL_METHODS.index(m)] if m in FINAL_METHODS else "#999999"
                  for m in methods_used]
        hatches = ["//" if m in FINAL_METHODS else "" for m in methods_used]
        bars = ax_rough.bar(idx, roughness_vals, color=colors, hatch=None, edgecolor="black", linewidth=0.5)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        ax_rough.set_xticks(idx)
        ax_rough.set_xticklabels([SHORT_LABELS.get(m, m) for m in methods_used], rotation=45, ha="right", fontsize=7)
        ax_rough.set_ylabel("roughness 降低率 (%)")
        ax_rough.set_title(f"{gname}\nroughness 降低率", fontsize=10)
        ax_rough.axhline(0, color="black", linewidth=0.5)
        ax_rough.grid(axis="y", alpha=0.3)

        ax_mean = axes[1, col]
        mean_colors = [FINAL_COLORS[FINAL_METHODS.index(m)] if m in FINAL_METHODS else "#999999"
                       for m in methods_used]
        ax_mean.bar(idx, [m * 100 for m in mean_ratios], color=mean_colors,
                    edgecolor="black", linewidth=0.5)
        ax_mean.set_xticks(idx)
        ax_mean.set_xticklabels([SHORT_LABELS.get(m, m) for m in methods_used], rotation=45, ha="right", fontsize=7)
        ax_mean.set_ylabel("均值保持率 (%)")
        ax_mean.set_title(f"{gname}\n均值保持率", fontsize=10)
        ax_mean.axhline(100, color="black", linewidth=0.5, linestyle="--")
        ax_mean.grid(axis="y", alpha=0.3)

    axes[0, 0].legend(
        [plt.Rectangle((0, 0), 1, 1, fc=c) for c in [FINAL_COLORS[0], "#999999"]],
        ["推荐方案", "其他候选"],
        loc="upper right", fontsize=8
    )
    fig = axes[0, 0].figure
    fig.suptitle("beta4 织物垫：算法指标汇总（按录制场景分组）", fontsize=13)
    fig.savefig(str(OUTPUT_DIR / "fabric_metrics_summary.png"), dpi=160)
    plt.close(fig)
    print(f"  [ok] Saved fabric_metrics_summary.png")


def main() -> None:
    all_metric_rows = []
    cases = []

    success_count = 0
    fail_count = 0

    for rec in RECORDINGS:
        label = rec["label"]
        print(f"\n=== Processing: {label} ===")
        try:
            stress_cells, stress_meta = load_recording(str(rec["stress"]))
            bg_cells, bg_meta = load_recording(str(rec["background"]))
        except Exception as e:
            print(f"  [FAIL] Load error: {e}")
            fail_count += 1
            continue

        sample_rate = stress_meta["sample_freq_hz"]
        channels = stress_meta["channels"]
        n_frames = stress_meta["n_frames"]

        baseline = np.median(bg_cells, axis=0)
        corrected = np.maximum(stress_cells - baseline, 0.0)

        total_adc = corrected.sum(axis=1)
        methods = apply_algorithms(total_adc, corrected, sample_rate)

        time_axis = stress_meta.get("time_elapsed", None)
        if time_axis is None or len(time_axis) != n_frames:
            time_axis = np.arange(n_frames, dtype=float) / sample_rate

        cases.append({
            "label": label,
            "sample_rate": sample_rate,
            "channels": channels,
            "time": time_axis,
            "methods": methods,
        })

        metric_rows = calculate_metrics(
            rec["device"], label, sample_rate, channels, methods
        )
        all_metric_rows.extend(metric_rows)

        print(f"  frames={n_frames}, rate={sample_rate:.1f} Hz, cells={channels}")
        for row in metric_rows:
            print(f"    {row['method']:24s} mean={row['mean_total_adc']:.0f} "
                  f"std={row['std_total_adc']:.2f} roughness={row['first_difference_std']:.3f}")
        success_count += 1

    print(f"\n{'='*60}")
    print(f"Processed: {success_count} success, {fail_count} fail")

    if not all_metric_rows:
        print("No data to plot. Exiting.")
        return

    csv_path = DATA_DIR / "fabric_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_metric_rows)
    print(f"\n[ok] Metrics saved to {csv_path}")

    print("\n--- Generating plots ---")
    plot_all_algorithms(cases)
    plot_final_comparison(cases)
    plot_metrics_summary(all_metric_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
