from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from filters import (
    causal_median3,
    ema,
    hard_gate,
    local_crosstalk_correction,
    lowpass_iir,
    richardson_lucy,
    scalar_kalman,
    spatial_median,
)
from simulation_models import ForceCase, simulate_force_case, simulate_membrane_case


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
SEED = 20260713


def membrane_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    true_mask = truth > max(float(truth.max()) * 0.01, 1e-9)
    estimate_mask = estimate > max(float(estimate.max()) * 0.01, 1e-9)
    union = np.logical_or(true_mask, estimate_mask).sum()
    intersection = np.logical_and(true_mask, estimate_mask).sum()
    noncontact = np.logical_not(true_mask)
    true_sum = float(truth.sum())
    rows, columns = np.indices(truth.shape)

    def centroid(values: np.ndarray) -> np.ndarray:
        total = max(float(values.sum()), 1e-9)
        return np.array([(rows * values).sum() / total, (columns * values).sum() / total])

    return {
        "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
        "total_ratio": float(estimate.sum() / max(true_sum, 1e-9)),
        "peak_ratio": float(estimate.max() / max(float(truth.max()), 1e-9)),
        "area_ratio": float(estimate_mask.sum() / max(int(true_mask.sum()), 1)),
        "iou": float(intersection / max(int(union), 1)),
        "centroid_shift_cells": float(np.linalg.norm(centroid(estimate) - centroid(truth))),
        "noncontact_residual_ratio": float(estimate[noncontact].sum() / max(true_sum, 1e-9)),
    }


def run_membrane_study() -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED)
    rows = []
    representative = None
    case_names = ("local_center", "local_edge", "local_multi", "global_uniform", "global_gradient")
    for case_name in case_names:
        for leakage in (0.08, 0.16, 0.28):
            for model in ("matched", "mismatch"):
                case = simulate_membrane_case(case_name, leakage, rng, asymmetric=model == "mismatch")
                assumed_kernel = case.kernel if model == "matched" else simulate_membrane_case(
                    case_name, leakage * 0.75, np.random.default_rng(0), asymmetric=False
                ).kernel
                methods = {
                    "raw": case.observed,
                    "hard_gate": hard_gate(case.observed),
                    "spatial_median": spatial_median(case.observed),
                    "local_correction": local_crosstalk_correction(case.observed, assumed_kernel),
                    "richardson_lucy": richardson_lucy(case.observed, assumed_kernel),
                }
                for method, estimate in methods.items():
                    rows.append(
                        {
                            "case": case.name,
                            "load_type": case.load_type,
                            "model": model,
                            "leakage": leakage,
                            "method": method,
                            **membrane_metrics(case.truth, estimate),
                        }
                    )
                if case_name == "local_center" and leakage == 0.16 and model == "matched":
                    representative = (case, methods)
    plot_membrane(representative)
    return rows


def plot_membrane(representative) -> None:
    case, methods = representative
    panels = [("Truth", case.truth), ("Observed", case.observed)] + [
        (name.replace("_", " ").title(), values) for name, values in methods.items() if name != "raw"
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    limit = max(float(values.max()) for _, values in panels)
    for axis, (title, values) in zip(axes.flat, panels):
        image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=limit)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.8, label="Simulated response")
    figure.suptitle("64x64 membrane crosstalk: representative local load")
    figure.savefig(OUTPUT_DIR / "membrane_crosstalk_comparison.png", dpi=180)
    plt.close(figure)


def force_algorithms(values: np.ndarray, sample_rate: float, measurement_variance: float) -> dict[str, np.ndarray]:
    return {
        "raw": values,
        "ema_0.35s": ema(values, sample_rate, 0.35),
        "ema_0.8s": ema(values, sample_rate, 0.8),
        "median3_ema": ema(causal_median3(values), sample_rate, 0.5),
        "iir1_0.5hz": lowpass_iir(values, sample_rate, 0.5, 1),
        "iir2_0.5hz": lowpass_iir(values, sample_rate, 0.5, 2),
        "kalman_fast": scalar_kalman(values, measurement_variance * 0.012, measurement_variance),
        "kalman_stable": scalar_kalman(values, measurement_variance * 0.002, measurement_variance),
    }


def settling_time(
    estimate: np.ndarray,
    truth: np.ndarray,
    transition_index: int,
    next_transition: int,
    sample_rate: float,
) -> float:
    target = float(np.median(truth[min(transition_index + round(sample_rate * 3), next_transition - 1) : next_transition]))
    tolerance = max(abs(target) * 0.02, 0.15)
    window = max(round(sample_rate), 1)
    for index in range(transition_index, max(transition_index, next_transition - window)):
        if np.all(np.abs(estimate[index : index + window] - target) <= tolerance):
            return float((index - transition_index) / sample_rate)
    return float((next_transition - transition_index) / sample_rate)


def force_metrics(case: ForceCase, estimate: np.ndarray) -> dict[str, float]:
    truth = case.true_cells.sum(axis=1)
    guard = round(case.sample_rate * 5.0)
    segments = []
    boundaries = (0,) + case.transition_indices + (len(truth),)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start > guard * 2:
            segments.append(slice(start + guard, end - guard))
    steady_estimate = np.concatenate([estimate[segment] for segment in segments])
    steady_truth = np.concatenate([truth[segment] for segment in segments])
    settling = []
    for position, transition in enumerate(case.transition_indices):
        following = case.transition_indices[position + 1] if position + 1 < len(case.transition_indices) else len(truth)
        settling.append(settling_time(estimate, truth, transition, following, case.sample_rate))
    error = steady_estimate - steady_truth
    return {
        "steady_bias": float(np.mean(error)),
        "steady_rmse": float(np.sqrt(np.mean(error**2))),
        "steady_std": float(np.std(error)),
        "steady_peak_to_peak": float(np.ptp(error)),
        "error95": float(np.quantile(np.abs(error), 0.95)),
        "mean_settling_s": float(np.mean(settling)),
        "max_settling_s": float(np.max(settling)),
        "final_drift": float(np.mean(error[-round(case.sample_rate * 10) :]) - np.mean(error[: round(case.sample_rate * 10)])),
    }


def run_force_study() -> tuple[list[dict[str, object]], dict[str, tuple[ForceCase, dict[str, np.ndarray]]]]:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    plotted = {}
    for load_type in ("local", "global"):
        case = simulate_force_case(load_type, rng)
        truth_total = case.true_cells.sum(axis=1)
        raw_total = case.observed_cells.sum(axis=1)
        baseline = raw_total[: case.transition_indices[0]]
        total_variance = max(float(np.var(baseline)), 1e-6)
        total_methods = force_algorithms(raw_total, case.sample_rate, total_variance)
        cell_variance = np.maximum(np.var(case.observed_cells[: case.transition_indices[0]], axis=0), 1e-6)
        cell_methods = {
            "cell_ema_0.35s": ema(case.observed_cells, case.sample_rate, 0.35).sum(axis=1),
            "cell_median3_ema": ema(causal_median3(case.observed_cells), case.sample_rate, 0.5).sum(axis=1),
            "cell_kalman": scalar_kalman(
                case.observed_cells, cell_variance * 0.012, cell_variance
            ).sum(axis=1),
        }
        methods = {**total_methods, **cell_methods}
        for method, estimate in methods.items():
            pipeline = "per_cell_then_sum" if method.startswith("cell_") else "sum_then_filter"
            rows.append(
                {
                    "case": case.name,
                    "load_type": load_type,
                    "pipeline": pipeline,
                    "method": method,
                    **force_metrics(case, estimate),
                }
            )
        plotted[load_type] = (case, {"truth": truth_total, **methods})
    plot_force(plotted)
    plot_force_all_algorithms(plotted)
    plot_pareto(rows)
    return rows, plotted


def plot_force(plotted: dict[str, tuple[ForceCase, dict[str, np.ndarray]]]) -> None:
    selected = ("truth", "raw", "median3_ema", "iir2_0.5hz", "kalman_fast", "cell_ema_0.35s")
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for axis, load_type in zip(axes, ("local", "global")):
        case, methods = plotted[load_type]
        for method in selected:
            axis.plot(case.time, methods[method], label=method, linewidth=1.2 if method != "truth" else 2.2)
        axis.set_title(f"{load_type.title()} loading")
        axis.set_ylabel("Total force (simulated units)")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    figure.savefig(OUTPUT_DIR / "force_stability_comparison.png", dpi=180)
    plt.close(figure)


def plot_force_all_algorithms(plotted: dict[str, tuple[ForceCase, dict[str, np.ndarray]]]) -> None:
    methods = (
        "raw",
        "ema_0.35s",
        "ema_0.8s",
        "median3_ema",
        "iir1_0.5hz",
        "iir2_0.5hz",
        "kalman_fast",
        "kalman_stable",
        "cell_ema_0.35s",
        "cell_median3_ema",
        "cell_kalman",
    )
    titles = {
        "raw": "Raw",
        "ema_0.35s": "EMA 0.35 s",
        "ema_0.8s": "EMA 0.8 s",
        "median3_ema": "Median-3 + EMA",
        "iir1_0.5hz": "1st-order IIR 0.5 Hz",
        "iir2_0.5hz": "2nd-order IIR 0.5 Hz",
        "kalman_fast": "Kalman, fast",
        "kalman_stable": "Kalman, stable",
        "cell_ema_0.35s": "Per-cell EMA 0.35 s",
        "cell_median3_ema": "Per-cell Median-3 + EMA",
        "cell_kalman": "Per-cell Kalman",
    }
    figure, axes = plt.subplots(
        len(methods),
        2,
        figsize=(15, 25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for column, load_type in enumerate(("local", "global")):
        case, values = plotted[load_type]
        for row, method in enumerate(methods):
            axis = axes[row, column]
            axis.plot(case.time, values["truth"], color="black", linewidth=1.4, label="Truth")
            axis.plot(case.time, values[method], color="tab:blue", linewidth=0.9, label="Estimate")
            axis.set_title(f"{titles[method]} - {load_type}", fontsize=9)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel("Force")
            if row == len(methods) - 1:
                axis.set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    figure.suptitle("All causal force-stability algorithms", fontsize=15)
    figure.savefig(OUTPUT_DIR / "force_stability_all_algorithms.png", dpi=180)
    plt.close(figure)


def plot_pareto(rows: list[dict[str, object]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    markers = {"local": "o", "global": "s"}
    for row in rows:
        axis.scatter(
            row["mean_settling_s"],
            row["steady_std"],
            marker=markers[str(row["load_type"])],
            s=45,
            alpha=0.75,
        )
        axis.annotate(str(row["method"]), (row["mean_settling_s"], row["steady_std"]), fontsize=6, alpha=0.8)
    axis.set_xlabel("Mean settling time (s)")
    axis.set_ylabel("Steady-state error standard deviation")
    axis.set_title("Stability-response trade-off")
    axis.grid(alpha=0.25)
    figure.savefig(OUTPUT_DIR / "force_stability_pareto.png", dpi=180)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(membrane_rows: list[dict[str, object]], force_rows: list[dict[str, object]]) -> None:
    print("Membrane mean RMSE by method:")
    for method in sorted({str(row["method"]) for row in membrane_rows}):
        values = [float(row["rmse"]) for row in membrane_rows if row["method"] == method]
        print(f"  {method:20s} {np.mean(values):.4f}")
    print("Force mean steady std / settling by method:")
    for method in sorted({str(row["method"]) for row in force_rows}):
        selected = [row for row in force_rows if row["method"] == method]
        print(
            f"  {method:20s} std={np.mean([float(row['steady_std']) for row in selected]):.4f} "
            f"settling={np.mean([float(row['mean_settling_s']) for row in selected]):.3f}s"
        )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    membrane_rows = run_membrane_study()
    force_rows, _ = run_force_study()
    write_csv(DATA_DIR / "membrane_crosstalk_metrics.csv", membrane_rows)
    write_csv(DATA_DIR / "force_stability_metrics.csv", force_rows)
    print_summary(membrane_rows, force_rows)


if __name__ == "__main__":
    main()
