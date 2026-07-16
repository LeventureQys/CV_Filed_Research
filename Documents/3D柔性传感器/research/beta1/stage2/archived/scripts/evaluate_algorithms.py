"""
evaluate_algorithms.py — Evaluate multiple denoising algorithms on synthetic data
using the metrics defined in 评估指标设计文档.md.

Algorithms compared:
  0. Raw (no processing)
  1. Stat-based (Stage1 approach, k_sigma=3.0)
  2. Stat-based (k_sigma=2.0, more aggressive)
  3. Temporal median filter (window=3)
  4. Gaussian spatial filter (5x5)
  5. Bilateral spatial filter (preserves edges)
  6. Stat + temporal smoothing (hybrid)
  7. Stat + spatial smoothing (hybrid)
  8. Non-local means (if scikit-image available)

Usage:
    python scripts/evaluate_algorithms.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "..", "src"))

from alg.noise_suppressor import Analyzer, Processor

DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS, COLS = 64, 64


# ---------------------------------------------------------------------------
# Metrics (see 评估指标设计文档.md)
# ---------------------------------------------------------------------------

def compute_rmse(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x_pred - x_true) ** 2)))


def compute_snr(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    mse = np.mean((x_pred - x_true) ** 2)
    var_true = np.var(x_true)
    if var_true < 1e-12 or mse < 1e-12:
        return 0.0
    return float(10.0 * np.log10(var_true / mse))


def compute_bnsr(x_pred: np.ndarray, x_true: np.ndarray,
                 x_input: np.ndarray) -> float:
    """Background Noise Suppression Ratio."""
    bg_mask = x_true < 1e-6
    if bg_mask.sum() == 0:
        return 0.0
    input_noise = x_input[bg_mask] - x_true[bg_mask]
    output_noise = x_pred[bg_mask] - x_true[bg_mask]
    var_in = np.var(input_noise)
    var_out = np.var(output_noise)
    if var_out < 1e-12 or var_in < 1e-12:
        return 0.0
    return float(10.0 * np.log10(var_in / var_out))


def compute_srr(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    """Signal Retention Rate (on stress regions)."""
    stress_mask = x_true >= 1e-6
    if stress_mask.sum() == 0:
        return 1.0
    pred_stress = np.abs(x_pred[stress_mask]).mean()
    true_stress = np.abs(x_true[stress_mask]).mean()
    if true_stress < 1e-12:
        return 1.0
    return float(pred_stress / true_stress)


def compute_ssim(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    """Simplified SSIM for a single 64x64 frame."""
    def ssim_single(pred_2d, true_2d):
        mu_x = true_2d.mean()
        mu_y = pred_2d.mean()
        sigma_x_sq = true_2d.var()
        sigma_y_sq = pred_2d.var()
        sigma_xy = np.mean((true_2d - mu_x) * (pred_2d - mu_y))

        C1, C2 = 1e-4, 1e-4
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        if denominator < 1e-12:
            return 0.0
        return float(numerator / denominator)

    # Average SSIM across all frames
    ssim_vals = []
    for i in range(x_pred.shape[0]):
        pred_2d = x_pred[i].reshape(ROWS, COLS)
        true_2d = x_true[i].reshape(ROWS, COLS)
        ssim_vals.append(ssim_single(pred_2d, true_2d))
    return float(np.mean(ssim_vals))


def compute_centroid_shift(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    """Weighted centroid shift (in pixels)."""
    def centroid(frame_2d):
        ys, xs = np.mgrid[0:ROWS, 0:COLS]
        total = frame_2d.sum()
        if total < 1e-6:
            return ROWS / 2, COLS / 2
        cx = (xs * frame_2d).sum() / total
        cy = (ys * frame_2d).sum() / total
        return cx, cy

    shifts = []
    for i in range(x_pred.shape[0]):
        pred_2d = x_pred[i].reshape(ROWS, COLS)
        true_2d = x_true[i].reshape(ROWS, COLS)
        cx_p, cy_p = centroid(pred_2d)
        cx_t, cy_t = centroid(true_2d)
        shift = np.sqrt((cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2)
        shifts.append(shift)

    return float(np.mean(shifts))


def compute_contact_area_retention(x_pred: np.ndarray, x_true: np.ndarray,
                                   threshold_ratio: float = 0.1) -> float:
    """Contact area retention ratio."""
    ratios = []
    for i in range(x_pred.shape[0]):
        pred_2d = x_pred[i].reshape(ROWS, COLS)
        true_2d = x_true[i].reshape(ROWS, COLS)
        thr_pred = pred_2d.max() * threshold_ratio
        thr_true = true_2d.max() * threshold_ratio
        area_pred = (pred_2d >= thr_pred).sum()
        area_true = (true_2d >= thr_true).sum()
        if area_true == 0:
            ratios.append(1.0)
        else:
            ratios.append(area_pred / area_true)
    return float(np.mean(ratios))


def compute_temp_smoothness(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    """Temporal smoothness ratio (ideal = 1.0)."""
    if x_pred.shape[0] < 2:
        return 1.0
    pred_diffs = np.abs(np.diff(x_pred, axis=0)).mean()
    true_diffs = np.abs(np.diff(x_true, axis=0)).mean()
    if true_diffs < 1e-12:
        return 1.0
    return float(pred_diffs / true_diffs)


def compute_composite_score(metrics: dict) -> float:
    """Weighted composite score from 评估指标设计文档.md."""
    w1, w2, w3, w4 = 0.25, 0.25, 0.30, 0.20

    # Normalize SNR: assume typical range [-10, 40] dB
    snr = metrics.get("snr", 0)
    snr_norm = max(0.0, min(1.0, (snr + 10.0) / 50.0))

    srr = min(metrics.get("srr", 0), 1.0)
    ssim_val = max(0.0, metrics.get("ssim", 0))
    centroid = metrics.get("centroid_shift", 32.0)
    centroid_score = max(0.0, 1.0 - centroid / 32.0)

    return w1 * snr_norm + w2 * srr + w3 * ssim_val + w4 * centroid_score


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_one(noisy_frames: np.ndarray, gt_frames: np.ndarray,
                 algo_name: str, model=None, apply_fn=None) -> dict:
    """Run one algorithm and compute all metrics."""
    t0 = time.perf_counter()
    if apply_fn is not None:
        out_frames = apply_fn(noisy_frames, model)
    else:
        out_frames = noisy_frames.copy()
    elapsed_s = time.perf_counter() - t0

    out_frames = np.asarray(out_frames, dtype=np.float64)

    metrics = {
        "algo": algo_name,
        "rmse": compute_rmse(out_frames, gt_frames),
        "snr": compute_snr(out_frames, gt_frames),
        "bnsr": compute_bnsr(out_frames, gt_frames, noisy_frames),
        "srr": compute_srr(out_frames, gt_frames),
        "ssim": compute_ssim(out_frames, gt_frames),
        "centroid_shift": compute_centroid_shift(out_frames, gt_frames),
        "area_retention": compute_contact_area_retention(out_frames, gt_frames),
        "temp_smoothness": compute_temp_smoothness(out_frames, gt_frames),
        "composite": 0.0,
        "process_time_ms": elapsed_s / noisy_frames.shape[0] * 1000.0,
    }
    metrics["composite"] = compute_composite_score(metrics)
    return metrics


def print_metrics_table(all_metrics: list[dict], dataset_label: str):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  {dataset_label}")
    print(f"{'=' * 90}")
    header = f"{'Algo':<18s} {'RMSE':>8s} {'SNR(dB)':>8s} {'BNSR':>8s} {'SRR':>7s} {'SSIM':>7s} {'Centr':>7s} {'AreaRet':>7s} {'CS':>6s}"
    print(header)
    print("-" * 90)
    for m in all_metrics:
        line = (
            f"{m['algo']:<18s}"
            f" {m['rmse']:>8.2f}"
            f" {m['snr']:>8.2f}"
            f" {m['bnsr']:>8.2f}"
            f" {m['srr']:>7.4f}"
            f" {m['ssim']:>7.4f}"
            f" {m['centroid_shift']:>7.2f}"
            f" {m['area_retention']:>7.4f}"
            f" {m['composite']:>6.4f}"
        )
        print(line)
    print("-" * 90)


def save_results(all_results: dict, out_path: str):
    """Save all results to CSV (rows=datasets, cols=algo+metric combinations)."""
    rows = []
    for ds_label, metrics_list in all_results.items():
        row = {"dataset": ds_label}
        for m in metrics_list:
            prefix = m["algo"]
            for key in ["rmse", "snr", "bnsr", "srr", "ssim",
                        "centroid_shift", "area_retention",
                        "temp_smoothness", "composite"]:
                row[f"{prefix}_{key}"] = m.get(key, 0)
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[results] saved to {out_path}")


# ---------------------------------------------------------------------------
# Algorithm definitions
# ---------------------------------------------------------------------------

def algo_raw(frames, model):
    return frames.copy()


def algo_stat_k3(frames, model):
    if model is None:
        return frames.copy()
    proc = Processor(model)
    return proc.process_batch(frames)


def algo_stat_k2(frames, model):
    if model is None:
        return frames.copy()
    proc = Processor(model, noise_gate_decay=0.0)
    return proc.process_batch(frames)


def algo_temporal_median(frames, model):
    """Per-channel temporal median filter (window=3)."""
    from scipy.ndimage import median_filter
    return median_filter(frames, size=(3, 1))


def algo_gaussian_spatial(frames, model):
    """Per-frame 2D Gaussian spatial filter (sigma=1.5, 5x5 kernel)."""
    from scipy.ndimage import gaussian_filter
    out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        out[i] = gaussian_filter(out[i], sigma=1.5, mode="reflect")
    return out.reshape(frames.shape)


def algo_bilateral_spatial(frames, model):
    """Per-frame bilateral filter (edge-preserving spatial smoothing)."""
    try:
        from skimage.restoration import denoise_bilateral
    except ImportError:
        print("  [warn] scikit-image not available, falling back to gaussian")
        return algo_gaussian_spatial(frames, model)

    out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        out[i] = denoise_bilateral(out[i], sigma_color=10, sigma_spatial=2,
                                   mode="reflect", channel_axis=None)
    return out.reshape(frames.shape)


def algo_stat_temporal_hybrid(frames, model):
    """Stat-k3 then temporal median (hybrid)."""
    frames_s = algo_stat_k3(frames, model)
    return algo_temporal_median(frames_s, None)


def algo_stat_spatial_hybrid(frames, model):
    """Stat-k3 then gaussian spatial (hybrid)."""
    frames_s = algo_stat_k3(frames, model)
    return algo_gaussian_spatial(frames_s, None)


def algo_non_local_means(frames, model):
    """Non-local means denoising (if available)."""
    try:
        from skimage.restoration import denoise_nl_means
    except ImportError:
        print("  [warn] NL-means not available, falling back to bilateral")
        return algo_bilateral_spatial(frames, model)

    out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        out[i] = denoise_nl_means(out[i], h=10, sigma=5,
                                  patch_size=5, patch_distance=7,
                                  mode="reflect", channel_axis=None)
    return out.reshape(frames.shape)


ALGORITHMS = {
    "raw": algo_raw,
    "stat_k3": algo_stat_k3,
    "stat_k2": algo_stat_k2,
    "temporal_median3": algo_temporal_median,
    "gaussian_spatial": algo_gaussian_spatial,
    "bilateral_spatial": algo_bilateral_spatial,
    "stat_k3+temporal": algo_stat_temporal_hybrid,
    "stat_k3+spatial": algo_stat_spatial_hybrid,
    "nl_means": algo_non_local_means,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_synthetic(csv_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load synthetic data: noisy from CSV, ground truth from NPZ."""
    from alg.data_loader import load_csv
    noisy, meta = load_csv(csv_path)
    gt_path = csv_path.replace(".csv", "_gt.npz")
    with np.load(gt_path) as data:
        gt = data["ground_truth"]
    return noisy, gt, meta


def main():
    # Find all CSV files in synthetic data dir
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"Found {len(csv_files)} synthetic datasets.\n")

    all_results = {}

    for csv_name in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_name)
        label = csv_name.replace(".csv", "")

        print(f"Loading {csv_name}...")
        noisy, gt, meta = load_synthetic(csv_path)
        n_frames, n_channels = noisy.shape

        # Train Analyzer on first N frames (assume they contain background)
        n_bg = min(10, n_frames // 2)
        bg_frames = noisy[:n_bg]

        try:
            analyzer = Analyzer(k_sigma=3.0)
            model = analyzer.analyze(bg_frames, meta.get("sample_freq_hz", 2.0))
        except Exception as e:
            print(f"  Analyzer failed: {e}")
            model = None

        dataset_metrics = []
        for algo_name, apply_fn in ALGORITHMS.items():
            metrics = evaluate_one(noisy, gt, algo_name, model, apply_fn)
            dataset_metrics.append(metrics)

        all_results[label] = dataset_metrics
        print_metrics_table(dataset_metrics, label)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "algorithm_comparison.csv")
    save_results(all_results, out_path)

    # Find best composite score per dataset
    print(f"\n{'=' * 90}")
    print("  BEST ALGORITHM PER DATASET (by Composite Score)")
    print(f"{'=' * 90}")
    for ds_label, metrics_list in all_results.items():
        best = max(metrics_list, key=lambda m: m["composite"])
        print(f"  {ds_label:<30s} → {best['algo']:<18s} (CS={best['composite']:.4f}, SNR={best['snr']:.1f}dB, SSIM={best['ssim']:.4f})")


if __name__ == "__main__":
    main()
