"""
algo_experiments.py — Improved denoising algorithms for 64x64 tactile sensor array.

Based on evaluation results from evaluate_algorithms.py, the Stage1 stat-based
approach fails completely for 64x64 membrane (background ADC=0 → baseline=0 →
all signals treated as noise). This module implements and tests improved approaches.

Key insights from evaluation:
1. temporal_median3 works well for moderate noise (CS > 0.93)
2. Gaussian spatial smoothing works well for large-area contacts (CS > 0.97)
3. Low SNR scenarios (SNR < 5dB) need more sophisticated methods
4. 64x64 membrane has NO background noise, so the "gate" approach is entirely wrong
5. The real noise comes from: (a) per-channel read noise, (b) temporal jitter,
   (c) channel crosstalk — these are better handled by spatial/temporal smoothing
   than by baseline thresholding

Strategies to try:
  A. Adaptive spatial filter (sigma based on local gradient)
  B. Temporal + spatial hybrid (median temporal, then guided spatial)
  C. Non-local means with adaptive parameters
  D. Edge-preserving total variation denoising
  E. Wavelet soft-thresholding (for comparison)
  F. Adaptive Wiener filter in spatial domain
  G. Multi-frame averaging + spatial smoothing (for static scenes)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "..", "src"))

from alg.noise_suppressor import Analyzer, Processor
from evaluate_algorithms import (
    compute_rmse, compute_snr, compute_bnsr, compute_srr,
    compute_ssim, compute_centroid_shift, compute_contact_area_retention,
    compute_temp_smoothness, compute_composite_score,
    load_synthetic, print_metrics_table, save_results, ROWS, COLS,
)

DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Improved algorithms
# ---------------------------------------------------------------------------

def algo_temporal_median5(frames, model):
    """Per-channel temporal median filter (window=5)."""
    from scipy.ndimage import median_filter
    return median_filter(frames, size=(5, 1))


def algo_adaptive_spatial(frames, model):
    """Adaptive spatial filter: use local variance to control blur strength.

    Low variance → more blur (homogeneous area, likely noise)
    High variance → less blur (edge/contact area, preserve detail)
    """
    from scipy.ndimage import gaussian_filter, uniform_filter

    out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        frame = out[i]

        # Local variance estimate
        local_mean = uniform_filter(frame, size=5)
        local_sq_mean = uniform_filter(frame ** 2, size=5)
        local_var = np.maximum(0, local_sq_mean - local_mean ** 2)

        # Noise variance estimate (assume homogeneous regions)
        noise_var = np.median(local_var[local_var > 0])

        # Adaptive sigma: more smoothing where local_var ≈ noise_var
        sigma_map = np.clip(noise_var / (local_var + 1e-6) * 2.0, 0.5, 4.0)

        # Apply Gaussian filter with spatially varying sigma (approximate)
        # Use fixed sigma as approximation
        sigma_eff = float(np.mean(sigma_map))
        out[i] = gaussian_filter(frame, sigma=sigma_eff, mode="reflect")
    return out.reshape(frames.shape)


def algo_then_adaptive_spatial(frames, model):
    """Temporal median (w=3) then adaptive spatial."""
    frames_t = algo_temporal_median5(frames, model)
    return algo_adaptive_spatial(frames_t, model)


def algo_then_gaussian_spatial(frames, model):
    """Temporal median (w=3) then gaussian spatial."""
    from scipy.ndimage import median_filter
    from scipy.ndimage import gaussian_filter

    frames_t = median_filter(frames, size=(3, 1))
    out = frames_t.reshape(frames_t.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        out[i] = gaussian_filter(out[i], sigma=1.5, mode="reflect")
    return out.reshape(frames.shape)


def algo_bilateral_strong(frames, model):
    """Stronger bilateral filtering with automatic sigma estimation."""
    try:
        from skimage.restoration import denoise_bilateral
        out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
        for i in range(out.shape[0]):
            frame = out[i]
            sigma_est = np.std(frame[frame > 0])
            if sigma_est < 0.01:
                sigma_est = 5.0
            # More aggressive for low SNR, gentler for high SNR
            sigma_color = max(5, sigma_est * 0.5)
            out[i] = denoise_bilateral(frame, sigma_color=sigma_color,
                                       sigma_spatial=2, mode="reflect",
                                       channel_axis=None)
        return out.reshape(frames.shape)
    except ImportError:
        return algo_adaptive_spatial(frames, model)


def algo_wavelet_soft(frames, model):
    """Wavelet soft-thresholding denoising."""
    try:
        import pywt
    except ImportError:
        print("  [warn] PyWavelets not available")
        return frames.copy()

    out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        frame = out[i]
        # Estimate noise std using median absolute deviation of finest scale
        coeffs = pywt.wavedec2(frame, "db4", level=3)
        sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
        if sigma < 0.01:
            sigma = 1.0
        # Universal threshold
        threshold = sigma * np.sqrt(2 * np.log(frame.size))
        # Soft threshold
        coeffs_th = [coeffs[0]]
        for detail_level in coeffs[1:]:
            detail_th = tuple(
                pywt.threshold(d, threshold, mode="soft") for d in detail_level
            )
            coeffs_th.append(detail_th)
        out[i] = pywt.waverec2(coeffs_th, "db4")[:ROWS, :COLS]
    return out.reshape(frames.shape)


def algo_guided_filter(frames, model):
    """Guided filter using temporal average as guidance."""
    try:
        from skimage.filters import rank
        from skimage.morphology import disk
    except ImportError:
        return algo_then_gaussian_spatial(frames, model)

    # Compute temporal average as guide
    guide = frames.mean(axis=0).reshape(ROWS, COLS)
    guide = (guide / guide.max() * 255).astype(np.uint8) if guide.max() > 0 else guide.astype(np.uint8)

    out = frames.copy().reshape(frames.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        frame_u8 = (out[i] / max(out[i].max(), 1) * 255).astype(np.uint8)
        filtered = rank.mean_bilateral(frame_u8, disk(5), s0=10, s1=10)
        out[i] = filtered.astype(np.float64) / 255.0 * out[i].max()
    return out.reshape(frames.shape)


def algo_multi_frame_avg(frames, model):
    """Multi-frame averaging (sliding window of 3)."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(frames, size=3, axis=0)


def algo_then_then(frames, model):
    """Multi-frame avg → temporal median → spatial gaussian."""
    from scipy.ndimage import median_filter, gaussian_filter, uniform_filter1d

    frames_avg = uniform_filter1d(frames, size=3, axis=0)
    frames_m = median_filter(frames_avg, size=(3, 1))
    out = frames_m.reshape(frames_m.shape[0], ROWS, COLS)
    for i in range(out.shape[0]):
        out[i] = gaussian_filter(out[i], sigma=1.0, mode="reflect")
    return out.reshape(frames.shape)


IMPROVED_ALGORITHMS = {
    "temporal_median5": algo_temporal_median5,
    "adaptive_spatial": algo_adaptive_spatial,
    "temporal5+adaptive": algo_then_adaptive_spatial,
    "temporal3+gaussian": algo_then_gaussian_spatial,
    "bilateral_strong": algo_bilateral_strong,
    "wavelet_soft": algo_wavelet_soft,
    "multi_frame_avg3": algo_multi_frame_avg,
    "avg3+median3+gauss": algo_then_then,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    print(f"Evaluating {len(IMPROVED_ALGORITHMS)} improved algorithms on {len(csv_files)} datasets...\n")

    all_results = {}

    for csv_name in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_name)
        label = csv_name.replace(".csv", "")

        print(f"Loading {csv_name}...")
        noisy, gt, meta = load_synthetic(csv_path)

        dataset_metrics = []
        for algo_name, apply_fn in IMPROVED_ALGORITHMS.items():
            metrics = _evaluate_one(noisy, gt, algo_name, None, apply_fn)
            dataset_metrics.append(metrics)

        all_results[label] = dataset_metrics
        print_metrics_table(dataset_metrics, f"IMPROVED — {label}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "algorithm_improved.csv")
    save_results(all_results, out_path)

    print(f"\n{'=' * 90}")
    print("  BEST IMPROVED ALGORITHM PER DATASET (by Composite Score)")
    print(f"{'=' * 90}")
    for ds_label, metrics_list in all_results.items():
        best = max(metrics_list, key=lambda m: m["composite"])
        print(f"  {ds_label:<30s} → {best['algo']:<18s} (CS={best['composite']:.4f}, SNR={best['snr']:.1f}dB, SSIM={best['ssim']:.4f})")


def _evaluate_one(noisy_frames, gt_frames, algo_name, model, apply_fn):
    t0 = time.perf_counter()
    out_frames = apply_fn(noisy_frames, model)
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


if __name__ == "__main__":
    main()
