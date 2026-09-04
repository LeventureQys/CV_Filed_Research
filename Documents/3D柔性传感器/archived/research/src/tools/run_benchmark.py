"""
run_benchmark.py — Compare noise suppression algorithms on all recording data.

Algorithms:
  0. Raw (no suppression) — baseline
  1. Moving Average (window=3) — simple smooth
  2. Median Filter (window=3) — robust smooth
  3. Stat-based (this paper, k_sigma=3.0)
  4. Stat-based (this paper, k_sigma=2.0) — more aggressive

Metrics per algorithm per CSV:
  - background_total_mean:    mean sum of channels on background frames
  - background_nz_mean:       mean non-zero channels on background
  - stress_total_mean:        mean sum on stress frames
  - stress_peak_mean:         mean peak value on stress frames
  - stress_total_retention:   processed / original total (stress only)
  - stress_peak_retention:    processed / original peak (stress only)
  - suppression_db:           20*log10(orig_total / proc_total) on background
  - rtf:                      real-time factor
  - avg_process_time_ms:      per frame

Outputs:
  - Console summary table
  - beta1/stats/benchmark_results.csv
  
Usage:
    python tools/run_benchmark.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alg.data_loader import load_dataset, load_csv, parse_csv_header, guess_sampling_freq
from alg.noise_suppressor import Analyzer, Processor, AnalyzerResult


ALGORITHMS = {
    "raw":           lambda bg, freq: (None, False),
    "moving_avg_3":  lambda bg, freq: (None, True),
    "median_3":      lambda bg, freq: (None, True),
    "stat_k3.0":     lambda bg, freq: build_stat_processor(bg, freq, 3.0, 0.0),
    "stat_k2.0":     lambda bg, freq: build_stat_processor(bg, freq, 2.0, 0.0),
    "stat_k3.0_d1":  lambda bg, freq: build_stat_processor(bg, freq, 3.0, 1.0),
}


def build_stat_processor(bg_frames, freq, k_sigma, decay):
    analyzer = Analyzer(k_sigma=k_sigma)
    model = analyzer.analyze(bg_frames, freq)
    proc = Processor(model, noise_gate_decay=decay)
    return (model, proc)


def apply_raw(frames, model, is_active):
    return frames.copy()


def apply_moving_avg_3(frames, model, is_active):
    out = frames.copy()
    for i in range(1, frames.shape[0] - 1):
        out[i] = (frames[i - 1] + frames[i] + frames[i + 1]) / 3.0
    return out


def apply_median_3(frames, model, is_active):
    from scipy.ndimage import median_filter
    return median_filter(frames, size=(3, 1))


def apply_stat_processor(frames, model, is_active):
    if model is None or not isinstance(model, AnalyzerResult):
        return frames.copy()
    proc = Processor(model, noise_gate_decay=is_active if isinstance(is_active, (int, float)) else 0.0)
    out = proc.process_batch(frames)
    return out


APPLY_FUNCS = {
    "raw": apply_raw,
    "moving_avg_3": apply_moving_avg_3,
    "median_3": apply_median_3,
    "stat_k3.0": apply_stat_processor,
    "stat_k2.0": apply_stat_processor,
    "stat_k3.0_d1": apply_stat_processor,
}


def identify_background_stress_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heuristic: frames with total sum > threshold are "stress"; rest are "background".

    For near-zero data (all ADC=0), fallback to first 30%/last 70% split.
    Returns (bg_indices, stress_indices) as arrays.
    """
    N = frames.shape[0]
    totals = frames.sum(axis=1)
    if totals.max() <= 0:
        split = max(1, N // 3)
        return np.arange(split), np.arange(split, N)

    median_total = np.median(totals)
    mad = np.median(np.abs(totals - median_total))
    threshold = median_total + max(2.0 * mad, median_total * 0.3)

    stress_mask = totals >= threshold
    bg_mask = ~stress_mask

    # Fallback if either partition is too small
    if stress_mask.sum() < 1 or bg_mask.sum() < 1:
        split = max(1, N // 3)
        bg = np.arange(split)
        stress = np.arange(split, N)
        if len(bg) == 0:
            bg = np.array([0])
        if len(stress) == 0:
            stress = np.array([N - 1])
        return bg, stress

    return np.where(bg_mask)[0], np.where(stress_mask)[0]


def benchmark_one(frames: np.ndarray, freq: float) -> dict:
    """Run all algorithms on one CSV and return metrics dict."""
    N, C = frames.shape
    bg_idx, stress_idx = identify_background_stress_frames(frames)

    frames_bg = frames[bg_idx]
    frames_stress = frames[stress_idx]

    results = {"csv_frames": N, "csv_channels": C, "freq_hz": freq}

    for algo_name in ALGORITHMS:
        model, is_active = ALGORITHMS[algo_name](frames_bg, freq)
        t_start = time.perf_counter()
        out_frames = APPLY_FUNCS[algo_name](frames, model, is_active)
        elapsed = time.perf_counter() - t_start

        out_bg = out_frames[bg_idx]
        out_stress = out_frames[stress_idx]
        orig_bg = frames[bg_idx]
        orig_stress = frames[stress_idx]

        def safe_mean(arr):
            if arr.size == 0:
                return 0.0
            v = arr.mean()
            return 0.0 if np.isnan(v) else float(v)

        total_bg_orig = safe_mean(orig_bg.sum(axis=1))
        total_bg_proc = safe_mean(out_bg.sum(axis=1))
        nz_bg_orig = int(safe_mean((orig_bg > 0).sum(axis=1)))
        nz_bg_proc = int(safe_mean((out_bg > 0).sum(axis=1)))

        total_stress_orig = safe_mean(orig_stress.sum(axis=1))
        total_stress_proc = safe_mean(out_stress.sum(axis=1))
        peak_stress_orig = safe_mean(orig_stress.max(axis=1))
        peak_stress_proc = safe_mean(out_stress.max(axis=1))

        # Retention ratios
        total_ret = total_stress_proc / max(total_stress_orig, 1)
        peak_ret = peak_stress_proc / max(peak_stress_orig, 1)

        # Suppression on background (dB)
        suppress_db = 0.0
        if total_bg_orig > 0 and total_bg_proc > 0:
            suppress_db = 20 * np.log10(total_bg_orig / total_bg_proc)

        # RTF
        frame_time_s = 1.0 / max(freq, 1.0) if freq > 0 else 0.001
        if frame_time_s > 0:
            rtf = elapsed / max(frame_time_s * N, 1e-9)
        else:
            rtf = 0.0
        per_frame_ms = elapsed / max(N, 1) * 1000.0

        results[f"{algo_name}_bg_total_mean"] = round(total_bg_proc, 2)
        results[f"{algo_name}_bg_nz_mean"] = nz_bg_proc
        results[f"{algo_name}_stress_total_mean"] = round(total_stress_proc, 2)
        results[f"{algo_name}_stress_peak_mean"] = round(peak_stress_proc, 2)
        results[f"{algo_name}_total_retention"] = round(total_ret, 4)
        results[f"{algo_name}_peak_retention"] = round(peak_ret, 4)
        results[f"{algo_name}_suppression_db"] = round(suppress_db, 2)
        results[f"{algo_name}_rtf"] = round(rtf, 4)
        results[f"{algo_name}_ms_per_frame"] = round(per_frame_ms, 4)

    return results


def print_summary_table(all_results: list[dict], csv_labels: list[str]):
    """Print a compact table for quick comparison."""
    algo_names = list(ALGORITHMS.keys())
    metrics_to_show = [
        ("bg_total_mean", "BgTotal", "8.1f"),
        ("bg_nz_mean", "BgNzCh", "6.0f"),
        ("stress_total_mean", "StrTotal", "9.1f"),
        ("suppression_db", "Sup(dB)", "7.2f"),
        ("total_retention", "T-Ret", "6.3f"),
        ("peak_retention", "P-Ret", "6.3f"),
        ("ms_per_frame", "ms/fr", "7.4f"),
    ]

    # Per-algo
    for algo in algo_names:
        print(f"\n── {algo} ──")
        header = f"{'CSV':<30s}"
        for _, label, _ in metrics_to_show:
            header += f" {label:>8s}"
        print(header)
        for i, res in enumerate(all_results):
            line = f"{csv_labels[i][:30]:<30s}"
            for key, _, fmt in metrics_to_show:
                val = res.get(f"{algo}_{key}", 0)
                line += f" {val:>8{fmt}}"
            print(line)


def save_csv(all_results: list[dict], csv_labels: list[str], out_path: str):
    """Save full benchmark results to CSV."""
    cols = list({"csv_frames", "csv_channels", "freq_hz"})
    for res in all_results:
        for k in res:
            if k not in cols:
                cols.append(k)
    cols.append("csv_label")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for label, res in zip(csv_labels, all_results):
            res["csv_label"] = label
            writer.writerow(res)

    print(f"\n[benchmark] saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run noise suppression benchmark on all CSV data")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "DataSet"))
    print(f"Scanning: {base_dir}")
    all_csv = []
    for root, dirs, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".csv"):
                all_csv.append(os.path.join(root, fn))
    all_csv.sort()

    print(f"Found {len(all_csv)} CSV files.")

    all_results = []
    csv_labels = []

    for path in all_csv:
        try:
            meta = parse_csv_header(path)
            label = f"{meta.get('设备名称','?')} {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
            print(f"\n[{len(all_results)+1}/{len(all_csv)}] {label}")

            frames, meta2 = load_csv(path, max_frames=500)  # cap for speed
            freq = meta2.get("sample_freq_hz", 0.0)
            if freq == 0:
                freq = guess_sampling_freq(path)

            res = benchmark_one(frames, freq)
            res["csv_path"] = path
            all_results.append(res)
            csv_labels.append(label)
            print(f"  frames={frames.shape[0]} channels={frames.shape[1]} freq={freq:.1f}Hz")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No benchmark results.")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print_summary_table(all_results, csv_labels)

    # Save CSV
    out_path = args.out
    if not out_path:
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stats"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "benchmark_results.csv")
    save_csv(all_results, csv_labels, out_path)


if __name__ == "__main__":
    main()
