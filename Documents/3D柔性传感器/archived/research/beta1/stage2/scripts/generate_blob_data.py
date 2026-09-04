"""
generate_blob_data.py — Generate 64x64 blob/cluster pressure data with
per-channel background white noise (glove-like pattern).

Two-phase structure (critical for noise model learning):
  Phase 1 (bg_frames): baseline + white noise only (no pressure)
  Phase 2 (sig_frames): baseline + pressure blobs + white noise

Ground truth: Phase 1 = all zeros, Phase 2 = pressure patch pattern.

This matches the real device workflow: sample background first, then apply
pressure. Enables proper noise model learning (no signal contamination).
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "synthetic_blob")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS, COLS = 64, 64
N_CHANNELS = ROWS * COLS
ADC_MAX = 4095


def gaussian_blob(rows, cols, cx, cy, sigma, amplitude):
    ys, xs = np.mgrid[0:rows, 0:cols]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return amplitude * np.exp(-dist2 / (2 * sigma ** 2))


def make_pressure_map(blobs, rows=ROWS, cols=COLS):
    canvas = np.zeros((rows, cols), dtype=np.float64)
    for b in blobs:
        canvas += gaussian_blob(rows, cols, b["cx"], b["cy"], b["sigma"], b["amplitude"])
    return canvas


def generate_dataset(label, n_bg, n_sig, blobs, noise_std, baseline_mean, baseline_spread, desc):
    """
    n_bg  : number of pure-background frames (baseline + noise only)
    n_sig : number of signal frames (baseline + pressure + noise)
    """
    np.random.seed(hash(label) % (2 ** 31))

    baselines = np.random.normal(baseline_mean, baseline_spread, size=N_CHANNELS)
    baselines[baselines < 2.0] = 2.0

    pressure_map = make_pressure_map(blobs)

    # Background frames: baselines + noise, gt = zeros
    bg_noise = np.random.normal(0, noise_std, size=(n_bg, N_CHANNELS))
    bg_noisy = bg_noise + baselines[np.newaxis, :]
    bg_gt = np.zeros((n_bg, N_CHANNELS), dtype=np.float64)

    # Signal frames: baselines + pressure + noise, gt = pressure
    sig_noise = np.random.normal(0, noise_std, size=(n_sig, N_CHANNELS))
    sig_noisy = sig_noise + baselines[np.newaxis, :] + pressure_map.ravel()[np.newaxis, :]
    sig_gt = np.tile(pressure_map.ravel()[np.newaxis, :], (n_sig, 1))

    noisy = np.concatenate([bg_noisy, sig_noisy], axis=0)
    gt = np.concatenate([bg_gt, sig_gt], axis=0)
    noisy = np.clip(noisy, 0, ADC_MAX)
    gt = np.clip(gt, 0, ADC_MAX)

    return noisy, gt, baselines, n_bg


def save_dataset(noisy, gt, baselines, filepath, label, desc, freq_hz=2.0):
    N, C = noisy.shape

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["##设备信息"])
        writer.writerow(["设备名称", f"Simulated Blob 64x64 - {label}"])
        writer.writerow(["行数", str(ROWS)])
        writer.writerow(["列数", str(COLS)])
        writer.writerow(["数据点数", str(N)])
        writer.writerow(["显示模式", "ADC"])
        writer.writerow(["下位机基准单位", "raw"])
        writer.writerow(["上位机展示单位", "raw"])
        writer.writerow(["录制频率", "raw"])
        writer.writerow(["录制时间", "2026年07月11日_14时00分00秒"])
        writer.writerow(["噪声描述", desc])
        writer.writerow(["##数据开始"])
        col_header = ["时间戳"] + [f"通道{i+1}" for i in range(C)]
        writer.writerow(col_header)

        ts_base = datetime(2026, 7, 11, 14, 0, 0)
        interval_s = 1.0 / max(freq_hz, 1.0)
        for i in range(N):
            ts = ts_base + __import__("datetime").timedelta(seconds=i * interval_s)
            ts_str = f"{ts.hour}时{ts.minute}分{ts.second}秒.{ts.microsecond // 1000:03d}"
            vals = [int(v) for v in noisy[i]]
            writer.writerow([ts_str] + vals)

    gt_path = filepath.replace(".csv", "_gt.npz")
    np.savez_compressed(gt_path, ground_truth=gt, noisy=noisy, baselines=baselines)
    print(f"  CSV: {os.path.basename(filepath)}  NPZ: {os.path.basename(gt_path)}")


def make_datasets():
    datasets = []

    datasets.append({
        "label": "blob_single_center",
        "desc": "single_center_blob, bg_base~15adc, noise_std=3, n_bg=15",
        "bg_frames": 15, "sig_frames": 15,
        "blobs": [{"cx": 32, "cy": 32, "sigma": 10, "amplitude": 500}],
        "noise_std": 3.0, "baseline_mean": 15.0, "baseline_spread": 5.0,
    })

    datasets.append({
        "label": "blob_two",
        "desc": "two_blobs, bg_base~15adc, noise_std=5, n_bg=15",
        "bg_frames": 15, "sig_frames": 15,
        "blobs": [
            {"cx": 18, "cy": 20, "sigma": 7, "amplitude": 400},
            {"cx": 46, "cy": 44, "sigma": 7, "amplitude": 350},
        ],
        "noise_std": 5.0, "baseline_mean": 15.0, "baseline_spread": 5.0,
    })

    datasets.append({
        "label": "blob_five_palm",
        "desc": "five_blobs_palm, bg_base~15adc, noise_std=5, n_bg=15",
        "bg_frames": 15, "sig_frames": 15,
        "blobs": [
            {"cx": 16, "cy": 20, "sigma": 5, "amplitude": 300},
            {"cx": 48, "cy": 20, "sigma": 5, "amplitude": 280},
            {"cx": 16, "cy": 44, "sigma": 5, "amplitude": 320},
            {"cx": 48, "cy": 44, "sigma": 5, "amplitude": 290},
            {"cx": 32, "cy": 32, "sigma": 7, "amplitude": 400},
        ],
        "noise_std": 5.0, "baseline_mean": 15.0, "baseline_spread": 5.0,
    })

    datasets.append({
        "label": "blob_large_area",
        "desc": "large_area_press, bg_base~20adc, noise_std=8, n_bg=15",
        "bg_frames": 15, "sig_frames": 15,
        "blobs": [
            {"cx": 32, "cy": 32, "sigma": 14, "amplitude": 700},
            {"cx": 24, "cy": 24, "sigma": 9, "amplitude": 250},
            {"cx": 40, "cy": 40, "sigma": 9, "amplitude": 250},
        ],
        "noise_std": 8.0, "baseline_mean": 20.0, "baseline_spread": 8.0,
    })

    datasets.append({
        "label": "blob_edge",
        "desc": "edge_blobs, bg_base~12adc, noise_std=3, n_bg=15",
        "bg_frames": 15, "sig_frames": 15,
        "blobs": [
            {"cx": 7, "cy": 7, "sigma": 5, "amplitude": 350},
            {"cx": 56, "cy": 56, "sigma": 5, "amplitude": 350},
        ],
        "noise_std": 3.0, "baseline_mean": 12.0, "baseline_spread": 4.0,
    })

    datasets.append({
        "label": "blob_weak_high_noise",
        "desc": "weak_blobs, bg_base~20adc, noise_std=12, n_bg=15",
        "bg_frames": 15, "sig_frames": 15,
        "blobs": [
            {"cx": 22, "cy": 32, "sigma": 5, "amplitude": 120},
            {"cx": 42, "cy": 32, "sigma": 5, "amplitude": 100},
        ],
        "noise_std": 12.0, "baseline_mean": 20.0, "baseline_spread": 8.0,
    })

    return datasets


def main():
    datasets = make_datasets()
    print(f"Generating {len(datasets)} blob datasets (bg_frames + sig_frames)...\n")

    for i, ds in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] {ds['label']}: {ds['desc']}")

        noisy, gt, baselines, n_bg = generate_dataset(
            ds["label"], ds["bg_frames"], ds["sig_frames"],
            ds["blobs"], ds["noise_std"],
            ds["baseline_mean"], ds["baseline_spread"],
            ds["desc"],
        )

        filepath = os.path.join(OUTPUT_DIR, f"{ds['label']}.csv")
        save_dataset(noisy, gt, baselines, filepath, ds["label"], ds["desc"])

        gt_bg = gt[:n_bg].sum(axis=1).mean()
        gt_sig = gt[n_bg:].sum(axis=1).mean()
        noisy_bg = noisy[:n_bg].sum(axis=1).mean()
        noisy_sig = noisy[n_bg:].sum(axis=1).mean()
        print(f"  bg: gt_mean={gt_bg:.0f}, noisy_mean={noisy_bg:.0f} | "
              f"sig: gt_mean={gt_sig:.0f}, noisy_mean={noisy_sig:.0f}")
        print(f"  baseline_range=[{baselines.min():.1f}, {baselines.max():.1f}]")

    print(f"\nDone. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
