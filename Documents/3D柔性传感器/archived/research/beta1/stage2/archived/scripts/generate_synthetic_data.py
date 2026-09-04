"""
generate_synthetic_data.py — Generate synthetic 64×64 tactile sensor data with
known ground truth and controllable white noise.

Output: CSV files matching the real device recording format (compatible with
data_loader.py from Stage1), plus NPZ files with ground truth arrays.

Usage:
    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROWS, COLS = 64, 64
N_CHANNELS = ROWS * COLS
ADC_MAX = 4095


def gaussian_patch(rows, cols, cx, cy, sigma, amplitude):
    """Generate a 2D Gaussian pressure patch."""
    ys, xs = np.mgrid[0:rows, 0:cols]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    patch = amplitude * np.exp(-dist2 / (2 * sigma ** 2))
    return patch


def generate_ground_truth_frames(
    n_frames: int,
    patches: list[dict],
    rows: int = ROWS,
    cols: int = COLS,
) -> np.ndarray:
    """Generate ground truth frames with given pressure patches.

    Parameters
    ----------
    n_frames : int
        Number of frames to generate.
    patches : list of dict
        Each dict: {cx, cy, sigma (pixels), amplitude (ADC)}.
        Patches are stable throughout the sequence (no temporal variation).

    Returns
    -------
    frames : ndarray, shape (N, ROWS*COLS)
    """
    patch_sum = np.zeros((rows, cols), dtype=np.float64)
    for p in patches:
        patch_sum += gaussian_patch(rows, cols, p["cx"], p["cy"], p["sigma"], p["amplitude"])
    patch_sum = np.clip(patch_sum, 0, ADC_MAX)
    frames_2d = np.tile(patch_sum[np.newaxis, :, :], (n_frames, 1, 1))
    return frames_2d.reshape(n_frames, rows * cols)


def add_white_noise(
    frames: np.ndarray,
    noise_std: float = 5.0,
    noise_floor: float = 0.5,
) -> np.ndarray:
    """Add per-channel white Gaussian noise.

    Parameters
    ----------
    frames : ndarray, shape (N, C)
        Ground truth frames.
    noise_std : float
        Standard deviation of additive white noise (ADC units).
    noise_floor : float
        Minimum noise floor per channel to prevent all-zero.

    Returns
    -------
    noisy : ndarray, shape (N, C)
    """
    noise = np.random.normal(0, max(noise_std, noise_floor), size=frames.shape)
    noisy = frames + noise
    noisy = np.clip(noisy, 0, ADC_MAX)
    return noisy


def add_baseline_drift(
    frames: np.ndarray,
    drift_amplitude: float = 3.0,
    n_base_points: int = 5,
) -> np.ndarray:
    """Add slow baseline drift (simulating temperature drift).

    Interpolates random drift points across the frame sequence.
    """
    N, C = frames.shape
    t = np.linspace(0, 1, N)
    drift_signals = np.zeros((N, C))
    for c in range(C):
        bp = np.random.uniform(-drift_amplitude, drift_amplitude, n_base_points)
        drift_signals[:, c] = np.interp(t, np.linspace(0, 1, n_base_points), bp)
    drifted = frames + drift_signals
    drifted = np.clip(drifted, 0, ADC_MAX)
    return drifted


def add_per_channel_offset(
    frames: np.ndarray,
    offset_std: float = 2.0,
) -> np.ndarray:
    """Add per-channel constant offset (simulating channel DC bias)."""
    offsets = np.random.normal(0, offset_std, size=frames.shape[1])
    offsetted = frames + offsets[np.newaxis, :]
    offsetted = np.clip(offsetted, 0, ADC_MAX)
    return offsetted


def save_as_csv(
    noisy_frames: np.ndarray,
    gt_frames: np.ndarray,
    filepath: str,
    label: str,
    noise_desc: str,
    freq_hz: float = 2.0,
):
    """Save noisy frames in the real-device CSV format.

    The CSV includes only the noisy data (as real recordings would have).
    Ground truth is saved separately as NPZ for evaluation.
    """
    n_frames, n_channels = noisy_frames.shape
    rows = cols = int(np.sqrt(n_channels))

    ts_base = datetime(2026, 7, 10, 14, 0, 0)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["##设备信息"])
        writer.writerow(["设备名称", f"Simulated 64x64 - {label}"])
        writer.writerow(["行数", str(rows)])
        writer.writerow(["列数", str(cols)])
        writer.writerow(["数据点数", str(n_frames)])
        writer.writerow(["显示模式", "ADC"])
        writer.writerow(["下位机基准单位", "raw"])
        writer.writerow(["上位机展示单位", "raw"])
        writer.writerow(["录制频率", "raw"])
        writer.writerow(["录制时间", f"2026年07月10日_14时00分00秒"])
        writer.writerow(["噪声描述", noise_desc])
        writer.writerow(["##数据开始"])

        # Column header
        col_header = ["时间戳"] + [f"通道{i+1}" for i in range(n_channels)]
        writer.writerow(col_header)

        # Data rows (simulate ~2Hz recording)
        interval_s = 1.0 / max(freq_hz, 1.0)
        for i in range(n_frames):
            ts = ts_base + __import__("datetime").timedelta(seconds=i * interval_s)
            ts_str = (
                f"{ts.hour}时{ts.minute}分{ts.second}秒"
                f".{ts.microsecond // 1000:03d}"
            )
            vals = [int(v) for v in noisy_frames[i]]
            row = [ts_str] + vals
            writer.writerow(row)

    # Save ground truth as NPZ
    gt_path = filepath.replace(".csv", "_gt.npz")
    np.savez_compressed(gt_path, ground_truth=gt_frames, noisy=noisy_frames)
    print(f"  CSV: {os.path.basename(filepath)}")
    print(f"  GT:  {os.path.basename(gt_path)}")


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = []

# Dataset 1: Single large central patch (like a single finger press)
DATASETS.append({
    "label": "single_center_patch",
    "noise_desc": "white_noise_std=5.0, baseline_drift=2.0",
    "n_frames": 30,
    "freq_hz": 2.0,
    "patches": [
        {"cx": 32, "cy": 32, "sigma": 8, "amplitude": 500},
    ],
    "noise_std": 5.0,
    "drift_amplitude": 2.0,
})

# Dataset 2: Two separate patches (like two fingers)
DATASETS.append({
    "label": "two_patches",
    "noise_desc": "white_noise_std=8.0, per_ch_offset=1.0",
    "n_frames": 30,
    "freq_hz": 2.0,
    "patches": [
        {"cx": 20, "cy": 20, "sigma": 6, "amplitude": 600},
        {"cx": 44, "cy": 44, "sigma": 6, "amplitude": 400},
    ],
    "noise_std": 8.0,
    "per_ch_offset_std": 1.0,
})

# Dataset 3: Multiple patches across the sensor (5 patches, palm-like)
DATASETS.append({
    "label": "five_patches_palm",
    "noise_desc": "white_noise_std=10.0, baseline_drift=3.0, per_ch_offset=2.0",
    "n_frames": 30,
    "freq_hz": 2.0,
    "patches": [
        {"cx": 16, "cy": 16, "sigma": 5, "amplitude": 300},
        {"cx": 16, "cy": 48, "sigma": 5, "amplitude": 350},
        {"cx": 32, "cy": 32, "sigma": 7, "amplitude": 500},
        {"cx": 48, "cy": 16, "sigma": 5, "amplitude": 280},
        {"cx": 48, "cy": 48, "sigma": 5, "amplitude": 320},
    ],
    "noise_std": 10.0,
    "drift_amplitude": 3.0,
    "per_ch_offset_std": 2.0,
})

# Dataset 4: Large area contact (like palm press)
DATASETS.append({
    "label": "large_area_press",
    "noise_desc": "white_noise_std=15.0, high_noise",
    "n_frames": 30,
    "freq_hz": 2.0,
    "patches": [
        {"cx": 32, "cy": 32, "sigma": 16, "amplitude": 800},
        {"cx": 24, "cy": 24, "sigma": 10, "amplitude": 300},
        {"cx": 40, "cy": 40, "sigma": 10, "amplitude": 300},
    ],
    "noise_std": 15.0,
})

# Dataset 5: Edge contact (near sensor boundary)
DATASETS.append({
    "label": "edge_contact",
    "noise_desc": "white_noise_std=5.0, edge_patches",
    "n_frames": 30,
    "freq_hz": 2.0,
    "patches": [
        {"cx": 8, "cy": 8, "sigma": 5, "amplitude": 400},
        {"cx": 56, "cy": 56, "sigma": 5, "amplitude": 400},
    ],
    "noise_std": 5.0,
})

# Dataset 6 (bonus): Low SNR scenario
DATASETS.append({
    "label": "low_snr_small_patches",
    "noise_desc": "white_noise_std=20.0, weak_signal",
    "n_frames": 30,
    "freq_hz": 2.0,
    "patches": [
        {"cx": 20, "cy": 32, "sigma": 4, "amplitude": 150},
        {"cx": 44, "cy": 32, "sigma": 4, "amplitude": 120},
    ],
    "noise_std": 20.0,
    "drift_amplitude": 5.0,
    "per_ch_offset_std": 3.0,
})


def main():
    print(f"Generating {len(DATASETS)} synthetic datasets...")
    for ds in DATASETS:
        print(f"\n[{ds['label']}]")
        np.random.seed(hash(ds["label"]) % (2 ** 31))

        gt = generate_ground_truth_frames(ds["n_frames"], ds["patches"])
        noisy = add_white_noise(gt, noise_std=ds["noise_std"])

        if ds.get("drift_amplitude", 0) > 0:
            noisy = add_baseline_drift(noisy, drift_amplitude=ds["drift_amplitude"])

        if ds.get("per_ch_offset_std", 0) > 0:
            noisy = add_per_channel_offset(noisy, offset_std=ds["per_ch_offset_std"])

        filename = f"{ds['label']}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        save_as_csv(
            noisy, gt, filepath,
            label=ds["label"],
            noise_desc=ds["noise_desc"],
            freq_hz=ds["freq_hz"],
        )

    print(f"\nDone. All files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
