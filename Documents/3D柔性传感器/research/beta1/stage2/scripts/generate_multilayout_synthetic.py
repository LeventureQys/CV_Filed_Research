from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

import numpy as np


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "data", "synthetic_multilayout")
os.makedirs(OUT_DIR, exist_ok=True)


def gaussian_blob(rows, cols, cx, cy, sigma, amplitude):
    ys, xs = np.mgrid[0:rows, 0:cols]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return amplitude * np.exp(-dist2 / (2 * sigma * sigma))


def make_profile(n_frames):
    t = np.linspace(0, 1, n_frames)
    # small white noise assumption -> slowly varying signal profile
    return np.clip(np.sin(np.pi * t) ** 1.2, 0.0, 1.0)


def boundary_mask(frame2d, thr_ratio=0.08):
    from scipy.ndimage import binary_dilation
    mask = frame2d > frame2d.max() * thr_ratio if frame2d.max() > 0 else np.zeros_like(frame2d, dtype=bool)
    dil = binary_dilation(mask, iterations=1)
    return dil ^ mask


def generate_sequence(rows, cols, mode, n_frames, blobs, baseline_mean, baseline_spread, noise_std):
    channels = rows * cols
    profile = make_profile(n_frames)
    static = np.zeros((rows, cols), dtype=np.float64)
    for b in blobs:
        static += gaussian_blob(rows, cols, b["cx"], b["cy"], b["sigma"], b["amplitude"])

    gt_frames = []
    noisy_frames = []
    base = np.random.normal(baseline_mean, baseline_spread, size=channels)
    base[base < 0] = 0.0

    edge = boundary_mask(static)
    for scale in profile:
        gt2d = static * scale
        gt = gt2d.ravel()
        if mode == "glove_like":
            noise = np.random.normal(0.0, noise_std, size=channels)
            noisy = np.maximum(base + gt + noise, 0.0)
        else:
            # membrane-like: no global background, small edge-local noise only
            noise2d = np.zeros((rows, cols), dtype=np.float64)
            edge_noise = np.random.normal(0.0, noise_std, size=edge.sum())
            noise2d[edge] = edge_noise
            noisy = np.maximum(gt + noise2d.ravel(), 0.0)
        gt_frames.append(gt)
        noisy_frames.append(noisy)

    return np.asarray(noisy_frames), np.asarray(gt_frames)


def save_csv(path, rows, cols, noisy_frames, desc):
    n_frames, channels = noisy_frames.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["##设备信息"])
        writer.writerow(["设备名称", f"SyntheticMultiLayout {rows}x{cols}"])
        writer.writerow(["行数", str(rows)])
        writer.writerow(["列数", str(cols)])
        writer.writerow(["数据点数", str(n_frames)])
        writer.writerow(["显示模式", "ADC"])
        writer.writerow(["下位机基准单位", "raw"])
        writer.writerow(["上位机展示单位", "raw"])
        writer.writerow(["录制频率", "raw"])
        writer.writerow(["录制时间", "2026年07月11日_16时00分00秒"])
        writer.writerow(["噪声描述", desc])
        writer.writerow(["##数据开始"])
        writer.writerow(["时间戳"] + [f"通道{i+1}" for i in range(channels)])
        t0 = datetime(2026, 7, 11, 16, 0, 0)
        for i in range(n_frames):
            ts = t0 + timedelta(milliseconds=50 * i)
            ts_str = f"{ts.hour}时{ts.minute}分{ts.second}秒.{ts.microsecond // 1000:03d}"
            writer.writerow([ts_str] + [int(v) for v in noisy_frames[i]])


def main():
    specs = [
        {
            "label": "glove_like_12x8",
            "rows": 12,
            "cols": 8,
            "mode": "glove_like",
            "baseline_mean": 8.0,
            "baseline_spread": 2.0,
            "noise_std": 0.8,
            "blobs": [{"cx": 3, "cy": 5, "sigma": 1.6, "amplitude": 26}],
        },
        {
            "label": "membrane_small_12x8",
            "rows": 12,
            "cols": 8,
            "mode": "membrane_like",
            "baseline_mean": 0.0,
            "baseline_spread": 0.0,
            "noise_std": 1.5,
            "blobs": [{"cx": 3, "cy": 5, "sigma": 1.7, "amplitude": 40}],
        },
        {
            "label": "membrane_mid_32x16",
            "rows": 32,
            "cols": 16,
            "mode": "membrane_like",
            "baseline_mean": 0.0,
            "baseline_spread": 0.0,
            "noise_std": 2.0,
            "blobs": [{"cx": 8, "cy": 15, "sigma": 3.5, "amplitude": 120}],
        },
        {
            "label": "membrane_large_64x64",
            "rows": 64,
            "cols": 64,
            "mode": "membrane_like",
            "baseline_mean": 0.0,
            "baseline_spread": 0.0,
            "noise_std": 3.0,
            "blobs": [{"cx": 32, "cy": 32, "sigma": 8.0, "amplitude": 220}],
        },
    ]

    for spec in specs:
        np.random.seed(abs(hash(spec["label"])) % (2 ** 31))
        noisy, gt = generate_sequence(
            spec["rows"], spec["cols"], spec["mode"], 48,
            spec["blobs"], spec["baseline_mean"], spec["baseline_spread"], spec["noise_std"],
        )
        csv_path = os.path.join(OUT_DIR, f"{spec['label']}.csv")
        save_csv(csv_path, spec["rows"], spec["cols"], noisy, f"mode={spec['mode']} noise_std={spec['noise_std']}")
        np.savez_compressed(csv_path.replace(".csv", "_gt.npz"), ground_truth=gt, noisy=noisy)
        print(f"saved: {os.path.basename(csv_path)}")


if __name__ == "__main__":
    main()
