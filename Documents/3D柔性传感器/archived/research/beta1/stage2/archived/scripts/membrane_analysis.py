"""
membrane_analysis.py — 64x64 membrane-specific noise analysis and processing.

Key finding from data analysis:
  - 90% of channels have ZERO background (bg_mean=0, bg_std=0)
  - Only pressure-area channels (~8%) have non-zero values
  - Noise only exists at pressure edges (cross-talk/jitter), NOT globally
  - This is fundamentally different from glove/fabric (global white noise)

Suitable approaches for membrane:
  [1] MedianGate — spatial median filter (3x3), preserves edges, no blobbing
  [2] Bilateral   — edge-preserving smoothing (5x5, sigma_color=25, sigma_space=1.5)

Not suitable:
  [X] WienerGate  — needs global baseline, which doesn't exist on membrane
  [X] Gaussian    — causes blobbing (user confirmed)
  [X] StatGate    — hard gate cuts meaningful edge pressure signal
"""
from __future__ import annotations

import os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_membrane")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_SET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "DataSet"))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic_blob")

sys.path.insert(0, os.path.join(BASE_DIR, "..", "..", "src"))
from alg.noise_suppressor import Analyzer, Processor


def _load_csv_raw(path):
    rows_list = []
    n_ch = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            if row[0] == "##数据开始":
                header_row = next(reader, [])
                n_ch = sum(1 for h in header_row if h.strip().startswith("通道"))
                in_data = True
                continue
            if in_data and len(row) > n_ch:
                try:
                    rows_list.append([float(v) for v in row[1:1 + n_ch]])
                except (ValueError, IndexError):
                    continue
    return np.array(rows_list, dtype=np.float64)


def median_filter_2d(frame, rows, cols, size=3):
    """Apply spatial median filter. Preserves edges, no blobbing."""
    from scipy.ndimage import median_filter
    f2d = frame.reshape(rows, cols)
    return median_filter(f2d, size=size).ravel()


def bilateral_filter_2d(frame, rows, cols, sigma_color=25, sigma_space=1.5):
    """Edge-preserving bilateral filter."""
    from skimage.restoration import denoise_bilateral
    f2d = frame.reshape(rows, cols)
    return denoise_bilateral(f2d, sigma_color=sigma_color,
                             sigma_spatial=sigma_space,
                             mode="reflect", channel_axis=None).ravel()


def gaussian_spatial(frame, rows, cols, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    f2d = frame.reshape(rows, cols)
    return gaussian_filter(f2d, sigma=sigma, mode="reflect").ravel()


class WienerGate:
    def __init__(self, k_sigma=1.0, over_sub=1.0, dd_alpha=0.98, min_snr=0.1):
        self.k_sigma = k_sigma
        self.over_sub = over_sub
        self.dd_alpha = dd_alpha
        self.min_snr = min_snr
        self.baseline = None
        self.noise_std = None
        self.n_channels = 0
        self.prior_snr = None

    def fit(self, bg_frames):
        N, C = bg_frames.shape
        self.n_channels = C
        self.baseline = np.median(bg_frames, axis=0)
        self.noise_std = np.std(bg_frames, axis=0, ddof=1)
        self.noise_std = np.maximum(self.noise_std, 0.5)
        self.prior_snr = np.full(C, self.min_snr, dtype=np.float64)

    def process_frame(self, frame):
        residual = np.maximum(frame - self.baseline, 0.0)
        noise_power = self.noise_std * self.noise_std
        post_snr_power = residual * residual / noise_power
        post_snr = np.maximum(post_snr_power - 1.0, 0.0)
        self.prior_snr = (
            self.dd_alpha * self.prior_snr
            + (1.0 - self.dd_alpha) * post_snr
        )
        self.prior_snr = np.maximum(self.prior_snr, self.min_snr)
        gain = self.prior_snr / (self.over_sub + self.prior_snr)
        return residual * gain


def _render_one(ax, data, title, vmin, vmax, rows, cols):
    ax.imshow(data.reshape(rows, cols), cmap="hot", aspect="auto",
              vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def render_membrane_compare(label, frame_raw, rows, cols):
    from scipy.ndimage import median_filter as mf
    from scipy.ndimage import gaussian_filter

    out_median3 = median_filter_2d(frame_raw, rows, cols, 3)
    out_median5 = median_filter_2d(frame_raw, rows, cols, 5)
    out_gauss = gaussian_spatial(frame_raw, rows, cols, 1.5)
    out_raw = frame_raw.copy()

    vmax = max(np.percentile(frame_raw, 99.5), 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    _render_one(axes[0], out_raw, "Original (raw)", 0, vmax, rows, cols)
    _render_one(axes[1], out_median3, "Median 3×3", 0, vmax, rows, cols)
    _render_one(axes[2], out_median5, "Median 5×5", 0, vmax, rows, cols)
    _render_one(axes[3], out_gauss, "Gaussian σ=1.5 (v1)", 0, vmax, rows, cols)

    fig.suptitle(f"{label} — Membrane-specific comparison", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.90, wspace=0.04)

    out_path = os.path.join(OUTPUT_DIR, f"membrane_{label}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")
    return out_path


def main():
    print("Membrane-specific noise analysis & comparison\n")

    # === Real film64 data ===
    film_path = os.path.join(DATA_SET_DIR, "64x32膜片", "1kg砝码压力",
                             "录制数据_20260710145522_part0.csv")
    if os.path.isfile(film_path):
        print("── 64x64 membrane (real) ──")
        data = _load_csv_raw(film_path)
        N, C = data.shape
        rows, cols = 64, 64
        print(f"  frames={N}, channels={C}")

        mid_idx = N // 2
        frame = data[mid_idx]

        render_membrane_compare("film64_1kg_real", frame, rows, cols)

    # === Synthetic blob data (membrane-like: low baseline) ===
    print("\n── Synthetic blob (membrane-like) ──")
    for csv_name in sorted(os.listdir(SYNTHETIC_DIR)):
        if not csv_name.endswith(".csv") or csv_name.startswith("._"):
            continue
        path = os.path.join(SYNTHETIC_DIR, csv_name)
        label = csv_name.replace(".csv", "")

        data = _load_csv_raw(path)
        sig = data[15:]
        mid = sig.shape[0] // 2
        frame = sig[mid]
        rows, cols = 64, 64

        render_membrane_compare(label, frame, rows, cols)

    print(f"\nDone. PNGs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
