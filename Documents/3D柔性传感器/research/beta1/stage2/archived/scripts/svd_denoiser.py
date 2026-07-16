"""
svd_denoiser.py — SVD-based frame decomposition denoiser.

Core insight: pressure signals form spatially coherent blobs (LOW RANK),
while noise is spatially uncorrelated (FULL RANK). Per-frame SVD truncation
separates the two components naturally.

For each frame:
  1. Subtract per-channel baseline (from bg frames, if available)
  2. Compute SVD of the 2D matrix
  3. Keep first r singular values (adaptive threshold or fixed)
  4. Reconstruct denoised frame

This replaces per-channel independent processing with frame-level
component analysis — exactly what the user requested.
"""
from __future__ import annotations

import os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_svd")
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


def _resolve_shape(n_channels):
    if n_channels == 96:
        return 12, 8
    if n_channels == 4096:
        return 64, 64
    side = int(np.sqrt(n_channels))
    if side * side == n_channels:
        return side, side
    return 64, 64


class SVDDenoiser:
    """Per-frame SVD truncation denoiser."""

    def __init__(self, mode="adaptive", fixed_rank=5, energy_ratio=0.90):
        """
        mode: "fixed" (use fixed_rank) or "adaptive" (use energy_ratio)
        fixed_rank: number of singular values to keep
        energy_ratio: keep enough components to reach this cumulative energy
        """
        self.mode = mode
        self.fixed_rank = fixed_rank
        self.energy_ratio = energy_ratio
        self.baseline = None

    def fit_baseline(self, bg_frames):
        """Learn per-channel baseline from background frames."""
        self.baseline = np.median(bg_frames, axis=0)

    def process_frame(self, frame, rows, cols):
        if self.baseline is not None:
            residual = np.maximum(frame - self.baseline, 0.0)
        else:
            residual = frame.copy()

        mat = residual.reshape(rows, cols)
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)

        if self.mode == "fixed":
            r = min(self.fixed_rank, len(s))
        else:
            energy = np.cumsum(s * s) / np.sum(s * s)
            r = int(np.searchsorted(energy, self.energy_ratio) + 1)
            r = min(r, len(s))

        s_truncated = np.zeros_like(s)
        s_truncated[:r] = s[:r]
        denoised = (U * s_truncated) @ Vt
        return denoised.ravel().astype(np.float64)


class SVDSoftDenoiser:
    """SVD with soft thresholding (Wiener-like in singular value space)."""

    def __init__(self, tau=1.0):
        self.tau = tau
        self.baseline = None

    def fit_baseline(self, bg_frames):
        self.baseline = np.median(bg_frames, axis=0)

    def process_frame(self, frame, rows, cols):
        if self.baseline is not None:
            residual = np.maximum(frame - self.baseline, 0.0)
        else:
            residual = frame.copy()

        mat = residual.reshape(rows, cols)
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)

        sigma_est = np.median(s) if len(s) > 0 else 1.0
        threshold = self.tau * sigma_est

        s_soft = np.maximum(s - threshold, 0.0)
        denoised = (U * s_soft) @ Vt
        return denoised.ravel().astype(np.float64)


def _render_one(ax, data, title, vmin, vmax, rows, cols):
    ax.imshow(data.reshape(rows, cols), cmap="hot", aspect="auto",
              vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def render_svd_compare(label, frame_raw, rows, cols, bg=None, gt=None):
    """Compare: Raw | SVD_fixed(r) | SVD_adaptive | WienerGate | Median3x3"""
    svd_r3 = SVDDenoiser(mode="fixed", fixed_rank=min(3, min(rows, cols)))
    svd_adp = SVDDenoiser(mode="adaptive", energy_ratio=0.90)
    svd_soft = SVDSoftDenoiser(tau=1.5)

    if bg is not None:
        svd_r3.fit_baseline(bg)
        svd_adp.fit_baseline(bg)
        svd_soft.fit_baseline(bg)

    out_raw = frame_raw.copy()
    out_r3 = svd_r3.process_frame(frame_raw, rows, cols)
    out_adp = svd_adp.process_frame(frame_raw, rows, cols)
    out_soft = svd_soft.process_frame(frame_raw, rows, cols)

    vmax = max(np.percentile(frame_raw, 99.5), 1.0)

    ncols = 5
    has_gt = gt is not None
    if has_gt:
        ncols += 1

    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4.5))
    _render_one(axes[0], out_raw, "Original", 0, vmax, rows, cols)
    _render_one(axes[1], out_r3, "SVD r=3", 0, vmax, rows, cols)
    _render_one(axes[2], out_adp, "SVD adaptive", 0, vmax, rows, cols)
    _render_one(axes[3], out_soft, "SVD soft(t=1.5)", 0, vmax, rows, cols)
    _render_one(axes[4], _median3(frame_raw, rows, cols), "Median 3x3", 0, vmax, rows, cols)

    if has_gt:
        _render_one(axes[5], gt, "Ground Truth", 0, vmax, rows, cols)

    fig.suptitle(f"{label} — SVD Denoising", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.90, wspace=0.04)

    out_path = os.path.join(OUTPUT_DIR, f"svd_{label}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")
    return out_path


def _median3(frame, rows, cols):
    from scipy.ndimage import median_filter
    return median_filter(frame.reshape(rows, cols), size=3).ravel()


def main():
    print("SVD-based frame denoising comparison\n")

    # === Synthetic blob data ===
    print("── Synthetic blob ──")
    for csv_name in sorted(os.listdir(SYNTHETIC_DIR)):
        if not csv_name.endswith(".csv") or csv_name.startswith("._"):
            continue
        path = os.path.join(SYNTHETIC_DIR, csv_name)
        label = csv_name.replace(".csv", "")
        data = _load_csv_raw(path)
        bg = data[:15]
        sig = data[15:]
        frame = sig[sig.shape[0] // 2]
        gt_path = path.replace(".csv", "_gt.npz")
        gt_frame = None
        if os.path.isfile(gt_path):
            gt_all = np.load(gt_path)["ground_truth"]
            gt_frame = gt_all[15:][sig.shape[0] // 2]
        rows, cols = 64, 64
        render_svd_compare(label, frame, rows, cols, bg=bg, gt=gt_frame)

    # === Real device data ===
    print("\n── Real devices ──")
    real_cases = [
        ("glove_1kg",  "手套", "单手指1kg", "录制数据_20260710141208_part0.csv", 60),
        ("fabric_500g","织物垫","500g砝码压力","录制数据_20260710143312_part0.csv", 50),
        ("film64_1kg", "64x32膜片","1kg砝码压力","录制数据_20260710145522_part0.csv", 0),
    ]
    for label, dev, subdir, fn, n_bg in real_cases:
        path = os.path.join(DATA_SET_DIR, dev, subdir, fn)
        if not os.path.isfile(path):
            continue
        data = _load_csv_raw(path)
        N = data.shape[0]
        rows, cols = _resolve_shape(data.shape[1])

        if n_bg > 0:
            bg = data[:n_bg]
            sig = data[n_bg:]
        else:
            bg = None
            sig = data[N // 3:]

        frame = sig[sig.shape[0] // 2]
        render_svd_compare(label, frame, rows, cols, bg=bg)

    print(f"\nDone. All PNGs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
