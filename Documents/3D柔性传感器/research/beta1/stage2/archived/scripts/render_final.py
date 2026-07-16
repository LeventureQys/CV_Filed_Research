"""
render_final.py — Final unified comparison with PCA route included.

Layout per dataset:
  Row 1: Original | PCA Subspace | PCA Soft | SVD adaptive | GT / WienerGate
  Row 2: WienerGate | Median 3x3 | SVD hard r=3 | SVD soft | reserved

This is the final stage2 visual summary.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca_subspace_denoiser import PCASubspaceDenoiser


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_SET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "DataSet"))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic_blob")


def _load_csv_raw(path: str) -> np.ndarray:
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
            if in_data and n_ch is not None and len(row) > n_ch:
                try:
                    rows_list.append([float(v) for v in row[1:1 + n_ch]])
                except (ValueError, IndexError):
                    continue
    return np.asarray(rows_list, dtype=np.float64)


def _resolve_shape(n_channels: int) -> tuple[int, int]:
    if n_channels == 96:
        return 12, 8
    if n_channels == 4096:
        return 64, 64
    side = int(np.sqrt(n_channels))
    if side * side == n_channels:
        return side, side
    return 64, 64


class SVDDenoiser:
    def __init__(self, mode: str = "adaptive", fixed_rank: int = 3, energy_ratio: float = 0.90):
        self.mode = mode
        self.fixed_rank = fixed_rank
        self.energy_ratio = energy_ratio
        self.baseline = None

    def fit_baseline(self, bg_frames: np.ndarray):
        self.baseline = np.median(bg_frames, axis=0)

    def process_frame(self, frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
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
        s_trunc = np.zeros_like(s)
        s_trunc[:r] = s[:r]
        return ((U * s_trunc) @ Vt).ravel().astype(np.float64)


class SVDSoftDenoiser:
    def __init__(self, tau: float = 1.0):
        self.tau = tau
        self.baseline = None

    def fit_baseline(self, bg_frames: np.ndarray):
        self.baseline = np.median(bg_frames, axis=0)

    def process_frame(self, frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
        if self.baseline is not None:
            residual = np.maximum(frame - self.baseline, 0.0)
        else:
            residual = frame.copy()
        mat = residual.reshape(rows, cols)
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        sigma_est = np.median(s) if len(s) > 0 else 1.0
        s_soft = np.maximum(s - self.tau * sigma_est, 0.0)
        return ((U * s_soft) @ Vt).ravel().astype(np.float64)


class WienerGate:
    def __init__(self, k_sigma: float = 1.0, over_sub: float = 1.0, dd_alpha: float = 0.98, min_snr: float = 0.1):
        self.k_sigma = k_sigma
        self.over_sub = over_sub
        self.dd_alpha = dd_alpha
        self.min_snr = min_snr
        self.baseline = None
        self.noise_std = None
        self.prior_snr = None

    def fit(self, bg_frames: np.ndarray):
        self.baseline = np.median(bg_frames, axis=0)
        self.noise_std = np.std(bg_frames, axis=0, ddof=1)
        self.noise_std = np.maximum(self.noise_std, 0.5)
        self.prior_snr = np.full(bg_frames.shape[1], self.min_snr, dtype=np.float64)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        residual = np.maximum(frame - self.baseline, 0.0)
        noise_power = self.noise_std * self.noise_std
        post_snr = np.maximum(residual * residual / noise_power - 1.0, 0.0)
        self.prior_snr = self.dd_alpha * self.prior_snr + (1.0 - self.dd_alpha) * post_snr
        self.prior_snr = np.maximum(self.prior_snr, self.min_snr)
        gain = self.prior_snr / (self.over_sub + self.prior_snr)
        return residual * gain


def _median3(frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
    from scipy.ndimage import median_filter
    return median_filter(frame.reshape(rows, cols), size=3).ravel()


def _render_one(ax, data, title, rows, cols, vmax):
    ax.imshow(data.reshape(rows, cols), cmap="hot", aspect="auto", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def render_unified(label: str, frames: np.ndarray, bg_count: int, gt_frame: np.ndarray | None = None):
    rows, cols = _resolve_shape(frames.shape[1])
    bg = frames[:bg_count] if bg_count > 0 else None
    sig = frames[bg_count:] if bg_count > 0 else frames[frames.shape[0] // 3:]
    frame = sig[sig.shape[0] // 2]

    pca_main = PCASubspaceDenoiser(energy_ratio=0.90, max_components=12)
    pca_soft = PCASubspaceDenoiser(energy_ratio=0.97, max_components=16)
    info_main = pca_main.fit(frames, bg_count)
    pca_soft.fit(frames, bg_count)
    out_pca_main = pca_main.process_frame(frame)
    out_pca_soft = pca_soft.process_frame(frame)

    svd_r3 = SVDDenoiser(mode="fixed", fixed_rank=3)
    svd_adp = SVDDenoiser(mode="adaptive", energy_ratio=0.90)
    svd_soft = SVDSoftDenoiser(tau=1.0)
    wg = WienerGate(k_sigma=1.0) if bg is not None else None
    if bg is not None:
        svd_r3.fit_baseline(bg)
        svd_adp.fit_baseline(bg)
        svd_soft.fit_baseline(bg)
        wg.fit(bg)

    out_svd_r3 = svd_r3.process_frame(frame, rows, cols)
    out_svd_adp = svd_adp.process_frame(frame, rows, cols)
    out_svd_soft = svd_soft.process_frame(frame, rows, cols)
    out_wg = wg.process_frame(frame) if wg is not None else frame.copy()
    out_med3 = _median3(frame, rows, cols)

    vmax = max(np.percentile(frame, 99.5), 1.0)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.6))

    _render_one(axes[0, 0], frame,         "Original", rows, cols, vmax)
    _render_one(axes[0, 1], out_pca_main,  "PCA Subspace", rows, cols, vmax)
    _render_one(axes[0, 2], out_pca_soft,  "PCA Soft", rows, cols, vmax)
    _render_one(axes[0, 3], out_svd_adp,   "SVD adaptive", rows, cols, vmax)
    if gt_frame is not None:
        _render_one(axes[0, 4], gt_frame,  "Ground Truth", rows, cols, vmax)
    else:
        _render_one(axes[0, 4], out_wg,    "WienerGate", rows, cols, vmax)

    _render_one(axes[1, 0], out_wg,       "WienerGate", rows, cols, vmax)
    _render_one(axes[1, 1], out_med3,     "Median 3x3", rows, cols, vmax)
    _render_one(axes[1, 2], out_svd_r3,   "SVD hard r=3", rows, cols, vmax)
    _render_one(axes[1, 3], out_svd_soft, "SVD soft", rows, cols, vmax)
    axes[1, 4].axis("off")

    fig.suptitle(
        f"{label}  |  mode={info_main.device_mode}  k={info_main.kept_components}  baseline_sub={info_main.baseline_subtracted}",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.90, wspace=0.04, hspace=0.20)
    out_path = os.path.join(OUTPUT_DIR, f"final_{label}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")


def main():
    print("=== Final unified comparison with PCA included ===\n")

    print("── Synthetic blob (6 datasets) ──")
    for csv_name in sorted(os.listdir(SYNTHETIC_DIR)):
        if not csv_name.endswith(".csv") or csv_name.startswith("._"):
            continue
        path = os.path.join(SYNTHETIC_DIR, csv_name)
        label = csv_name.replace(".csv", "")
        frames = _load_csv_raw(path)
        gt_frame = None
        gt_path = path.replace(".csv", "_gt.npz")
        if os.path.isfile(gt_path):
            gt_all = np.load(gt_path)["ground_truth"]
            gt_frame = gt_all[15:][(frames.shape[0] - 15) // 2]
        render_unified(label, frames, 15, gt_frame)

    print("\n── Real devices (3 datasets) ──")
    real_cases = [
        ("glove_1kg", "手套", "单手指1kg", "录制数据_20260710141208_part0.csv", 60),
        ("fabric_500g", "织物垫", "500g砝码压力", "录制数据_20260710143312_part0.csv", 50),
        ("film64_1kg", "64x32膜片", "1kg砝码压力", "录制数据_20260710145522_part0.csv", 0),
    ]
    for label, dev, subdir, fn, bg_count in real_cases:
        path = os.path.join(DATA_SET_DIR, dev, subdir, fn)
        if not os.path.isfile(path):
            continue
        frames = _load_csv_raw(path)
        render_unified(label, frames, bg_count, None)

    print(f"\nDone. All PNGs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
