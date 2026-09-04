from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nmf_temporal_denoiser import NMFTemporalDenoiser
from pca_subspace_denoiser import PCASubspaceDenoiser, load_csv_channels_only


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "output_nmf")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_SET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "DataSet"))
SYN_MULTI_DIR = os.path.join(BASE_DIR, "data", "synthetic_multilayout")


def _resolve_shape(channels: int) -> tuple[int, int]:
    if channels == 96:
        return 12, 8
    if channels == 4096:
        return 64, 64
    if channels == 512:
        return 32, 16
    side = int(np.sqrt(channels))
    if side * side == channels:
        return side, side
    return 64, 64


def _median3(frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
    from scipy.ndimage import median_filter
    return median_filter(frame.reshape(rows, cols), size=3).ravel()


def _render(ax, frame, title, rows, cols, vmax):
    ax.imshow(frame.reshape(rows, cols), cmap="hot", aspect="auto", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def render_one(label: str, frames: np.ndarray, bg_count: int, gt_frame: np.ndarray | None = None):
    rows, cols = _resolve_shape(frames.shape[1])
    sig = frames[bg_count:] if bg_count > 0 else frames[frames.shape[0] // 3:]
    rel_idx = sig.shape[0] // 2
    frame_idx = (bg_count + rel_idx) if bg_count > 0 else (frames.shape[0] // 3 + rel_idx)
    frame = frames[frame_idx]

    nmf = NMFTemporalDenoiser(max_components=8, target_rel_error=0.12, activation_smooth=5)
    info_nmf = nmf.fit(frames, bg_count)
    out_nmf = nmf.process_frame(frame_idx)

    pca = PCASubspaceDenoiser(energy_ratio=0.90, max_components=12)
    info_pca = pca.fit(frames, bg_count)
    out_pca = pca.process_frame(frame)

    out_med = _median3(frame, rows, cols)
    vmax = max(np.percentile(frame, 99.5), 1.0)

    fig, axes = plt.subplots(1, 4 if gt_frame is not None else 3, figsize=(16 if gt_frame is not None else 12, 4.2))
    _render(axes[0], frame, "Original", rows, cols, vmax)
    _render(axes[1], out_nmf, f"NMF k={info_nmf.n_components}", rows, cols, vmax)
    _render(axes[2], out_pca, f"PCA k={info_pca.kept_components}", rows, cols, vmax)
    if gt_frame is not None:
        _render(axes[3], gt_frame, "Ground Truth", rows, cols, vmax)
    else:
        _render(axes[2], out_med, "Median 3x3", rows, cols, vmax)

    fig.suptitle(f"{label} | nmf_mode={info_nmf.device_mode} pca_mode={info_pca.device_mode}", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.88, wspace=0.05)
    out_path = os.path.join(OUT_DIR, f"nmf_{label}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")


def main():
    print("Generating NMF comparison figures...\n")

    print("-- synthetic multilayout --")
    for csv_name in sorted(os.listdir(SYN_MULTI_DIR)):
        if not csv_name.endswith(".csv"):
            continue
        path = os.path.join(SYN_MULTI_DIR, csv_name)
        frames = load_csv_channels_only(path)
        gt = np.load(path.replace(".csv", "_gt.npz"))["ground_truth"]
        label = csv_name.replace(".csv", "")
        render_one(label, frames, 12, gt[12:][(frames.shape[0] - 12) // 2])

    print("\n-- real devices --")
    real_cases = [
        ("glove_1kg", "手套", "单手指1kg", "录制数据_20260710141208_part0.csv", 60),
        ("fabric_500g", "织物垫", "500g砝码压力", "录制数据_20260710143312_part0.csv", 50),
        ("film64_1kg", "64x32膜片", "1kg砝码压力", "录制数据_20260710145522_part0.csv", 0),
    ]
    for label, dev, subdir, fn, bg_count in real_cases:
        path = os.path.join(DATA_SET_DIR, dev, subdir, fn)
        if not os.path.isfile(path):
            continue
        frames = load_csv_channels_only(path)
        render_one(label, frames, bg_count, None)

    print(f"\nDone. PNGs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
