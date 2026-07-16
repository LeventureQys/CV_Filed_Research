from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca_subspace_denoiser import PCASubspaceDenoiser, load_csv_channels_only


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_pca")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_SET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "DataSet"))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic_blob")


def _resolve_shape(n_channels: int) -> tuple[int, int]:
    if n_channels == 96:
        return 12, 8
    if n_channels == 4096:
        return 64, 64
    side = int(np.sqrt(n_channels))
    if side * side == n_channels:
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


def render_one(label: str, frame: np.ndarray, rows: int, cols: int, pca_main: np.ndarray, pca_soft: np.ndarray, median: np.ndarray, gt: np.ndarray | None = None, info_text: str = ""):
    vmax = max(np.percentile(frame, 99.5), 1.0)
    has_gt = gt is not None
    fig, axes = plt.subplots(1, 4 if has_gt else 3, figsize=(16 if has_gt else 12, 4.2))
    _render(axes[0], frame, "Original", rows, cols, vmax)
    _render(axes[1], pca_main, "PCA Subspace", rows, cols, vmax)
    _render(axes[2], pca_soft, "PCA Soft", rows, cols, vmax)
    if has_gt:
        _render(axes[3], gt, "Ground Truth", rows, cols, vmax)
    else:
        _render(axes[2], median, "Median 3x3", rows, cols, vmax)

    fig.suptitle(f"{label}  |  {info_text}", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.88, wspace=0.05)
    out_path = os.path.join(OUTPUT_DIR, f"pca_{label}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")


def main():
    print("Rendering PCA-subspace comparisons...")

    # synthetic
    for csv_name in sorted(os.listdir(SYNTHETIC_DIR)):
        if not csv_name.endswith(".csv") or csv_name.startswith("._"):
            continue
        path = os.path.join(SYNTHETIC_DIR, csv_name)
        label = csv_name.replace(".csv", "")
        frames = load_csv_channels_only(path)
        bg_count = 15
        rows, cols = 64, 64
        gt = None
        gt_path = path.replace(".csv", "_gt.npz")
        if os.path.isfile(gt_path):
            gt_all = np.load(gt_path)["ground_truth"]
            gt = gt_all[bg_count:][(frames.shape[0] - bg_count) // 2]

        pca_main = PCASubspaceDenoiser(energy_ratio=0.90, max_components=8)
        info = pca_main.fit(frames, bg_count)
        sig = frames[bg_count:]
        frame = sig[sig.shape[0] // 2]
        out_main = pca_main.process_frame(frame)

        pca_soft = PCASubspaceDenoiser(energy_ratio=0.97, max_components=12)
        pca_soft.fit(frames, bg_count)
        out_soft = pca_soft.process_frame(frame)
        out_median = _median3(frame, rows, cols)

        render_one(
            label,
            frame,
            rows,
            cols,
            out_main,
            out_soft,
            out_median,
            gt=gt,
            info_text=f"mode={info.device_mode} k={info.kept_components} baseline_sub={info.baseline_subtracted}",
        )

    # real
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
        rows, cols = _resolve_shape(frames.shape[1])

        pca_main = PCASubspaceDenoiser(energy_ratio=0.90, max_components=12)
        info = pca_main.fit(frames, bg_count)
        sig = frames[bg_count:] if bg_count > 0 else frames[frames.shape[0] // 3:]
        frame = sig[sig.shape[0] // 2]
        out_main = pca_main.process_frame(frame)

        pca_soft = PCASubspaceDenoiser(energy_ratio=0.97, max_components=16)
        pca_soft.fit(frames, bg_count)
        out_soft = pca_soft.process_frame(frame)
        out_median = _median3(frame, rows, cols)

        render_one(
            label,
            frame,
            rows,
            cols,
            out_main,
            out_soft,
            out_median,
            gt=None,
            info_text=f"mode={info.device_mode} k={info.kept_components} baseline_sub={info.baseline_subtracted}",
        )

    print(f"Done. All PNGs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
