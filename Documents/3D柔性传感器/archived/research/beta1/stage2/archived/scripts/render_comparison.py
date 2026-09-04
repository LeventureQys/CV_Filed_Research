"""
render_comparison.py — Single-frame before/after comparison heatmaps for all
algorithms across 3 real device types + 6 synthetic datasets.

Output: stage2/output/*.png  — one figure per dataset, showing original vs
processed side-by-side for each algorithm.

Usage:
    python scripts/render_comparison.py
"""
from __future__ import annotations

import os, sys, csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_SET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "DataSet"))
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic_blob")

sys.path.insert(0, os.path.join(BASE_DIR, "..", "..", "src"))
from alg.noise_suppressor import Analyzer, Processor


# ============================================================
# WienerGate (same as wiener_gate.py)
# ============================================================
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

    def process_batch(self, frames):
        N = frames.shape[0]
        out = np.empty_like(frames)
        for i in range(N):
            out[i] = self.process_frame(frames[i])
        return out


# ============================================================
# Data loading
# ============================================================
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
                    vals = [float(v) for v in row[1:1 + n_ch]]
                    rows_list.append(vals)
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


def _pick_mid_frame(frames):
    """Pick middle frame from the signal portion."""
    M = frames.shape[0]
    idx = M // 2 + M // 4
    return frames[min(idx, M - 1)]


# ============================================================
# Algorithms returning a single output frame
# ============================================================
def _apply_statgate(frame, bg, k):
    analyzer = Analyzer(k_sigma=k)
    model = analyzer.analyze(bg)
    proc = Processor(model)
    return proc.process(frame)

def _apply_spatial(frame):
    from scipy.ndimage import gaussian_filter
    rows, cols = _resolve_shape(len(frame))
    f2d = frame.reshape(rows, cols)
    s2d = gaussian_filter(f2d, sigma=1.5, mode="reflect")
    return s2d.ravel()

def _apply_wiener(frame, bg, k):
    wg = WienerGate(k_sigma=k)
    wg.fit(bg)
    return wg.process_frame(frame)

def _apply_hybrid(frame, bg, k):
    out1 = _apply_statgate(frame, bg, k)
    rows, cols = _resolve_shape(len(frame))
    from scipy.ndimage import gaussian_filter
    f2d = out1.reshape(rows, cols)
    s2d = gaussian_filter(f2d, sigma=1.5, mode="reflect")
    return s2d.ravel()


# ============================================================
# Rendering
# ============================================================
def _render_one(ax, data, title, vmin, vmax, rows, cols):
    im = ax.imshow(
        data.reshape(rows, cols), cmap="hot", aspect="auto",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def render_dataset(label, frame_raw, bg, rows, cols, gt=None):
    """
    Create a 2-row figure:
      Top row: Raw | WienerGate_k1 | WienerGate_k1.5
      Bottom row: StatGate_k1 | Spatial | Hybrid
    Plus a row for ground truth if available.
    """
    n_channels = len(frame_raw)
    if rows * cols != n_channels:
        rows, cols = _resolve_shape(n_channels)

    out_wg1 = _apply_wiener(frame_raw, bg, 1.0)
    out_wg15 = _apply_wiener(frame_raw, bg, 1.5)
    out_stat = _apply_statgate(frame_raw, bg, 1.0)
    out_spatial = _apply_spatial(frame_raw)
    out_hybrid = _apply_hybrid(frame_raw, bg, 1.0)

    has_gt = gt is not None
    nrows = 3 if has_gt else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4.5 * nrows))
    if nrows == 2:
        axes = np.atleast_2d(axes)

    raw_vmax = max(np.percentile(frame_raw, 99), 1.0)
    proc_vmax = max(raw_vmax * 0.8, 1.0)

    _render_one(axes[0, 0], frame_raw, "Original (raw)", 0, raw_vmax, rows, cols)
    _render_one(axes[0, 1], out_wg1, "WienerGate k=1.0", 0, proc_vmax, rows, cols)
    _render_one(axes[0, 2], out_wg15, "WienerGate k=1.5", 0, proc_vmax, rows, cols)
    _render_one(axes[1, 0], out_stat, "StatGate k=1.0 (old)", 0, proc_vmax, rows, cols)
    _render_one(axes[1, 1], out_spatial, "Spatial Gauss (old)", 0, raw_vmax, rows, cols)
    _render_one(axes[1, 2], out_hybrid, "Hybrid Stat+Space (old)", 0, proc_vmax, rows, cols)

    if has_gt:
        _render_one(axes[2, 0], gt, "Ground Truth", 0, raw_vmax, rows, cols)
        for j in range(1, 3):
            axes[2, j].axis("off")

    suptitle = f"{label}  |  channels={n_channels}  {rows}x{cols}"
    if bg is not None:
        suptitle += f"  |  bg_mean={bg.mean():.1f}  σ_noise={np.std(bg, axis=0).mean():.1f}"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.93, wspace=0.06, hspace=0.15)

    out_path = os.path.join(OUTPUT_DIR, f"comparison_{label}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(out_path)}")
    return out_path


# ============================================================
# Dataset definitions
# ============================================================
def get_real_datasets():
    ds = DATA_SET_DIR
    datasets = []

    glove_path = os.path.join(ds, "手套", "单手指1kg", "录制数据_20260710141208_part0.csv")
    if os.path.isfile(glove_path):
        datasets.append(("glove_1kg", glove_path, 60))

    fabric_path = os.path.join(ds, "织物垫", "500g砝码压力", "录制数据_20260710143312_part0.csv")
    if os.path.isfile(fabric_path):
        datasets.append(("fabric_500g", fabric_path, 60))

    film_path = os.path.join(ds, "64x32膜片", "1kg砝码压力", "录制数据_20260710145522_part0.csv")
    if os.path.isfile(film_path):
        datasets.append(("film64_1kg", film_path, 60))

    return datasets


def get_syn_datasets():
    datasets = []
    for csv_name in sorted(os.listdir(SYNTHETIC_DIR)):
        if not csv_name.endswith(".csv") or csv_name.startswith("._"):
            continue
        path = os.path.join(SYNTHETIC_DIR, csv_name)
        label = csv_name.replace(".csv", "")
        datasets.append((label, path, 15))
    return datasets


# ============================================================
# Main
# ============================================================
def main():
    print("Rendering comparison heatmaps...")
    print(f"Output: {OUTPUT_DIR}\n")

    # === Synthetic datasets ===
    print("── Synthetic blob datasets ──")
    for label, path, n_bg in get_syn_datasets():
        data = _load_csv_raw(path)
        gt_path = path.replace(".csv", "_gt.npz")
        gt_all = np.load(gt_path)["ground_truth"]
        bg = data[:n_bg]
        sig = data[n_bg:]
        sig_gt = gt_all[n_bg:]
        mid = sig.shape[0] // 2
        frame_raw = sig[mid]
        gt_frame = sig_gt[mid]
        rows, cols = _resolve_shape(frame_raw.shape[0])
        render_dataset(label, frame_raw, bg, rows, cols, gt=gt_frame)

    # === Real device datasets ===
    print("\n── Real device datasets ──")
    for label, path, n_bg in get_real_datasets():
        data = _load_csv_raw(path)
        bg = data[:n_bg]
        sig = data[n_bg:]
        mid = sig.shape[0] // 2
        frame_raw = sig[mid]
        rows, cols = _resolve_shape(frame_raw.shape[0])
        render_dataset(label, frame_raw, bg, rows, cols)

    print(f"\nDone. All PNGs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
