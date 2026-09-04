"""
render_edge_comparison.py — EdgeGate vs raw comparison across all datasets.
"""
from __future__ import annotations

import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edge_gate import EdgeGate

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_SET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "DataSet"))
SYN_MULTI_DIR = os.path.join(BASE_DIR, "data", "synthetic_multilayout")


def _load_csv(path):
    rows_list = []; n_ch = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row: continue
            if row[0] == "##数据开始":
                header_row = next(reader, [])
                n_ch = sum(1 for h in header_row if h.strip().startswith("通道"))
                in_data = True; continue
            if in_data and n_ch is not None and len(row) > n_ch:
                try: rows_list.append([float(v) for v in row[1:1 + n_ch]])
                except (ValueError, IndexError): continue
    return np.asarray(rows_list, dtype=np.float64)


def _shape(C):
    if C == 96: return 12, 8
    if C == 4096: return 64, 64
    if C == 512: return 32, 16
    s = int(np.sqrt(C))
    return (s, s) if s * s == C else (64, 64)


def _render(ax, data, title, r, c, vmax):
    ax.imshow(data.reshape(r, c), cmap="hot", aspect="auto", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])


def render(label, frames, bg_count, gt_frame=None):
    rows, cols = _shape(frames.shape[1])
    bg = frames[:bg_count] if bg_count > 0 else None
    sig = frames[bg_count:] if bg_count > 0 else frames[frames.shape[0] // 3:]
    mid = sig.shape[0] // 2
    frame = sig[mid]

    eg = EdgeGate(window=3, edge_ratio=2.5, signal_ratio=3.0)
    if bg is not None: eg.fit(bg)
    out = eg.process_frame(frame, rows, cols)

    vmax = max(np.percentile(frame, 99.5), 1.0)

    has_gt = gt_frame is not None
    fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(12 if has_gt else 8, 4.2))
    _render(axes[0], frame, "Original", rows, cols, vmax)
    _render(axes[1], out, "EdgeGate", rows, cols, vmax)
    if has_gt:
        _render(axes[2], gt_frame, "Ground Truth", rows, cols, vmax)

    info = f"{rows}x{cols}"
    fig.suptitle(f"{label}  |  {info}", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.88, wspace=0.05)
    path = os.path.join(OUT_DIR, f"edge_{label}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {os.path.basename(path)}")


def main():
    print("EdgeGate comparison\n")

    print("-- synthetic --")
    for fn in sorted(os.listdir(SYN_MULTI_DIR)):
        if not fn.endswith(".csv"): continue
        fp = os.path.join(SYN_MULTI_DIR, fn); label = fn.replace(".csv", "")
        frames = _load_csv(fp)
        gt = np.load(fp.replace(".csv", "_gt.npz"))["ground_truth"]
        gt_mid = gt[12:][(frames.shape[0] - 12) // 2]
        render(label, frames, 12, gt_mid)

    print("\n-- real --")
    cases = [
        ("glove_1kg",   "手套","单手指1kg","录制数据_20260710141208_part0.csv",60),
        ("fabric_500g", "织物垫","500g砝码压力","录制数据_20260710143312_part0.csv",50),
        ("film64_1kg",  "64x32膜片","1kg砝码压力","录制数据_20260710145522_part0.csv",0),
    ]
    for label, dev, sub, fn, bg_count in cases:
        fp = os.path.join(DATA_SET_DIR, dev, sub, fn)
        if not os.path.isfile(fp): continue
        render(label, _load_csv(fp), bg_count)

    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
