from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = STAGE_DIR.parents[1]
sys.path.insert(0, str(RESEARCH_DIR / "src"))

from alg.data_loader import load_csv
from membrane_spatiotemporal import MembraneSpatioTemporalGate


def recording_label(path: Path) -> str:
    load = path.parent.name.replace("砝码压力", "")
    return f"{load}_{path.stem.split('_')[1]}"


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 1.0


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return safe_ratio(np.count_nonzero(left & right), union)


def evaluate_recording(path: Path, frames: np.ndarray, sample_rate_hz: float) -> tuple[dict, np.ndarray]:
    gate = MembraneSpatioTemporalGate()
    offline, mask = gate.process_offline(frames)
    causal = gate.process_prefix_causal(frames)
    high = frames >= 100.0
    low = (frames > 0.0) & (frames < 10.0)
    changed = offline != frames
    row = {
        "recording": recording_label(path),
        "frames": frames.shape[0],
        "sample_rate_hz": sample_rate_hz,
        "duration_s": frames.shape[0] / sample_rate_hz,
        "raw_nonzero_channels": int(np.count_nonzero(np.max(frames, axis=0) > 0)),
        "offline_mask_channels": int(np.count_nonzero(mask)),
        "raw_total": float(np.sum(frames)),
        "offline_total_retention": safe_ratio(float(np.sum(offline)), float(np.sum(frames))),
        "causal_total_retention": safe_ratio(float(np.sum(causal)), float(np.sum(frames))),
        "offline_high_value_retention": safe_ratio(float(np.sum(offline[high])), float(np.sum(frames[high]))),
        "offline_low_value_removed": safe_ratio(float(np.sum(frames[low] - offline[low])), float(np.sum(frames[low]))),
        "changed_sample_ratio": float(np.mean(changed)),
        "offline_causal_disagreement": float(np.mean(offline != causal)),
        "startup_delay_frames": 2,
        "startup_delay_ms": 2000.0 / sample_rate_hz,
    }
    return row, mask


def threshold_sweep(frames_by_label: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for threshold in (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0):
        for label, frames in frames_by_label.items():
            gate = MembraneSpatioTemporalGate(spatial_threshold=threshold)
            output, mask = gate.process_offline(frames)
            high = frames >= 100.0
            rows.append({
                "recording": label,
                "spatial_threshold": threshold,
                "mask_channels": int(np.count_nonzero(mask)),
                "total_retention": safe_ratio(float(np.sum(output)), float(np.sum(frames))),
                "high_value_retention": safe_ratio(float(np.sum(output[high])), float(np.sum(frames[high]))),
            })
    return rows


def cross_recording(rows: list[tuple[str, np.ndarray, np.ndarray]]) -> list[dict]:
    results = []
    gate = MembraneSpatioTemporalGate()
    for source_label, _, source_mask in rows:
        for target_label, target_frames, target_mask in rows:
            output = gate.process_with_mask(target_frames, source_mask)
            results.append({
                "source_mask": source_label,
                "target_recording": target_label,
                "mask_jaccard": jaccard(source_mask, target_mask),
                "target_total_retention": safe_ratio(float(np.sum(output)), float(np.sum(target_frames))),
                "target_peak_retention": safe_ratio(float(np.max(output)), float(np.max(target_frames))),
            })
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(real_rows: list[dict], cross_rows: list[dict], sweep_rows: list[dict]) -> None:
    output_dir = STAGE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["recording"] for row in real_rows]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(labels, [row["offline_total_retention"] for row in real_rows], label="offline")
    axes[0, 0].bar(labels, [row["causal_total_retention"] for row in real_rows], alpha=0.55, label="prefix-causal")
    axes[0, 0].set_title("Raw ADC total retention")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=35)
    matrix = np.array([row["target_total_retention"] for row in cross_rows]).reshape(len(labels), len(labels))
    image = axes[0, 1].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axes[0, 1].set_title("Cross-recording mask retention")
    axes[0, 1].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axes[0, 1].set_yticks(range(len(labels)), labels)
    figure.colorbar(image, ax=axes[0, 1])
    thresholds = sorted({row["spatial_threshold"] for row in sweep_rows})
    for label in labels:
        selected = [row for row in sweep_rows if row["recording"] == label]
        axes[1, 0].plot(thresholds, [row["total_retention"] for row in selected], marker="o", label=label)
        axes[1, 1].plot(thresholds, [row["mask_channels"] for row in selected], marker="o", label=label)
    axes[1, 0].set_title("Spatial threshold sensitivity: retention")
    axes[1, 0].set_xlabel("threshold (ADC)")
    axes[1, 0].set_ylabel("retention")
    axes[1, 1].set_title("Spatial threshold sensitivity: mask size")
    axes[1, 1].set_xlabel("threshold (ADC)")
    axes[1, 1].set_ylabel("channels")
    axes[1, 1].legend(fontsize=7, ncol=2)
    figure.savefig(output_dir / "membrane_evaluation_summary.png", dpi=170)
    plt.close(figure)


def main() -> None:
    data_dir = RESEARCH_DIR / "DataSet" / "64x64膜片"
    pressure_paths = sorted(path for path in data_dir.rglob("*.csv") if "背景" not in str(path))
    background_path = next(path for path in data_dir.rglob("*.csv") if "背景" in str(path))
    real_rows = []
    recording_rows = []
    frames_by_label = {}
    for path in pressure_paths:
        frames, metadata = load_csv(str(path))
        label = recording_label(path)
        row, mask = evaluate_recording(path, frames, float(metadata["sample_freq_hz"]))
        real_rows.append(row)
        recording_rows.append((label, frames, mask))
        frames_by_label[label] = frames
    background, background_metadata = load_csv(str(background_path))
    background_row = {
        "recording": "background",
        "frames": background.shape[0],
        "sample_rate_hz": float(background_metadata["sample_freq_hz"]),
        "duration_s": background.shape[0] / float(background_metadata["sample_freq_hz"]),
        "nonzero_samples": int(np.count_nonzero(background)),
        "maximum_adc": float(np.max(background)),
    }
    cross_rows = cross_recording(recording_rows)
    sweep_rows = threshold_sweep(frames_by_label)
    write_csv(STAGE_DIR / "data" / "membrane_real_metrics.csv", real_rows)
    write_csv(STAGE_DIR / "data" / "membrane_cross_recording.csv", cross_rows)
    write_csv(STAGE_DIR / "data" / "membrane_threshold_sweep.csv", sweep_rows)
    write_csv(STAGE_DIR / "data" / "membrane_background_metrics.csv", [background_row])
    render_summary(real_rows, cross_rows, sweep_rows)
    for row in real_rows:
        print(row)
    print(background_row)


if __name__ == "__main__":
    main()
