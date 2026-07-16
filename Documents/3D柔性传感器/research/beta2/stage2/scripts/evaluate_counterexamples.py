from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from membrane_spatiotemporal import MembraneSpatioTemporalGate


def retention(raw: np.ndarray, output: np.ndarray) -> float:
    total = float(np.sum(raw))
    return float(np.sum(output) / total) if total else 1.0


def write_csv(rows: list[dict]) -> None:
    path = STAGE_DIR / "data" / "membrane_synthetic_counterexamples.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    gate = MembraneSpatioTemporalGate()
    rows = []

    for duration in (1, 2, 3, 5, 10):
        frames = np.zeros((20, 64 * 64), dtype=np.float32)
        frames[5:5 + duration, 32 * 64 + 32] = 100.0
        output, _ = gate.process_offline(frames)
        rows.append({"scenario": "short_contact", "parameter": duration, "retention": retention(frames, output)})

    for amplitude in (5, 8, 10, 12, 20, 100):
        frames = np.zeros((20, 64 * 64), dtype=np.float32)
        frames[5:15, 32 * 64 + 32] = amplitude
        output, _ = gate.process_offline(frames)
        rows.append({"scenario": "low_amplitude_contact", "parameter": amplitude, "retention": retention(frames, output)})

    source = np.zeros((20, 64 * 64), dtype=np.float32)
    source[5:15, 20 * 64 + 20] = 100.0
    _, source_mask = gate.process_offline(source)
    target = np.zeros_like(source)
    target[5:15, 44 * 64 + 44] = 100.0
    external = gate.process_with_mask(target, source_mask)
    own, _ = gate.process_offline(target)
    rows.append({"scenario": "new_position_external_mask", "parameter": 44, "retention": retention(target, external)})
    rows.append({"scenario": "new_position_own_mask", "parameter": 44, "retention": retention(target, own)})

    moving = np.zeros((20, 64 * 64), dtype=np.float32)
    for frame_index in range(5, 15):
        position = 20 + frame_index - 5
        moving[frame_index, position * 64 + position] = 100.0
    moving_output, _ = gate.process_offline(moving)
    rows.append({"scenario": "moving_contact", "parameter": 10, "retention": retention(moving, moving_output)})

    write_csv(rows)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    groups = [
        ("short_contact", "Contact duration", "frames"),
        ("low_amplitude_contact", "Contact amplitude", "ADC"),
        ("new_position", "Mask position transfer", "case"),
    ]
    for axis, (prefix, title, xlabel) in zip(axes, groups):
        selected = [row for row in rows if row["scenario"].startswith(prefix)]
        axis.bar([str(row["parameter"]) for row in selected], [row["retention"] for row in selected])
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("ADC retention")
    output_dir = STAGE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "membrane_counterexamples.png", dpi=170)
    plt.close(figure)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
