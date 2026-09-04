from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


STAGE_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = STAGE_DIR.parents[1]
sys.path.insert(0, str(RESEARCH_DIR / "src"))

from alg.data_loader import load_csv
from temporal_spectral_gate import TemporalSpectralGate


def find_recording(timestamp: str) -> Path:
    matches = list((RESEARCH_DIR / "DataSet").rglob(f"*{timestamp}*.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one recording for {timestamp}, found {len(matches)}")
    return matches[0]


def sample_noise(background: np.ndarray, frame_count: int, rng: np.random.Generator) -> np.ndarray:
    residual = background - np.median(background, axis=0)
    return residual[rng.integers(0, residual.shape[0], size=frame_count)]


def synthetic_cases(frame_count: int, channels: int, sample_rate_hz: float) -> dict[str, np.ndarray]:
    time = np.arange(frame_count) / sample_rate_hz
    cases = {}
    steady = np.zeros((frame_count, channels))
    steady[np.ix_((time >= 3.0) & (time < 13.0), [5, 27, 61])] = np.array([8.0, 14.0, 22.0])
    cases["sparse_steady"] = steady
    ramp = np.zeros((frame_count, channels))
    envelope = np.clip((time - 2.0) / 3.0, 0.0, 1.0) * np.clip((15.0 - time) / 3.0, 0.0, 1.0)
    ramp[:, [9, 40, 78]] = envelope[:, None] * np.array([10.0, 18.0, 30.0])
    cases["slow_ramp"] = ramp
    short = np.zeros((frame_count, channels))
    short[(time >= 4.0) & (time < 4.8), 14] = 18.0
    short[(time >= 8.0) & (time < 9.2), 53] = 25.0
    short[(time >= 12.0) & (time < 12.6), 82] = 12.0
    cases["short_contacts"] = short
    multi = np.zeros((frame_count, channels))
    multi[np.ix_((time >= 2.0) & (time < 7.0), [2, 18, 35, 70])] = np.array([7.0, 11.0, 16.0, 24.0])
    multi[np.ix_((time >= 6.0) & (time < 14.0), [12, 48, 90])] = np.array([9.0, 15.0, 20.0])
    cases["overlap_contacts"] = multi
    low_snr = np.zeros((frame_count, channels))
    low_snr[np.ix_((time >= 3.0) & (time < 14.0), [7, 31, 66])] = np.array([3.0, 4.0, 5.0])
    cases["low_snr"] = low_snr
    return cases


def calculate_metrics(noisy: np.ndarray, output: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    active = truth > 0
    background = ~active
    truth_energy = float(np.sum(truth ** 2))
    return {
        "rmse": float(np.sqrt(np.mean((output - truth) ** 2))),
        "srr": float(np.sum(output[active] ** 2) / max(truth_energy, 1e-12)),
        "bnsr_db": float(10.0 * np.log10(max(np.mean(noisy[background] ** 2), 1e-12) / max(np.mean(output[background] ** 2), 1e-12))),
        "peak_retention": float(np.max(output) / max(np.max(truth), 1e-12)),
    }


def run_synthetic(device: str, background: np.ndarray, sample_rate_hz: float, rng: np.random.Generator):
    frame_count = int(round(sample_rate_hz * 18.0))
    gate = TemporalSpectralGate().fit(background, sample_rate_hz)
    baseline = np.median(background, axis=0)
    rows = []
    examples = {}
    for case_name, truth in synthetic_cases(frame_count, background.shape[1], sample_rate_hz).items():
        noisy = truth + sample_noise(background, frame_count, rng)
        output = gate.process(baseline + noisy)
        row = {"device": device, "case": case_name}
        row.update(calculate_metrics(noisy, output, truth))
        rows.append(row)
        examples[case_name] = (truth, noisy, output)
    return rows, examples


def run_real(device: str, background: np.ndarray, stress: np.ndarray, sample_rate_hz: float):
    gate = TemporalSpectralGate().fit(background, sample_rate_hz)
    baseline = np.median(background, axis=0)
    background_output = gate.process(background)
    stress_output = gate.process(stress)
    background_residual = np.maximum(background - baseline, 0.0)
    stress_residual = np.maximum(stress - baseline, 0.0)
    metrics = {
        "device": device,
        "background_suppression_db": float(10.0 * np.log10(max(np.mean(background_residual ** 2), 1e-12) / max(np.mean(background_output ** 2), 1e-12))),
        "stress_energy_retention": float(np.sum(stress_output ** 2) / max(np.sum(stress_residual ** 2), 1e-12)),
        "stress_peak_retention": float(np.max(stress_output) / max(np.max(stress_residual), 1e-12)),
    }
    return metrics, (background_residual, background_output, stress_residual, stress_output)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_synthetic(path: Path, examples, sample_rate_hz: float, title: str):
    case_names = ["sparse_steady", "slow_ramp", "short_contacts", "low_snr"]
    case_labels = {
        "sparse_steady": "持续稀疏接触",
        "slow_ramp": "缓慢加载与卸载",
        "short_contacts": "短时接触",
        "low_snr": "低信噪比接触",
    }
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    for axis, case_name in zip(axes, case_names):
        truth, noisy, output = examples[case_name]
        channel = int(np.argmax(np.max(truth, axis=0)))
        time = np.arange(truth.shape[0]) / sample_rate_hz
        axis.plot(time, noisy[:, channel], color="#999999", linewidth=0.8, label="含噪输入残差")
        axis.plot(time, truth[:, channel], color="#1f77b4", linewidth=1.5, label="压力真值")
        axis.plot(time, output[:, channel], color="#d62728", linewidth=1.2, label="谱门控输出")
        axis.set_title(f"{case_labels[case_name]}（通道 {channel + 1}）")
        axis.set_ylabel("ADC")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("时间（秒）")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def render_real(path: Path, arrays, sample_rate_hz: float, title: str):
    background_raw, background_output, stress_raw, stress_output = arrays
    channel = int(np.argmax(np.mean(stress_raw, axis=0)))
    time = np.arange(stress_raw.shape[0]) / sample_rate_hz
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(time, stress_raw[:, channel], color="#777777", linewidth=0.8, label="背景扣除后原始数据")
    axes[0].plot(time, stress_output[:, channel], color="#d62728", linewidth=1.1, label="谱门控输出")
    axes[0].set_title(f"压力录制代表通道：通道 {channel + 1}")
    axes[0].legend()
    limit = min(300, len(background_raw))
    axes[1].plot(background_raw[:limit, channel], color="#777777", linewidth=0.8, label="背景原始残差")
    axes[1].plot(background_output[:limit, channel], color="#d62728", linewidth=1.1, label="背景处理结果")
    axes[1].set_title("无压力背景片段")
    axes[1].legend()
    for axis in axes:
        axis.set_ylabel("ADC")
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("帧序号")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def render_real_heatmaps(path: Path, arrays, rows: int, cols: int, title: str):
    background_raw, background_output, stress_raw, stress_output = arrays
    if rows * cols != background_raw.shape[1]:
        raise ValueError(f"layout {rows}x{cols} does not match {background_raw.shape[1]} channels")

    background_index = int(np.argmax(np.sum(background_raw, axis=1)))
    stress_totals = np.sum(stress_raw, axis=1)
    stress_target = float(np.quantile(stress_totals, 0.9))
    stress_index = int(np.argmin(np.abs(stress_totals - stress_target)))
    pairs = [
        ("背景残差最大帧", background_raw[background_index], background_output[background_index], background_index),
        ("典型压力帧", stress_raw[stress_index], stress_output[stress_index], stress_index),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for row_index, (label, before, after, frame_index) in enumerate(pairs):
        vmax = max(float(np.max(before)), float(np.max(after)), 1.0)
        images = []
        for column_index, (column_label, values) in enumerate((("处理前", before), ("处理后", after))):
            axis = axes[row_index, column_index]
            matrix = values.reshape(rows, cols)
            image = axis.imshow(matrix, cmap="hot", aspect="equal", vmin=0.0, vmax=vmax, interpolation="nearest")
            images.append(image)
            axis.set_title(f"{label} - {column_label}\n第 {frame_index} 帧，总值 {np.sum(values):.1f} ADC")
            axis.set_xlabel("列")
            axis.set_ylabel("行")
            axis.set_xticks(np.arange(cols))
            axis.set_yticks(np.arange(rows))
            for cell_row in range(rows):
                for cell_col in range(cols):
                    value = float(matrix[cell_row, cell_col])
                    if abs(value) >= 100.0:
                        text = f"{value:.0f}"
                    elif abs(value) >= 10.0:
                        text = f"{value:.1f}"
                    else:
                        text = f"{value:.2f}"
                    label_text = axis.text(
                        cell_col,
                        cell_row,
                        text,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=6.5,
                        fontweight="bold",
                    )
                    label_text.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground="black")])
        figure.colorbar(images[-1], ax=axes[row_index, :], shrink=0.85, label="背景扣除后 ADC")

    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    recordings = {"glove": ("20260710135959", "20260710141208"), "fabric": ("20260710143202", "20260710143312")}
    device_labels = {"glove": "手套", "fabric": "织物垫"}
    rng = np.random.default_rng(20260711)
    synthetic_rows = []
    real_rows = []
    for device, (background_id, stress_id) in recordings.items():
        background, metadata = load_csv(str(find_recording(background_id)))
        stress, _ = load_csv(str(find_recording(stress_id)))
        sample_rate_hz = float(metadata["sample_freq_hz"])
        rows, examples = run_synthetic(device, background, sample_rate_hz, rng)
        synthetic_rows.extend(rows)
        render_synthetic(STAGE_DIR / "output" / f"synthetic_{device}.png", examples, sample_rate_hz, f"{device_labels[device]}：时域谱门控模拟对比")
        metrics, arrays = run_real(device, background, stress, sample_rate_hz)
        real_rows.append(metrics)
        render_real(STAGE_DIR / "output" / f"real_{device}.png", arrays, sample_rate_hz, f"{device_labels[device]}：真实录制处理前后对比")
        render_real_heatmaps(
            STAGE_DIR / "output" / f"real_{device}_heatmap_compare.png",
            arrays,
            int(metadata["rows"]),
            int(metadata["cols"]),
            f"{device_labels[device]}：矩阵 ADC 处理前后对比",
        )
    write_csv(STAGE_DIR / "data" / "synthetic_metrics.csv", synthetic_rows)
    write_csv(STAGE_DIR / "data" / "real_metrics.csv", real_rows)
    for row in synthetic_rows + real_rows:
        print(row)


if __name__ == "__main__":
    main()
