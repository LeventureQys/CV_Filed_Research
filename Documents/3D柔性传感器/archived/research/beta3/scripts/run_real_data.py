from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT.parent
sys.path.insert(0, str(RESEARCH_DIR / "src"))

from alg.data_loader import load_csv
from filters import causal_median3, ema, lowpass_iir, scalar_kalman


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


RECORDINGS = (
    {
        "device": "fabric",
        "label": "织物垫（500 g）",
        "background": ("织物垫", "背景噪声", "录制数据_20260710143202_part0.csv"),
        "stress": ("织物垫", "500g砝码压力", "录制数据_20260710143312_part0.csv"),
    },
    {
        "device": "glove",
        "label": "手套（单手指 1 kg）",
        "background": ("手套", "背景噪声", "录制数据_20260710135959_part0.csv"),
        "stress": ("手套", "单手指1kg", "录制数据_20260710141208_part0.csv"),
    },
    {
        "device": "membrane",
        "label": "64×64 膜片（1 kg）",
        "background": ("64x64膜片", "背景噪音", "录制数据_20260710144907_part0.csv"),
        "stress": ("64x64膜片", "1kg砝码压力", "录制数据_20260710145358_part0.csv"),
    },
)


METHOD_LABELS = {
    "raw": "原始",
    "ema_0.35s": "EMA 0.35 s",
    "ema_0.8s": "EMA 0.8 s",
    "median3_ema": "Median-3 + EMA",
    "iir1_0.5hz": "一阶 IIR 0.5 Hz",
    "iir2_0.5hz": "二阶 IIR 0.5 Hz",
    "kalman_fast": "卡尔曼（快响应）",
    "kalman_stable": "卡尔曼（稳态型）",
    "cell_median3_ema": "逐 Cell Median-3 + EMA",
}


def recording_path(parts: tuple[str, str, str]) -> Path:
    return RESEARCH_DIR / "DataSet" / Path(*parts)


def apply_algorithms(cells: np.ndarray, sample_rate: float) -> dict[str, np.ndarray]:
    total = cells.sum(axis=1)
    measurement_variance = max(float(np.var(np.diff(total))) / 2.0, 1e-6)
    return {
        "raw": total,
        "ema_0.35s": ema(total, sample_rate, 0.35),
        "ema_0.8s": ema(total, sample_rate, 0.8),
        "median3_ema": ema(causal_median3(total), sample_rate, 0.5),
        "iir1_0.5hz": lowpass_iir(total, sample_rate, 0.5, 1),
        "iir2_0.5hz": lowpass_iir(total, sample_rate, 0.5, 2),
        "kalman_fast": scalar_kalman(total, measurement_variance * 0.012, measurement_variance),
        "kalman_stable": scalar_kalman(total, measurement_variance * 0.002, measurement_variance),
        "cell_median3_ema": ema(causal_median3(cells), sample_rate, 0.5).sum(axis=1),
    }


def calculate_metrics(
    device: str,
    label: str,
    sample_rate: float,
    channel_count: int,
    methods: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    guard = min(max(round(sample_rate * 2.0), 1), len(methods["raw"]) // 4)
    evaluation = slice(guard, len(methods["raw"]) - guard)
    raw_roughness = max(float(np.std(np.diff(methods["raw"][evaluation]))), 1e-12)
    for method, values in methods.items():
        segment = values[evaluation]
        roughness = float(np.std(np.diff(segment)))
        rows.append(
            {
                "device": device,
                "recording": label,
                "sample_rate_hz": sample_rate,
                "channels": channel_count,
                "frames": len(values),
                "method": method,
                "mean_total_adc": float(np.mean(segment)),
                "std_total_adc": float(np.std(segment)),
                "peak_to_peak_adc": float(np.ptp(segment)),
                "first_difference_std": roughness,
                "roughness_retention": roughness / raw_roughness,
            }
        )
    return rows


def plot_comparison(cases: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for axis, case in zip(axes, cases):
        time = case["time"]
        methods = case["methods"]
        for index, (method, values) in enumerate(methods.items()):
            if method == "raw":
                axis.plot(time, values, color="#777777", linewidth=0.8, alpha=0.75, label=METHOD_LABELS[method])
            else:
                axis.plot(time, values, color=colors[(index - 1) % len(colors)], linewidth=1.15, label=METHOD_LABELS[method])
        axis.set_title(f"{case['label']}，{case['sample_rate']:.1f} Hz，{case['channels']} Cell")
        axis.set_ylabel("背景扣除后总 ADC")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8, loc="best")
    axes[-1].set_xlabel("时间（秒）")
    figure.suptitle("三类传感器真实录制：beta3 因果时间算法同图对比", fontsize=15)
    figure.savefig(OUTPUT_DIR / "real_data_all_algorithms_comparison.png", dpi=180)
    plt.close(figure)


def write_csv(rows: list[dict[str, object]]) -> None:
    path = DATA_DIR / "real_data_filter_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    metric_rows = []
    for recording in RECORDINGS:
        background, _ = load_csv(str(recording_path(recording["background"])))
        stress, metadata = load_csv(str(recording_path(recording["stress"])))
        sample_rate = float(metadata["sample_freq_hz"])
        baseline = np.median(background, axis=0)
        corrected_cells = np.maximum(stress - baseline, 0.0)
        methods = apply_algorithms(corrected_cells, sample_rate)
        cases.append(
            {
                "label": recording["label"],
                "sample_rate": sample_rate,
                "channels": corrected_cells.shape[1],
                "time": np.arange(len(stress), dtype=float) / sample_rate,
                "methods": methods,
            }
        )
        metric_rows.extend(
            calculate_metrics(
                str(recording["device"]),
                str(recording["label"]),
                sample_rate,
                corrected_cells.shape[1],
                methods,
            )
        )
    plot_comparison(cases)
    write_csv(metric_rows)
    for row in metric_rows:
        print(
            f"{row['device']:8s} {row['method']:24s} "
            f"roughness={row['first_difference_std']:.3f} "
            f"retention={row['roughness_retention']:.3f}"
        )


if __name__ == "__main__":
    main()
