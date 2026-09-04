from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve


@dataclass(frozen=True)
class MembraneCase:
    name: str
    load_type: str
    truth: np.ndarray
    observed: np.ndarray
    kernel: np.ndarray


@dataclass(frozen=True)
class ForceCase:
    name: str
    load_type: str
    sample_rate: float
    time: np.ndarray
    true_cells: np.ndarray
    observed_cells: np.ndarray
    transition_indices: tuple[int, ...]


def crosstalk_kernel(leakage: float, asymmetric: bool = False) -> np.ndarray:
    if not 0.0 <= leakage < 0.45:
        raise ValueError("leakage must be in [0, 0.45)")
    if asymmetric:
        weights = np.array(
            [[0.03, 0.08, 0.04], [0.10, 0.0, 0.16], [0.04, 0.07, 0.03]],
            dtype=float,
        )
        weights /= weights.sum()
        kernel = leakage * weights
    else:
        kernel = leakage * np.array(
            [[0.05, 0.10, 0.05], [0.10, 0.0, 0.10], [0.05, 0.10, 0.05]],
            dtype=float,
        ) / 0.6
    kernel[1, 1] = 1.0 - leakage
    return kernel


def _disc(shape: tuple[int, int], center: tuple[float, float], radius: float, level: float) -> np.ndarray:
    rows, columns = np.indices(shape)
    distance = np.sqrt((rows - center[0]) ** 2 + (columns - center[1]) ** 2)
    edge = np.clip(radius + 0.5 - distance, 0.0, 1.0)
    return level * edge


def membrane_truth(case_name: str, shape: tuple[int, int] = (64, 64)) -> tuple[str, np.ndarray]:
    if case_name == "local_center":
        return "local", _disc(shape, (31.5, 31.5), 5.5, 1000.0)
    if case_name == "local_edge":
        return "local", _disc(shape, (6.0, 31.5), 5.0, 900.0)
    if case_name == "local_multi":
        first = _disc(shape, (20.0, 20.0), 4.5, 800.0)
        second = _disc(shape, (43.0, 42.0), 7.0, 1100.0)
        return "local", np.maximum(first, second)
    if case_name == "global_uniform":
        truth = np.zeros(shape, dtype=float)
        truth[5:-5, 5:-5] = 600.0
        return "global", truth
    if case_name == "global_gradient":
        rows, columns = np.indices(shape)
        mask = (rows >= 5) & (rows < shape[0] - 5) & (columns >= 5) & (columns < shape[1] - 5)
        truth = np.zeros(shape, dtype=float)
        truth[mask] = 350.0 + 450.0 * columns[mask] / (shape[1] - 1)
        return "global", truth
    raise ValueError(f"unknown membrane case: {case_name}")


def simulate_membrane_case(
    case_name: str,
    leakage: float,
    rng: np.random.Generator,
    asymmetric: bool = False,
) -> MembraneCase:
    load_type, truth = membrane_truth(case_name)
    kernel = crosstalk_kernel(leakage, asymmetric)
    observed = convolve(truth, kernel, mode="constant", cval=0.0)
    active_noise = rng.normal(0.0, 1.5, truth.shape) * (observed > 0.0)
    observed = np.maximum(observed + active_noise, 0.0)
    return MembraneCase(case_name, load_type, truth, observed, kernel)


def _force_profile(sample_rate: float, duration: float) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    time = np.arange(round(duration * sample_rate), dtype=float) / sample_rate
    total = np.zeros_like(time)
    schedule = ((15.0, 20.0), (55.0, 40.0), (105.0, 25.0), (150.0, 0.0))
    transition_indices = []
    current = 0.0
    for transition_time, level in schedule:
        index = int(round(transition_time * sample_rate))
        transition_indices.append(index)
        total[index:] += level - current
        current = level
    loaded = total > 0.0
    total[loaded] *= 1.0 + 0.012 * (1.0 - np.exp(-(time[loaded] - 15.0) / 80.0))
    return time, total, tuple(transition_indices)


def simulate_force_case(
    load_type: str,
    rng: np.random.Generator,
    sample_rate: float = 55.0,
    duration: float = 180.0,
    cell_count: int = 64,
) -> ForceCase:
    if load_type not in {"local", "global"}:
        raise ValueError("load_type must be local or global")
    time, true_total, transitions = _force_profile(sample_rate, duration)
    if load_type == "local":
        active_count = 9
        weights = np.exp(-0.5 * (np.arange(active_count) - 4.0) ** 2 / 2.0)
        weights /= weights.sum()
        spatial_weights = np.zeros(cell_count)
        spatial_weights[:active_count] = weights
    else:
        spatial_weights = rng.uniform(0.8, 1.2, cell_count)
        spatial_weights /= spatial_weights.sum()
    true_cells = true_total[:, None] * spatial_weights[None, :]

    sensitivity = rng.uniform(70.0, 100.0, cell_count)
    high_range = true_cells > 0.45
    adc = true_cells * sensitivity
    adc[high_range] = 0.45 * sensitivity[np.newaxis, :].repeat(len(time), axis=0)[high_range] + (
        true_cells[high_range] - 0.45
    ) * (sensitivity[np.newaxis, :].repeat(len(time), axis=0)[high_range] * 0.58)
    fixed_bias = rng.normal(10.0, 2.0, cell_count)
    common_noise = rng.normal(0.0, 1.8, len(time))[:, None]
    independent_noise = rng.normal(0.0, 3.0, adc.shape)
    drift = (0.8 * np.sin(2.0 * np.pi * time / 140.0))[:, None]
    adc_observed = adc + fixed_bias + common_noise + independent_noise + drift
    spike_mask = rng.random(adc.shape) < 0.0008
    adc_observed += spike_mask * rng.normal(0.0, 70.0, adc.shape)
    adc_corrected = np.maximum(adc_observed - fixed_bias, 0.0)

    breakpoint = 0.45 * sensitivity
    observed_cells = adc_corrected / sensitivity
    upper = adc_corrected > breakpoint
    repeated_sensitivity = sensitivity[np.newaxis, :].repeat(len(time), axis=0)
    repeated_breakpoint = breakpoint[np.newaxis, :].repeat(len(time), axis=0)
    observed_cells[upper] = 0.45 + (adc_corrected[upper] - repeated_breakpoint[upper]) / (
        repeated_sensitivity[upper] * 0.58
    )
    return ForceCase(
        name=f"force_{load_type}",
        load_type=load_type,
        sample_rate=sample_rate,
        time=time,
        true_cells=true_cells,
        observed_cells=observed_cells,
        transition_indices=transitions,
    )
