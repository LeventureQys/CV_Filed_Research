from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, lfilter


def causal_median3(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = array.copy()
    for index in range(2, len(array)):
        output[index] = np.median(array[index - 2 : index + 1], axis=0)
    return output


def ema(values: np.ndarray, sample_rate: float, time_constant: float) -> np.ndarray:
    if sample_rate <= 0.0 or time_constant <= 0.0:
        raise ValueError("sample_rate and time_constant must be positive")
    array = np.asarray(values, dtype=float)
    alpha = 1.0 - np.exp(-1.0 / (sample_rate * time_constant))
    output = np.empty_like(array)
    output[0] = array[0]
    for index in range(1, len(array)):
        output[index] = output[index - 1] + alpha * (array[index] - output[index - 1])
    return output


def lowpass_iir(values: np.ndarray, sample_rate: float, cutoff: float, order: int) -> np.ndarray:
    if not 0.0 < cutoff < sample_rate / 2.0:
        raise ValueError("cutoff must be between zero and Nyquist")
    numerator, denominator = butter(order, cutoff, btype="low", fs=sample_rate)
    initial = np.asarray(values, dtype=float)[0]
    centered = np.asarray(values, dtype=float) - initial
    return lfilter(numerator, denominator, centered, axis=0) + initial


def scalar_kalman(values: np.ndarray, process_variance: float, measurement_variance: float) -> np.ndarray:
    process_variance = np.asarray(process_variance, dtype=float)
    measurement_variance = np.asarray(measurement_variance, dtype=float)
    if np.any(process_variance <= 0.0) or np.any(measurement_variance <= 0.0):
        raise ValueError("Kalman variances must be positive")
    array = np.asarray(values, dtype=float)
    output = np.empty_like(array)
    estimate = array[0].copy()
    covariance = np.full_like(estimate, measurement_variance, dtype=float)
    output[0] = estimate
    for index in range(1, len(array)):
        covariance = covariance + process_variance
        gain = covariance / (covariance + measurement_variance)
        estimate = estimate + gain * (array[index] - estimate)
        covariance = (1.0 - gain) * covariance
        output[index] = estimate
    return output


def hard_gate(image: np.ndarray, relative_threshold: float = 0.08) -> np.ndarray:
    threshold = float(np.max(image)) * relative_threshold
    return np.where(image >= threshold, image, 0.0)


def spatial_median(image: np.ndarray) -> np.ndarray:
    return median_filter(image, size=3, mode="nearest")


def local_crosstalk_correction(observed: np.ndarray, kernel: np.ndarray, iterations: int = 12) -> np.ndarray:
    center = float(kernel[1, 1])
    neighbors = kernel.copy()
    neighbors[1, 1] = 0.0
    estimate = np.maximum(observed / center, 0.0)
    for _ in range(iterations):
        leaked = _convolve_same(estimate, neighbors)
        estimate = np.maximum((observed - leaked) / center, 0.0)
    return estimate


def richardson_lucy(observed: np.ndarray, kernel: np.ndarray, iterations: int = 20) -> np.ndarray:
    estimate = np.maximum(observed, 1e-6)
    mirrored = kernel[::-1, ::-1]
    normalization = _convolve_same(np.ones_like(observed), mirrored)
    for _ in range(iterations):
        prediction = np.maximum(_convolve_same(estimate, kernel), 1e-6)
        estimate *= _convolve_same(observed / prediction, mirrored) / np.maximum(normalization, 1e-6)
    return np.maximum(estimate, 0.0)


def _convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    return convolve(image, kernel, mode="constant", cval=0.0)
