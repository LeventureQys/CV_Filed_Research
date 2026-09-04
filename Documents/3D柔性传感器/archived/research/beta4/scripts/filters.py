from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

_b3_path = Path(__file__).resolve().parents[2] / "beta3" / "scripts"
_b3_spec = importlib.util.spec_from_file_location("_beta3_filters", _b3_path / "filters.py")
_b3 = importlib.util.module_from_spec(_b3_spec)
_b3_spec.loader.exec_module(_b3)
causal_median3 = _b3.causal_median3
ema = _b3.ema
lowpass_iir = _b3.lowpass_iir
scalar_kalman = _b3.scalar_kalman
hard_gate = _b3.hard_gate
spatial_median = _b3.spatial_median
local_crosstalk_correction = _b3.local_crosstalk_correction
richardson_lucy = _b3.richardson_lucy


def adaptive_baseline(cells: np.ndarray, sample_rate: float, window_s: float = 10.0) -> np.ndarray:
    window = max(1, round(sample_rate * window_s))
    out = np.empty_like(cells)
    for i in range(len(cells)):
        if i >= window:
            baseline = np.median(cells[i - window : i + 1], axis=0)
            out[i] = cells[i] - baseline
        else:
            out[i] = cells[i]
    return np.maximum(out, 0.0)


def dual_state_kalman(values: np.ndarray, sample_rate: float,
                      q_force: float = 0.01, q_drift: float = 1e-4,
                      r_measure: float = 1.0) -> np.ndarray:
    dt = 1.0 / max(sample_rate, 1e-6)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([q_force, q_drift])
    R = np.array([[r_measure]])

    arr = np.asarray(values, dtype=float)
    x = np.array([arr[0], 0.0])
    P = np.eye(2) * r_measure
    out = np.empty_like(arr)
    out[0] = x[0]

    for k in range(1, len(arr)):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = (P @ H.T) / S[0, 0]
        innov = arr[k] - (H @ x)[0]
        x = x + K.flatten() * innov
        P = (np.eye(2) - np.outer(K, H[0])) @ P
        out[k] = x[0]
    return out


def huber_ema(values: np.ndarray, sample_rate: float,
              time_constant: float = 0.8,
              threshold_sigma: float = 3.0) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / (sample_rate * max(time_constant, 0.01)))
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    out[0] = arr[0]
    residual_history = []

    for k in range(1, len(arr)):
        residual = arr[k] - out[k - 1]
        residual_history.append(abs(residual))
        max_history = max(50, round(sample_rate * 5))
        if len(residual_history) > max_history:
            residual_history.pop(0)

        if len(residual_history) > 5:
            sigma = float(np.median(residual_history)) * 1.4826
        else:
            sigma = abs(residual)
        sigma = max(sigma, 1e-6)

        if abs(residual) > threshold_sigma * sigma:
            residual = np.sign(residual) * threshold_sigma * sigma

        out[k] = out[k - 1] + alpha * residual
    return out
