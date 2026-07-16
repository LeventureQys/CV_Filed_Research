"""
temporal_gate.py — Temporal consistency gate.

Core principle:
  White noise is frame-to-frame independent (flashes randomly).
  Real pressure persists across consecutive frames (physical contact).

  For each channel:
    if value > baseline + k*noise_std for N consecutive frames → signal → keep
    else → noise → zero

This uses the ONE property that cleanly separates signal from noise
regardless of spatial structure, SNR, or baseline level.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing


class TemporalGate:
    def __init__(
        self,
        min_consecutive: int = 3,
        k_sigma: float = 1.5,
        closing_radius: int = 1,
        min_area: int = 2,
    ):
        self.min_consecutive = min_consecutive
        self.k_sigma = k_sigma
        self.closing_radius = closing_radius
        self.min_area = min_area

        self.baseline = None
        self.noise_std = None
        self.threshold = None
        self.consecutive_count = None

    def fit(self, bg_frames: np.ndarray):
        self.baseline = np.median(bg_frames, axis=0)
        self.noise_std = np.std(bg_frames, axis=0, ddof=1)
        self.noise_std = np.maximum(self.noise_std, 0.5)
        self.threshold = self.baseline + self.k_sigma * self.noise_std
        self.consecutive_count = np.zeros(bg_frames.shape[1], dtype=int)

    def process_frame(self, frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
        if self.baseline is not None:
            residual = np.maximum(frame - self.baseline, 0.0)
            threshold = self.threshold if self.threshold is not None else self.baseline + self.k_sigma * self.noise_std
        else:
            # No baseline → absolute threshold (membrane mode)
            residual = frame.copy()
            threshold = np.full_like(frame, 3.0)
            if self.consecutive_count is None:
                self.consecutive_count = np.zeros(len(frame), dtype=int)
            if self.threshold is None:
                self.threshold = threshold

        above = frame >= threshold
        self.consecutive_count = np.where(
            above,
            np.minimum(self.consecutive_count + 2, self.min_consecutive * 2),
            np.maximum(self.consecutive_count - 1, 0))

        active = self.consecutive_count >= self.min_consecutive

        # Spatial cleanup: closing + remove small islands
        if rows > 1 and cols > 1:
            mask2d = active.reshape(rows, cols)
            mask2d = binary_closing(mask2d, structure=np.ones((3, 3), dtype=bool),
                                    iterations=self.closing_radius)
            active = mask2d.ravel()

        out = residual * active
        return out.astype(np.float64)

    def process_batch(self, frames: np.ndarray, rows: int, cols: int) -> np.ndarray:
        out = np.empty_like(frames, dtype=np.float64)
        # Reset state
        self.consecutive_count = np.zeros(frames.shape[1], dtype=int)
        for i in range(frames.shape[0]):
            out[i] = self.process_frame(frames[i], rows, cols)
        return out
