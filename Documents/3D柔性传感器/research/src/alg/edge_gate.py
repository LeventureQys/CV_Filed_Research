"""
edge_gate.py — Spatial-structure-aware edge-preserving gate.

Core idea (user's insight):
  - Pressure signal forms contiguous blobs with clear edges.
  - Outside these blobs, values are noise -> suppress to zero.
  - This is closer to edge detection + spatial thresholding than
    per-channel statistical noise modeling.

Algorithm:
  1. Compute per-channel local mean and local variance in a window.
  2. Classify each channel:
     - High local variance + high local mean  -> near edge -> keep
     - Low local variance + high local mean  -> homogeneous pressure -> keep
     - Low local variance + low local mean   -> homogeneous noise -> zero
  3. Morphological cleanup to fill small holes and remove isolated speckles.
  4. Output = raw * mask (preserves original ADC values, no smoothing).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, binary_closing


class EdgeGate:
    def __init__(
        self,
        window: int = 3,
        edge_ratio: float = 2.5,
        signal_ratio: float = 3.0,
        closing_radius: int = 2,
        min_signal_area: int = 3,
    ):
        self.window = window
        self.edge_ratio = edge_ratio
        self.signal_ratio = signal_ratio
        self.closing_radius = closing_radius
        self.min_signal_area = min_signal_area

        self.baseline = None
        self.noise_std = None

    def fit(self, bg_frames: np.ndarray):
        self.baseline = np.median(bg_frames, axis=0)
        self.noise_std = np.std(bg_frames, axis=0, ddof=1)
        self.noise_std = np.maximum(self.noise_std, 0.5)

    def _build_mask(self, frame_2d: np.ndarray) -> np.ndarray:
        """Build binary keep mask from local structure analysis."""
        local_mean = uniform_filter(frame_2d, size=self.window)
        local_sq = uniform_filter(frame_2d ** 2, size=self.window)
        local_var = np.maximum(local_sq - local_mean ** 2, 0.0)

        noise_var = np.median(local_var)
        if noise_var < 1e-6:
            noise_var = 1.0

        edge_score = local_var / (noise_var + 1e-6)
        signal_score = local_mean / (np.sqrt(noise_var) + 1e-6)

        keep_mask = (edge_score > self.edge_ratio) | (signal_score > self.signal_ratio)

        structure = np.ones((3, 3), dtype=bool)
        keep_mask = binary_closing(keep_mask, structure=structure, iterations=self.closing_radius)

        return keep_mask

    def _clean_small_regions(self, mask_2d: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than min_signal_area."""
        from scipy.ndimage import label
        labeled, n_labels = label(mask_2d)
        for i in range(1, n_labels + 1):
            if (labeled == i).sum() < self.min_signal_area:
                mask_2d[labeled == i] = False
        return mask_2d

    def process_frame(self, frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
        if self.baseline is not None:
            residual = np.maximum(frame - self.baseline, 0.0)
        else:
            residual = frame.copy()

        frame_2d = residual.reshape(rows, cols)
        mask = self._build_mask(frame_2d)
        mask = self._clean_small_regions(mask)

        out_2d = frame_2d * mask
        return out_2d.ravel().astype(np.float64)

    def process_batch(self, frames: np.ndarray, rows: int, cols: int) -> np.ndarray:
        out = np.empty_like(frames, dtype=np.float64)
        for i in range(frames.shape[0]):
            out[i] = self.process_frame(frames[i], rows, cols)
        return out
