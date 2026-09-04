from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing


class MembraneSpatioTemporalGate:
    def __init__(self, rows: int = 64, cols: int = 64, spatial_threshold: float = 10.0,
                 temporal_threshold: float = 3.0, activation_level: int = 5):
        self.rows = rows
        self.cols = cols
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.activation_level = activation_level

    def build_mask(self, frames: np.ndarray) -> np.ndarray:
        mask = np.max(frames, axis=0).reshape(self.rows, self.cols) >= self.spatial_threshold
        return binary_closing(mask, np.ones((3, 3), dtype=bool), iterations=1).ravel()

    def process_with_mask(self, frames: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
        counter = np.zeros(frames.shape[1], dtype=np.int16)
        output = np.zeros_like(frames, dtype=np.float64)
        for frame_index, frame in enumerate(frames):
            above = frame >= self.temporal_threshold
            counter = np.where(above, np.minimum(counter + 2, self.activation_level * 2),
                               np.maximum(counter - 1, 0))
            active = counter >= self.activation_level
            combined = (active & spatial_mask).reshape(self.rows, self.cols)
            combined = binary_closing(combined, np.ones((3, 3), dtype=bool), iterations=1)
            output[frame_index] = frame * combined.ravel()
        return output

    def process_offline(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = self.build_mask(frames)
        return self.process_with_mask(frames, mask), mask

    def process_prefix_causal(self, frames: np.ndarray) -> np.ndarray:
        counter = np.zeros(frames.shape[1], dtype=np.int16)
        running_max = np.zeros(frames.shape[1], dtype=np.float64)
        output = np.zeros_like(frames, dtype=np.float64)
        for frame_index, frame in enumerate(frames):
            running_max = np.maximum(running_max, frame)
            spatial = binary_closing(
                (running_max >= self.spatial_threshold).reshape(self.rows, self.cols),
                np.ones((3, 3), dtype=bool),
                iterations=1,
            ).ravel()
            above = frame >= self.temporal_threshold
            counter = np.where(above, np.minimum(counter + 2, self.activation_level * 2),
                               np.maximum(counter - 1, 0))
            active = counter >= self.activation_level
            combined = binary_closing(
                (active & spatial).reshape(self.rows, self.cols),
                np.ones((3, 3), dtype=bool),
                iterations=1,
            ).ravel()
            output[frame_index] = frame * combined
        return output

