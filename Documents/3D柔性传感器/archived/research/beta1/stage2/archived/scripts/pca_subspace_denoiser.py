"""
pca_subspace_denoiser.py — Frame-vector PCA subspace denoiser.

Core idea:
  - Treat one frame as one high-dimensional vector x ∈ R^C.
  - Learn a signal subspace from MANY frames of the same recording.
  - Denoise by projecting each frame onto the learned subspace with
    Wiener-like soft shrinkage in component space.

This differs from per-frame 2D SVD:
  - SVD-hard decomposes a single matrix and imposes row/column low-rank.
  - PCA-subspace learns dominant whole-layout components across time.
  - This better matches the user's "same timestamp, whole layout as a whole"
    component-analysis requirement.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np


def load_csv_channels_only(path: str) -> np.ndarray:
    rows_list = []
    n_ch = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            if row[0] == "##数据开始":
                header_row = next(reader, [])
                n_ch = sum(1 for h in header_row if h.strip().startswith("通道"))
                in_data = True
                continue
            if in_data and n_ch is not None and len(row) > n_ch:
                try:
                    rows_list.append([float(v) for v in row[1:1 + n_ch]])
                except (ValueError, IndexError):
                    continue
    return np.asarray(rows_list, dtype=np.float64)


@dataclass
class PCAFitInfo:
    baseline_subtracted: bool
    bg_count: int
    kept_components: int
    energy_ratio: float
    device_mode: str


class PCASubspaceDenoiser:
    def __init__(
        self,
        energy_ratio: float = 0.95,
        sigma_floor: float = 0.5,
        max_components: int = 16,
        active_frame_limit: int = 600,
    ):
        self.energy_ratio = energy_ratio
        self.sigma_floor = sigma_floor
        self.max_components = max_components
        self.active_frame_limit = active_frame_limit

        self.baseline: np.ndarray | None = None
        self.noise_std: np.ndarray | None = None
        self.components: np.ndarray | None = None  # shape [k, C]
        self.eigenvalues: np.ndarray | None = None
        self.keep_k: int = 0
        self.baseline_subtracted: bool = True
        self.device_mode: str = "unknown"

    def _infer_mode(self, bg_frames: np.ndarray) -> str:
        bg_mean = bg_frames.mean(axis=0)
        zero_like_ratio = float((bg_mean < 0.5).sum()) / float(bg_mean.size)
        if zero_like_ratio > 0.80:
            return "membrane"
        return "baseline_white_noise"

    def fit(self, frames: np.ndarray, bg_count: int = 0) -> PCAFitInfo:
        if frames.ndim != 2:
            raise ValueError("frames must be [T, C]")
        t_count, channels = frames.shape
        if t_count < 4:
            raise ValueError("need at least 4 frames")

        if bg_count <= 0:
            bg_count = max(1, min(t_count // 4, 60))

        bg_frames = frames[:bg_count]
        self.device_mode = self._infer_mode(bg_frames)
        self.baseline_subtracted = self.device_mode != "membrane"

        if self.baseline_subtracted:
            self.baseline = np.median(bg_frames, axis=0)
            residual = np.maximum(frames - self.baseline[np.newaxis, :], 0.0)
        else:
            self.baseline = np.zeros(channels, dtype=np.float64)
            residual = frames.copy()

        self.noise_std = np.std(bg_frames, axis=0, ddof=1)
        self.noise_std = np.maximum(self.noise_std, self.sigma_floor)

        whitened = residual / self.noise_std[np.newaxis, :]

        active = whitened[bg_count:]
        if active.shape[0] > self.active_frame_limit:
            step = max(1, active.shape[0] // self.active_frame_limit)
            active = active[::step]

        # Whole-layout subspace learning: rows=frames, cols=channels.
        # Do NOT mean-center by channel; baseline subtraction already removed
        # the DC offset for glove/fabric, and membrane has natural zero bg.
        _, singular_values, vt = np.linalg.svd(active, full_matrices=False)
        eigenvalues = (singular_values * singular_values) / max(active.shape[0], 1)

        total = float(np.sum(eigenvalues))
        if total <= 1e-12:
            keep_k = 1
        else:
            cumulative = np.cumsum(eigenvalues) / total
            keep_k = int(np.searchsorted(cumulative, self.energy_ratio) + 1)

        keep_k = max(1, min(keep_k, self.max_components, vt.shape[0]))
        self.keep_k = keep_k
        self.components = vt[:keep_k]
        self.eigenvalues = eigenvalues[:keep_k]

        return PCAFitInfo(
            baseline_subtracted=self.baseline_subtracted,
            bg_count=bg_count,
            kept_components=keep_k,
            energy_ratio=self.energy_ratio,
            device_mode=self.device_mode,
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.components is None or self.eigenvalues is None or self.noise_std is None or self.baseline is None:
            raise RuntimeError("call fit() before process_frame()")

        if self.baseline_subtracted:
            residual = np.maximum(frame - self.baseline, 0.0)
        else:
            residual = frame.copy()

        whitened = residual / self.noise_std
        coeffs = self.components @ whitened

        # In whitened space, pure noise variance is ~1.
        # Soft Wiener-like shrinkage in component domain.
        gains = np.maximum((self.eigenvalues - 1.0) / np.maximum(self.eigenvalues, 1e-9), 0.0)
        coeffs_shrunk = gains * coeffs

        recon_whitened = self.components.T @ coeffs_shrunk
        recon = np.maximum(recon_whitened * self.noise_std, 0.0)
        return recon.astype(np.float64)

    def process_batch(self, frames: np.ndarray) -> np.ndarray:
        out = np.empty_like(frames, dtype=np.float64)
        for i in range(frames.shape[0]):
            out[i] = self.process_frame(frames[i])
        return out
