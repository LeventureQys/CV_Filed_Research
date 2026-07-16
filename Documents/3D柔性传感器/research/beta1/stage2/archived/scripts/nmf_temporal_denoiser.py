"""
nmf_temporal_denoiser.py — Whole-layout temporal NMF denoiser.

Frame sequence X is modeled as:
    X ≈ W @ H

Where:
  - rows of X are frames (time)
  - columns of X are channels (whole layout as one vector)
  - W are time activations
  - H are non-negative spatial basis patterns

This directly uses temporal information and whole-layout structure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import NMF


@dataclass
class NMFFitInfo:
    baseline_subtracted: bool
    bg_count: int
    device_mode: str
    n_components: int
    recon_error: float


class NMFTemporalDenoiser:
    def __init__(
        self,
        max_components: int = 8,
        target_rel_error: float = 0.10,
        activation_smooth: int = 5,
        sigma_floor: float = 0.5,
        random_state: int = 0,
        max_iter: int = 600,
    ):
        self.max_components = max_components
        self.target_rel_error = target_rel_error
        self.activation_smooth = activation_smooth
        self.sigma_floor = sigma_floor
        self.random_state = random_state
        self.max_iter = max_iter

        self.baseline = None
        self.noise_std = None
        self.device_mode = "unknown"
        self.model = None
        self.W = None
        self.H = None
        self.frames_input = None
        self.bg_count = 0

    def _infer_mode(self, bg_frames: np.ndarray) -> str:
        bg_mean = bg_frames.mean(axis=0)
        zero_like_ratio = float((bg_mean < 0.5).sum()) / float(bg_mean.size)
        if zero_like_ratio > 0.80:
            return "membrane"
        return "baseline_white_noise"

    def _prepare_frames(self, frames: np.ndarray, bg_count: int) -> np.ndarray:
        bg = frames[:bg_count] if bg_count > 0 else frames[: max(1, frames.shape[0] // 4)]
        self.device_mode = self._infer_mode(bg)
        baseline_subtracted = self.device_mode != "membrane"
        if baseline_subtracted:
            self.baseline = np.median(bg, axis=0)
            residual = np.maximum(frames - self.baseline[np.newaxis, :], 0.0)
        else:
            self.baseline = np.zeros(frames.shape[1], dtype=np.float64)
            residual = frames.copy()

        self.noise_std = np.std(bg, axis=0, ddof=1)
        self.noise_std = np.maximum(self.noise_std, self.sigma_floor)
        return residual

    def fit(self, frames: np.ndarray, bg_count: int = 0) -> NMFFitInfo:
        if frames.ndim != 2:
            raise ValueError("frames must be [T, C]")
        if bg_count <= 0:
            bg_count = max(1, min(frames.shape[0] // 4, 60))
        self.bg_count = bg_count
        self.frames_input = frames.copy()
        X = self._prepare_frames(frames, bg_count)

        # Fit on active region if available; keep time structure.
        X_fit = X[bg_count:] if bg_count < X.shape[0] else X
        X_fit = np.maximum(X_fit, 0.0)

        best_model = None
        best_W = None
        best_H = None
        best_k = 1
        best_err = float("inf")

        denom = np.linalg.norm(X_fit) + 1e-9
        for k in range(1, min(self.max_components, X_fit.shape[0], X_fit.shape[1]) + 1):
            model = NMF(
                n_components=k,
                init="nndsvda",
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
            W = model.fit_transform(X_fit)
            H = model.components_
            recon = W @ H
            rel_err = float(np.linalg.norm(X_fit - recon) / denom)
            best_model, best_W, best_H, best_k, best_err = model, W, H, k, rel_err
            if rel_err <= self.target_rel_error:
                break

        self.model = best_model
        self.H = best_H

        # transform full sequence using learned basis
        W_full = self.model.transform(np.maximum(X, 0.0))
        if self.activation_smooth > 1:
            from scipy.ndimage import median_filter
            W_full = median_filter(W_full, size=(self.activation_smooth, 1))
        self.W = W_full

        return NMFFitInfo(
            baseline_subtracted=(self.device_mode != "membrane"),
            bg_count=bg_count,
            device_mode=self.device_mode,
            n_components=best_k,
            recon_error=best_err,
        )

    def process_frame(self, frame_index: int) -> np.ndarray:
        if self.W is None or self.H is None:
            raise RuntimeError("call fit() first")
        recon = self.W[frame_index] @ self.H
        return np.maximum(recon, 0.0).astype(np.float64)

    def process_batch(self) -> np.ndarray:
        if self.W is None or self.H is None:
            raise RuntimeError("call fit() first")
        return np.maximum(self.W @ self.H, 0.0).astype(np.float64)
