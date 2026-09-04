"""
noise_suppressor.py — TactileSense Noise Suppressor
Reference: WebRTC NoiseSuppressor Analyze/Process architecture.
Adapted for multi-channel low-frequency ADC tactile sensor frames.

Architecture
────────────
  Analyzer (offline/first-N)
    ├─ learn per-channel baseline, std, noise floor
    └─ compute per-channel noise estimates

  Processor (online per frame)
    ├─ Mode::StatGate    — baseline subtract + threshold gate (Stage1, for glove/fabric)
    ├─ Mode::Spatial     — temporal+spatial smoothing (Stage2, for 64x64 membrane)
    └─ Mode::Hybrid      — combine both: gate then smooth residual

  NoiseKind: White (active), Pink/Brown (reserved placeholder)
"""

from __future__ import annotations

import math
import time
from enum import IntEnum
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# NoiseKind — only White implemented; Pink and Brown are placeholders
# ---------------------------------------------------------------------------
class NoiseKind:
    """Enumeration of supported noise types (reserved for future)."""
    White = "white"
    Pink = "pink"     # not implemented
    Brown = "brown"   # not implemented


# ---------------------------------------------------------------------------
# ProcessorMode — denoising strategy selection
# ---------------------------------------------------------------------------
class ProcessorMode(IntEnum):
    StatGate = 0   # Stage1: baseline subtract + threshold gate
    Spatial = 1    # Stage2: temporal avg + spatial gaussian (for 64x64 membrane)
    Hybrid = 2     # Stage2: gate first, then smooth residuals
    EdgeGate = 3   # beta2: spatial structure-aware edge-preserving gate
    TemporalGate = 4  # beta2: temporal consistency gate (membrane)
    SpatioTemporal = 5  # beta2: spatial + temporal combined (membrane final)


# ---------------------------------------------------------------------------
# AnalyzerResult — learned noise model (serialisable dict)
# ---------------------------------------------------------------------------
class AnalyzerResult:
    """Output of Analyzer.analyze() — per-channel noise model."""

    def __init__(self):
        self.n_channels: int = 0
        self.baseline: np.ndarray = np.array([], dtype=np.float64)
        self.noise_std: np.ndarray = np.array([], dtype=np.float64)
        self.noise_floor: np.ndarray = np.array([], dtype=np.float64)
        self.max_adc: np.ndarray = np.array([], dtype=np.float64)
        self.channel_active: np.ndarray = np.array([], dtype=bool)
        self.sample_freq_hz: float = 0.0
        self.rows: int = 0
        self.cols: int = 0
        self.noise_kind: str = NoiseKind.White
        self.n_frames_analyzed: int = 0
        self.metadata: dict = {}

    def to_dict(self) -> dict:
        return {
            "n_channels": self.n_channels,
            "baseline": self.baseline.tolist(),
            "noise_std": self.noise_std.tolist(),
            "noise_floor": self.noise_floor.tolist(),
            "max_adc": self.max_adc.tolist(),
            "channel_active": self.channel_active.tolist(),
            "sample_freq_hz": self.sample_freq_hz,
            "rows": self.rows,
            "cols": self.cols,
            "noise_kind": self.noise_kind,
            "n_frames_analyzed": self.n_frames_analyzed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalyzerResult":
        obj = cls()
        obj.n_channels = d["n_channels"]
        obj.baseline = np.array(d["baseline"])
        obj.noise_std = np.array(d["noise_std"])
        obj.noise_floor = np.array(d["noise_floor"])
        obj.max_adc = np.array(d["max_adc"])
        obj.channel_active = np.array(d["channel_active"])
        obj.sample_freq_hz = d["sample_freq_hz"]
        obj.rows = d.get("rows", 0)
        obj.cols = d.get("cols", 0)
        obj.noise_kind = d["noise_kind"]
        obj.n_frames_analyzed = d["n_frames_analyzed"]
        obj.metadata = d.get("metadata", {})
        return obj


# ---------------------------------------------------------------------------
# Analyzer — learn noise model from background frames
# ---------------------------------------------------------------------------
class Analyzer:
    """
    Learn per-channel baseline and noise profile from background frames.

    Parameters
    ----------
    k_sigma : float
        Multiplier on noise_std for the noise floor threshold.
        Default 3.0 (covers ~99.7% of white noise under Gaussian assumption).
    min_noise_std : float
        Minimum noise standard deviation to prevent division by zero (in ADC units).
    rows, cols : int
        Sensor layout dimensions (0 = unknown, no spatial processing).
    """

    def __init__(self, k_sigma: float = 3.0, min_noise_std: float = 0.5,
                 rows: int = 0, cols: int = 0):
        self.k_sigma = k_sigma
        self.min_noise_std = min_noise_std
        self.rows = rows
        self.cols = cols

    def analyze(self, frames: np.ndarray, sample_freq_hz: float = 0.0) -> AnalyzerResult:
        """
        Parameters
        ----------
        frames : ndarray, shape (N, C)
            Background noise frames (N frames, C channels).
        sample_freq_hz : float
            Estimated sampling frequency.

        Returns
        -------
        AnalyzerResult
        """
        N, C = frames.shape
        baseline = np.median(frames, axis=0)
        noise_std = np.std(frames, axis=0, ddof=1)
        noise_std = np.maximum(noise_std, self.min_noise_std)
        noise_floor = baseline + self.k_sigma * noise_std
        max_adc = np.max(frames, axis=0)
        channel_active = noise_std > self.min_noise_std * 1.5

        # Infer rows/cols from metadata or channel count
        rows = self.rows
        cols = self.cols
        if rows <= 0 or cols <= 0:
            side = int(math.isqrt(C))
            if side * side == C:
                rows = cols = side

        result = AnalyzerResult()
        result.n_channels = C
        result.baseline = baseline
        result.noise_std = noise_std
        result.noise_floor = noise_floor
        result.max_adc = max_adc
        result.channel_active = channel_active
        result.sample_freq_hz = sample_freq_hz
        result.rows = rows
        result.cols = cols
        result.noise_kind = NoiseKind.White
        result.n_frames_analyzed = N
        result.metadata = {
            "k_sigma": self.k_sigma,
            "min_noise_std": self.min_noise_std,
        }
        return result


# ---------------------------------------------------------------------------
# Processor — apply learned noise model to suppress white noise
# ---------------------------------------------------------------------------
class Processor:
    """
    Apply white noise suppression. Three modes available via ProcessorMode.

    StatGate (mode=0, default):
        Stage1 algorithm: baseline subtract + threshold gate.
        Best for glove/fabric sensors with non-zero background.

    Spatial (mode=1):
        Multi-frame average + temporal median + spatial gaussian.
        Best for 64x64 membrane (near-zero background, spatial noise).
        No baseline subtraction needed.

    Hybrid (mode=2):
        StatGate first, then smooth residuals with spatial filter.
        Best for mixed scenarios.
    """

    def __init__(self, noise_model: AnalyzerResult,
                 noise_gate_decay: float = 0.0,
                 enable_suppression: bool = True,
                 mode: ProcessorMode = ProcessorMode.StatGate,
                 temporal_window: int = 3,
                 spatial_sigma: float = 1.5,
                 # EdgeGate params
                 edge_window: int = 3,
                 edge_ratio: float = 2.5,
                 signal_ratio: float = 3.0):
        """
        Parameters
        ----------
        noise_model : AnalyzerResult
        noise_gate_decay : float
            Subtracted from each channel after gating (StatGate mode only).
        enable_suppression : bool
            If False, pass through raw data.
        mode : ProcessorMode
            Denoising strategy.
        temporal_window : int
            Temporal averaging window size (Spatial mode).
        spatial_sigma : float
            Gaussian spatial filter sigma (Spatial mode).
        edge_window : int
            Local window size for EdgeGate.
        edge_ratio : float
            Edge detection threshold ratio for EdgeGate.
        signal_ratio : float
            Signal detection threshold ratio for EdgeGate.
        """
        self.noise_model = noise_model
        self.noise_gate_decay = noise_gate_decay
        self.enable_suppression = enable_suppression
        self.mode = ProcessorMode(mode)
        self.temporal_window = max(1, temporal_window)
        self.spatial_sigma = max(0.1, spatial_sigma)
        self.edge_window = edge_window
        self.edge_ratio = edge_ratio
        self.signal_ratio = signal_ratio

        self._timing_total_s = 0.0
        self._frames_processed = 0

        # Internal state for temporal processing
        self._frame_buffer = []

    def _apply_statgate(self, data: np.ndarray) -> np.ndarray:
        residual = data - self.noise_model.baseline
        threshold = self.noise_model.noise_floor - self.noise_model.baseline
        out = np.where(residual >= threshold, residual, 0.0)
        out = np.maximum(out - self.noise_gate_decay, 0.0)
        return out

    def _apply_spatial(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal averaging + spatial gaussian smoothing.

        For 64x64 membrane data where background ADC=0, this avoids the
        destructive baseline subtraction that kills all pressure signals.
        """
        rows = self.noise_model.rows
        cols = self.noise_model.cols
        if rows <= 0 or cols <= 0:
            return data.copy()

        out = np.asarray(data, dtype=np.float64)

        # 1. Multi-frame averaging (temporal)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        N, C = out.shape

        if N > 1:
            from scipy.ndimage import uniform_filter1d
            try:
                out = uniform_filter1d(out, size=min(self.temporal_window, N), axis=0)
            except ImportError:
                pass

        # 2. Per-frame spatial gaussian smoothing
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return out.reshape(-1, C) if out.ndim == 3 else out

        for i in range(N):
            frame_2d = out[i].reshape(rows, cols)
            smoothed = gaussian_filter(frame_2d, sigma=self.spatial_sigma, mode="reflect")
            out[i] = smoothed.ravel()

        return out.reshape(-1, C) if out.ndim == 3 else out

    def _apply_hybrid(self, data: np.ndarray) -> np.ndarray:
        """StatGate then smooth residuals."""
        out = self._apply_statgate(data)
        # Smooth the non-zero residual channels spatially
        rows = self.noise_model.rows
        cols = self.noise_model.cols
        if rows <= 0 or cols <= 0:
            return out

        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            return out

        if out.ndim == 1:
            out = out.reshape(1, -1)
        N, C = out.shape

        for i in range(N):
            frame_2d = out[i].reshape(rows, cols)
            smoothed = gaussian_filter(frame_2d, sigma=self.spatial_sigma, mode="reflect")
            out[i] = smoothed.ravel()

        return out

    def process(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if not self.enable_suppression:
            out = np.asarray(frame, dtype=np.float64).copy()
        else:
            if self.mode == ProcessorMode.StatGate:
                out = self._apply_statgate(np.asarray(frame, dtype=np.float64))
            elif self.mode == ProcessorMode.Spatial:
                out = self._apply_spatial(np.asarray(frame, dtype=np.float64))
            elif self.mode == ProcessorMode.Hybrid:
                out = self._apply_hybrid(np.asarray(frame, dtype=np.float64))
            elif self.mode == ProcessorMode.EdgeGate:
                out = self._apply_edgegate(np.asarray(frame, dtype=np.float64))
            elif self.mode == ProcessorMode.TemporalGate:
                out = self._apply_temporalgate(np.asarray(frame, dtype=np.float64))
            elif self.mode == ProcessorMode.SpatioTemporal:
                out = self._apply_spatiotemporal(np.asarray(frame, dtype=np.float64))
            else:
                out = np.asarray(frame, dtype=np.float64).copy()
        self._timing_total_s += time.perf_counter() - t0
        self._frames_processed += 1
        return out

    def process_batch(self, frames: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if not self.enable_suppression:
            out = np.asarray(frames, dtype=np.float64).copy()
        else:
            if self.mode == ProcessorMode.StatGate:
                out = self._apply_statgate(np.asarray(frames, dtype=np.float64))
            elif self.mode == ProcessorMode.Spatial:
                out = self._apply_spatial(np.asarray(frames, dtype=np.float64))
            elif self.mode == ProcessorMode.Hybrid:
                out = self._apply_hybrid(np.asarray(frames, dtype=np.float64))
            elif self.mode == ProcessorMode.EdgeGate:
                out = self._apply_edgegate(np.asarray(frames, dtype=np.float64))
            elif self.mode == ProcessorMode.TemporalGate:
                out = self._apply_temporalgate(np.asarray(frames, dtype=np.float64))
            elif self.mode == ProcessorMode.SpatioTemporal:
                out = self._apply_spatiotemporal(np.asarray(frames, dtype=np.float64))
            else:
                out = np.asarray(frames, dtype=np.float64).copy()
        self._timing_total_s += time.perf_counter() - t0
        self._frames_processed += frames.shape[0]
        return out

    def _apply_edgegate(self, data: np.ndarray) -> np.ndarray:
        from .edge_gate import EdgeGate
        eg = EdgeGate(
            window=self.edge_window,
            edge_ratio=self.edge_ratio,
            signal_ratio=self.signal_ratio)
        # Only set baseline if meaningful (non-zero for most channels → glove/fabric type)
        bl = self.noise_model.baseline
        if bl.size > 0:
            nonzero_ratio = float((bl > 0.5).sum()) / float(bl.size)
            if nonzero_ratio > 0.5:
                eg.baseline = bl
                eg.noise_std = self.noise_model.noise_std
        rows = self.noise_model.rows
        cols = self.noise_model.cols
        if rows <= 0 or cols <= 0:
            return np.asarray(data, dtype=np.float64).copy()
        if data.ndim == 1:
            return eg.process_frame(data, rows, cols)
        return eg.process_batch(data, rows, cols)

    def _apply_temporalgate(self, data: np.ndarray) -> np.ndarray:
        from .temporal_gate import TemporalGate
        rows = self.noise_model.rows
        cols = self.noise_model.cols
        if rows <= 0 or cols <= 0:
            return np.asarray(data, dtype=np.float64).copy()
        tg = TemporalGate(min_consecutive=5, k_sigma=1.0)
        bl = self.noise_model.baseline
        if bl.size > 0:
            nonzero_ratio = float((bl > 0.5).sum()) / float(bl.size)
            if nonzero_ratio > 0.5:
                # glove/fabric: learn threshold from baseline
                tg.baseline = bl
                tg.noise_std = self.noise_model.noise_std
                tg.threshold = tg.baseline + tg.k_sigma * tg.noise_std
                tg.consecutive_count = np.zeros(bl.size, dtype=int)
        if data.ndim == 1:
            return tg.process_frame(data, rows, cols)
        return tg.process_batch(data, rows, cols)

    def _apply_spatiotemporal(self, data: np.ndarray) -> np.ndarray:
        """Spatial + Temporal combined gate for membrane.

        Spatial: max_across_frames >= 10 ADC defines pressure blob regions.
        Temporal: leaky counter (threshold=3, min_cons=5) confirms persistence.
        Output: spatial AND temporal → keep; otherwise → zero.
        """
        from scipy.ndimage import binary_closing
        rows = self.noise_model.rows
        cols = self.noise_model.cols
        if rows <= 0 or cols <= 0:
            return np.asarray(data, dtype=np.float64).copy()

        is_batch = data.ndim == 2
        if not is_batch:
            data = data.reshape(1, -1)

        C = data.shape[1]
        spatial_thr = 10.0
        spatial_mask = (data.max(axis=0) >= spatial_thr).reshape(rows, cols)
        spatial_mask = binary_closing(spatial_mask, np.ones((3, 3), dtype=bool), iterations=1)
        spatial_flat = spatial_mask.ravel()

        threshold = 3.0
        min_cons = 5
        counter = np.zeros(C, dtype=int)
        out = np.empty_like(data)

        for i in range(data.shape[0]):
            above = data[i] >= threshold
            counter = np.where(above, np.minimum(counter + 2, min_cons * 2),
                               np.maximum(counter - 1, 0))
            temporal_active = counter >= min_cons
            combined = temporal_active & spatial_flat
            mask = combined.reshape(rows, cols)
            mask = binary_closing(mask, np.ones((3, 3), dtype=bool), iterations=1)
            out[i] = data[i] * mask.ravel()

        return out if is_batch else out[0]

    @property
    def avg_process_time_ms(self) -> float:
        if self._frames_processed == 0:
            return 0.0
        return self._timing_total_s / self._frames_processed * 1000.0

    @property
    def rtf(self) -> float:
        if self._frames_processed == 0 or self.noise_model.sample_freq_hz <= 0:
            return float("inf")
        frame_duration_s = 1.0 / self.noise_model.sample_freq_hz
        total_frames = self._frames_processed
        return self._timing_total_s / (total_frames * frame_duration_s)

    def reset_timing(self):
        self._timing_total_s = 0.0
        self._frames_processed = 0


# ---------------------------------------------------------------------------
# Convenience: build and run full pipeline
# ---------------------------------------------------------------------------
def build_full_noise_suppressor(
    background_frames: np.ndarray,
    sample_freq_hz: float = 0.0,
    k_sigma: float = 3.0,
    noise_gate_decay: float = 0.0,
    mode: ProcessorMode = ProcessorMode.StatGate,
    rows: int = 0,
    cols: int = 0,
    temporal_window: int = 3,
    spatial_sigma: float = 1.5,
) -> tuple[AnalyzerResult, Processor]:
    """One-shot: analyze background → build processor."""
    analyzer = Analyzer(k_sigma=k_sigma, rows=rows, cols=cols)
    model = analyzer.analyze(background_frames, sample_freq_hz)
    proc = Processor(model, noise_gate_decay=noise_gate_decay,
                     mode=mode, temporal_window=temporal_window,
                     spatial_sigma=spatial_sigma)
    return model, proc
