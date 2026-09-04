from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


class TemporalSpectralGate:
    def __init__(
        self,
        window_seconds: float = 1.6,
        overlap_ratio: float = 0.75,
        oversubtraction: float = 1.0,
        gain_floor: float = 0.08,
        gain_power: float = 0.7,
    ):
        self.window_seconds = window_seconds
        self.overlap_ratio = overlap_ratio
        self.oversubtraction = oversubtraction
        self.gain_floor = gain_floor
        self.gain_power = gain_power
        self.sample_rate_hz = None
        self.baseline = None
        self.noise_psd = None
        self.nperseg = None
        self.noverlap = None

    def fit(self, background_frames: np.ndarray, sample_rate_hz: float):
        frames = np.asarray(background_frames, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[0] < 8:
            raise ValueError("background_frames must have shape (frames, channels) with at least 8 frames")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        self.sample_rate_hz = float(sample_rate_hz)
        self.baseline = np.median(frames, axis=0)
        self.nperseg = min(frames.shape[0], max(8, int(round(self.window_seconds * self.sample_rate_hz))))
        self.noverlap = min(self.nperseg - 1, int(round(self.nperseg * self.overlap_ratio)))

        residual = frames - self.baseline
        _, _, spectrum = stft(
            residual.T,
            fs=self.sample_rate_hz,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary="zeros",
            padded=True,
            axis=-1,
        )
        power = np.abs(spectrum) ** 2
        self.noise_psd = np.median(power, axis=-1, keepdims=True)
        floor = np.quantile(self.noise_psd, 0.25, axis=1, keepdims=True)
        self.noise_psd = np.maximum(self.noise_psd, np.maximum(floor * 0.1, 1e-8))
        return self

    def process(self, frames: np.ndarray) -> np.ndarray:
        if self.noise_psd is None or self.baseline is None:
            raise RuntimeError("fit must be called before process")

        values = np.asarray(frames, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.baseline.shape[0]:
            raise ValueError("frames must have shape (frames, fitted_channels)")

        residual = values - self.baseline
        _, _, spectrum = stft(
            residual.T,
            fs=self.sample_rate_hz,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            boundary="zeros",
            padded=True,
            axis=-1,
        )
        power = np.abs(spectrum) ** 2
        posterior_snr = power / self.noise_psd
        signal_snr = np.maximum(posterior_snr - self.oversubtraction, 0.0)
        gain = signal_snr / (signal_snr + 1.0)
        gain = np.maximum(gain, self.gain_floor) ** self.gain_power
        enhanced = spectrum * gain

        _, restored = istft(
            enhanced,
            fs=self.sample_rate_hz,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            input_onesided=True,
            boundary=True,
            time_axis=-1,
            freq_axis=-2,
        )
        restored = restored[:, :values.shape[0]].T
        return np.maximum(restored, 0.0)
