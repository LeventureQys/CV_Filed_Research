"""
replay_animator.py — Interactive animation tool for visualizing noise suppression.

Usage:
    python tools/replay_animator.py <path_to_csv>

Controls:
    Space         — pause/resume
    R             — reset to first frame
    Q / Esc       — quit
    Left/Right    — step one frame back/forward
    Up/Down       — speed up/slow down
    O             — toggle overlay: show original vs processed side by side
    S             — save current frame as PNG
    A             — toggle auto-crop (hide dead channels in heatmap)

Displays:
  Left panel:   Original frame (heatmap)
  Right panel:  Processed frame (heatmap)
  Below:        Per-frame statistics (total, peak, active channels, suppression ratio)
  Title:        Frame index / total, timestamp
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backend_bases import KeyEvent
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alg.data_loader import load_csv
from alg.noise_suppressor import Analyzer, Processor, ProcessorMode, AnalyzerResult


class ReplayAnimator:
    """Interactive heatmap animation viewer for noise suppression."""

    def __init__(self, csv_path: str,
                 k_sigma: float = 1.0,
                 noise_gate_decay: float = 0.0,
                 fps: int = 10,
                 vmin: float = 0.0,
                 vmax: float = None,
                 auto_crop: bool = True,
                 overlay: bool = True,
                 mode: int = 0,
                 temporal_window: int = 3,
                 spatial_sigma: float = 1.5):
        self.csv_path = csv_path
        self.k_sigma = k_sigma
        self.noise_gate_decay = noise_gate_decay
        self.fps = fps
        self.auto_crop = auto_crop
        self.show_overlay = overlay
        self.mode = ProcessorMode(mode)
        self.temporal_window = temporal_window
        self.spatial_sigma = spatial_sigma

        # Load data
        print(f"[load] {csv_path}")
        self.frames_orig, self.meta = load_csv(csv_path)
        self.n_frames, self.n_channels = self.frames_orig.shape
        print(f"  frames={self.n_frames} channels={self.n_channels} "
              f"rows={self.meta.get('rows','?')} cols={self.meta.get('cols','?')} "
              f"freq={self.meta.get('sample_freq_hz',0):.1f} Hz")

        # Infer rows/cols from channels
        rows = self.meta.get("rows", 1)
        cols = self.meta.get("cols", 1)
        if rows * cols != self.n_channels:
            # fallback: try to infer; for known patterns
            if self.n_channels == 96:
                rows, cols = 12, 8
            elif self.n_channels == 4096:
                rows, cols = 64, 64
            else:
                cols = int(np.sqrt(self.n_channels))
                rows = self.n_channels // cols
                if rows * cols < self.n_channels:
                    rows += 1
        self.nrows, self.ncols = rows, cols

        # Background frames: first N frames as noise model
        n_bg = min(100, self.n_frames // 2)
        if n_bg < 10:
            n_bg = self.n_frames
        bg_frames = self.frames_orig[:n_bg]

        # Analyze + Process
        freq = self.meta.get("sample_freq_hz", 0.0)
        self.analyzer = Analyzer(k_sigma=k_sigma, rows=self.nrows, cols=self.ncols)
        self.noise_model = self.analyzer.analyze(bg_frames, freq)
        self.processor = Processor(self.noise_model, noise_gate_decay=noise_gate_decay,
                                   mode=self.mode, temporal_window=self.temporal_window,
                                   spatial_sigma=self.spatial_sigma)
        self.frames_processed = self.processor.process_batch(self.frames_orig)

        print(f"  noise model: {n_bg} frames analyzed")
        print(f"  baseline:  mean={self.noise_model.baseline.mean():.1f} "
              f"min={self.noise_model.baseline.min():.1f} max={self.noise_model.baseline.max():.1f}")
        print(f"  noise_std: mean={self.noise_model.noise_std.mean():.1f} "
              f"min={self.noise_model.noise_std.min():.1f}")

        # Compute display scales independently. Processed values are residuals after
        # baseline subtraction and usually much smaller than raw ADC values.
        self.vmax = vmax
        if self.vmax is None:
            self.vmax = np.percentile(self.frames_orig, 99.5)
            self.vmax = max(self.vmax, 1.0)
        self.processed_vmax = max(np.percentile(self.frames_processed, 99.5), 1.0)

        # State
        self.idx = 0
        self.paused = False
        self.speed_mult = 1.0
        self._stats_counter = 0
        self._running = False

        # Build figure
        self._build_figure()

        # Timer-driven animation (FuncAnimation causes Tk blit crashes
        # when the timer fires before the window is mapped).
        self._anim_timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self._anim_timer.add_callback(self._step)
        self._anim_timer.stop()

    def _build_figure(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        self.fig = fig
        self.ax_orig = axes[0]
        self.ax_proc = axes[1]

        self.im_orig = self.ax_orig.imshow(
            np.zeros((self.nrows, self.ncols)), cmap="hot", aspect="auto",
            vmin=0, vmax=self.vmax, interpolation="nearest")
        self.ax_orig.set_title("Original")
        self.ax_orig.set_xlabel("Column")
        self.ax_orig.set_ylabel("Row")

        self.im_proc = self.ax_proc.imshow(
            np.zeros((self.nrows, self.ncols)), cmap="hot", aspect="auto",
            vmin=0, vmax=self.processed_vmax, interpolation="nearest")
        self.ax_proc.set_title("Processed")
        self.ax_proc.set_xlabel("Column")
        self.ax_proc.set_ylabel("Row")

        fig.subplots_adjust(left=0.06, right=0.92, bottom=0.12, top=0.88, wspace=0.25)

        # Colorbar
        self.cbar = fig.colorbar(self.im_proc, ax=axes, shrink=0.7, label="Processed residual ADC")

        # Stats text
        self.stats_text = fig.text(0.02, 0.02, "", fontsize=9, family="monospace",
                                   verticalalignment="bottom",
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Full draw once to establish backgrounds for blit
        fig.canvas.draw()

        # Save background regions for manual blitting
        self._bg_orig = self.fig.canvas.copy_from_bbox(self.ax_orig.bbox)
        self._bg_proc = self.fig.canvas.copy_from_bbox(self.ax_proc.bbox)
        self._bg_fig  = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        # Connect keys
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Show controls in console
        print("\nControls:")
        print("  Space   — pause/resume")
        print("  R       — reset")
        print("  Q/Esc   — quit")
        print("  ← / →   — step frame")
        print("  ↑ / ↓   — speed (x0.5 / x2)")
        print("  O       — toggle overlay mode")
        print("  S       — save PNG")
        print("  A       — toggle auto-crop\n")

    def _reshape(self, frame: np.ndarray) -> np.ndarray:
        if frame.size == self.nrows * self.ncols:
            return frame.reshape(self.nrows, self.ncols)
        return np.zeros((self.nrows, self.ncols))

    def _render_frame(self):
        """Paint current self.idx onto the figure using manual blit."""
        i = self.idx
        orig = self.frames_orig[i]
        proc = self.frames_processed[i]

        self.im_orig.set_data(self._reshape(orig))
        self.im_proc.set_data(self._reshape(proc))

        # Manual blit: restore background, render artists, update
        canvas = self.fig.canvas
        canvas.restore_region(self._bg_orig)
        canvas.restore_region(self._bg_proc)
        self.ax_orig.draw_artist(self.im_orig)
        self.ax_proc.draw_artist(self.im_proc)
        canvas.blit(self.ax_orig.bbox)
        canvas.blit(self.ax_proc.bbox)

        self._stats_counter += 1
        if (self._stats_counter % 10 == 0) or (i == 0):
            orig_total = float(orig.sum())
            proc_total = float(proc.sum())
            orig_peak = float(orig.max())
            proc_peak = float(proc.max())
            nz_orig = int((orig > 0).sum())
            nz_proc = int((proc > 0).sum())
            suppress_db = 0
            if orig_total > 0:
                suppress_db = max(0, 20 * np.log10(orig_total / max(proc_total, 1)))

            self.fig.suptitle(
                f"Frame {i+1}/{self.n_frames}  |  "
                f"{'PAUSED' if self.paused else f'{self.fps*self.speed_mult:.1f} fps'}  |  "
                f"Speed x{self.speed_mult:.1f}",
                fontsize=13, fontweight="bold")
            self.stats_text.set_text(
                f"Original:  total={orig_total:.0f}  peak={orig_peak:.0f}  nz_ch={nz_orig}\n"
                f"Processed: total={proc_total:.0f}  peak={proc_peak:.0f}  nz_ch={nz_proc}\n"
                f"Suppression: {suppress_db:.1f} dB  |  "
                f"Active channels: {self.noise_model.channel_active.sum():.0f}/{self.n_channels}")
            canvas.draw_idle()

    def _step(self):
        """Called by the timer on each tick."""
        if not self.paused:
            self.idx = (self.idx + 1) % self.n_frames
        self._render_frame()

    def _set_interval(self):
        self._anim_timer.interval = int(1000 / (self.fps * self.speed_mult))

    def _on_key(self, event: KeyEvent):
        if event.key in (" ",):
            self.paused = not self.paused
            self._render_frame()
            self.fig.canvas.draw_idle()
        elif event.key in ("r", "R"):
            self.idx = 0
            self.paused = True
            self._render_frame()
            self.fig.canvas.draw_idle()
        elif event.key in ("q", "Q", "escape"):
            self._anim_timer.stop()
            plt.close(self.fig)
        elif event.key in ("left",):
            self.idx = (self.idx - 1) % self.n_frames
            self.paused = True
            self._render_frame()
            self.fig.canvas.draw_idle()
        elif event.key in ("right",):
            self.idx = (self.idx + 1) % self.n_frames
            self.paused = True
            self._render_frame()
            self.fig.canvas.draw_idle()
        elif event.key in ("up",):
            self.speed_mult = min(self.speed_mult * 2.0, 16.0)
            self._set_interval()
            self._render_frame()
            self.fig.canvas.draw_idle()
        elif event.key in ("down",):
            self.speed_mult = max(self.speed_mult / 2.0, 0.125)
            self._set_interval()
            self._render_frame()
            self.fig.canvas.draw_idle()
        elif event.key in ("o", "O"):
            self.show_overlay = not self.show_overlay
            print(f"[toggle overlay] {'on' if self.show_overlay else 'off'}")
        elif event.key in ("s", "S"):
            out_path = f"frame_{self.idx:05d}.png"
            self.fig.savefig(out_path, dpi=150)
            print(f"[save] {out_path}")
        elif event.key in ("a", "A"):
            self.auto_crop = not self.auto_crop
            print(f"[toggle auto-crop] {'on' if self.auto_crop else 'off'}")

    def show(self):
        self._render_frame()
        self._anim_timer.start()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="TactileSense Noise Suppression Replay Animator")
    parser.add_argument("csv_path", type=str, nargs="?", default=None, help="Path to CSV recording file")
    parser.add_argument("--k-sigma", type=float, default=1.0, help="Noise floor multiplier (default: 1.0)")
    parser.add_argument("--decay", type=float, default=0.0, help="Noise gate decay subtraction")
    parser.add_argument("--fps", type=int, default=10, help="Initial playback FPS")
    parser.add_argument("--mode", type=int, default=0, help="Processor mode: 0=StatGate, 1=Spatial, 2=Hybrid")
    parser.add_argument("--temporal-window", type=int, default=3, help="Temporal averaging window (Spatial mode)")
    parser.add_argument("--spatial-sigma", type=float, default=1.5, help="Gaussian spatial sigma (Spatial mode)")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed heatmap max")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.askopenfilename(
            title="Select CSV recording file",
            filetypes=[("CSV files", "*.csv")])
        root.destroy()
        if not csv_path:
            print("No file selected. Exiting.")
            return

    app = ReplayAnimator(csv_path, k_sigma=args.k_sigma,
                         noise_gate_decay=args.decay,
                         fps=args.fps, vmax=args.vmax,
                         mode=args.mode,
                         temporal_window=args.temporal_window,
                         spatial_sigma=args.spatial_sigma)
    app.show()


if __name__ == "__main__":
    main()
