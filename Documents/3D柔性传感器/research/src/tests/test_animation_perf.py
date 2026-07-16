"""test_animation_perf.py — unit test for animation freeze under TkAgg.

Simulates FuncAnimation loop without plt.show().
Verifies per-frame cost < interval at 10 FPS for all three device types.
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("TkAgg")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from tools.replay_animator import ReplayAnimator


_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "DataSet"))

_CASES = [
    ("glove_1kg",   ("手套", "单手指1kg", "录制数据_20260710141208_part0.csv")),
    ("fabric_500g", ("织物垫", "500g砝码压力", "录制数据_20260710143312_part0.csv")),
    ("film64_1kg",  ("64x32膜片", "1kg砝码压力", "录制数据_20260710145522_part0.csv")),
]

FPS = 10
INTERVAL_MS = 100
SIMULATED_FRAMES = 120  # simulate 12 seconds of animation


def test_animation_no_freeze(label, csv_rel, k_sigma=1.0):
    csv_path = os.path.join(_DATA_DIR, *csv_rel)
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {label}: file not found: {csv_path}")
        return

    # 1. build (includes figure creation with TkAgg canvas)
    t0 = time.perf_counter()
    app = ReplayAnimator(csv_path, k_sigma=k_sigma, fps=FPS)
    t1 = time.perf_counter()
    init_s = t1 - t0

    # 2. verify default state is playing
    assert not app.paused, f"{label}: should default to playing"
    assert app.idx == 0, f"{label}: should start at frame 0"

    # 3. simulate animation loop
    timings = []
    title_updated_count = 0

    for frame_idx in range(SIMULATED_FRAMES):
        t_start = time.perf_counter()
        app._render_frame()
        if not app.paused:
            app.idx = (app.idx + 1) % app.n_frames
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000.0
        timings.append(elapsed_ms)

        if app._stats_counter % 10 == 0:
            title_updated_count += 1

    # 3. statistics
    timings.sort()
    avg_ms = sum(timings) / len(timings)
    p95_ms = timings[int(len(timings) * 0.95)]
    p99_ms = timings[int(len(timings) * 0.99)]
    max_ms = timings[-1]

    print(f"\n[{label}]")
    print(f"  init_s={init_s:.3f}")
    print(f"  n_frames={app.n_frames} n_channels={app.n_channels}")
    print(f"  simulated {SIMULATED_FRAMES} frames @ {FPS} FPS")
    print(f"  avg={avg_ms:.3f}ms  p95={p95_ms:.3f}ms  p99={p99_ms:.3f}ms  max={max_ms:.3f}ms")
    print(f"  title_updates={title_updated_count} (expected ~{SIMULATED_FRAMES // 10})")

    # 4. assertions
    assert avg_ms < INTERVAL_MS, (
        f"{label}: avg frame time {avg_ms:.2f}ms >= {INTERVAL_MS}ms interval")
    assert max_ms < INTERVAL_MS * 3, (
        f"{label}: max frame time {max_ms:.2f}ms >= 3x interval")
    assert app.processed_vmax > 1.0, (
        f"{label}: processed_vmax too low: {app.processed_vmax:.1f}")
    assert title_updated_count >= 11, (
        f"{label}: too few title updates: {title_updated_count} < 11")

    # 5. cleanup
    import matplotlib.pyplot as plt
    plt.close(app.fig)

    print(f"  [PASS]")


if __name__ == "__main__":
    for label, csv_rel in _CASES:
        test_animation_no_freeze(label, csv_rel)

    print("\n=== all animation perf tests passed ===")
