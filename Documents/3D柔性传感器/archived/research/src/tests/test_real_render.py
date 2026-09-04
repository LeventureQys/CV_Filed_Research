"""test_real_render.py — opens a real TkAgg window and measures frame render time.

WARNING: this opens a window! It will auto-close after 5 seconds.
Measures both the blit time and the total Tk event processing time.
"""

from __future__ import annotations

import os
import sys
import time
import threading

import matplotlib
matplotlib.use("TkAgg")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from tools.replay_animator import ReplayAnimator

_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "DataSet"))

_CASES = [
    ("glove_1kg",   ("手套", "单手指1kg", "录制数据_20260710141208_part0.csv")),
    ("film64_1kg",  ("64x32膜片", "1kg砝码压力", "录制数据_20260710145522_part0.csv")),
]


def test_real_render_loop(label, csv_rel, k_sigma=1.0):
    csv_path = os.path.join(_DATA_DIR, *csv_rel)
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {label}: file not found: {csv_path}")
        return

    app = ReplayAnimator(csv_path, k_sigma=k_sigma, fps=10)
    app.paused = False

    # Start measuring in a thread
    results = []

    def measure():
        time.sleep(0.5)  # wait for window to map
        for _ in range(50):
            t0 = time.perf_counter()
            app._render_frame()
            # Simulate the timer advancing idx
            app.idx = (app.idx + 1) % app.n_frames
            t1 = time.perf_counter()
            results.append((t1 - t0) * 1000.0)
            time.sleep(0.09)  # ~10 FPS but leave room

    t = threading.Thread(target=measure, daemon=True)
    t.start()

    # Show window, auto-close after 6s
    def auto_close():
        time.sleep(5)
        app._anim_timer.stop()
        import matplotlib.pyplot as plt
        plt.close(app.fig)

    threading.Thread(target=auto_close, daemon=True).start()

    app.show()

    if not results:
        print(f"[{label}] no measurements collected!")
        return

    results.sort()
    avg_ms = sum(results) / len(results)
    p95_ms = results[int(len(results) * 0.95)]
    max_ms = results[-1]

    print(f"\n[{label}] REAL WINDOW RENDER (n={len(results)})")
    print(f"  channels={app.n_channels}")
    print(f"  avg={avg_ms:.3f}ms  p95={p95_ms:.3f}ms  max={max_ms:.3f}ms")

    if avg_ms > 50:
        print(f"  WARNING: avg frame time {avg_ms:.1f}ms is high — may feel sluggish")
    elif avg_ms < 16:
        print(f"  GREAT: avg frame time {avg_ms:.1f}ms — smooth at 10 FPS")
    else:
        print(f"  OK: avg frame time {avg_ms:.1f}ms")

    if avg_ms > 100:
        raise AssertionError(f"[{label}] real render too slow: avg {avg_ms:.1f}ms >= 100ms interval")


if __name__ == "__main__":
    test_real_render_loop(*_CASES[0])
    test_real_render_loop(*_CASES[1])
    print("\n=== real render test done ===")
