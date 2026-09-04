import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from tools.replay_animator import ReplayAnimator


def test_case(label, csv_rel, k_sigma=1.0, init_only=False):
    data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "DataSet"))
    csv_path = os.path.join(data_dir, *csv_rel)
    if not os.path.isfile(csv_path):
        print(f"[skip] {label}: file not found: {csv_path}")
        return

    t0 = time.perf_counter()
    app = ReplayAnimator(csv_path, k_sigma=k_sigma, fps=10)
    t1 = time.perf_counter()
    app.idx = min(10, app.n_frames - 1)
    app._render_frame()
    t2 = time.perf_counter()

    out_path = os.path.join(BASE_DIR, "stats", f"acceptance_replay_{label}.png")
    app.fig.savefig(out_path, dpi=120)
    t3 = time.perf_counter()

    print(f"[{label}] init={t1-t0:.3f}s render_one={t2-t1:.3f}s savefig={t3-t2:.3f}s "
          f"frames={app.n_frames} ch={app.n_channels} "
          f"vmax={app.vmax:.1f} pvmax={app.processed_vmax:.1f}")

    if t1 - t0 > 10.0:
        raise AssertionError(f"{label}: init too slow: {t1-t0:.3f}s")
    if app.processed_vmax <= 1.0:
        raise AssertionError(f"{label}: processed display scale too low: {app.processed_vmax:.1f}")

    if init_only:
        return

    # Short animation loop: simulate ~30 frames
    t4 = time.perf_counter()
    for f_idx in range(30):
        app._render_frame()
        app.idx = (app.idx + 1) % app.n_frames
    t5 = time.perf_counter()
    anim_s = t5 - t4
    avg_ms = anim_s / 30.0 * 1000.0
    print(f"  anim_30_frames={anim_s:.3f}s avg={avg_ms:.3f}ms/frame")

    if avg_ms > 100.0:
        raise AssertionError(f"{label}: blit too slow: {avg_ms:.1f}ms/frame")


if __name__ == "__main__":
    test_case("glove_1kg", ("手套", "单手指1kg", "录制数据_20260710141208_part0.csv"))
    test_case("fabric_500g", ("织物垫", "500g砝码压力", "录制数据_20260710143312_part0.csv"))
    test_case("film64_1kg", ("64x32膜片", "1kg砝码压力", "录制数据_20260710145522_part0.csv"))
    print("replay animator acceptance passed")
