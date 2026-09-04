import os
import sys
import time

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from alg.data_loader import load_csv
from alg.noise_suppressor import build_full_noise_suppressor


def check_case(label, path, k_sigma=1.0):
    t0 = time.perf_counter()
    frames, meta = load_csv(path)
    t1 = time.perf_counter()
    bg = frames[:min(100, frames.shape[0] // 2)]
    model, proc = build_full_noise_suppressor(bg, meta.get("sample_freq_hz", 0), k_sigma=k_sigma)
    out = proc.process_batch(frames)
    t2 = time.perf_counter()

    raw_vmax = max(np.percentile(frames, 99.5), 1.0)
    proc_vmax = max(np.percentile(out, 99.5), 1.0)
    proc_mean = float(out.sum(axis=1).mean())
    proc_nz = float((out > 0).sum(axis=1).mean())
    process_s = t2 - t1
    load_s = t1 - t0

    print(f"[{label}]")
    print(f"  shape={frames.shape} load_s={load_s:.3f} process_s={process_s:.3f}")
    print(f"  raw_sum_mean={frames.sum(axis=1).mean():.2f} proc_sum_mean={proc_mean:.2f} proc_nz_mean={proc_nz:.2f}")
    print(f"  raw_vmax={raw_vmax:.2f} processed_vmax={proc_vmax:.2f} avg_ms={proc.avg_process_time_ms:.4f} rtf={proc.rtf:.4f}")

    if proc_nz <= 0:
        raise AssertionError(f"{label}: processed output is all zero")
    if process_s > 0.5:
        raise AssertionError(f"{label}: processing too slow: {process_s:.3f}s")
    if proc_vmax < 1.0:
        raise AssertionError(f"{label}: processed display scale too small")


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "DataSet"))
    check_case("glove", os.path.join(data_dir, "手套", "单手指1kg", "录制数据_20260710141208_part0.csv"))
    check_case("fabric", os.path.join(data_dir, "织物垫", "500g砝码压力", "录制数据_20260710143312_part0.csv"))
    check_case("film64", os.path.join(data_dir, "64x32膜片", "1kg砝码压力", "录制数据_20260710145522_part0.csv"))
    print("acceptance passed")
