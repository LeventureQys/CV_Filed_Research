import os
import sys

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from alg.noise_suppressor import Analyzer, Processor


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_signal_above_noise_floor_is_preserved_after_baseline_subtract():
    background = np.array([
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
        [10.0, 10.0],
    ])
    model = Analyzer(k_sigma=3.0, min_noise_std=0.5).analyze(background, sample_freq_hz=20.0)
    processor = Processor(model)

    processed = processor.process(np.array([20.0, 20.0]))

    assert_true(processed[0] > 0.0, f"expected channel 0 to retain signal, got {processed[0]}")
    assert_true(processed[1] > 0.0, f"expected channel 1 to retain signal, got {processed[1]}")


def test_background_near_baseline_is_suppressed():
    background = np.array([
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
        [10.0, 10.0],
    ])
    model = Analyzer(k_sigma=3.0, min_noise_std=0.5).analyze(background, sample_freq_hz=20.0)
    processor = Processor(model)

    processed = processor.process(np.array([10.5, 10.5]))

    assert_true(np.allclose(processed, [0.0, 0.0]), f"expected background suppression, got {processed}")


def test_batch_processing_matches_single_frame_processing():
    background = np.array([
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
        [10.0, 10.0],
    ])
    model = Analyzer(k_sigma=3.0, min_noise_std=0.5).analyze(background, sample_freq_hz=20.0)
    processor = Processor(model)
    frames = np.array([
        [10.5, 10.5],
        [20.0, 20.0],
        [30.0, 12.0],
    ])

    batch = processor.process_batch(frames)
    processor.reset_timing()
    singles = np.array([processor.process(frame) for frame in frames])

    assert_true(np.allclose(batch, singles), f"batch differs from single processing: {batch} vs {singles}")


if __name__ == "__main__":
    test_signal_above_noise_floor_is_preserved_after_baseline_subtract()
    test_background_near_baseline_is_suppressed()
    test_batch_processing_matches_single_frame_processing()
    print("noise_suppressor behavior tests passed")
