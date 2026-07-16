import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from filters import ema, local_crosstalk_correction, scalar_kalman
from simulation_models import crosstalk_kernel


class FilterTests(unittest.TestCase):
    def test_ema_reduces_steady_noise(self):
        rng = np.random.default_rng(3)
        values = 10.0 + rng.normal(0.0, 1.0, 2000)
        filtered = ema(values, 50.0, 0.5)
        self.assertLess(np.std(filtered[500:]), np.std(values[500:]))

    def test_kalman_reduces_steady_noise(self):
        rng = np.random.default_rng(4)
        values = 5.0 + rng.normal(0.0, 0.5, 2000)
        filtered = scalar_kalman(values, 0.001, 0.25)
        self.assertLess(np.std(filtered[500:]), np.std(values[500:]))

    def test_local_correction_inverts_known_kernel(self):
        from scipy.ndimage import convolve

        truth = np.zeros((32, 32))
        truth[14:18, 14:18] = 100.0
        kernel = crosstalk_kernel(0.15)
        observed = convolve(truth, kernel, mode="constant", cval=0.0)
        corrected = local_crosstalk_correction(observed, kernel)
        self.assertLess(np.sqrt(np.mean((corrected - truth) ** 2)), 1.0)


if __name__ == "__main__":
    unittest.main()
