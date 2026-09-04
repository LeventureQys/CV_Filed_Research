from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_spectral_gate import TemporalSpectralGate


class TemporalSpectralGateTest(unittest.TestCase):
    def test_requires_fit(self):
        gate = TemporalSpectralGate()
        with self.assertRaises(RuntimeError):
            gate.process(np.zeros((20, 4)))

    def test_shape_and_nonnegative_output(self):
        rng = np.random.default_rng(7)
        background = 5.0 + rng.normal(0.0, 1.0, size=(200, 8))
        signal = np.zeros((160, 8))
        signal[30:130, 3] = 12.0
        observed = 5.0 + signal + rng.normal(0.0, 1.0, size=signal.shape)
        output = TemporalSpectralGate().fit(background, 20.0).process(observed)
        self.assertEqual(output.shape, signal.shape)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertGreater(float(output[50:110, 3].mean()), 8.0)

    def test_channel_permutation_invariance(self):
        rng = np.random.default_rng(11)
        background = rng.normal(4.0, 1.0, size=(240, 12))
        observed = rng.normal(4.0, 1.0, size=(180, 12))
        observed[40:150, [2, 9]] += np.array([8.0, 15.0])
        permutation = rng.permutation(observed.shape[1])
        inverse = np.argsort(permutation)

        direct = TemporalSpectralGate().fit(background, 20.0).process(observed)
        permuted = TemporalSpectralGate().fit(background[:, permutation], 20.0).process(observed[:, permutation])

        np.testing.assert_allclose(direct, permuted[:, inverse], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
