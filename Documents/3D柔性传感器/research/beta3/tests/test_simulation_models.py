import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simulation_models import crosstalk_kernel, simulate_force_case, simulate_membrane_case


class SimulationModelTests(unittest.TestCase):
    def test_crosstalk_kernel_is_conservative(self):
        kernel = crosstalk_kernel(0.2)
        self.assertAlmostEqual(float(kernel.sum()), 1.0)
        self.assertAlmostEqual(float(kernel[1, 1]), 0.8)

    def test_membrane_observation_has_neighbor_leakage(self):
        case = simulate_membrane_case("local_center", 0.2, np.random.default_rng(1))
        self.assertEqual(case.truth.shape, (64, 64))
        self.assertGreater(np.count_nonzero(case.observed), np.count_nonzero(case.truth))

    def test_force_case_dimensions_and_truth(self):
        case = simulate_force_case("local", np.random.default_rng(2), duration=20.0)
        self.assertEqual(case.observed_cells.shape, case.true_cells.shape)
        self.assertEqual(case.observed_cells.shape[1], 64)
        self.assertTrue(np.all(case.observed_cells >= 0.0))


if __name__ == "__main__":
    unittest.main()
