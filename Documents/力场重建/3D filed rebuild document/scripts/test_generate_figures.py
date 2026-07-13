import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("generate_figures.py")
SPEC = importlib.util.spec_from_file_location("generate_figures", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SurfaceFieldFigureTests(unittest.TestCase):
    def test_step_geometry_evidence(self):
        evidence = MODULE.verify_step_geometry()
        self.assertEqual(evidence["radius_mm"], 6.0)
        self.assertEqual(evidence["axial_min_mm"], -8.0)
        self.assertEqual(evidence["axial_max_mm"], 8.0)

    def test_wendland_boundary_and_range(self):
        distances = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        weights = MODULE.wendland_c2(distances, support=8.0)
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[-2], 0.0)
        self.assertAlmostEqual(weights[-1], 0.0)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights <= 1.0))
        self.assertTrue(np.all(np.diff(weights) <= 1e-12))

    def test_selected_points_lie_on_cylinder(self):
        x, y, z = MODULE.sensor_xyz(offset_mm=0.0)
        radius_squared = x**2 + (y - MODULE.AXIS_CENTER_Y_MM) ** 2
        self.assertTrue(np.allclose(radius_squared, MODULE.RADIUS_MM**2))
        self.assertTrue(np.all(z >= -MODULE.LENGTH_MM / 2.0))
        self.assertTrue(np.all(z <= MODULE.LENGTH_MM / 2.0))

    def test_constant_samples_reconstruct_constant_interpolation(self):
        sensors = MODULE.SENSORS.copy()
        sensors[:, 2] = 0.63
        _, _, interpolated, coverage, effective, weight_sum = MODULE.reconstruct(
            np.linspace(-8.0, 8.0, 80),
            np.linspace(-90.0, 90.0, 100),
            sensors=sensors,
        )
        covered = weight_sum > 1e-12
        self.assertTrue(np.allclose(interpolated[covered], 0.63))
        self.assertTrue(np.all((coverage >= 0.0) & (coverage <= 1.0)))
        self.assertTrue(np.all(effective <= interpolated + 1e-12))

    def test_interpolation_stays_in_sample_range(self):
        _, _, interpolated, coverage, effective, weight_sum = MODULE.reconstruct(
            np.linspace(-8.0, 8.0, 80), np.linspace(-90.0, 90.0, 100)
        )
        covered = weight_sum > 1e-12
        self.assertGreaterEqual(
            interpolated[covered].min(), MODULE.SENSORS[:, 2].min() - 1e-12
        )
        self.assertLessEqual(
            interpolated[covered].max(), MODULE.SENSORS[:, 2].max() + 1e-12
        )
        self.assertTrue(np.isfinite(coverage).all())
        self.assertTrue(np.isfinite(effective).all())

    def test_surface_distance_not_shorter_than_chord(self):
        angle = np.deg2rad(np.linspace(0.0, 180.0, 1000))
        surface_distance = MODULE.RADIUS_MM * angle
        chord = 2.0 * MODULE.RADIUS_MM * np.sin(angle / 2.0)
        self.assertTrue(np.all(surface_distance + 1e-12 >= chord))


if __name__ == "__main__":
    unittest.main()
