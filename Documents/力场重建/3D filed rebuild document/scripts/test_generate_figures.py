import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("generate_figures.py")
SPEC = importlib.util.spec_from_file_location("generate_figures", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullModelSurfaceFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.mesh, cls.centers, cls.normals, cls.values = MODULE.load_project()
        cls.result = MODULE.reconstruct(cls.mesh, cls.centers, cls.values)

    def test_full_model_and_real_cells_loaded(self):
        self.assertEqual(len(self.mesh.vertices), 814)
        self.assertEqual(len(self.mesh.faces), 1656)
        self.assertTrue(self.mesh.is_watertight)
        self.assertEqual(len(self.centers), 31)

    def test_wendland_boundary_and_range(self):
        distances = np.array([0.0, 7.5, 15.0, 22.5, 30.0, 35.0])
        weights = MODULE.wendland_c2(distances, support=30.0)
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[-2], 0.0)
        self.assertAlmostEqual(weights[-1], 0.0)
        self.assertTrue(np.all((weights >= 0.0) & (weights <= 1.0)))
        self.assertTrue(np.all(np.diff(weights) <= 1e-12))

    def test_cells_attach_to_complete_surface(self):
        self.assertLess(self.result["attachment_error"].max(), 1e-3)
        self.assertEqual(len(self.result["seeds"]), len(self.centers))

    def test_constant_samples_preserve_constant_interpolation(self):
        constant = np.full(len(self.centers), 0.63)
        result = MODULE.reconstruct(self.mesh, self.centers, constant)
        covered = result["weight_sum"] > 1e-12
        self.assertTrue(np.allclose(result["interpolated"][covered], 0.63))
        self.assertTrue(np.all((result["coverage"] >= 0.0) & (result["coverage"] <= 1.0)))

    def test_interpolation_stays_in_sample_range(self):
        covered = self.result["weight_sum"] > 1e-12
        interpolated = self.result["interpolated"][covered]
        self.assertGreaterEqual(interpolated.min(), self.values.min() - 1e-12)
        self.assertLessEqual(interpolated.max(), self.values.max() + 1e-12)
        self.assertTrue(np.isfinite(self.result["effective"]).all())

    def test_surface_path_not_shorter_than_euclidean(self):
        vertices = np.asarray(self.mesh.vertices)
        for sample_index, source in enumerate(self.result["seeds"]):
            distances = self.result["distances"][sample_index]
            finite = np.isfinite(distances)
            chord = np.linalg.norm(vertices - vertices[source], axis=1)
            self.assertTrue(np.all(distances[finite] + 1e-9 >= chord[finite]))

    def test_overlay_is_independent_and_denser_than_roi(self):
        active = MODULE.ACTIVE_CELL_INDICES
        roi_mesh, overlay, _, roi_faces = MODULE.build_overlay(
            self.mesh,
            self.result["closest"][active],
            self.result["face_ids"][active],
        )
        self.assertGreater(roi_faces.sum(), 0)
        self.assertLess(roi_faces.sum(), len(self.mesh.faces))
        self.assertGreater(len(overlay.faces), len(roi_mesh.faces))
        self.assertGreater(len(overlay.vertices), len(roi_mesh.vertices))

    def test_overlay_reconstruction_and_top_k(self):
        active = MODULE.ACTIVE_CELL_INDICES
        roi_mesh, overlay, _, _ = MODULE.build_overlay(
            self.mesh,
            self.result["closest"][active],
            self.result["face_ids"][active],
        )
        overlay_result = MODULE.reconstruct_overlay(
            overlay,
            self.result["closest"][active],
            self.values[active],
            top_k=2,
        )
        self.assertTrue(np.all(overlay_result["retained_counts"] <= 2))
        self.assertTrue(np.all((overlay_result["coverage"] >= 0.0) & (overlay_result["coverage"] <= 1.0)))
        self.assertTrue(np.isfinite(overlay_result["effective"]).all())


if __name__ == "__main__":
    unittest.main()
