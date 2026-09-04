from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from membrane_spatiotemporal import MembraneSpatioTemporalGate


class MembraneSpatioTemporalGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = MembraneSpatioTemporalGate(rows=8, cols=8)

    def test_persistent_low_offset_is_removed(self):
        frames = np.zeros((20, 64))
        frames[:, 9] = 7.0
        output, _ = self.gate.process_offline(frames)
        self.assertEqual(float(output.sum()), 0.0)

    def test_sustained_pressure_is_preserved_after_activation(self):
        frames = np.zeros((20, 64))
        frames[:, 27:29] = 100.0
        output, _ = self.gate.process_offline(frames)
        np.testing.assert_allclose(output[2:, 27:29], 100.0)
        self.assertEqual(float(output[:2].sum()), 0.0)

    def test_two_frame_contact_is_completely_lost(self):
        frames = np.zeros((20, 64))
        frames[5:7, 27:29] = 100.0
        output, _ = self.gate.process_offline(frames)
        self.assertEqual(float(output.sum()), 0.0)

    def test_offline_mask_contains_future_contact(self):
        frames = np.zeros((20, 64))
        frames[10:15, 27:29] = 100.0
        _, mask = self.gate.process_offline(frames)
        self.assertTrue(bool(mask[27]))
        self.assertTrue(bool(mask[28]))

    def test_external_mask_cannot_generalize_to_new_location(self):
        training = np.zeros((20, 64))
        training[:, 9:11] = 100.0
        evaluation = np.zeros((20, 64))
        evaluation[:, 45:47] = 100.0
        mask = self.gate.build_mask(training)
        output = self.gate.process_with_mask(evaluation, mask)
        self.assertEqual(float(output.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

