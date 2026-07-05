"""Tests for utils/stages.py — the core, framework-agnostic logic
behind 'advance to the next stage when validation plateaus'.
"""

from __future__ import annotations

import unittest

from utils.stages import PlateauDetector, StageAction, StageController


class TestPlateauDetector(unittest.TestCase):
    def test_improving_never_plateaus(self):
        d = PlateauDetector(patience=3)
        for v in [1.0, 0.8, 0.6, 0.4, 0.2]:
            self.assertFalse(d.update(v))

    def test_plateaus_after_patience_epochs(self):
        d = PlateauDetector(patience=3, min_delta=0.0)
        self.assertFalse(d.update(1.0))  # sets best=1.0
        self.assertFalse(d.update(1.0))  # wait=1
        self.assertFalse(d.update(1.0))  # wait=2
        self.assertTrue(d.update(1.0))  # wait=3 -> patience reached

    def test_min_delta_requires_meaningful_improvement(self):
        d = PlateauDetector(patience=2, min_delta=0.1)
        self.assertFalse(d.update(1.0))
        self.assertFalse(
            d.update(0.95)
        )  # improvement < min_delta -> counts as no improvement, wait=1
        self.assertTrue(d.update(0.94))  # wait=2 -> plateau

    def test_reset_clears_state(self):
        d = PlateauDetector(patience=1, min_delta=0.0)
        d.update(1.0)
        d.update(1.0)  # would now be at the plateau boundary
        d.reset()
        self.assertIsNone(d.best)
        self.assertEqual(d.wait, 0)


class TestStageController(unittest.TestCase):
    def test_continues_while_improving(self):
        c = StageController(patience=3)
        for v in [2.0, 1.5, 1.0]:
            self.assertEqual(c.step(v), StageAction.CONTINUE)

    def test_advances_on_plateau(self):
        c = StageController(patience=2, min_delta=0.0)
        self.assertEqual(c.step(1.0), StageAction.CONTINUE)
        self.assertEqual(c.step(1.0), StageAction.CONTINUE)
        self.assertEqual(c.step(1.0), StageAction.ADVANCE)


if __name__ == "__main__":
    unittest.main()
