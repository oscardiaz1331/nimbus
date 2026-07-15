"""Tests for utils/benchmark/trials.py — pure YAML/dataclass parsing, no GPU needed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.benchmark.trials import Trial, load_trials


class TestLoadTrials(unittest.TestCase):
    def test_parses_list_into_trial_dataclasses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.yaml"
            path.write_text(
                "trials:\n"
                "  - name: YOLO11-N-seg\n"
                "    framework: yolo\n"
                "    checkpoint: finetuned_checkpoints/yolo-11-n.pt\n"
                "    variant: yolo11n-seg\n"
                "  - name: RF-DETR Nano\n"
                "    framework: rfdetr\n"
                "    checkpoint: finetuned_checkpoints/rf-detr-n.pth\n"
                "    variant: nano\n",
                encoding="utf-8",
            )
            trials = load_trials(path)
            self.assertEqual(len(trials), 2)
            self.assertEqual(trials[0], Trial("YOLO11-N-seg", "yolo", "finetuned_checkpoints/yolo-11-n.pt", "yolo11n-seg"))
            self.assertEqual(trials[1].framework, "rfdetr")

    def test_rejects_unknown_framework(self):
        with self.assertRaises(ValueError):
            Trial(name="bad", framework="tensorflow", checkpoint="x.pt", variant="n")


if __name__ == "__main__":
    unittest.main()
