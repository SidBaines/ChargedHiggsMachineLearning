"""Regression checks for the compact student prediction cache."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from physics_students_sb200 import cached_student_predictions, teacher_predictions


class StudentCacheTests(unittest.TestCase):
    def test_student_cache_requires_matching_checkpoint_and_inference_inputs(self):
        predictions = {"score": np.array([1.]), "channel": np.array([1]), "perfect": np.array([True])}
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "student.pt"
            checkpoint.write_bytes(b"original checkpoint")
            cache = Path(temp) / "student.npz"
            np.savez(cache, **dict(predictions, score=[-1.]))
            with (
                patch("physics_students_sb200.selection_model"),
                patch("physics_students_sb200.student_predictions", return_value=predictions) as predict,
            ):
                first = cached_student_predictions(cache, checkpoint, "readout-key", 2, 1024, {})
                again = cached_student_predictions(cache, checkpoint, "readout-key", 2, 1024, {})
                np.testing.assert_array_equal(first["score"], [1.])
                np.testing.assert_array_equal(again["score"], [1.])
                self.assertEqual(predict.call_count, 1)

                checkpoint.write_bytes(b"different checkpoint")
                for source_key, residue, batch in (
                    ("readout-key", 2, 1024), ("readout-key", 6, 1024),
                    ("readout-key", 6, 512), ("new-readout-key", 6, 512),
                ):
                    cached_student_predictions(cache, checkpoint, source_key, residue, batch, {})
                self.assertEqual(predict.call_count, 5)

    def test_teacher_cache_requires_exact_source_key_and_row_ids(self):
        sample = {"dsid": [510115], "row": [2], "label": [1], "weight": [0.5]}
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            np.savez(directory / "readouts_teacher-key_2_16000_355000_2701.npz",
                     **{f"ev_{key}": value for key, value in sample.items()},
                     fr_native=[1.], fr_channel=[1], fr_perfect=[True])
            np.testing.assert_array_equal(teacher_predictions(directory, "teacher-key", 2, sample)["score"], [1.])
            with self.assertRaises(FileNotFoundError):
                teacher_predictions(directory, "different-key", 2, sample)
            with self.assertRaises(AssertionError):
                teacher_predictions(directory, "teacher-key", 2, dict(sample, row=[10]))


if __name__ == "__main__":
    unittest.main()
