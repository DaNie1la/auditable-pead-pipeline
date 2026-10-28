import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pead_pipeline.finbert import (  # noqa: E402
    ScoreConfig,
    aggregate_window_probabilities,
    compute_window_weights,
)


class Phase3Tests(unittest.TestCase):
    def test_overlapping_windows_count_only_new_tokens(self):
        weights = compute_window_weights(
            [510, 510, 100, 200], [0, 0, 0, 1], stride=64
        )
        self.assertEqual(weights, [510, 446, 36, 200])

    def test_probability_aggregation_and_tone(self):
        scores = aggregate_window_probabilities(
            probabilities=[[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]],
            sample_mapping=[0, 0],
            content_lengths=[510, 164],
            sample_count=1,
            stride=64,
            label_indices={"positive": 0, "negative": 1, "neutral": 2},
        )
        expected_positive = (0.8 * 510 + 0.2 * 100) / 610
        expected_negative = (0.1 * 510 + 0.6 * 100) / 610
        self.assertAlmostEqual(scores[0].positive, expected_positive)
        self.assertAlmostEqual(scores[0].negative, expected_negative)
        self.assertAlmostEqual(
            scores[0].tone, expected_positive - expected_negative
        )
        self.assertEqual(scores[0].tokens, 610)
        self.assertEqual(scores[0].windows, 2)

    def test_score_config_hash_is_stable_and_sensitive(self):
        self.assertEqual(ScoreConfig().sha256, ScoreConfig().sha256)
        self.assertNotEqual(ScoreConfig().sha256, ScoreConfig(stride=32).sha256)
        self.assertNotEqual(
            ScoreConfig().sha256, ScoreConfig(window_batch_size=32).sha256
        )


if __name__ == "__main__":
    unittest.main()
