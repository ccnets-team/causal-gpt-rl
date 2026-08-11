"""Survival statistics over long-horizon rollout lengths."""
import unittest

from examples.deploy.survival import format_survival_table, survival_stats


def _humanoid_kv5000_lengths():
    """Lengths reproducing the published Humanoid KV5000 survival table.

    8 episodes die in each of the first two 1,000-step intervals, then 4, 5, 4,
    and 21 run the full 5,000 steps.
    """
    return (
        [500] * 8
        + [1500] * 8
        + [2500] * 4
        + [3500] * 5
        + [4500] * 4
        + [5000] * 21
    )


class SurvivalStatsTest(unittest.TestCase):
    def test_matches_published_humanoid_table(self):
        stats = survival_stats(
            _humanoid_kv5000_lengths(), horizon=5000, bucket=1000
        )

        self.assertEqual(stats["num_episodes"], 50)
        self.assertEqual(stats["completers"], 21)
        self.assertAlmostEqual(stats["completion_rate"], 0.42)

        entered = [row["entered"] for row in stats["intervals"]]
        died = [row["died"] for row in stats["intervals"]]
        survived = [row["survived"] for row in stats["intervals"]]
        self.assertEqual(entered, [50, 42, 34, 30, 25])
        self.assertEqual(died, [8, 8, 4, 5, 4])
        self.assertEqual(survived, [42, 34, 30, 25, 21])

        conditional = [row["conditional"] for row in stats["intervals"]]
        for value, expected in zip(conditional, [0.840, 0.810, 0.882, 0.833, 0.840]):
            self.assertAlmostEqual(value, expected, places=3)

        cumulative = [row["cumulative"] for row in stats["intervals"]]
        for value, expected in zip(cumulative, [0.84, 0.68, 0.60, 0.50, 0.42]):
            self.assertAlmostEqual(value, expected, places=2)

    def test_constant_rate_prediction_tracks_measured_completion(self):
        """The flatness check: a constant hazard predicts the measured rate."""
        stats = survival_stats(
            _humanoid_kv5000_lengths(), horizon=5000, bucket=1000
        )
        self.assertAlmostEqual(stats["conditional_mean"], 0.8412, places=3)
        # 0.84 ** 5 = 0.418 against a measured 0.42 — flat hazard.
        self.assertAlmostEqual(stats["constant_rate_prediction"], 0.42, places=2)

    def test_completer_return_per_step_excludes_early_deaths(self):
        lengths = [100, 100, 1000, 1000]
        # The short episodes earn 1.0/step; the completers earn 5.0/step.
        returns = [100.0, 100.0, 5000.0, 5000.0]
        stats = survival_stats(lengths, returns, horizon=1000, bucket=500)

        self.assertAlmostEqual(stats["return_per_step_completers"], 5.0)
        self.assertAlmostEqual(stats["return_per_step_all"], 10200.0 / 2200.0)
        self.assertLess(stats["return_per_step_all"], stats["return_per_step_completers"])

    def test_no_completers_leaves_return_per_step_undefined(self):
        stats = survival_stats([10, 20], [1.0, 2.0], horizon=1000, bucket=500)
        self.assertEqual(stats["completers"], 0)
        self.assertIsNone(stats["return_per_step_completers"])

    def test_interval_with_no_entrants_reports_none(self):
        stats = survival_stats([10, 10], horizon=1000, bucket=500)
        self.assertEqual(stats["intervals"][0]["conditional"], 0.0)
        self.assertIsNone(stats["intervals"][1]["conditional"])

    def test_final_interval_truncates_to_horizon(self):
        stats = survival_stats([300] * 3, horizon=300, bucket=200)
        self.assertEqual(
            [(row["start"], row["end"]) for row in stats["intervals"]],
            [(1, 200), (201, 300)],
        )

    def test_defaults_use_max_length_and_five_buckets(self):
        stats = survival_stats([1000] * 4)
        self.assertEqual(stats["horizon"], 1000)
        self.assertEqual(stats["bucket"], 200)
        self.assertEqual(len(stats["intervals"]), 5)

    def test_rejects_mismatched_returns(self):
        with self.assertRaises(ValueError):
            survival_stats([1, 2, 3], [1.0, 2.0])

    def test_rejects_empty_lengths(self):
        with self.assertRaises(ValueError):
            survival_stats([])

    def test_table_renders_measured_and_predicted(self):
        stats = survival_stats(
            _humanoid_kv5000_lengths(), [1.0] * 50, horizon=5000, bucket=1000
        )
        table = format_survival_table(stats)
        self.assertIn("completed 21/50 (42.0%)", table)
        self.assertIn("return/step, completers:", table)


if __name__ == "__main__":
    unittest.main()
