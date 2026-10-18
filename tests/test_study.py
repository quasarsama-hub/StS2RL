import unittest

from spire2_rl.study import (
    MAX_SEED_INDEX,
    SEEDS_PER_SPLIT,
    evaluation_seeds,
    paired_compare,
    seed_for,
    summarize,
    wilson_interval,
)


def episode(seed, outcome="death", floor=10, hp=0):
    return {"seed": seed, "outcome": outcome, "floor": floor, "hp": hp}


class SeedPartitionTests(unittest.TestCase):
    def test_large_indices_and_different_experiments_never_cross_splits(self):
        indices = [0, 1, 2, 10_000_000, MAX_SEED_INDEX - 1, MAX_SEED_INDEX]
        windows = {"train": 0, "validation": SEEDS_PER_SPLIT, "test": 2 * SEEDS_PER_SPLIT}
        all_splits = {}
        for split, offset in windows.items():
            observed = set()
            for experiment_seed in (7, 8, -7, 1 << 40):
                seeds = [seed_for(index, split, experiment_seed) for index in indices]
                self.assertEqual(len(set(seeds)), len(indices))
                for seed in seeds:
                    self.assertGreater(seed, offset)
                    self.assertLessEqual(seed, offset + SEEDS_PER_SPLIT)
                    self.assertLess(seed, 1 << 31)
                observed.update(seeds)
            all_splits[split] = observed
        self.assertTrue(all_splits["train"].isdisjoint(all_splits["validation"]))
        self.assertTrue(all_splits["train"].isdisjoint(all_splits["test"]))
        self.assertTrue(all_splits["validation"].isdisjoint(all_splits["test"]))

    def test_seed_slices_resume_without_reusing_training_episodes(self):
        first = evaluation_seeds("train", 500, 7)
        resumed = evaluation_seeds("train", 500, 7, start=500)
        self.assertTrue(set(first).isdisjoint(resumed))
        self.assertEqual(first + resumed, evaluation_seeds("train", 1000, 7))
        self.assertEqual(first, evaluation_seeds("train", 500, 7))
        self.assertNotEqual(first, evaluation_seeds("train", 500, 8))
        self.assertEqual(evaluation_seeds("test", 0, start=SEEDS_PER_SPLIT), [])
        self.assertEqual(evaluation_seeds("test", 1, start=MAX_SEED_INDEX), [seed_for(MAX_SEED_INDEX, "test")])

    def test_invalid_indices_and_slices_cannot_wrap(self):
        for index in (-1, SEEDS_PER_SPLIT, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                seed_for(index, "train")
        for kwargs in (
            {"split": "testing", "count": 1},
            {"split": "test", "count": -1},
            {"split": "test", "count": 2, "start": MAX_SEED_INDEX},
            {"split": "test", "count": 0, "start": SEEDS_PER_SPLIT + 1},
            {"split": "test", "count": True},
            {"split": "test", "count": 1, "experiment_seed": "7"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluation_seeds(**kwargs)


class SummaryTests(unittest.TestCase):
    def test_infrastructure_failures_are_visible_without_becoming_game_losses(self):
        report = summarize([
            episode(1, "win", 50, 20),
            episode(2, "death", 10, 0),
            episode(3, "timeout", 99, 99),
            episode(4, "aborted", None, None),
        ])
        self.assertEqual(report["episodes"], 4)
        self.assertEqual(report["valid_game_runs"], 2)
        self.assertEqual(report["outcomes"], {"win": 1, "death": 1, "timeout": 1, "aborted": 1})
        self.assertEqual(report["win_rate"], 0.5)
        self.assertEqual(report["mean_floor"], 30)
        self.assertEqual(report["mean_hp"], 10)
        self.assertEqual(report["infrastructure_failure_rate"], 0.25)
        self.assertEqual(report["incomplete_run_rate"], 0.5)
        self.assertEqual(report["floor_samples"], 2)
        self.assertAlmostEqual(report["win_rate_ci95"][0], 0.0945312057)
        self.assertAlmostEqual(report["win_rate_ci95"][1], 0.9054687943)

    def test_empty_or_entirely_aborted_batches_do_not_claim_zero_win_rate(self):
        empty = summarize([])
        failures = summarize([episode(1, "aborted", None, None)])
        for report in (empty, failures):
            self.assertIsNone(report["win_rate"])
            self.assertIsNone(report["win_rate_ci95"])
            self.assertIsNone(report["mean_floor"])
            self.assertEqual(report["valid_game_runs"], 0)
        self.assertEqual(failures["aborted_rate"], 1)
        self.assertGreater(wilson_interval(0, 100)[1], 0)
        self.assertLess(wilson_interval(100, 100)[0], 1)

    def test_unknown_outcomes_and_nonfinite_metrics_are_rejected(self):
        for row in (
            {"seed": 1},
            episode(1, "lost"),
            episode(1, floor=float("nan")),
            episode(1, hp=float("inf")),
            episode(1, floor=-1),
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                summarize([row])


class PairedComparisonTests(unittest.TestCase):
    def test_comparison_matches_seeds_and_bootstraps_pairs_not_independent_sides(self):
        baseline = [episode(seed, "win" if seed % 2 else "death", seed % 11) for seed in range(100)]
        candidate = [dict(row, floor=row["floor"] + 2) for row in baseline]
        report = paired_compare(candidate, baseline, bootstrap_samples=300)
        self.assertEqual(report["win_delta"], 0)
        self.assertEqual(report["win_delta_ci95"], [0, 0])
        self.assertEqual(report["floor_delta"], 2)
        self.assertEqual(report["floor_delta_ci95"], [2, 2])
        self.assertTrue(report["promote"])
        self.assertEqual(report, paired_compare(reversed(candidate), baseline[::2] + baseline[1::2], bootstrap_samples=300))

    def test_real_outcome_differences_measure_win_improvement(self):
        baseline = [episode(seed) for seed in range(100)]
        candidate = [episode(seed, "win", 50, 10) for seed in range(100)]
        report = paired_compare(candidate, baseline, bootstrap_samples=100)
        self.assertEqual(report["win_delta"], 1)
        self.assertTrue(report["promote"])

    def test_mixed_results_have_reproducible_uncertainty_and_do_not_promote_a_tie(self):
        baseline = [episode(seed, "win" if seed % 2 else "death", 20) for seed in range(100)]
        candidate = [episode(seed, "death" if seed % 2 else "win", 20) for seed in range(100)]
        report = paired_compare(candidate, baseline, bootstrap_samples=1000, bootstrap_seed=19)
        self.assertEqual(report, paired_compare(candidate, baseline, bootstrap_samples=1000, bootstrap_seed=19))
        self.assertEqual(report["win_delta"], 0)
        self.assertLess(report["win_delta_ci95"][0], 0)
        self.assertGreater(report["win_delta_ci95"][1], 0)
        self.assertEqual(report["floor_delta_ci95"], [0, 0])
        self.assertFalse(report["promote"])

    def test_duplicate_missing_and_unknown_records_do_not_silently_intersect(self):
        clean = [episode(1), episode(2)]
        invalid_batches = [
            [episode(1), episode(1)],
            [episode(1)],
            [episode(1), episode(3)],
            [{"outcome": "death", "floor": 10}, episode(2)],
            [episode(1, "error"), episode(2)],
            [dict(episode(1), floor=None), episode(2)],
            [episode(True), episode(2)],
        ]
        for invalid in invalid_batches:
            for candidate, baseline in ((invalid, clean), (clean, invalid)):
                with self.subTest(candidate=candidate, baseline=baseline), self.assertRaises(ValueError):
                    paired_compare(candidate, baseline, bootstrap_samples=100)

    def test_failures_in_either_policy_block_promotion_and_are_not_losses(self):
        baseline = [episode(seed) for seed in range(102)]
        candidate = [episode(seed, "win", 50, 10) for seed in range(102)]
        candidate[0] = episode(0, "aborted", None, None)
        baseline[1] = episode(1, "timeout", None, None)
        report = paired_compare(candidate, baseline, bootstrap_samples=100)
        self.assertEqual(report["pairs"], 102)
        self.assertEqual(report["eligible_pairs"], 100)
        self.assertEqual(report["excluded_seeds"], [0, 1])
        self.assertEqual(report["win_delta"], 1)
        self.assertEqual(report["candidate"]["win_rate"], 1)
        self.assertEqual(report["candidate"]["outcomes"]["aborted"], 1)
        self.assertEqual(report["baseline"]["outcomes"]["timeout"], 1)
        self.assertFalse(report["promote"])
        self.assertIn("timeout or aborted", report["reasons"][0])

    def test_small_batches_and_floor_gains_with_win_regressions_cannot_promote(self):
        baseline = [episode(seed, "win", 40, 10) for seed in range(100)]
        candidate = [episode(seed, "death", 45) for seed in range(100)]
        report = paired_compare(candidate, baseline, bootstrap_samples=100)
        self.assertGreater(report["floor_delta_ci95"][0], 0)
        self.assertLess(report["win_delta_ci95"][0], 0)
        self.assertFalse(report["promote"])
        small = paired_compare([episode(1, "win", 50)], [episode(1)], bootstrap_samples=100)
        self.assertFalse(small["promote"])
        self.assertIn("100 clean pairs", small["reasons"][0])
        empty = paired_compare([], [], bootstrap_samples=100)
        self.assertIsNone(empty["win_delta_ci95"])
        self.assertFalse(empty["promote"])


if __name__ == "__main__":
    unittest.main()
