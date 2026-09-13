from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from censorcast_v05.features import DesignPanel, FeatureBuilder, _fill_price_matrix
from censorcast_v05.metrics import (
    exact_item_cluster_delta_curve,
    fixed_policy_familywise_bootstrap,
    search_two_block_exact_threshold,
    wape,
)
from censorcast_v05.models import censored_poisson_conditional_mean


class PriceTests(unittest.TestCase):
    def test_forward_then_backward_fill(self):
        matrix = np.asarray([[np.nan, 2.0, np.nan, 3.0, np.nan], [1.0, np.nan, np.nan, np.nan, 4.0]])
        actual = _fill_price_matrix(matrix)
        np.testing.assert_allclose(actual[0], [2, 2, 2, 3, 3])
        np.testing.assert_allclose(actual[1], [1, 1, 1, 1, 4])


class CensoredPoissonTests(unittest.TestCase):
    def test_conditional_mean_exceeds_capacity_only_when_censored(self):
        mean = np.asarray([2.0, 5.0, 8.0])
        capacity = np.asarray([3.0, 4.0, 10.0])
        flag = np.asarray([1, 0, 1])
        actual = censored_poisson_conditional_mean(mean, capacity, flag, cap_multiplier=8.0)
        self.assertGreaterEqual(actual[0], 4.0)
        self.assertAlmostEqual(float(actual[1]), 5.0, places=5)
        self.assertGreaterEqual(actual[2], 11.0)


class FeatureLeakageTests(unittest.TestCase):
    def make_panel(self, truth):
        n, days = truth.shape
        metadata = {
            "id": np.asarray([f"id_{i}" for i in range(n)]),
            "item_id": np.asarray([f"item_{i // 2}" for i in range(n)]),
            "dept_id": np.asarray(["dept" for _ in range(n)]),
            "cat_id": np.asarray(["cat" for _ in range(n)]),
            "store_id": np.asarray([f"store_{i % 2}" for i in range(n)]),
            "state_id": np.asarray(["CA" for _ in range(n)]),
        }
        calendar = pd.DataFrame({"d": [f"d_{i}" for i in range(1, days + 1)]})
        arrays = {
            "wday": ((np.arange(days) % 7) + 1).astype(np.float32),
            "month": ((np.arange(days) // 30) % 12 + 1).astype(np.float32),
            "year": np.full(days, 2011, dtype=np.float32),
            "event_any": np.zeros(days, dtype=np.float32),
            "event_type": np.zeros(days, dtype=np.float32),
            "snap_CA": np.zeros(days, dtype=np.float32),
            "snap_TX": np.zeros(days, dtype=np.float32),
            "snap_WI": np.zeros(days, dtype=np.float32),
        }
        observed = truth.copy().astype(np.float32)
        return DesignPanel(
            truth=truth.astype(np.float32),
            observed=observed,
            capacity=np.maximum(observed, 1),
            censored=np.zeros_like(observed, dtype=np.uint8),
            metadata=metadata,
            calendar=calendar,
            price_matrix=np.ones((n, 20), dtype=np.float32),
            day_to_week_index=np.minimum(np.arange(days) // 7, 19).astype(np.int16),
            static_codes={name: np.zeros(n, dtype=np.int16) for name in ("dept_id", "cat_id", "store_id", "state_id")},
            calendar_arrays=arrays,
        )

    def test_future_truth_change_does_not_change_origin_features(self):
        rng = np.random.default_rng(4)
        truth = rng.poisson(3, size=(8, 120)).astype(np.float32)
        panel_a = self.make_panel(truth)
        panel_b = self.make_panel(truth.copy())
        panel_b.truth[:, 100:] += 500
        series = np.arange(8, dtype=np.int32)
        day = np.full(8, 90, dtype=np.int32)
        a = FeatureBuilder(panel_a).make_features(series, day, include_censor=True)
        b = FeatureBuilder(panel_b).make_features(series, day, include_censor=True)
        np.testing.assert_array_equal(a, b)


class ExactFrontierTests(unittest.TestCase):
    def test_exact_prefix_metrics_match_brute_force(self):
        rng = np.random.default_rng(8)
        clusters = np.repeat(np.arange(12), 20)
        y = rng.gamma(2.0, 2.0, size=len(clusters))
        p = np.clip(y + rng.normal(0, 1, size=len(y)), 0, None)
        score = rng.normal(size=len(y))
        baseline_wape = 0.5
        curve = exact_item_cluster_delta_curve(
            score=score,
            truth=y,
            prediction=p,
            item_cluster=clusters,
            baseline_selection_wape=baseline_wape,
        )
        index = len(curve.threshold) // 2
        threshold = curve.threshold[index]
        accepted = score <= threshold
        self.assertAlmostEqual(curve.coverage[index], float(accepted.mean()), places=12)
        self.assertAlmostEqual(curve.accepted_wape[index], wape(y[accepted], p[accepted]), places=12)

    def test_two_block_search_reports_every_unique_threshold(self):
        rng = np.random.default_rng(11)
        cluster = np.repeat(np.arange(15), 30)
        y = rng.gamma(2, 2, len(cluster))
        p = y + rng.normal(0, 0.2, len(cluster))
        score_a = rng.normal(size=len(cluster))
        score_b = rng.normal(size=len(cluster))
        a = exact_item_cluster_delta_curve(
            score=score_a, truth=y, prediction=p, item_cluster=cluster, baseline_selection_wape=0.5
        )
        b = exact_item_cluster_delta_curve(
            score=score_b, truth=y, prediction=p, item_cluster=cluster, baseline_selection_wape=0.5
        )
        result, _ = search_two_block_exact_threshold(
            a, b, maximum_ratio_ucb=0.85, minimum_coverage_lcb=0.2
        )
        expected = len(np.unique(np.r_[a.threshold, b.threshold]))
        self.assertEqual(result["thresholds_examined"], expected)


class BootstrapTests(unittest.TestCase):
    def test_familywise_bootstrap_is_deterministic(self):
        rng = np.random.default_rng(1)
        values = np.column_stack(
            [np.full(20, 100.0), np.full(20, 50.0), rng.uniform(50, 100, 20), rng.uniform(5, 10, 20)]
        )
        kwargs = dict(
            aggregates={"a": values, "b": values * np.asarray([1, 1, 1, 1])},
            baseline_selection_wape=0.4,
            reps=200,
            seed=4,
            familywise_alpha=0.05,
            minimum_coverage_lcb=0.3,
            maximum_ratio_ucb=0.85,
        )
        first = fixed_policy_familywise_bootstrap(**kwargs)
        second = fixed_policy_familywise_bootstrap(**kwargs)
        self.assertEqual(first, second)


class CommandBoundaryTests(unittest.TestCase):
    def test_runner_exposes_no_guardian_or_external_command(self):
        text = (ROOT / "run_v0_5_design.py").read_text(encoding="utf-8")
        self.assertNotIn('sub.add_parser("guardian")', text)
        self.assertNotIn('sub.add_parser("external")', text)


if __name__ == "__main__":
    unittest.main()
