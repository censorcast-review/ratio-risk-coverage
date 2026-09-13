"""Synthetic-only origin, coding, and historical numerical-equivalence checks."""
from dataclasses import replace
from pathlib import Path
import io
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict as ext
from censorcast_v05.features import DesignPanel, FeatureBuilder
from run_selective_study import construct, CUTOFF


def save_npz(path, **values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    ext.write(path, stream.getvalue())


class ExternalPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        rng = np.random.default_rng(31851)
        n = 100
        truth = rng.poisson(3, (n, 1941)).astype(np.float32)
        capacity = rng.integers(1, 6, (n, 1941)).astype(np.float32)
        observed = np.minimum(truth, capacity)
        censored = (truth > capacity).astype(np.uint8)
        cls.meta = {
            "id": np.array([f"SYNTHETIC_EXTERNAL_{i}" for i in range(n)]),
            "item_id": np.array([f"SYNTHETIC_EXTERNAL_ITEM_{i}" for i in range(n)]),
            "dept_id": np.array([f"D{i % 4}" for i in range(n)]),
            "cat_id": np.array([f"C{i % 2}" for i in range(n)]),
            "store_id": np.array([f"S{i % 5}" for i in range(n)]),
            "state_id": np.array([("CA", "TX", "WI")[i % 3] for i in range(n)]),
        }
        cls.design_metadata = dict(cls.meta)
        cls.design_metadata["item_id"] = np.array([f"SYNTHETIC_DESIGN_ITEM_{i}" for i in range(n)])
        cls.context = dict(cls.meta, observed=observed[:, :1913], capacity=capacity[:, :1913],
                           censored=censored[:, :1913], day_start=np.array([1]), day_end=np.array([1913]))
        cls.outcomes = dict(cls.meta, truth=truth[:, 1913:], observed=observed[:, 1913:],
                            capacity=capacity[:, 1913:], censored=censored[:, 1913:],
                            day_start=np.array([1914]), day_end=np.array([1941]))
        days = np.arange(1, 1942)
        cls.calendar = pd.DataFrame({
            "d": [f"d_{d}" for d in days], "wm_yr_wk": (days - 1) // 7 + 1000,
            "wday": (days - 1) % 7 + 1, "month": ((days - 1) // 30) % 12 + 1,
            "year": 2011 + (days - 1) // 365,
            "event_name_1": np.where(days % 17 == 0, "EVENT", None),
            "event_name_2": np.where(days % 19 == 0, "SECOND_EVENT", None),
            "event_type_1": np.where(days % 17 == 0, "Cultural", None),
            "snap_CA": (days % 10 == 0).astype(int), "snap_TX": (days % 11 == 0).astype(int),
            "snap_WI": (days % 12 == 0).astype(int),
        })
        cls.calendar_path = cls.root / "calendar.csv"
        cls.calendar.to_csv(cls.calendar_path, index=False)
        weeks = cls.calendar["wm_yr_wk"].unique()
        prices = pd.DataFrame({"store_id": np.repeat(cls.meta["store_id"], len(weeks)),
                               "item_id": np.repeat(cls.meta["item_id"], len(weeks)),
                               "wm_yr_wk": np.tile(weeks, n),
                               "sell_price": np.tile(2 + (weeks % 9) / 10, n)})
        cls.prices_path = cls.root / "sell_prices.csv"
        prices.to_csv(cls.prices_path, index=False)
        save_npz(cls.root / "historical.npz", **dict(cls.context, truth=truth[:, :1913]))
        cls.old_panel = DesignPanel.load(cls.root / "historical.npz", cls.calendar_path, cls.prices_path)
        cls.panel = ext.build_extended_panel(cls.context, cls.outcomes, cls.calendar_path,
                                            cls.prices_path, cls.design_metadata)
        shape = (n, 1941 - 1314 + 1)
        base = rng.uniform(.1, 6, shape).astype(np.float32)
        em = rng.uniform(.1, 9, shape).astype(np.float32)
        probability = rng.uniform(0, 1, shape).astype(np.float32)
        cls.forecasts = {"baseline": base, "raw_poisson_histgb": (base * .8).astype(np.float32),
                         "censored_poisson_em": em, "censor_probability": probability,
                         "proposal": base + 2 * probability ** 2 * np.maximum(em - base, 0)}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_historical_point_features_are_bitwise_identical(self):
        days = np.array([1314, 1317, 1433, 1554, 1674, 1794, 1913])
        series = np.repeat(np.arange(self.panel.n_series), len(days))
        targets = np.tile(days, self.panel.n_series)
        for include_censor in (False, True):
            expected = FeatureBuilder(self.old_panel).make_features(series, targets, include_censor=include_censor)
            actual = ext.FrozenFeatureBuilder(self.panel).make_features(series, targets, include_censor=include_censor)
            np.testing.assert_array_equal(actual, expected)

    def test_extension_keeps_original_time_fraction_denominator(self):
        days = np.array([1914, 1920, 1941])
        features = ext.FrozenFeatureBuilder(self.panel).make_features(np.array([0, 1, 2]), days, include_censor=True)
        i = FeatureBuilder.COMMON_NAMES.index("time_fraction")
        np.testing.assert_array_equal(features[:, i], (days / 1913).astype(np.float32))
        self.assertTrue(np.all(features[:, i] > 1))

    def test_historical_risk_features_match_existing_construct(self):
        raw_root, output = self.root / "risk_input", self.root / "risk_output"
        raw_root.mkdir(exist_ok=True)
        output.mkdir(exist_ok=True)
        save_npz(raw_root / "design_outcomes_v0_5.npz", **dict(self.context, truth=self.old_panel.truth))
        blocks = {"selection_base": (1314, 1433), "risk_train": (1434, 1553),
                  "calibration_a": (1554, 1673), "calibration_b": (1674, 1793), "shadow": (1794, 1913)}
        for name, (start, end) in blocks.items():
            days = np.arange(start, end + 1)
            idx = days - 1314
            data = {key: value[:, idx] for key, value in self.forecasts.items()}
            data.update(observed=self.old_panel.observed[:, days - 1], truth=self.old_panel.truth[:, days - 1],
                        censored=self.old_panel.censored[:, days - 1], target_days=days,
                        prior_observed_wape=np.zeros((self.panel.n_series, len(days)), np.float32),
                        instantaneous_absolute_error=np.zeros((self.panel.n_series, len(days)), np.float32))
            save_npz(raw_root / f"{name}_v0_5.npz", **data)
        construct(raw_root, output)
        for block in ("risk_train", "calibration_a", "calibration_b", "shadow"):
            start, end = blocks[block]
            days = np.arange(start, end + 1)
            days = days[ext.origin(days) >= CUTOFF[block]]
            expected = np.load(output / f"{block}_features.npy")
            actual = ext.risk_features(self.panel, self.forecasts, days)
            np.testing.assert_array_equal(actual, expected)

    def test_after_origin_observed_sales_and_flags_cannot_change_features(self):
        days = np.array([1914, 1915, 1916, 1917, 1918])
        fixed_origin = int(ext.origin(days)[0])
        self.assertTrue(np.all(ext.origin(days) == fixed_origin))
        observed = self.panel.observed.copy()
        observed[:, fixed_origin:] += 1000
        capacity = self.panel.capacity.copy()
        capacity[:, fixed_origin:] += 2000
        censored = self.panel.censored.copy()
        censored[:, fixed_origin:] = 1 - censored[:, fixed_origin:]
        changed = replace(self.panel, observed=observed, capacity=capacity, censored=censored)
        series = np.repeat(np.arange(self.panel.n_series), len(days))
        targets = np.tile(days, self.panel.n_series)
        for include_censor in (False, True):
            original = ext.FrozenFeatureBuilder(self.panel).make_features(series, targets, include_censor=include_censor)
            altered = ext.FrozenFeatureBuilder(changed).make_features(series, targets, include_censor=include_censor)
            np.testing.assert_array_equal(altered, original)
        np.testing.assert_array_equal(ext.risk_features(changed, self.forecasts, days),
                                      ext.risk_features(self.panel, self.forecasts, days))

    def test_external_truth_is_never_in_feature_panel(self):
        changed = dict(self.outcomes)
        changed["truth"] = changed["truth"].copy()
        changed["truth"][changed["censored"].astype(bool)] += 1000
        panel = ext.build_extended_panel(self.context, changed, self.calendar_path, self.prices_path, self.design_metadata)
        np.testing.assert_array_equal(panel.truth, self.panel.observed)
        np.testing.assert_array_equal(panel.observed, self.panel.observed)
        np.testing.assert_array_equal(panel.censored, self.panel.censored)

    def test_static_categories_use_training_vocabulary_in_subset(self):
        selection = np.array([0, 2, 4, 6])
        meta = {name: array[selection] for name, array in self.meta.items()}
        codes = ext._category_codes(meta, self.design_metadata)
        for name in ext.STATIC:
            np.testing.assert_array_equal(codes[name], self.panel.static_codes[name][selection])

    def test_unseen_static_category_and_event_type_fail_closed(self):
        meta = dict(self.meta)
        meta["cat_id"] = np.repeat("NEVER_FITTED", self.panel.n_series)
        with self.assertRaisesRegex(ValueError, "Unknown frozen"):
            ext._category_codes(meta, self.design_metadata)
        calendar = self.calendar.copy()
        calendar.loc[1940, "event_type_1"] = "UNSEEN_FUTURE_TYPE"
        with self.assertRaisesRegex(ValueError, "outside the frozen vocabulary"):
            ext._calendar_values(calendar)

    def test_context_truth_rejected(self):
        bad = dict(self.context, truth=self.old_panel.truth)
        with self.assertRaisesRegex(ValueError, "without truth"):
            ext.build_extended_panel(bad, self.outcomes, self.calendar_path, self.prices_path, self.design_metadata)

    def test_overlap_with_training_items_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            ext.build_extended_panel(self.context, self.outcomes, self.calendar_path, self.prices_path, self.meta)

    def test_opening_gate_checked_before_input_or_model_access(self):
        with self.assertRaises(PermissionError):
            ext.predict_external(None, None, None, None, None, {}, "0" * 64)
        receipt = {"status": "AUTHORIZED_EXTERNAL_PREDICTION", "freeze_sha256": "a" * 64,
                   "external_use_count": 1, "original_external_gate_pass": False, "certificate_issued": False}
        ext._assert_gate(receipt, "a" * 64)
        for field, bad in (("external_use_count", 2), ("freeze_sha256", "b" * 64),
                           ("original_external_gate_pass", True), ("certificate_issued", True)):
            changed = dict(receipt)
            changed[field] = bad
            with self.assertRaises(PermissionError):
                ext._assert_gate(changed, "a" * 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
