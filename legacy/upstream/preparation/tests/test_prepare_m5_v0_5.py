from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_m5_v0_5", ROOT / "prepare_m5_v0_5.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PartitionTests(unittest.TestCase):
    def test_partition_is_exact_disjoint_and_deterministic(self):
        items = [f"item_{index:03d}" for index in range(17)]
        specification = {
            "salt": "fixed",
            "design_items": 9,
            "fresh_guardian_items": 4,
            "external_items": 4,
        }
        first = MODULE.partition_items(items, specification)
        second = MODULE.partition_items(list(reversed(items)), specification)
        self.assertEqual(first, second)
        self.assertEqual({name: len(values) for name, values in first.items()}, {
            "design": 9,
            "fresh_guardian": 4,
            "external": 4,
        })
        joined = first["design"] + first["fresh_guardian"] + first["external"]
        self.assertEqual(set(joined), set(items))
        self.assertEqual(len(joined), len(set(joined)))

    def test_partition_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            MODULE.partition_items(
                ["a", "a"],
                {"salt": "x", "design_items": 1, "fresh_guardian_items": 1, "external_items": 0},
            )


class CensoringTests(unittest.TestCase):
    def test_rule_is_prequential(self):
        rng = np.random.default_rng(7)
        truth = rng.poisson(2.0, size=(8, 100)).astype(np.int32)
        changed = truth.copy()
        changed[:, 80:] += 100
        observed_a, capacity_a, censored_a = MODULE.simulate_controlled_censoring(truth, warmup_days=10)
        observed_b, capacity_b, censored_b = MODULE.simulate_controlled_censoring(changed, warmup_days=10)
        np.testing.assert_array_equal(observed_a[:, :80], observed_b[:, :80])
        np.testing.assert_array_equal(capacity_a[:, :80], capacity_b[:, :80])
        np.testing.assert_array_equal(censored_a[:, :80], censored_b[:, :80])

    def test_observation_and_censoring_semantics(self):
        truth = np.tile(np.arange(1, 81, dtype=np.int32), (3, 1))
        observed, capacity, censored = MODULE.simulate_controlled_censoring(truth, warmup_days=8)
        self.assertTrue(np.all(observed <= truth))
        self.assertTrue(np.all(observed <= capacity))
        np.testing.assert_array_equal(censored.astype(bool), truth > capacity)
        np.testing.assert_array_equal(observed[:, :8], truth[:, :8])


class ShardTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({
            "id": ["a", "b"],
            "item_id": ["i1", "i2"],
            "dept_id": ["d1", "d1"],
            "cat_id": ["c1", "c1"],
            "store_id": ["s1", "s1"],
            "state_id": ["x", "x"],
        })
        self.truth = np.arange(24, dtype=np.int32).reshape(2, 12)
        self.observed = self.truth.copy()
        self.capacity = np.maximum(self.truth, 1)
        self.censored = np.zeros_like(self.truth, dtype=np.uint8)

    def test_context_shard_omits_truth(self):
        payload = MODULE.shard_payload(
            self.metadata, self.truth, self.observed, self.capacity, self.censored, 1, 8, False
        )
        self.assertNotIn("truth", payload)
        self.assertEqual(payload["observed"].shape, (2, 8))

    def test_outcome_shard_has_exact_days(self):
        payload = MODULE.shard_payload(
            self.metadata, self.truth, self.observed, self.capacity, self.censored, 9, 12, True
        )
        self.assertEqual(payload["truth"].shape, (2, 4))
        self.assertEqual(int(payload["day_start"][0]), 9)
        self.assertEqual(int(payload["day_end"][0]), 12)

    def test_atomic_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            MODULE.atomic_npz(path, truth=self.truth, id=np.asarray(["a", "b"]))
            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["truth"], self.truth)
                np.testing.assert_array_equal(loaded["id"], np.asarray(["a", "b"]))


class ProtocolTests(unittest.TestCase):
    def test_protocol_has_no_v04_data_reuse(self):
        protocol = json.loads((ROOT / "configs" / "PROTOCOL_v0_5.json").read_text())
        self.assertFalse(protocol["independence"]["freshretail_v04_guardian_reused"])
        self.assertFalse(protocol["independence"]["freshretail_eval_or_test_authorized"])
        counts = protocol["item_cluster_partition"]
        self.assertEqual(counts["design_items"] + counts["fresh_guardian_items"] + counts["external_items"], 3049)
        self.assertEqual(protocol["time_partition"]["external_final"], [1914, 1941])


if __name__ == "__main__":
    unittest.main()
