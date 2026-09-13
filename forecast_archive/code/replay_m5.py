"""Replay saved M5 forecasts and calibrators without fitting or selecting anything.

This additive repair deliberately does not import the historical executable
scripts: their archived dependencies are incomplete.  The hierarchy feature
formulas below are transcribed from the bundled run_hierarchy_pilot.py.  Risk
prediction and policy application are an independent reconstruction of the
documented operation, not a recovered copy of the missing helper module.

Correctness is checked against saved validation predictions, frozen policy
edges/cell populations, per-item sufficient statistics, and reported metrics.
The absent q-as-feature and truth-label models cannot be replayed by this file.
All writes are directed to a new --output directory outside the input package.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "m5"))
from legacy_features.features import DesignPanel, FeatureBuilder


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class HierarchyFeatures:
    """The seven observable hierarchy features preserved in the bundled source."""
    def __init__(self, panel):
        _, self.inv = np.unique(panel.metadata["item_id"], return_inverse=True)
        n = self.inv.max() + 1
        count = np.bincount(self.inv).astype(float)[:, None]
        agg = np.zeros((n, panel.n_days), np.float32)
        hits = np.zeros_like(agg)
        np.add.at(agg, self.inv, panel.observed)
        np.add.at(hits, self.inv, (panel.observed >= panel.capacity).astype(np.float32))
        self.mean = agg / count
        self.hit = hits / count
        self.cs = np.pad(np.cumsum(self.mean.astype(float), axis=1), ((0, 0), (1, 0)))
        self.hcs = np.pad(np.cumsum(self.hit.astype(float), axis=1), ((0, 0), (1, 0)))

    def make(self, series, target_days):
        item = self.inv[series]
        origin = target_days - (1 + ((target_days - 1) % 7))
        lag = [self.mean[item, origin - k] for k in [1, 7, 28]]
        means = [(self.cs[item, origin] - self.cs[item, origin - w]) / w for w in [7, 28, 56]]
        hit28 = (self.hcs[item, origin] - self.hcs[item, origin - 28]) / 28
        return np.column_stack([*lag, *means, hit28]).astype(np.float32)


def predict_saved(builder, hierarchy, base, hit, days, threads, score_features=False):
    n = builder.panel.n_series
    forecasts = np.empty((n, len(days)), np.float32)
    risks = np.empty_like(forecasts)
    extra = {k: np.empty_like(forecasts) for k in
             ["historical_hit_rate_28", "origin_fill_ratio", "capacity_pressure"]} if score_features else {}
    for start in range(0, len(days), 7):
        ds = days[start:start + 7]
        s = np.repeat(np.arange(n, dtype=np.int32), len(ds))
        d = np.tile(ds, n)
        common = builder.make_features(s, d, include_censor=True)
        x = np.column_stack([common, hierarchy.make(s, d)])
        forecasts[:, start:start + len(ds)] = base.predict(xgb.DMatrix(x)).reshape(n, -1)
        risks[:, start:start + len(ds)] = np.clip(hit.predict(common, num_threads=threads), 0, 1).reshape(n, -1)
        if score_features:
            for name, index, sign in [("historical_hit_rate_28", 29, 1),
                                      ("origin_fill_ratio", 31, 1), ("capacity_pressure", 32, -1)]:
                extra[name][:, start:start + len(ds)] = (sign * common[:, index]).reshape(n, -1)
    return np.maximum(forecasts - 1, 0), risks, extra


def apply_saved_policy(forecast, score, categories, policy):
    """Apply the saved bin edges and scales; no quantile fitting or selection."""
    bins = np.searchsorted(np.asarray(policy["edges"]), score, side="right")
    multiplier = np.empty_like(forecast, dtype=float)
    for cat in np.unique(categories):
        row = categories == cat
        scale = np.asarray([policy["scales"][f"{cat}|{j}"] for j in range(len(policy["edges"]) + 1)])
        multiplier[row] = scale[bins[row]]
    return multiplier * forecast


def scores(y, f):
    error = y - f
    mass = y.sum()
    return {"wape": float(np.abs(error).sum() / mass), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(-error.sum() / mass)}


def main(args):
    started = time.monotonic()
    root = args.package.resolve()
    out = args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError("--output must be outside the immutable source package")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": "saved_model_replay",
               "training_calls": 0, "policy_fit_calls": 0, "policy_selection_calls": 0,
               "new_holdout_accesses": 0, "input_package": str(root), "checks": [], "input_sha256": {},
               "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                            "xgboost": xgb.__version__, "lightgbm": lgb.__version__},
               "limitations": ["q-as-feature and truth-label model files are absent; their forecast metrics are not independently replayed",
                               "Missing helper implementation is reconstructed, with numeric equivalence tested only for the bundled inputs"]}
    checks = receipt["checks"]

    def check(name, actual, expected, tolerance=5e-10):
        a, e = np.asarray(actual), np.asarray(expected)
        if a.shape != e.shape:
            checks.append({"name": name, "pass": False, "actual_shape": list(a.shape), "expected_shape": list(e.shape)})
            return
        if a.dtype.kind in "OUS" or e.dtype.kind in "OUS":
            ok, diff = bool(np.array_equal(a, e)), None
        else:
            diff = float(np.max(np.abs(a.astype(float) - e.astype(float)))) if a.size else 0.0
            ok = bool(np.isfinite(diff) and diff <= tolerance)
        checks.append({"name": name, "pass": ok, "max_absolute_difference": diff, "tolerance": tolerance})

    def record(path):
        path = Path(path)
        key = str(path.relative_to(root)) if root in path.parents else str(path.relative_to(out))
        receipt["input_sha256"][key] = sha256(path)

    archive = root / "inputs/M5_DEVELOPMENT_INPUTS.zip"
    record(archive)
    extracted = out / "extracted_inputs"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name in ["data/design_outcomes_v0_5.npz", "data/calendar.csv", "data/sell_prices.csv"] + [f"cache/{b}_aligned.npz" for b in args.blocks]:
            z.extract(name, extracted)
            record(extracted / name)
    attribution_path = root / "evidence/risk_calibration/attribution_controls/RESULTS.json"
    attribution = load_json(attribution_path)
    record(attribution_path)
    attr_protocol = load_json(root / "evidence/risk_calibration/attribution_controls/PROTOCOL.json")
    for label, filename in [("design", "design_outcomes_v0_5.npz"), ("calendar", "calendar.csv"), ("prices", "sell_prices.csv")]:
        check(f"input_hash:{label}", sha256(extracted / "data" / filename), attr_protocol["input_sha256"][label])
    panel = DesignPanel.load(extracted / "data/design_outcomes_v0_5.npz", extracted / "data/calendar.csv", extracted / "data/sell_prices.csv")
    panel.censored = (panel.observed >= panel.capacity).astype(np.uint8)
    builder, hierarchy = FeatureBuilder(panel), HierarchyFeatures(panel)
    cats = panel.metadata["cat_id"]
    items, inv = np.unique(panel.metadata["item_id"], return_inverse=True)
    validation_path = root / "evidence/risk_calibration/primary_base/validation.npz"
    ref_validation = np.load(validation_path)
    valid_days = ref_validation["days"]
    record(validation_path)
    results = {"rows": [], "matched_score_controls": {}, "validation": {}, "unreplayable_controls": ["q_as_feature_l1", "truth_label_oracle_l1"]}
    print(f"Loaded {panel.n_series} series; beginning saved-model inference", flush=True)
    for seed in args.seeds:
        if seed == 20260906:
            model_root = root / "evidence/risk_calibration/primary_base"
            policy_root = root / "evidence/risk_calibration/primary_policy"
            ref_rows = load_json(policy_root / "RESULTS.json")["rows"]
        else:
            model_root = root / f"evidence/risk_calibration/seed_replication/seed_{seed}"
            policy_root = model_root
            ref_rows = [r for r in load_json(root / "evidence/risk_calibration/seed_replication/RESULTS.json")["rows"] if r["seed"] == seed]
        model_path = model_root / "model.ubj"
        hit_path = root / f"evidence/revision/m5_observable/seed_{seed}/hit.txt"
        policy_path = policy_root / "POLICY_FREEZE.json"
        for p in [model_path, hit_path, policy_path]:
            record(p)
        policy_blob = load_json(policy_path)
        policy = policy_blob["risk_policy"]
        base = xgb.Booster(params={"nthread": args.threads})
        base.load_model(model_path)
        base.set_param({"nthread": args.threads})
        hit = lgb.Booster(model_file=str(hit_path))
        check(f"seed{seed}:feature_count", base.num_features(), 40, 0)
        if seed == 20260906:
            check("primary:model_hash", sha256(model_path), attr_protocol["input_sha256"]["hierarchy_base"])
            check("primary:hit_hash", sha256(hit_path), attr_protocol["input_sha256"]["hit_model"])
            fv, qv, _ = predict_saved(builder, hierarchy, base, hit, valid_days, args.threads)
            check("primary:validation_predictions", fv, ref_validation["prediction"], 0)
            expected_edges = np.unique(np.quantile(qv, np.arange(1, policy["nbins"]) / policy["nbins"]))
            check("primary:validation_risk_quantiles", expected_edges, policy["edges"], 0)
            groups = np.searchsorted(policy["edges"], qv, side="right")
            yv = panel.truth[:, valid_days - 1].astype(float)
            for cell in [v for v in attribution["cells"] if v["population"] == "point_validation"]:
                mask = (cats == cell["category"])[:, None] & (groups == cell["bin"] - 1)
                name = f"validation_cell:{cell['category']}:{cell['bin']}"
                check(name + ":rows", int(mask.sum()), cell["rows"], 0)
                check(name + ":demand_mass", float(yv[mask].sum()), cell["demand_mass"], 0)
                if mask.any():
                    check(name + ":mean_risk", float(qv[mask].mean()), cell["mean_risk"], 0)
            results["validation"] = {"rows": int(fv.size), "saved_prediction_max_abs_diff": float(np.max(np.abs(fv - ref_validation["prediction"]))) }
            del fv, qv, groups, yv
        for block in args.blocks:
            days = np.load(extracted / f"cache/{block}_aligned.npz")["target_days"]
            y = panel.truth[:, days - 1].astype(float)
            f, q, extras = predict_saved(builder, hierarchy, base, hit, days, args.threads,
                                         score_features=(seed == 20260906 and block == "shadow"))
            category_scales = np.asarray([policy_blob["base_category_scales"][str(c)] for c in cats])[:, None]
            p0 = category_scales * f
            p1 = apply_saved_policy(f, q, cats, policy)
            row = {"seed": seed, "block": block, "base": scores(y, p0), "risk": scores(y, p1)}
            results["rows"].append(row)
            expected = next(r for r in ref_rows if r["block"] == block)
            for method in ["base", "risk"]:
                for metric in ["wape", "mae", "rmse", "bias"]:
                    check(f"seed{seed}:{block}:{method}:{metric}", row[method][metric], expected[method][metric])
            item_stats = {"items": items, "mass": np.bincount(inv, weights=y.sum(axis=1)),
                          "base_error": np.bincount(inv, weights=np.abs(y - p0).sum(axis=1)),
                          "risk_error": np.bincount(inv, weights=np.abs(y - p1).sum(axis=1))}
            ref_stat_path = policy_root / f"{block}_item_stats.npz"
            record(ref_stat_path)
            ref_stats = np.load(ref_stat_path)
            for key, value in item_stats.items():
                check(f"seed{seed}:{block}:item_stats:{key}", value, ref_stats[key], 5e-7)
            np.savez_compressed(out / f"seed_{seed}_{block}_item_stats.npz", **item_stats)
            if seed == 20260906 and block == "shadow":
                extras.update({"learned_hit_risk": q, "base_forecast": f})
                for name, score in extras.items():
                    saved = attribution["corrected_score_ablation"]["scores"][name]
                    forecast = apply_saved_policy(f, score, cats, saved["policy"])
                    actual = scores(y, forecast)
                    results["matched_score_controls"][name] = actual
                    for metric in actual:
                        check(f"matched_score:{name}:{metric}", actual[metric], saved["later"][metric])
                # The complete fixed policy and primary cells link the independent
                # application implementation to the public mechanistic audit.
                check("matched_learned_policy:edges", attribution["corrected_score_ablation"]["scores"]["learned_hit_risk"]["policy"]["edges"], policy["edges"], 0)
                groups = np.searchsorted(policy["edges"], q, side="right")
                for cell in [v for v in attribution["cells"] if v["population"] == "later"]:
                    mask = (cats == cell["category"])[:, None] & (groups == cell["bin"] - 1)
                    name = f"later_cell:{cell['category']}:{cell['bin']}"
                    check(name + ":rows", int(mask.sum()), cell["rows"], 0)
                    check(name + ":demand_mass", float(y[mask].sum()), cell["demand_mass"], 0)
            print(json.dumps(row, sort_keys=True), flush=True)
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["status"] = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    receipt["check_count"] = len(checks)
    receipt["failed_checks"] = [c for c in checks if not c["pass"]]
    write_json(out / "REPLAY_RESULTS.json", results)
    write_json(out / "REPLAY_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "elapsed_seconds": receipt["elapsed_seconds"], "failed": receipt["failed_checks"]}, indent=2), flush=True)
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", choices=[20260906, 20260907, 20260908], default=[20260906, 20260907, 20260908])
    parser.add_argument("--blocks", nargs="+", choices=["calibration_a", "calibration_b", "shadow"], default=["calibration_a", "calibration_b", "shadow"])
    main(parser.parse_args())
