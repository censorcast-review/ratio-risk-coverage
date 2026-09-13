"""Verify saved cross-fit membership, protocol ordering, statistics and forecasts.

Does not fit models, choose candidates or overwrite evidence. Uses existing M5
labels solely to reconstruct the already reported descriptive metrics.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from replay_m5 import load_json, sha256, write_json


def main(args):
    root = args.package.resolve()
    out = root / "evidence/cell_audit/crossfit"
    protocol, results = load_json(out / "PROTOCOL.json"), load_json(out / "RESULTS.json")
    checks = []

    def check(name, value):
        checks.append({"name": name, "pass": bool(value)})
        if not value:
            raise AssertionError(name)

    def close(name, actual, expected, tol=1e-11):
        check(name, np.allclose(actual, expected, rtol=0, atol=tol))

    hashes = load_json(out / "ARTIFACT_HASHES.json")
    for name, expected in hashes.items():
        check("sha256:" + name, sha256(out / name) == expected)
    check("code_matches_frozen_protocol", sha256(root / "code/run_crossfit_control.py") == protocol["script_sha256"])
    check("result_protocol_identity", results["protocol_sha256"] == sha256(out / "PROTOCOL.json"))
    check("exactly_six_fits", len(results["fits"]) == results["training_calls"] == 6)
    check("no_search_or_new_holdout", all(results[k] == 0 for k in
                                          ["hyperparameter_search_calls", "candidate_selection_calls", "new_holdout_accesses"]))
    frozen_at = datetime.fromisoformat(protocol["created_utc"])
    for fit in results["fits"]:
        check("protocol_precedes:" + fit["name"], frozen_at < datetime.fromisoformat(fit["started_utc"]))
    hit_protocol = load_json(root / "evidence/revision/m5_observable/PROTOCOL.json")
    base_protocol = load_json(root / "evidence/risk_calibration/attribution_controls/PROTOCOL.json")
    check("hit_parameters_exact", protocol["hit_parameters"] == hit_protocol["parameters"])
    check("base_parameters_exact", protocol["base_parameters"] == base_protocol["xgboost"])
    sample = np.load(out / "TRAINING_SAMPLE.npz")
    reference_sample = np.load(root / "evidence/reproducibility/m5_control_refit/TRAINING_SAMPLE.npz")
    for key in ["series", "days"]:
        check("archived_sample:" + key, np.array_equal(sample[key], reference_sample[key]))
    data = np.load(args.input_dir / "data/design_outcomes_v0_5.npz")
    check("design_hash", sha256(args.input_dir / "data/design_outcomes_v0_5.npz") == protocol["input_sha256"]["design_outcomes_v0_5.npz"])
    metadata_items = data["item_id"].astype(str)
    categories = data["cat_id"].astype(str)
    items, inv = np.unique(metadata_items, return_inverse=True)
    fold_frame = pd.read_csv(out / "ITEM_FOLDS.csv")
    check("fold_item_identity", np.array_equal(fold_frame.item_id.to_numpy(), items))
    sample_item = inv[sample["series"]]
    check("sample_item_identity", np.array_equal(sample_item, sample["item_index"]))
    item_cat = np.asarray([np.unique(categories[inv == i]).item() for i in range(len(items))])
    check("fold_categories", np.array_equal(item_cat, fold_frame.category.to_numpy()))
    expected_fold = np.full(len(items), -1)
    rng = np.random.default_rng(protocol["fold_seed"])
    for c in np.unique(item_cat):
        ids = rng.permutation(np.flatnonzero(item_cat == c))
        expected_fold[ids] = np.arange(len(ids)) % 5
    check("fixed_item_folds", np.array_equal(expected_fold, fold_frame.fold.to_numpy()))
    sample_fold = expected_fold[sample_item]
    check("sample_folds", np.array_equal(sample_fold, sample["fold"]))
    n = len(sample_fold)
    count = np.zeros(n, int)
    for j in range(5):
        membership = np.load(out / f"fold_{j}_membership.npz")
        train, held = membership["train_row_index"], membership["predicted_row_index"]
        check(f"fold{j}:full_training_complement", np.array_equal(train, np.flatnonzero(sample_fold != j)))
        check(f"fold{j}:exact_heldout_rows", np.array_equal(held, np.flatnonzero(sample_fold == j)))
        check(f"fold{j}:no_item_overlap", len(np.intersect1d(sample_item[train], sample_item[held])) == 0)
        count[held] += 1
    check("all_rows_have_exactly_one_oof_prediction", np.all(count == 1))
    q = np.load(out / "TRAINING_Q.npz")
    check("oof_finite_unit_interval", np.all(np.isfinite(q["q_oof"]) & (q["q_oof"] >= 0) & (q["q_oof"] <= 1)))
    observed = data["observed"][sample["series"], sample["days"] - 1]
    capacity = data["capacity"][sample["series"], sample["days"] - 1]
    check("observable_hit_labels", np.array_equal(sample["hit_target"], observed >= capacity))
    check("observed_base_labels", np.array_equal(sample["observed_target"], observed))
    close("oof_brier", np.mean((q["q_oof"] - (observed >= capacity)) ** 2), results["risk_training_diagnostics"]["out_of_fold_brier"])
    predictions = np.load(out / "PREDICTIONS.npz")
    scales = load_json(out / "CATEGORY_SCALE_FREEZE.json")["category_scales"]
    check("category_scale_identity", scales == results["category_scales"])
    for block in ["validation", "later"]:
        days = predictions[block + "_days"]
        y = data["truth"][:, days - 1].astype(float)
        raw, f = predictions[block + "_raw"], predictions[block + "_scaled"]
        close(block + ":scale_application", f, raw * np.asarray([scales[c] for c in categories])[:, None], 0)
        errors = np.abs(y - f)
        measured = {"wape": errors.sum() / y.sum(), "mae": errors.mean(),
                    "rmse": np.sqrt(np.mean((y - f) ** 2)), "bias": (f - y).sum() / y.sum()}
        for metric, value in measured.items():
            close(f"{block}:{metric}", value, results[block]["category_scaled"][metric])
        if block == "validation":
            check("purged_validation_origins", np.all(days - (1 + (days - 1) % 7) >= 1313))
            for c in np.unique(categories):
                yc, fc = y[categories == c].ravel(), raw[categories == c].ravel().astype(float)
                keep = fc > 0
                ratio, weight = yc[keep] / fc[keep], fc[keep]
                total = weight.sum()
                # Characterization is independent of sorting/tie convention.
                check("category_scale_is_weighted_median:" + c,
                      weight[ratio < scales[c]].sum() <= total / 2 + 1e-8 and
                      weight[ratio > scales[c]].sum() <= total / 2 + 1e-8)
        else:
            stats = np.load(out / "ITEM_STATISTICS.npz")
            mass = np.zeros(len(items))
            candidate = np.zeros(len(items))
            np.add.at(mass, inv, y.sum(axis=1))
            np.add.at(candidate, inv, errors.sum(axis=1))
            close("item_mass", mass, stats["mass"], 0)
            close("item_errors", candidate, stats["candidate_error"], 0)
            for name, key in [("vs_base", "base_error"), ("vs_rc", "rc_error")]:
                interval = results["later"]["paired_item_intervals"][name]
                rng = np.random.default_rng(interval["seed"])
                weight = rng.multinomial(len(items), np.full(len(items), 1 / len(items)), size=interval["draws"])
                values = weight @ (candidate - stats[key]) / (weight @ mass)
                close("paired_delta:" + name, (candidate - stats[key]).sum() / mass.sum(), interval["estimate"])
                close("paired_interval:" + name, np.quantile(values, [.025, .975]), interval["ci95"])
    report = {"status": "PASS", "checks": checks, "count": len(checks),
              "scope": "saved artifacts, item exclusion, fixed settings, protocol ordering and independent metric reconstruction; no new fits"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"PASS: {len(checks)} crossfit control checks")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
