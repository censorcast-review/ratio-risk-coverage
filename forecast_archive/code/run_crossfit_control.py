"""One fixed five-fold item-cross-fitted q-as-feature M5 control.

All M5 blocks are previously consumed. This is an exploratory counterfactual,
not a new holdout and not forward-time cross-fitting. The exact historical
1.2M sampled rows (not 600k) are recovered before fitting. Five observable
hit-risk heads and one 750-tree L1 base are fit; no candidate search is allowed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from replay_m5 import (DesignPanel, FeatureBuilder, HierarchyFeatures,
                      apply_saved_policy, load_json, scores, sha256, write_json)
from reproduce_m5_missing_controls import Progress, predict, weighted_scale


def now():
    return datetime.now(timezone.utc).isoformat()


def paired(items, mass, candidate, reference, seed=20260908, draws=4000):
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(items), np.full(len(items), 1 / len(items)), size=draws)
    vals = weights @ (candidate - reference) / (weights @ mass)
    return {"estimate": float((candidate - reference).sum() / mass.sum()),
            "ci95": np.quantile(vals, [.025, .975]).tolist(),
            "draws": draws, "seed": seed, "unit": "item across all stores"}


def main(args):
    started = time.monotonic()
    root, data, out = args.package.resolve(), args.input_dir.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed, fold_seed, kfold, rounds = 20260906, 20260908, 5, 750
    old_hit_path = root / "evidence/revision/m5_observable/PROTOCOL.json"
    old_base_path = root / "evidence/risk_calibration/attribution_controls/PROTOCOL.json"
    old_hit, old_base = load_json(old_hit_path), load_json(old_base_path)
    n_rows = int(old_hit["training_rows"])
    assert n_rows == old_base["training_rows"] == 1_200_000
    assert old_hit["training_days"] == old_base["training_days"] == [365, 1313]
    hit_params = dict(old_hit["parameters"])
    base_params = dict(old_base["xgboost"])
    assert hit_params["n_estimators"] == 300 and base_params["seed"] == seed
    full_hit_path = root / f"evidence/revision/m5_observable/seed_{seed}/hit.txt"
    sample_path = root / "evidence/reproducibility/m5_control_refit/TRAINING_SAMPLE.npz"
    input_hashes = {n: sha256(data / "data" / n) for n in old_hit["input_sha256"]}
    assert input_hashes == old_hit["input_sha256"]
    assert sha256(full_hit_path) == old_base["input_sha256"]["hit_model"]
    protocol = {
        "status": "FIXED_BEFORE_NEW_FITS", "created_utc": now(),
        "scope": "exploratory fixed comparison on previously consumed M5 development data",
        "new_holdout_accesses": 0, "forward_time_crossfit": False,
        "training_sample_rows": n_rows, "training_days": [365, 1313],
        "sample_seed": seed, "n_folds": kfold, "fold_seed": fold_seed,
        "fold_rule": "sorted unique items within each category; one seeded shuffle per category; round-robin fold assignment; all stores for an item stay together",
        "exclusion": "each risk head excludes every sampled row from its held-out items; only that head predicts q for those items' base-training rows",
        "same_sample_as_historical_hit_and_base": True,
        "historical_sample_size_correction": "both archived protocols specify 1.2M rows; 600k is not this M5 configuration",
        "hit_target": "observable H = 1{S >= C}", "base_target": "observed S + 1",
        "hit_parameters": hit_params, "hit_objective": "binary", "hit_random_state": seed,
        "base_parameters": base_params, "base_rounds": rounds,
        "planned_training_calls": {"crossfit_hit_heads": 5, "base_l1": 1},
        "inference_q": "reuse original full-training hit head on validation and Later; no sixth risk fit",
        "category_scale": "exact nonnegative weighted-median scale using only original validation rows; freeze before scoring Later",
        "validation_days": [1314, 1433], "validation_origin_purge": ">= 1313",
        "evaluation": "saved Later/shadow days; pooled and category WAPE, MAE, RMSE, signed bias",
        "paired_uncertainty": {"unit": "item across stores", "draws": 4000, "seed": fold_seed,
                                "contrasts": ["crossfit minus category-scaled hierarchy base", "crossfit minus RC"]},
        "hyperparameter_search_calls": 0, "candidate_selection_calls": 0,
        "limitation": "item-held-out q addresses same-row target fitting; it does not evaluate forward-time cross-fitting or all possible stackers, and fold risk heads have 80% of original sampled training rows",
        "input_sha256": input_hashes,
        "original_protocol_sha256": {"observable_hit": sha256(old_hit_path), "base": sha256(old_base_path)},
        "original_protocol_paths": [str(old_hit_path.relative_to(root)), str(old_base_path.relative_to(root))],
        "reference_sample_sha256": sha256(sample_path), "full_hit_model_sha256": sha256(full_hit_path),
        "script_sha256": sha256(__file__),
        "dependency_sha256": {n: sha256(root / "code" / n) for n in
                              ["replay_m5.py", "reproduce_m5_missing_controls.py", "m5/legacy_features/features.py"]},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "lightgbm": lgb.__version__, "xgboost": xgb.__version__}}
    write_json(out / "PROTOCOL.json", protocol)
    print("Fixed protocol written; loading panel and reconstructing exact sample", flush=True)
    panel = DesignPanel.load(data / "data/design_outcomes_v0_5.npz", data / "data/calendar.csv", data / "data/sell_prices.csv")
    panel.censored = (panel.observed >= panel.capacity).astype(np.uint8)
    builder, hierarchy = FeatureBuilder(panel), HierarchyFeatures(panel)
    series, days = builder.sample_pairs(365, 1313, n_rows, seed)
    reference_sample = np.load(sample_path)
    assert np.array_equal(series, reference_sample["series"])
    assert np.array_equal(days, reference_sample["days"])
    common = builder.make_features(series, days, include_censor=True)
    extra = hierarchy.make(series, days)
    observed = panel.observed[series, days - 1].astype(float)
    hit_labels = (observed >= panel.capacity[series, days - 1]).astype(int)
    items, inverse = np.unique(panel.metadata["item_id"], return_inverse=True)
    item_categories = np.asarray([np.unique(panel.metadata["cat_id"][inverse == i]).item() for i in range(len(items))])
    fold = np.full(len(items), -1, np.int8)
    rng = np.random.default_rng(fold_seed)
    for c in np.unique(item_categories):
        idx = np.flatnonzero(item_categories == c)
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % kfold
    train_fold = fold[inverse[series]]
    frame = pd.DataFrame({"item_id": items, "category": item_categories, "fold": fold,
                          "series_count": np.bincount(inverse),
                          "sampled_rows": np.bincount(inverse[series], minlength=len(items))})
    frame.to_csv(out / "ITEM_FOLDS.csv", index=False)
    np.savez_compressed(out / "TRAINING_SAMPLE.npz", series=series, days=days,
                        item_index=inverse[series], fold=train_fold, hit_target=hit_labels, observed_target=observed)
    # Hidden truth/strict indicator are neither labels nor features in these fits.
    check_n = min(2048, n_rows)
    old_truth, old_flag = panel.truth, panel.censored
    panel.truth = np.zeros_like(old_truth)
    panel.censored = 1 - old_flag
    assert np.array_equal(common[:check_n], builder.make_features(series[:check_n], days[:check_n], include_censor=True))
    assert np.array_equal(extra[:check_n], hierarchy.make(series[:check_n], days[:check_n]))
    panel.truth, panel.censored = old_truth, old_flag
    leakage = {"status": "PRE_FIT_CHECKS_PASS", "sample_exactly_matches_archived_reproduction": True,
               "sample_rows": n_rows, "item_count": len(items), "feature_count": common.shape[1] + extra.shape[1] + 1,
               "all_training_days_in_range": bool(np.all((days >= 365) & (days <= 1313))),
               "features_and_labels_use_only_observables": True,
               "hidden_truth_and_strict_flag_perturbation_feature_invariance": True,
               "validation_truth_used_in_model_fit": False,
               "held_out_item_training_overlap": [], "oof_assignment_count": None,
               "historical_feature_timing": "same audited origin-only feature builder; not forward-time model cross-fitting"}
    write_json(out / "LEAKAGE_CHECKS.json", leakage)
    q_oof = np.full(n_rows, np.nan, float)
    oof_count = np.zeros(n_rows, np.uint8)
    receipts = []
    for j in range(kfold):
        train, held = train_fold != j, train_fold == j
        train_items = np.unique(inverse[series[train]])
        held_items = np.unique(inverse[series[held]])
        overlap = int(np.intersect1d(train_items, held_items).size)
        assert overlap == 0
        leakage["held_out_item_training_overlap"].append({"fold": j, "overlap": overlap})
        train_row_index = np.flatnonzero(train)
        held_row_index = np.flatnonzero(held)
        np.savez_compressed(out / f"fold_{j}_membership.npz", train_row_index=train_row_index,
                            predicted_row_index=held_row_index)
        fit_start, tick = now(), time.monotonic()
        write_json(out / f"fold_{j}_FIT_START.json", {"started_utc": fit_start,
                   "protocol_sha256": sha256(out / "PROTOCOL.json"),
                   "training_rows": int(train.sum()), "held_out_rows": int(held.sum()),
                   "membership_sha256": sha256(out / f"fold_{j}_membership.npz")})
        print(f"Fitting risk fold {j + 1}/{kfold}: {train.sum()} training rows, {held.sum()} OOF predictions", flush=True)
        model = lgb.LGBMClassifier(objective="binary", random_state=seed, **hit_params).fit(common[train], hit_labels[train]).booster_
        model_path = out / f"hit_fold_{j}.txt"
        model.save_model(str(model_path))
        q_oof[held] = np.clip(model.predict(common[held], num_threads=hit_params["n_jobs"]), 0, 1)
        oof_count[held] += 1
        receipts.append({"name": f"hit_fold_{j}", "started_utc": fit_start, "finished_utc": now(),
                         "elapsed_seconds": time.monotonic() - tick, "training_rows": int(train.sum()),
                         "held_out_rows": int(held.sum()), "model_sha256": sha256(model_path)})
        write_json(out / "FIT_RECEIPTS.json", receipts)
        print(f"Risk fold {j + 1} complete in {time.monotonic() - tick:.1f}s", flush=True)
        del model
        gc.collect()
    assert np.all(oof_count == 1) and np.isfinite(q_oof).all()
    leakage["oof_assignment_count"] = {"min": int(oof_count.min()), "max": int(oof_count.max())}
    leakage["status"] = "PASS"
    write_json(out / "LEAKAGE_CHECKS.json", leakage)
    hit = lgb.Booster(model_file=str(full_hit_path))
    q_in = np.clip(hit.predict(common, num_threads=4), 0, 1)
    risk_diagnostics = {"in_sample_brier": float(np.mean((q_in - hit_labels) ** 2)),
                        "out_of_fold_brier": float(np.mean((q_oof - hit_labels) ** 2)),
                        "mean_absolute_q_difference": float(np.abs(q_oof - q_in).mean()),
                        "interpretation": "descriptive; fold heads use fewer rows and no held-out items"}
    np.savez_compressed(out / "TRAINING_Q.npz", q_oof=q_oof, q_full_in_sample=q_in,
                        assignment_count=oof_count)
    matrix = xgb.DMatrix(np.column_stack([common, extra, q_oof]), label=observed + 1.)
    del common, extra, observed, q_in, q_oof, train, held, train_row_index, held_row_index
    gc.collect()
    fit_start, tick = now(), time.monotonic()
    write_json(out / "base_FIT_START.json", {"started_utc": fit_start, "protocol_sha256": sha256(out / "PROTOCOL.json"),
               "training_sample_sha256": sha256(out / "TRAINING_SAMPLE.npz"), "training_q_sha256": sha256(out / "TRAINING_Q.npz")})
    print("Fitting one fixed OOF-q L1 base: 750 rounds", flush=True)
    base = xgb.train(base_params, matrix, num_boost_round=rounds, callbacks=[Progress("crossfit_q_base", rounds)])
    base_path = out / "crossfit_q_base.ubj"
    base.save_model(base_path)
    receipts.append({"name": "crossfit_q_base", "started_utc": fit_start, "finished_utc": now(),
                     "elapsed_seconds": time.monotonic() - tick, "model_sha256": sha256(base_path),
                     "training_rows": n_rows, "rounds": rounds})
    write_json(out / "FIT_RECEIPTS.json", receipts)
    del matrix
    gc.collect()
    cats = panel.metadata["cat_id"]
    validation = np.load(root / "evidence/risk_calibration/primary_base/validation.npz")
    valid_days = validation["days"]
    assert np.all(valid_days - builder.horizon_for_day(valid_days) >= 1313)
    assert valid_days.min() >= 1314 and valid_days.max() <= 1433
    yv = panel.truth[:, valid_days - 1].astype(float)
    fv = predict(builder, hierarchy, hit, base, valid_days, True, 8)
    scales = {str(c): weighted_scale(yv[cats == c], fv[cats == c]) for c in np.unique(cats)}
    write_json(out / "CATEGORY_SCALE_FREEZE.json", {"created_utc": now(), "category_scales": scales,
               "validation_days": valid_days.tolist(), "later_scoring_calls": 0,
               "model_sha256": sha256(base_path), "protocol_sha256": sha256(out / "PROTOCOL.json")})
    scale_vector = np.asarray([scales[str(c)] for c in cats])[:, None]
    fvs = scale_vector * fv
    validation_metrics = {"raw": scores(yv, fv), "category_scaled": scores(yv, fvs),
                          "categories": {str(c): scores(yv[cats == c], fvs[cats == c]) for c in np.unique(cats)}}
    frozen = load_json(root / "evidence/risk_calibration/primary_policy/POLICY_FREEZE.json")
    reference_base_v = validation["prediction"].astype(float) * np.asarray([frozen["base_category_scales"][str(c)] for c in cats])[:, None]
    qv = np.load(root / "evidence/revision/m5_observable/seed_20260906/validation_heads.npz")["hit"]
    reference_rc_v = apply_saved_policy(validation["prediction"], qv, cats, frozen["risk_policy"])
    validation_metrics["reference_base"] = scores(yv, reference_base_v)
    validation_metrics["reference_rc"] = scores(yv, reference_rc_v)
    print("Validation scaled WAPE:", validation_metrics["category_scaled"]["wape"], flush=True)
    later_days = np.load(data / "cache/shadow_aligned.npz")["target_days"]
    yl = panel.truth[:, later_days - 1].astype(float)
    fl = predict(builder, hierarchy, hit, base, later_days, True, 8)
    fls = scale_vector * fl
    reference = np.load(root / "evidence/risk_calibration/primary_policy/shadow_item_stats.npz")
    assert np.array_equal(items, reference["items"])
    mass = np.bincount(inverse, weights=yl.sum(axis=1))
    error = np.bincount(inverse, weights=np.abs(yl - fls).sum(axis=1))
    assert np.array_equal(mass, reference["mass"])
    uncertainty = {"vs_base": paired(items, mass, error, reference["base_error"]),
                   "vs_rc": paired(items, mass, error, reference["risk_error"])}
    np.savez_compressed(out / "PREDICTIONS.npz", validation_days=valid_days, validation_raw=fv,
                        validation_scaled=fvs, later_days=later_days, later_raw=fl, later_scaled=fls)
    np.savez_compressed(out / "ITEM_STATISTICS.npz", items=items, mass=mass, candidate_error=error,
                        base_error=reference["base_error"], rc_error=reference["risk_error"])
    results = {"status": "COMPLETED", "created_utc": now(), "elapsed_seconds": time.monotonic() - started,
               "training_calls": 6, "hyperparameter_search_calls": 0, "candidate_selection_calls": 0,
               "new_holdout_accesses": 0, "scope": protocol["scope"],
               "protocol_sha256": sha256(out / "PROTOCOL.json"), "fits": receipts,
               "risk_training_diagnostics": risk_diagnostics, "category_scales": scales,
               "validation": validation_metrics,
               "later": {"raw": scores(yl, fl), "category_scaled": scores(yl, fls),
                         "categories": {str(c): scores(yl[cats == c], fls[cats == c]) for c in np.unique(cats)},
                         "paired_item_intervals": uncertainty},
               "interpretation": "one fixed item-cross-fitted q feature control; no forward-time stacking or universal superiority claim"}
    write_json(out / "RESULTS.json", results)
    write_json(out / "ARTIFACT_HASHES.json", {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "ARTIFACT_HASHES.json"})
    print("COMPLETED", results["later"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing outputs are never overwritten")
    main(parser.parse_args())
