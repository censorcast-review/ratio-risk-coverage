"""Fixed-condition refit of the two M5 controls whose model bytes were omitted.

This is a NEW, exploratory reproduction on previously used M5 data. It does not
restore or rewrite the original freeze, and it does not turn M5 into unused data.
The archived table contained metrics and model hashes but no q-as-feature/oracle
model bytes. Both original configurations are run without outcome-based choice.
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

from replay_m5 import DesignPanel, FeatureBuilder, HierarchyFeatures, load_json, scores, sha256, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


class Progress(xgb.callback.TrainingCallback):
    def __init__(self, name, rounds):
        self.name, self.rounds, self.started = name, rounds, time.monotonic()

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % 100 == 0 or epoch + 1 == self.rounds:
            elapsed = time.monotonic() - self.started
            remaining = elapsed / (epoch + 1) * (self.rounds - epoch - 1)
            print(f"{self.name} round={epoch + 1}/{self.rounds} elapsed={elapsed:.1f}s estimated_remaining={remaining:.1f}s", flush=True)
        return False


def weighted_scale(y, f):
    """Independent exact L1 minimizer; no unknown historical helper is imported."""
    y, f = np.asarray(y, float).ravel(), np.asarray(f, float).ravel()
    positive = f > 0
    if not positive.any():
        return 1.0
    ratios, weights = y[positive] / f[positive], f[positive]
    order = np.argsort(ratios, kind="stable")
    return float(ratios[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() * 0.5)])


def predict(builder, hierarchy, hit, model, days, add_q, threads):
    n = builder.panel.n_series
    prediction = np.empty((n, len(days)), np.float32)
    for start in range(0, len(days), 7):
        ds = days[start:start + 7]
        series = np.repeat(np.arange(n, dtype=np.int32), len(ds))
        targets = np.tile(ds, n)
        common = builder.make_features(series, targets, include_censor=True)
        x = np.column_stack([common, hierarchy.make(series, targets)])
        if add_q:
            q = np.clip(hit.predict(common, num_threads=threads), 0, 1)
            x = np.column_stack([x, q])
        prediction[:, start:start + len(ds)] = model.predict(xgb.DMatrix(x)).reshape(n, -1)
    return np.maximum(prediction - 1.0, 0.0)


def main(args):
    started = time.monotonic()
    root, out, data = args.package.resolve(), args.output.resolve(), args.input_dir.resolve()
    if root == out or root in out.parents:
        raise ValueError("Output must be outside the immutable input package")
    out.mkdir(parents=True, exist_ok=False)
    source = root / "code/risk_calibration/run_attribution_and_integration_controls.py"
    old_protocol_path = root / "evidence/risk_calibration/attribution_controls/PROTOCOL.json"
    old_results_path = root / "evidence/risk_calibration/attribution_controls/RESULTS.json"
    old_protocol, old_results = load_json(old_protocol_path), load_json(old_results_path)
    seed, rounds, train_rows = 20260906, 750, 1_200_000
    params = dict(old_protocol["xgboost"])
    # Preserve the archived exact nthread=8 setting as well as all model settings.
    threads = int(params["nthread"])
    hit_path = root / f"evidence/revision/m5_observable/seed_{seed}/hit.txt"
    protocol = {
        "status": "FIXED_REPRODUCTION_PROTOCOL_RECORDED_BEFORE_NEW_FITS", "created_utc": now(),
        "purpose": "repair omitted model artifacts through a new fixed-condition reproduction",
        "evidence_status": "exploratory reproduction; all M5 evaluation blocks were used previously",
        "original_artifact_gap": "prior package retained q/oracle metrics and hashes but omitted their .ubj files; reason for omission is not documented",
        "not_original_freeze": "this new protocol does not replace the original protocol or certify the original run",
        "controls": ["q_as_feature_l1", "truth_label_oracle_l1"], "outcome_dependent_selection": False,
        "seed": seed, "training_days": [365, 1313], "training_rows": train_rows,
        "validation_days": [1314, 1433], "evaluation_block": "shadow (Later)",
        "model_parameters": params, "rounds": rounds,
        "scale_rule": "independent exact nonnegative weighted-median L1 category scaling on validation; also report original frozen scales",
        "helper_provenance": "observable feature implementation is bundled; hierarchy formulas transcribed; unknown original scale helper independently reconstructed and compared",
        "code_sha256": sha256(__file__), "replay_module_sha256": sha256(Path(__file__).with_name("replay_m5.py")),
        "original_source_sha256": sha256(source), "original_protocol_sha256": sha256(old_protocol_path),
        "original_results_sha256": sha256(old_results_path),
        "inputs_sha256": {name: sha256(data / "data" / name) for name in ["design_outcomes_v0_5.npz", "calendar.csv", "sell_prices.csv"]},
        "hit_model_sha256": sha256(hit_path),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__, "lightgbm": lgb.__version__},
    }
    for key, filename in [("design", "design_outcomes_v0_5.npz"), ("calendar", "calendar.csv"), ("prices", "sell_prices.csv")]:
        if protocol["inputs_sha256"][filename] != old_protocol["input_sha256"][key]:
            raise ValueError(f"Input hash mismatch: {key}")
    write_json(out / "NEW_REFIT_PROTOCOL.json", protocol)
    print("New protocol recorded before fitting; loading observed training panel", flush=True)
    panel = DesignPanel.load(data / "data/design_outcomes_v0_5.npz", data / "data/calendar.csv", data / "data/sell_prices.csv")
    panel.censored = (panel.observed >= panel.capacity).astype(np.uint8)
    builder, hierarchy = FeatureBuilder(panel), HierarchyFeatures(panel)
    hit = lgb.Booster(model_file=str(hit_path))
    series, train_days = builder.sample_pairs(365, 1313, train_rows, seed)
    common = builder.make_features(series, train_days, include_censor=True)
    x = np.column_stack([common, hierarchy.make(series, train_days)])
    q_train = np.clip(hit.predict(common, num_threads=threads), 0, 1)
    observed = panel.observed[series, train_days - 1].astype(float)
    truth = panel.truth[series, train_days - 1].astype(float)
    np.savez_compressed(out / "TRAINING_SAMPLE.npz", series=series, days=train_days)
    q_matrix = xgb.DMatrix(np.column_stack([x, q_train]), label=observed + 1.0)
    oracle_matrix = xgb.DMatrix(x, label=truth + 1.0)
    del x, common, q_train, observed, truth, series, train_days
    gc.collect()
    fit_receipts = []
    for name, matrix in [("q_as_feature_l1", q_matrix), ("truth_label_oracle_l1", oracle_matrix)]:
        t0 = time.monotonic()
        fit_start = now()
        write_json(out / f"{name}_FIT_START.json", {"started_utc": fit_start, "new_protocol_sha256": sha256(out / "NEW_REFIT_PROTOCOL.json"), "rounds": rounds})
        print(f"Fitting {name}: fixed {rounds} rounds", flush=True)
        model = xgb.train(params, matrix, num_boost_round=rounds, callbacks=[Progress(name, rounds)])
        model_path = out / f"{name}.ubj"
        model.save_model(model_path)
        old_hash_key = "pretrained_q_control" if name == "q_as_feature_l1" else "pretrained_oracle_control"
        fit_receipts.append({"name": name, "started_utc": fit_start, "finished_utc": now(),
                             "elapsed_seconds": time.monotonic() - t0, "model_sha256": sha256(model_path),
                             "original_declared_model_sha256": old_protocol["input_sha256"][old_hash_key],
                             "model_bytes_match_original_declared_hash": sha256(model_path) == old_protocol["input_sha256"][old_hash_key]})
        del model
        write_json(out / "FIT_RECEIPTS.json", fit_receipts)
    del q_matrix, oracle_matrix, matrix
    gc.collect()
    valid_days = np.load(root / "evidence/risk_calibration/primary_base/validation.npz")["days"]
    later_days = np.load(data / "cache/shadow_aligned.npz")["target_days"]
    yv, yl = panel.truth[:, valid_days - 1].astype(float), panel.truth[:, later_days - 1].astype(float)
    cats = panel.metadata["cat_id"]
    items, inv = np.unique(panel.metadata["item_id"], return_inverse=True)
    reference_stats = np.load(root / "evidence/risk_calibration/primary_policy/shadow_item_stats.npz")
    if not np.array_equal(items, reference_stats["items"]):
        raise ValueError("Item identity mismatch")
    controls = {}
    original = old_results["integration_and_oracle"]
    for name, old_key in [("q_as_feature_l1", "q_as_feature"), ("truth_label_oracle_l1", "truth_label_oracle")]:
        model = xgb.Booster(params={"nthread": threads})
        model.load_model(out / f"{name}.ubj")
        model.set_param({"nthread": threads})
        fv = predict(builder, hierarchy, hit, model, valid_days, name == "q_as_feature_l1", threads)
        scales = {str(c): weighted_scale(yv[cats == c], fv[cats == c]) for c in np.unique(cats)}
        old_scales = original[f"{old_key}_scales"]
        scale_difference = {c: scales[c] - old_scales[c] for c in scales}
        fl = predict(builder, hierarchy, hit, model, later_days, name == "q_as_feature_l1", threads)
        f = np.asarray([scales[str(c)] for c in cats])[:, None] * fl
        frozen_f = np.asarray([old_scales[str(c)] for c in cats])[:, None] * fl
        metric = scores(yl, f)
        item_error = np.bincount(inv, weights=np.abs(yl - f).sum(axis=1))
        mass = np.bincount(inv, weights=yl.sum(axis=1))
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(len(items), np.full(len(items), 1 / len(items)), size=4000)
        bootstrap = (weights @ (item_error - reference_stats["base_error"])) / (weights @ mass)
        delta = {"estimate": float((item_error - reference_stats["base_error"]).sum() / mass.sum()),
                 "ci95": np.quantile(bootstrap, [.025, .975]).tolist()}
        np.savez_compressed(out / f"{name}_item_stats.npz", items=items, mass=mass,
                            base_error=reference_stats["base_error"], candidate_error=item_error)
        np.savez_compressed(out / f"{name}_predictions.npz", valid_days=valid_days, validation_raw=fv,
                            later_days=later_days, later_raw=fl, later_scaled=f)
        controls[name] = {"metrics": metric, "original_reported_metrics": original[old_key],
                          "metric_differences_from_original": {k: metric[k] - original[old_key][k] for k in metric},
                          "validation_category_scales": scales, "scale_differences_from_original": scale_difference,
                          "metrics_with_original_frozen_scales": scores(yl, frozen_f), "paired_delta_vs_base": delta}
        print(name, controls[name], flush=True)
        del fv, fl, f, frozen_f, model, weights
    report = {"status": "COMPLETED_NEW_FIXED_CONDITION_REPRODUCTION", "finished_utc": now(),
              "elapsed_seconds": time.monotonic() - started, "training_calls": 2,
              "outcome_dependent_model_selection_calls": 0, "new_holdout_accesses": 0,
              "protocol_sha256": sha256(out / "NEW_REFIT_PROTOCOL.json"), "fits": fit_receipts, "controls": controls,
              "interpretation": "new exploratory reproduction of already reported comparisons; original frozen evidence unchanged"}
    write_json(out / "NEW_REFIT_RESULTS.json", report)
    print("Both fixed controls finished and saved with model bytes, row predictions, and item sufficient statistics", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--input-dir", type=Path, required=True, help="extracted_inputs directory produced by replay_m5.py")
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
