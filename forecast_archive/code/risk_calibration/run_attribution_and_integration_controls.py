"""Post-hoc attribution, feature-integration, and oracle controls for CENSORCAST-RC.

All M5 evaluation blocks were previously consumed.  This script is therefore a
diagnostic audit, not a new model-selection or confirmation experiment.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from run_aft_pilot import dump, now, scores, sha, weighted_scale
from run_hierarchy_pilot import HierarchyFeatures, make_x, predict
from evaluate_risk_calibration import apply_policy, fit_policy, predict_hit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accuracy_revision_20260907" / "code" / "m5"))
from legacy_features.features import DesignPanel, FeatureBuilder


SEED = 20260906
NBINS = 8
SHRINK = 0.75
TRAIN_ROWS = 1_200_000
PARAMS = {
    "objective": "reg:absoluteerror",
    "eta": 0.03,
    "max_depth": 8,
    "min_child_weight": 100,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 2.0,
    "tree_method": "hist",
    "max_bin": 255,
    "nthread": 8,
    "seed": SEED,
}


def feature_scores(builder: FeatureBuilder, days: np.ndarray, batch_days: int = 7):
    n = builder.panel.n_series
    keys = ["historical_hit_rate_28", "origin_fill_ratio", "capacity_pressure"]
    out = {key: np.empty((n, len(days)), np.float32) for key in keys}
    for start in range(0, len(days), batch_days):
        ds = days[start:start + batch_days]
        series = np.repeat(np.arange(n, dtype=np.int32), len(ds))
        target = np.tile(ds, n)
        x = builder.make_features(series, target, include_censor=True)
        out["historical_hit_rate_28"][:, start:start + len(ds)] = x[:, 29].reshape(n, -1)
        out["origin_fill_ratio"][:, start:start + len(ds)] = x[:, 31].reshape(n, -1)
        out["capacity_pressure"][:, start:start + len(ds)] = -x[:, 32].reshape(n, -1)
    return out


def predict_augmented(
    builder: FeatureBuilder,
    hierarchy: HierarchyFeatures,
    hit: lgb.Booster,
    model: xgb.Booster,
    days: np.ndarray,
    add_q: bool,
    batch_days: int = 7,
):
    n = builder.panel.n_series
    out = np.empty((n, len(days)), np.float32)
    for start in range(0, len(days), batch_days):
        ds = days[start:start + batch_days]
        series = np.repeat(np.arange(n, dtype=np.int32), len(ds))
        target = np.tile(ds, n)
        common = builder.make_features(series, target, include_censor=True)
        x = np.column_stack([common, hierarchy.make(series, target)])
        if add_q:
            q = np.clip(hit.predict(common, num_threads=8), 0, 1)
            x = np.column_stack([x, q])
        out[:, start:start + len(ds)] = model.predict(xgb.DMatrix(x)).reshape(n, -1)
    return np.maximum(out - 1.0, 0.0)


def category_scale(y: np.ndarray, f: np.ndarray, cats: np.ndarray):
    values = {str(c): weighted_scale(y[cats == c], f[cats == c]) for c in np.unique(cats)}
    multiplier = np.asarray([values[str(c)] for c in cats])[:, None]
    return values, multiplier * f


def item_interval(y: np.ndarray, reference: np.ndarray, candidate: np.ndarray, item_ids: np.ndarray):
    items, inverse = np.unique(item_ids, return_inverse=True)
    mass = np.bincount(inverse, weights=y.sum(axis=1))
    e0 = np.bincount(inverse, weights=np.abs(y - reference).sum(axis=1))
    e1 = np.bincount(inverse, weights=np.abs(y - candidate).sum(axis=1))
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(items), np.full(len(items), 1 / len(items)), size=4000)
    samples = (weights @ (e1 - e0)) / (weights @ mass)
    return {
        "estimate": float((e1 - e0).sum() / mass.sum()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def cell_rows(
    y: np.ndarray,
    base: np.ndarray,
    risk: np.ndarray,
    cats: np.ndarray,
    capacity: np.ndarray,
    policy: dict,
    population: str,
):
    edges = np.asarray(policy["edges"])
    groups = np.searchsorted(edges, risk, side="right")
    total_rows = y.size
    total_mass = y.sum()
    rows = []
    for cat in np.unique(cats):
        for group in range(len(edges) + 1):
            mask = (cats == cat)[:, None] & (groups == group)
            n = int(mask.sum())
            demand = float(y[mask].sum())
            strict = (y > capacity) & mask
            pred = base[mask] * policy["scales"][f"{cat}|{group}"]
            rows.append({
                "population": population,
                "category": str(cat),
                "bin": int(group + 1),
                "risk_lower": None if group == 0 else float(edges[group - 1]),
                "risk_upper": None if group == len(edges) else float(edges[group]),
                "multiplier": float(policy["scales"][f"{cat}|{group}"]),
                "rows": n,
                "row_share": float(n / total_rows),
                "demand_mass": demand,
                "demand_share": float(demand / total_mass),
                "mean_risk": float(risk[mask].mean()) if n else None,
                "strict_row_share": float(strict.sum() / n) if n else None,
                "base_wape": float(np.abs(y[mask] - base[mask]).sum() / demand) if demand else None,
                "calibrated_wape": float(np.abs(y[mask] - pred).sum() / demand) if demand else None,
            })
    return rows


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "FROZEN_POST_HOC_DIAGNOSTIC_BEFORE_FITS",
        "created_utc": now(),
        "evaluation_status": "all M5 blocks previously consumed; no confirmatory claim",
        "primary_questions": [
            "correct Table 2 on the identical hierarchy base",
            "display every category-by-risk calibration cell",
            "compare post-hoc calibration against q as a base-model feature",
            "measure a truth-label oracle ceiling",
        ],
        "matched_score_control": {"bins": NBINS, "shrinkage": SHRINK, "selection": "none"},
        "q_feature_control": "same hierarchy features and XGBoost configuration; learned q appended",
        "oracle_control": "same hierarchy features and XGBoost configuration; Y replaces S only as fit label",
        "training_days": [365, 1313],
        "training_rows": TRAIN_ROWS,
        "validation_days": [1314, 1433],
        "seed": SEED,
        "xgboost": PARAMS,
        "input_sha256": {
            "design": sha(args.data / "design_outcomes_v0_5.npz"),
            "calendar": sha(args.data / "calendar.csv"),
            "prices": sha(args.data / "sell_prices.csv"),
            "hierarchy_base": sha(args.base_model),
            "hit_model": sha(args.hit_model),
            "frozen_policy": sha(args.policy),
            "pretrained_q_control": sha(args.pretrained_dir / "q_as_feature_l1.ubj") if args.pretrained_dir else None,
            "pretrained_oracle_control": sha(args.pretrained_dir / "truth_label_oracle_l1.ubj") if args.pretrained_dir else None,
        },
        "script_sha256": sha(__file__),
    }
    dump(out / "PROTOCOL.json", protocol)

    panel = DesignPanel.load(
        args.data / "design_outcomes_v0_5.npz",
        args.data / "calendar.csv",
        args.data / "sell_prices.csv",
    )
    panel.censored = (panel.observed >= panel.capacity).astype(np.uint8)
    builder = FeatureBuilder(panel)
    hierarchy = HierarchyFeatures(panel)
    cats = panel.metadata["cat_id"]
    hit = lgb.Booster(model_file=str(args.hit_model))
    base_model = xgb.Booster()
    base_model.load_model(args.base_model)
    policy_blob = json.loads(args.policy.read_text())
    policy = policy_blob.get("risk_policy", policy_blob.get("final_policy"))

    valid_days = np.arange(1314, 1434, dtype=np.int32)
    valid_days = valid_days[valid_days - builder.horizon_for_day(valid_days) >= 1313]
    later_days = np.load(args.cache / "shadow_aligned.npz")["target_days"]
    y_valid = panel.truth[:, valid_days - 1].astype(float)
    y_later = panel.truth[:, later_days - 1].astype(float)
    base_valid = predict(builder, hierarchy, base_model, valid_days)
    base_later = predict(builder, hierarchy, base_model, later_days)
    q_valid = predict_hit(builder, hit, valid_days)
    q_later = predict_hit(builder, hit, later_days)

    # Corrected matched-score attribution: identical hierarchy base and family.
    valid_scores = feature_scores(builder, valid_days)
    later_scores = feature_scores(builder, later_days)
    valid_scores["learned_hit_risk"] = q_valid
    valid_scores["base_forecast"] = base_valid
    later_scores["learned_hit_risk"] = q_later
    later_scores["base_forecast"] = base_later
    score_rows = {}
    for name in valid_scores:
        fitted = fit_policy(y_valid, base_valid, valid_scores[name], cats, NBINS, SHRINK)
        forecast = apply_policy(base_later, later_scores[name], cats, fitted)
        score_rows[name] = {
            "later": scores(y_later, forecast),
            "policy": fitted,
        }
    base_scales, _ = category_scale(y_valid, base_valid, cats)
    base_multiplier = np.asarray([base_scales[str(c)] for c in cats])[:, None]
    scaled_base_later = base_multiplier * base_later

    # Exact 24-cell audit for the already-frozen proposed policy.
    cells = []
    cells.extend(cell_rows(
        y_valid, base_valid, q_valid, cats,
        panel.capacity[:, valid_days - 1], policy, "point_validation"
    ))
    cells.extend(cell_rows(
        y_later, base_later, q_later, cats,
        panel.capacity[:, later_days - 1], policy, "later"
    ))

    started = time.time()
    if args.pretrained_dir:
        print("Reusing completed fits from the interrupted post-fit scoring run", flush=True)
        q_model = xgb.Booster()
        q_model.load_model(args.pretrained_dir / "q_as_feature_l1.ubj")
        oracle_model = xgb.Booster()
        oracle_model.load_model(args.pretrained_dir / "truth_label_oracle_l1.ubj")
        q_model.save_model(out / "q_as_feature_l1.ubj")
        oracle_model.save_model(out / "truth_label_oracle_l1.ubj")
    else:
        # Feature-integration control and privileged-label oracle.
        series, train_days = builder.sample_pairs(365, 1313, TRAIN_ROWS, SEED)
        common = builder.make_features(series, train_days, include_censor=True)
        hierarchy_x = hierarchy.make(series, train_days)
        x = np.column_stack([common, hierarchy_x])
        observed = panel.observed[series, train_days - 1].astype(float)
        truth = panel.truth[series, train_days - 1].astype(float)
        q_train = np.clip(hit.predict(common, num_threads=8), 0, 1)
        q_matrix = xgb.DMatrix(np.column_stack([x, q_train]), label=observed + 1.0)
        oracle_matrix = xgb.DMatrix(x, label=truth + 1.0)
        del common, hierarchy_x, x, observed, truth, q_train
        gc.collect()
        print("Fitting q-as-feature L1 control", flush=True)
        q_model = xgb.train(PARAMS, q_matrix, num_boost_round=750)
        q_model.save_model(out / "q_as_feature_l1.ubj")
        del q_matrix
        gc.collect()
        print("Fitting truth-label L1 oracle", flush=True)
        oracle_model = xgb.train(PARAMS, oracle_matrix, num_boost_round=750)
        oracle_model.save_model(out / "truth_label_oracle_l1.ubj")
        del oracle_matrix
        gc.collect()

    q_valid_forecast = predict_augmented(builder, hierarchy, hit, q_model, valid_days, add_q=True)
    q_later_forecast = predict_augmented(builder, hierarchy, hit, q_model, later_days, add_q=True)
    oracle_valid = predict_augmented(builder, hierarchy, hit, oracle_model, valid_days, add_q=False)
    oracle_later = predict_augmented(builder, hierarchy, hit, oracle_model, later_days, add_q=False)
    q_scales, _ = category_scale(y_valid, q_valid_forecast, cats)
    q_multiplier = np.asarray([q_scales[str(c)] for c in cats])[:, None]
    q_later_scaled = q_multiplier * q_later_forecast
    oracle_scales, _ = category_scale(y_valid, oracle_valid, cats)
    oracle_multiplier = np.asarray([oracle_scales[str(c)] for c in cats])[:, None]
    oracle_later_scaled = oracle_multiplier * oracle_later
    proposed_later = apply_policy(base_later, q_later, cats, policy)

    results = {
        "status": "COMPLETED",
        "created_utc": now(),
        "elapsed_fit_seconds": time.time() - started,
        "table2_root_cause": {
            "finding": "the published matched-score table used a pre-hierarchy base despite its caption",
            "old_model_sha256": "c9426674a1a77ab0380d95ba75500504543b3d41d3efc586f61cd4992a734af3",
            "correct_hierarchy_model_sha256": sha(args.base_model),
        },
        "corrected_score_ablation": {
            "base_category_scales": base_scales,
            "base": scores(y_later, scaled_base_later),
            "scores": score_rows,
        },
        "cells": cells,
        "integration_and_oracle": {
            "hierarchy_base": scores(y_later, scaled_base_later),
            "proposed_posthoc_calibration": scores(y_later, proposed_later),
            "q_as_feature": scores(y_later, q_later_scaled),
            "q_as_feature_scales": q_scales,
            "q_as_feature_delta_vs_base": item_interval(
                y_later, scaled_base_later, q_later_scaled, panel.metadata["item_id"]
            ),
            "truth_label_oracle": scores(y_later, oracle_later_scaled),
            "truth_label_oracle_scales": oracle_scales,
            "truth_label_oracle_delta_vs_base": item_interval(
                y_later, scaled_base_later, oracle_later_scaled, panel.metadata["item_id"]
            ),
        },
    }
    dump(out / "RESULTS.json", results)
    np.savez_compressed(
        out / "LATER_PREDICTIONS.npz",
        days=later_days,
        base=scaled_base_later,
        proposed=proposed_later,
        q_as_feature=q_later_scaled,
        truth_label_oracle=oracle_later_scaled,
        learned_hit_risk=q_later,
    )
    print(json.dumps(results["corrected_score_ablation"], indent=2), flush=True)
    print(json.dumps(results["integration_and_oracle"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--hit-model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
