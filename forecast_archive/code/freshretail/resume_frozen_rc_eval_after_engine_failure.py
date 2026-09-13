"""Recover the frozen FreshRetailNet score after a pre-decoding engine failure.

The first run fitted and saved both models and FINAL_POLICY.json before it
downloaded eval.parquet. Pandas then stopped before decoding any row because no
Parquet engine was installed. This recovery verifies and reuses those frozen
objects; it performs no model fit or policy selection after decoding eval.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from censorcast.data import infer_split_dates, make_supervised, normalize_frame, split_supervised
from censorcast.models import SmoothedTargetEncoder
from run_frozen_rc_eval import (
    EXPECTED_EVAL_SHA256,
    SEED,
    apply_category_scale,
    apply_policy,
    cluster_interval,
    dump,
    fit_policy,
    grouped_metrics,
    metrics,
    now,
    sha,
)


TRAIN_SHA = "5dc9bcca4b42f25784895ba6327a06412860571a2c63a1fd6e285d521771bed3"
BASE_SHA = "b6a53c1d9e5181a14e7705f37b9c5cd28c1b6b3115532196f9807aec15ba3a4d"
RISK_SHA = "130b3ccb14378b867896c781836012d51697ff995ecffe310710b5f1b3f64e81"
POLICY_SHA = "7f483f1946aa6a3b1a404f3d563dd3644686e1109729623fa36a88894426db74"


def main(args: argparse.Namespace) -> None:
    output = args.output
    if (output / "RESULTS.json").exists():
        raise RuntimeError("Results already exist; refusing a second score")
    required = [output / "base_l1.ubj", output / "stockout_risk.ubj", output / "FINAL_POLICY.json"]
    if not all(path.exists() for path in required):
        raise RuntimeError("The pre-open frozen objects are incomplete")
    expected = {
        output / "base_l1.ubj": BASE_SHA,
        output / "stockout_risk.ubj": RISK_SHA,
        output / "FINAL_POLICY.json": POLICY_SHA,
    }
    for path, expected_hash in expected.items():
        if sha(path) != expected_hash:
            raise RuntimeError(f"Frozen object hash mismatch: {path.name}")
    if sha(args.train) != TRAIN_SHA or sha(args.eval) != EXPECTED_EVAL_SHA256:
        raise RuntimeError("Input hash mismatch")

    recovery_started = now()
    train = normalize_frame(pd.read_pickle(args.train))
    train["_source_split"] = "train"
    series_ids = set(train["series_id"].astype(str).unique())

    # Reconstruct only the deterministic train-fitted encoder and development
    # summaries. Models and the final calibration policy are loaded, not fit.
    train_dates = infer_split_dates(train)
    train_rows, numeric, categorical = make_supervised(
        train,
        horizons=range(1, 8),
        max_train_rows=600000,
        seed=SEED,
        development_holdout_days=35,
    )
    train_splits = split_supervised(train_rows, train_dates)
    fit = train_splits["fit"]
    y_fit = fit["sale_amount"].to_numpy(float)
    available_fit = fit["is_censored"].to_numpy(int) == 0
    encoder = SmoothedTargetEncoder(categorical=categorical, numeric=numeric, smoothing=30.0)
    encoder.fit(fit, y_fit, uncensored=available_fit)
    base = xgb.Booster()
    base.load_model(output / "base_l1.ubj")
    risk = xgb.Booster()
    risk.load_model(output / "stockout_risk.ubj")
    final_policy = json.loads((output / "FINAL_POLICY.json").read_text())

    predicted: dict[str, dict] = {}
    ordered = ["selection", "risk_train", "calibration_a", "calibration_b", "shadow"]
    for name in ordered:
        frame = train_splits[name].reset_index(drop=True)
        x = encoder.transform(frame)
        predicted[name] = {
            "frame": frame,
            "y": frame["sale_amount"].to_numpy(float),
            "available": frame["is_censored"].to_numpy(int) == 0,
            "base_raw": np.clip(base.predict(xgb.DMatrix(x)) - 1.0, 0.0, None),
            "q": np.clip(risk.predict(xgb.DMatrix(x)), 0.0, 1.0),
            "category": frame["management_group_id"].astype(str).to_numpy(),
        }
    dev_fit_names = ["selection", "risk_train"]
    dev_policy = fit_policy(
        np.concatenate([predicted[name]["y"] for name in dev_fit_names]),
        np.concatenate([predicted[name]["base_raw"] for name in dev_fit_names]),
        np.concatenate([predicted[name]["q"] for name in dev_fit_names]),
        np.concatenate([predicted[name]["category"] for name in dev_fit_names]),
        np.concatenate([predicted[name]["available"] for name in dev_fit_names]),
    )
    development = []
    for name in ["calibration_a", "calibration_b", "shadow"]:
        item = predicted[name]
        base_scaled = apply_category_scale(item["base_raw"], item["category"], dev_policy)
        rc = apply_policy(item["base_raw"], item["q"], item["category"], dev_policy)
        development.append({
            "block": name,
            "base": metrics(item["y"], base_scaled, item["available"]),
            "risk_conditioned": metrics(item["y"], rc, item["available"]),
        })
    del train_rows, train_splits, predicted

    # First successful row-level decode. Everything that can learn from labels
    # is already frozen above.
    eval_decoded_utc = now()
    raw_eval = pd.read_parquet(args.eval)
    raw_eval["_source_split"] = "eval"
    evaluation = normalize_frame(raw_eval)
    evaluation = evaluation[evaluation["series_id"].astype(str).isin(series_ids)].copy()
    if evaluation["series_id"].nunique() != 2000 or len(evaluation) != 14000:
        raise RuntimeError("Official eval population differs from the frozen 2,000-series cohort")
    combined = pd.concat([train, evaluation], ignore_index=True)
    dates = infer_split_dates(combined)
    rows, check_numeric, check_categorical = make_supervised(
        combined,
        horizons=range(1, 8),
        max_train_rows=7,
        seed=SEED,
        development_holdout_days=35,
    )
    if check_numeric != numeric or check_categorical != categorical:
        raise RuntimeError("Feature schema changed")
    frame = split_supervised(rows, dates)["test"].reset_index(drop=True)
    x = encoder.transform(frame)
    y = frame["sale_amount"].to_numpy(float)
    available = frame["is_censored"].to_numpy(int) == 0
    category = frame["management_group_id"].astype(str).to_numpy()
    base_raw = np.clip(base.predict(xgb.DMatrix(x)) - 1.0, 0.0, None)
    q = np.clip(risk.predict(xgb.DMatrix(x)), 0.0, 1.0)
    base_scaled = apply_category_scale(base_raw, category, final_policy)
    rc = apply_policy(base_raw, q, category, final_policy)
    all_rows = np.ones(len(y), dtype=bool)
    result = {
        "status": "COMPLETED_NO_POST_EVAL_SELECTION",
        "recovery_started_utc": recovery_started,
        "eval_first_successful_decode_utc": eval_decoded_utc,
        "completed_utc": now(),
        "protocol_sha256": sha(args.protocol),
        "train_sha256": sha(args.train),
        "eval_sha256": sha(args.eval),
        "eval_download_count": 1,
        "eval_successful_decode_count": 1,
        "fit_after_eval_successful_decode": 0,
        "selection_after_eval_successful_decode": 0,
        "interrupted_attempt": {
            "stage": "pd.read_parquet before any row was decoded",
            "error": "missing optional dependency pyarrow or fastparquet",
            "metrics_computed": 0,
            "models_reused_by_sha256": {"base": BASE_SHA, "risk": RISK_SHA, "policy": POLICY_SHA},
        },
        "population": {
            "series": int(frame["series_id"].nunique()),
            "rows": int(len(frame)),
            "fully_available_rows": int(available.sum()),
            "stockout_rows": int((~available).sum()),
            "stockout_row_fraction": float((~available).mean()),
            "recorded_sales_mass_on_stockout_rows": float(y[~available].sum() / y.sum()),
        },
        "development_transfer": development,
        "final_policy": final_policy,
        "primary_fully_available": {
            "base": metrics(y, base_scaled, available),
            "risk_conditioned": metrics(y, rc, available),
            "paired_series_bootstrap": cluster_interval(frame, y, base_scaled, rc, available),
        },
        "secondary_all_recorded_sales": {
            "base": metrics(y, base_scaled, all_rows),
            "risk_conditioned": metrics(y, rc, all_rows),
        },
        "by_management_group": grouped_metrics(frame, y, base_scaled, rc, available, "management_group_id"),
        "by_horizon": grouped_metrics(frame, y, base_scaled, rc, available, "horizon"),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
    }
    dump(output / "RESULTS.json", result)
    prediction = frame[["series_id", "dt", "horizon", "management_group_id", "sale_amount", "is_censored"]].copy()
    prediction["base"] = base_scaled
    prediction["risk_conditioned"] = rc
    prediction["predicted_stockout_risk"] = q
    prediction.to_csv(output / "EVAL_PREDICTIONS.csv.gz", index=False)
    dump(output / "RUN_RECEIPT.json", {
        "status": result["status"],
        "protocol_sha256": sha(args.protocol),
        "original_runner_sha256": "7607bd68b784e8e16a100b712cffd0cb91c8b7b3143dea86ba971ce28bcf3ea6",
        "recovery_runner_sha256": sha(Path(__file__)),
        "base_sha256": BASE_SHA,
        "risk_sha256": RISK_SHA,
        "policy_sha256": POLICY_SHA,
        "results_sha256": sha(output / "RESULTS.json"),
        "predictions_sha256": sha(output / "EVAL_PREDICTIONS.csv.gz"),
        "eval_sha256": sha(args.eval),
        "successful_eval_decodes": 1,
        "post_decode_fit_calls": 0,
        "post_decode_selection_calls": 0,
    })
    print(json.dumps(result["primary_fully_available"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
