"""One-shot FreshRetailNet evaluation for the frozen CENSORCAST-RC protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from censorcast.data import (
    infer_split_dates,
    make_supervised,
    normalize_frame,
    split_supervised,
)
from censorcast.models import SmoothedTargetEncoder


SEED = 20260907
EVAL_URL = (
    "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/"
    "08c1fab7f9257bc73679d415d65d644165d351d4/data/eval.parquet"
)
EXPECTED_EVAL_SHA256 = "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def weighted_scale(y: np.ndarray, f: np.ndarray, default: float = 1.0) -> float:
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    mask = np.isfinite(y) & np.isfinite(f) & (f > 1e-10)
    if not np.any(mask):
        return float(default)
    ratios = np.clip(y[mask] / f[mask], 0.0, 50.0)
    weights = f[mask]
    order = np.argsort(ratios, kind="mergesort")
    ratios = ratios[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return float(ratios[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def fit_policy(
    y: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    category: np.ndarray,
    eligible: np.ndarray,
    bins: int = 8,
    shrinkage: float = 0.75,
) -> dict:
    mask = eligible & np.isfinite(y) & np.isfinite(f) & np.isfinite(q)
    edges = np.unique(np.quantile(q[mask], np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    group = np.searchsorted(edges, q, side="right")
    category_scales: dict[str, float] = {}
    scales: dict[str, float] = {}
    for cat in np.unique(category):
        cm = mask & (category == cat)
        cs = weighted_scale(y[cm], f[cm])
        category_scales[str(cat)] = cs
        for cell in range(len(edges) + 1):
            gm = cm & (group == cell)
            gs = weighted_scale(y[gm], f[gm], default=cs)
            value = np.exp(
                (1.0 - shrinkage) * np.log(max(cs, 1e-8))
                + shrinkage * np.log(max(gs, 1e-8))
            )
            scales[f"{cat}|{cell}"] = float(value)
    return {
        "bins": int(len(edges) + 1),
        "edges": edges.tolist(),
        "shrinkage": float(shrinkage),
        "category_scales": category_scales,
        "scales": scales,
    }


def apply_policy(f: np.ndarray, q: np.ndarray, category: np.ndarray, policy: dict) -> np.ndarray:
    group = np.searchsorted(np.asarray(policy["edges"]), q, side="right")
    multiplier = np.empty(len(f), dtype=float)
    for i, (cat, cell) in enumerate(zip(category, group, strict=True)):
        multiplier[i] = policy["scales"][f"{cat}|{cell}"]
    return np.clip(f * multiplier, 0.0, None)


def apply_category_scale(f: np.ndarray, category: np.ndarray, policy: dict) -> np.ndarray:
    multiplier = np.asarray([policy["category_scales"][str(cat)] for cat in category])
    return np.clip(f * multiplier, 0.0, None)


def metrics(y: np.ndarray, f: np.ndarray, eligible: np.ndarray) -> dict:
    mask = eligible & np.isfinite(y) & np.isfinite(f)
    yy = y[mask]
    ff = f[mask]
    error = ff - yy
    mass = max(float(yy.sum()), 1e-12)
    return {
        "rows": int(mask.sum()),
        "demand_mass": float(yy.sum()),
        "wape": float(np.abs(error).sum() / mass),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "signed_percentage_bias": float(error.sum() / mass),
    }


def grouped_metrics(frame: pd.DataFrame, y: np.ndarray, base: np.ndarray, rc: np.ndarray, eligible: np.ndarray, key: str) -> list[dict]:
    rows = []
    values = frame[key].to_numpy()
    for value in np.unique(values):
        mask = eligible & (values == value)
        if not np.any(mask):
            continue
        rows.append({
            key: str(value),
            "base": metrics(y, base, mask),
            "risk_conditioned": metrics(y, rc, mask),
        })
    return rows


def cluster_interval(frame: pd.DataFrame, y: np.ndarray, base: np.ndarray, rc: np.ndarray, eligible: np.ndarray) -> dict:
    ids, inverse = np.unique(frame["series_id"].astype(str).to_numpy(), return_inverse=True)
    mass = np.bincount(inverse, weights=np.where(eligible, y, 0.0))
    e0 = np.bincount(inverse, weights=np.where(eligible, np.abs(y - base), 0.0))
    e1 = np.bincount(inverse, weights=np.where(eligible, np.abs(y - rc), 0.0))
    estimate = float((e1.sum() - e0.sum()) / mass.sum())
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=4000)
    sample = (draws @ (e1 - e0)) / np.maximum(draws @ mass, 1e-12)
    return {
        "clusters": int(len(ids)),
        "draws": 4000,
        "estimate": estimate,
        "ci95": np.quantile(sample, [0.025, 0.975]).tolist(),
        "share_negative": float(np.mean(sample < 0.0)),
    }


def main(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    protocol_hash = sha(args.protocol)
    start = now()

    train = pd.read_pickle(args.train)
    if sha(args.train) != "5dc9bcca4b42f25784895ba6327a06412860571a2c63a1fd6e285d521771bed3":
        raise RuntimeError("Train cache hash differs from frozen protocol")
    train = normalize_frame(train)
    train["_source_split"] = "train"
    series_ids = set(train["series_id"].astype(str).unique())
    if len(series_ids) != 2000:
        raise RuntimeError("Frozen train population is not 2,000 series")

    # All fitting and policy estimation below use train rows only.  The eval
    # file is not downloaded until both model files and FINAL_POLICY exist.
    dates = infer_split_dates(train)
    supervised, numeric, categorical = make_supervised(
        train,
        horizons=range(1, 8),
        max_train_rows=600000,
        seed=SEED,
        development_holdout_days=35,
    )
    splits = split_supervised(supervised, dates)
    encoder = SmoothedTargetEncoder(categorical=categorical, numeric=numeric, smoothing=30.0)
    fit = splits["fit"]
    y_fit = fit["sale_amount"].to_numpy(float)
    available_fit = fit["is_censored"].to_numpy(int) == 0
    encoder.fit(fit, y_fit, uncensored=available_fit)
    x_fit = encoder.transform(fit)

    base_params = {
        "objective": "reg:absoluteerror",
        "eta": 0.03,
        "max_depth": 8,
        "min_child_weight": 80,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 2.0,
        "tree_method": "hist",
        "max_bin": 255,
        "nthread": 8,
        "seed": SEED,
    }
    risk_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.04,
        "max_depth": 7,
        "min_child_weight": 80,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 2.0,
        "tree_method": "hist",
        "max_bin": 255,
        "nthread": 8,
        "seed": SEED,
    }
    base = xgb.train(
        base_params,
        xgb.DMatrix(x_fit[available_fit], label=y_fit[available_fit] + 1.0),
        num_boost_round=600,
    )
    risk = xgb.train(
        risk_params,
        xgb.DMatrix(x_fit, label=fit["is_censored"].to_numpy(int)),
        num_boost_round=450,
    )
    base.save_model(output / "base_l1.ubj")
    risk.save_model(output / "stockout_risk.ubj")

    predicted: dict[str, dict] = {}
    ordered = ["selection", "risk_train", "calibration_a", "calibration_b", "shadow"]
    for name in ordered:
        frame = splits[name].reset_index(drop=True)
        x = encoder.transform(frame)
        predicted[name] = {
            "frame": frame,
            "y": frame["sale_amount"].to_numpy(float),
            "available": frame["is_censored"].to_numpy(int) == 0,
            "base_raw": np.clip(base.predict(xgb.DMatrix(x)) - 1.0, 0.0, None),
            "q": np.clip(risk.predict(xgb.DMatrix(x)), 0.0, 1.0),
            "category": frame["management_group_id"].astype(str).to_numpy(),
        }

    # Development transfer check: fit on the first two tail blocks, score the next three.
    dev_fit_names = ["selection", "risk_train"]
    dev_score_names = ["calibration_a", "calibration_b", "shadow"]
    dev_y = np.concatenate([predicted[name]["y"] for name in dev_fit_names])
    dev_f = np.concatenate([predicted[name]["base_raw"] for name in dev_fit_names])
    dev_q = np.concatenate([predicted[name]["q"] for name in dev_fit_names])
    dev_cat = np.concatenate([predicted[name]["category"] for name in dev_fit_names])
    dev_available = np.concatenate([predicted[name]["available"] for name in dev_fit_names])
    dev_policy = fit_policy(dev_y, dev_f, dev_q, dev_cat, dev_available)
    development = []
    for name in dev_score_names:
        item = predicted[name]
        base_scaled = apply_category_scale(item["base_raw"], item["category"], dev_policy)
        rc = apply_policy(item["base_raw"], item["q"], item["category"], dev_policy)
        development.append({
            "block": name,
            "base": metrics(item["y"], base_scaled, item["available"]),
            "risk_conditioned": metrics(item["y"], rc, item["available"]),
        })

    # Frozen final refit uses every train-tail target, then scores eval once.
    train_tail_names = ["selection", "risk_train", "calibration_a", "calibration_b", "shadow"]
    final_y = np.concatenate([predicted[name]["y"] for name in train_tail_names])
    final_f = np.concatenate([predicted[name]["base_raw"] for name in train_tail_names])
    final_q = np.concatenate([predicted[name]["q"] for name in train_tail_names])
    final_cat = np.concatenate([predicted[name]["category"] for name in train_tail_names])
    final_available = np.concatenate([predicted[name]["available"] for name in train_tail_names])
    final_policy = fit_policy(final_y, final_f, final_q, final_cat, final_available)
    dump(output / "FINAL_POLICY.json", final_policy)
    fit_completed_utc = now()

    # This download and read are the sole row-level opening of the official
    # eval split. No fit, family selection, or threshold search follows it.
    if args.eval.exists():
        raise RuntimeError("Refusing to reuse a pre-existing eval file in the one-shot run")
    eval_opened_utc = now()
    urllib.request.urlretrieve(EVAL_URL, args.eval)
    if sha(args.eval) != EXPECTED_EVAL_SHA256:
        raise RuntimeError("Official eval file hash mismatch")
    raw_eval = pd.read_parquet(args.eval)
    raw_eval["_source_split"] = "eval"
    evaluation = normalize_frame(raw_eval)
    evaluation = evaluation[evaluation["series_id"].astype(str).isin(series_ids)].copy()
    if evaluation["series_id"].nunique() != 2000 or len(evaluation) != 14000:
        raise RuntimeError(
            f"Expected 14,000 rows over 2,000 eval series; got {len(evaluation)} over "
            f"{evaluation['series_id'].nunique()}"
        )
    combined = pd.concat([train, evaluation], ignore_index=True)
    eval_dates = infer_split_dates(combined)
    eval_supervised, eval_numeric, eval_categorical = make_supervised(
        combined,
        horizons=range(1, 8),
        max_train_rows=7,
        seed=SEED,
        development_holdout_days=35,
    )
    if eval_numeric != numeric or eval_categorical != categorical:
        raise RuntimeError("Feature schema changed when eval rows were appended")
    test_frame = split_supervised(eval_supervised, eval_dates)["test"].reset_index(drop=True)
    x_test = encoder.transform(test_frame)
    item = {
        "frame": test_frame,
        "y": test_frame["sale_amount"].to_numpy(float),
        "available": test_frame["is_censored"].to_numpy(int) == 0,
        "base_raw": np.clip(base.predict(xgb.DMatrix(x_test)) - 1.0, 0.0, None),
        "q": np.clip(risk.predict(xgb.DMatrix(x_test)), 0.0, 1.0),
        "category": test_frame["management_group_id"].astype(str).to_numpy(),
    }

    base_scaled = apply_category_scale(item["base_raw"], item["category"], final_policy)
    rc = apply_policy(item["base_raw"], item["q"], item["category"], final_policy)
    primary = item["available"]
    all_rows = np.ones(len(primary), dtype=bool)
    result = {
        "status": "COMPLETED_NO_POST_EVAL_SELECTION",
        "started_utc": start,
        "fit_and_policy_frozen_utc": fit_completed_utc,
        "eval_opened_utc": eval_opened_utc,
        "completed_utc": now(),
        "protocol_sha256": protocol_hash,
        "train_sha256": sha(args.train),
        "eval_sha256": sha(args.eval),
        "eval_open_count": 1,
        "fit_after_eval_open": 0,
        "selection_after_eval_open": 0,
        "population": {
            "series": int(item["frame"]["series_id"].nunique()),
            "rows": int(len(item["frame"])),
            "fully_available_rows": int(primary.sum()),
            "stockout_rows": int((~primary).sum()),
            "stockout_row_fraction": float((~primary).mean()),
            "recorded_sales_mass_on_stockout_rows": float(item["y"][~primary].sum() / item["y"].sum()),
        },
        "development_transfer": development,
        "final_policy": final_policy,
        "primary_fully_available": {
            "base": metrics(item["y"], base_scaled, primary),
            "risk_conditioned": metrics(item["y"], rc, primary),
            "paired_series_bootstrap": cluster_interval(item["frame"], item["y"], base_scaled, rc, primary),
        },
        "secondary_all_recorded_sales": {
            "base": metrics(item["y"], base_scaled, all_rows),
            "risk_conditioned": metrics(item["y"], rc, all_rows),
        },
        "by_management_group": grouped_metrics(item["frame"], item["y"], base_scaled, rc, primary, "management_group_id"),
        "by_horizon": grouped_metrics(item["frame"], item["y"], base_scaled, rc, primary, "horizon"),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    dump(output / "RESULTS.json", result)
    prediction_frame = item["frame"][["series_id", "dt", "horizon", "management_group_id", "sale_amount", "is_censored"]].copy()
    prediction_frame["base"] = base_scaled
    prediction_frame["risk_conditioned"] = rc
    prediction_frame["predicted_stockout_risk"] = item["q"]
    prediction_frame.to_csv(output / "EVAL_PREDICTIONS.csv.gz", index=False)
    dump(output / "RUN_RECEIPT.json", {
        "status": result["status"],
        "protocol_sha256": protocol_hash,
        "script_sha256": sha(Path(__file__)),
        "results_sha256": sha(output / "RESULTS.json"),
        "prediction_sha256": sha(output / "EVAL_PREDICTIONS.csv.gz"),
        "eval_sha256": sha(args.eval),
        "eval_open_count": 1,
        "post_eval_fit_calls": 0,
        "post_eval_selection_calls": 0,
    })
    print(json.dumps(result["primary_fully_available"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
