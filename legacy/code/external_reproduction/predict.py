"""Frozen external predictions; the parent opening gate owns access authorization.

There is deliberately no command-line entry point. ``predict_external`` takes
already opened arrays only after the parent has durably recorded its one-use
opening. No fitting, threshold choice, family search, or outcome-based adaptation
occurs here. Synthetic tests exercise the pure feature helpers without data access.
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream/design"))
sys.path.insert(0, str(ROOT / "code"))
from censorcast_v05.features import DesignPanel, FeatureBuilder, _fill_price_matrix
from r2_io import dump, sha, write

META = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
STATIC = ("dept_id", "cat_id", "store_id", "state_id")
RISK_STATIC = ("cat_id", "dept_id", "store_id", "state_id")
LEGACY_LAST_DAY = 1913
LAST_DAY = 1941
FORECAST_START = 1314
TARGET_DAYS = np.arange(1914, 1942, dtype=np.int32)
FEATURE_NAMES = (
    "log_baseline", "log_proposal", "log_raw", "log_em", "censor_probability",
    "positive_correction", "prior_observed_wape_origin", "horizon",
    "observed_mean7", "observed_mean28", "observed_mean56", "observed_std28",
    "observed_zero_fraction28", "past_censor_rate7", "past_censor_rate28",
    "observed_seasonal7", "observed_seasonal14", "cat_id", "dept_id",
    "store_id", "state_id",
)
MODEL_PATHS = {
    "bundle": "inputs/forecast_bundle_v0_5.joblib",
    "point": "results/point_baselines/observed_l1_s20260906.txt",
    "error": "results/selective_strong/mean_error_s20260906.txt",
    "demand": "results/review_revision/demand_s20260906.txt",
}
POINT_RUNTIME = {"numpy": "2.1.3", "scipy": "1.16.3", "scikit-learn": "1.6.1",
                 "joblib": "1.5.3", "lightgbm": "4.6.0", "pandas": "2.2.3",
                 "threadpoolctl": "3.6.0"}


def origin(days):
    days = np.asarray(days, dtype=np.int32)
    return days - (1 + (days - 1) % 7)


class FrozenFeatureBuilder(FeatureBuilder):
    """Extend the day range, preserving the numerical training-time scale."""

    def make_features(self, series, target_day, *, include_censor):
        features = super().make_features(series, target_day, include_censor=include_censor)
        column = self.COMMON_NAMES.index("time_fraction")
        features[:, column] = np.asarray(target_day) / float(LEGACY_LAST_DAY)
        return features


def _category_codes(meta, design_metadata):
    codes = {}
    for name in STATIC:
        levels = np.unique(np.asarray(design_metadata[name]).astype(str))
        labels = np.asarray(meta[name]).astype(str)
        unknown = sorted(set(labels) - set(levels))
        if unknown:
            raise ValueError(f"Unknown frozen {name} levels: {unknown}")
        mapping = {value: index for index, value in enumerate(levels)}
        codes[name] = np.array([mapping[value] for value in labels], dtype=np.int16)
    return codes


def _calendar_values(calendar):
    """Encode future calendar labels with the original first-1913-day map."""
    def numeric(name, default):
        values = calendar[name] if name in calendar else pd.Series(default, index=calendar.index)
        return pd.to_numeric(values, errors="coerce").fillna(default).to_numpy(dtype=np.float32)

    event1 = calendar.get("event_name_1", pd.Series(None, index=calendar.index, dtype=object))
    event2 = calendar.get("event_name_2", pd.Series(None, index=calendar.index, dtype=object))
    event_type = calendar.get("event_type_1", pd.Series(None, index=calendar.index, dtype=object)).fillna("NONE").astype(str)
    levels = sorted(event_type.iloc[:LEGACY_LAST_DAY].unique().tolist())
    mapping = {value: index for index, value in enumerate(levels)}
    unknown = sorted(set(event_type) - set(mapping))
    if unknown:
        raise ValueError(f"Future event types are outside the frozen vocabulary: {unknown}")
    return {
        "wday": numeric("wday", 1), "month": numeric("month", 1),
        "year": numeric("year", 2011),
        "event_any": np.maximum(event1.notna().to_numpy(dtype=np.float32), event2.notna().to_numpy(dtype=np.float32)),
        "event_type": event_type.map(mapping).to_numpy(dtype=np.float32),
        "snap_CA": numeric("snap_CA", 0), "snap_TX": numeric("snap_TX", 0),
        "snap_WI": numeric("snap_WI", 0),
    }


def _load_prices(metadata, calendar, prices_csv):
    weeks = sorted(calendar["wm_yr_wk"].astype(int).unique().tolist())
    week_map = {week: index for index, week in enumerate(weeks)}
    day_to_week = np.array([week_map[int(x)] for x in calendar["wm_yr_wk"]], dtype=np.int16)
    keys = np.char.add(np.char.add(metadata["store_id"].astype(str), "\0"), metadata["item_id"].astype(str))
    if len(set(keys.tolist())) != len(keys):
        raise ValueError("External store-item keys are not unique")
    series_map = {value: index for index, value in enumerate(keys.tolist())}
    raw_prices = np.full((len(keys), len(weeks)), np.nan, dtype=np.float32)
    for chunk in pd.read_csv(prices_csv, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
                             dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"}, chunksize=750_000):
        chunk_keys = chunk["store_id"].astype(str) + "\0" + chunk["item_id"].astype(str)
        rows = chunk_keys.map(series_map)
        keep = rows.notna() & chunk["wm_yr_wk"].isin(week_map)
        if keep.any():
            rr = rows.loc[keep].astype(np.int32).to_numpy()
            cc = chunk.loc[keep, "wm_yr_wk"].map(week_map).astype(np.int16).to_numpy()
            raw_prices[rr, cc] = chunk.loc[keep, "sell_price"].to_numpy(dtype=np.float32)
    prices = _fill_price_matrix(raw_prices)
    historic_week_count = int(day_to_week[LEGACY_LAST_DAY - 1]) + 1
    historical_prices = _fill_price_matrix(raw_prices[:, :historic_week_count])
    if not np.array_equal(prices[:, :historic_week_count], historical_prices):
        raise ValueError("Future price backfill changes historical features; frozen replay would differ")
    return prices, day_to_week


def build_extended_panel(context, outcomes, calendar_csv, prices_csv, design_metadata):
    """Construct the original panel type without changing its historical loader."""
    if "truth" in context:
        raise ValueError("Context must contain observed sales only, without truth")
    for values, start, end in ((context, 1, LEGACY_LAST_DAY), (outcomes, 1914, LAST_DAY)):
        if int(values["day_start"][0]) != start or int(values["day_end"][0]) != end:
            raise ValueError("Unexpected external block day range")
        for name in META:
            if not np.array_equal(context[name], outcomes[name]):
                raise ValueError(f"Context/outcome row alignment mismatch: {name}")
        n = len(values["id"])
        for name in ("observed", "capacity", "censored"):
            array = np.asarray(values[name])
            if array.shape != (n, end - start + 1) or not np.isfinite(array).all():
                raise ValueError(f"Invalid shape or finite values: {name}")
        if np.any(values["observed"] < 0) or np.any(values["capacity"] < 0):
            raise ValueError("Negative observed sales or capacity")
        if np.any(values["observed"] > values["capacity"]):
            raise ValueError("Observed sales exceed capacity")
        if not np.isin(values["censored"], [0, 1]).all():
            raise ValueError("Censor flags must be binary")
    truth = np.asarray(outcomes["truth"], dtype=np.float32)
    if truth.shape != outcomes["observed"].shape or not np.isfinite(truth).all() or np.any(truth < 0):
        raise ValueError("Invalid external truth")
    if not np.array_equal(outcomes["observed"], np.minimum(truth, outcomes["capacity"])):
        raise ValueError("External controlled-censoring observations differ from min(truth, capacity)")
    if not np.array_equal(outcomes["censored"].astype(bool), truth > outcomes["capacity"]):
        raise ValueError("External strict-censoring flags differ from truth > capacity")
    metadata = {name: np.asarray(context[name]).astype(str) for name in META}
    if set(metadata["item_id"]) & set(np.asarray(design_metadata["item_id"]).astype(str)):
        raise ValueError("External items overlap model-development items")
    static_codes = _category_codes(metadata, design_metadata)
    calendar = pd.read_csv(calendar_csv)
    expected = [f"d_{day}" for day in range(1, LAST_DAY + 1)]
    if calendar["d"].astype(str).tolist()[:LAST_DAY] != expected:
        raise ValueError("Calendar ordering or extension is invalid")
    calendar = calendar.iloc[:LAST_DAY].copy().reset_index(drop=True)
    prices, day_to_week = _load_prices(metadata, calendar, prices_csv)
    observed = np.concatenate([context["observed"], outcomes["observed"]], axis=1).astype(np.float32)
    capacity = np.concatenate([context["capacity"], outcomes["capacity"]], axis=1).astype(np.float32)
    censored = np.concatenate([context["censored"], outcomes["censored"]], axis=1).astype(np.uint8)
    # FeatureBuilder never accesses panel.truth; even external labels are kept out.
    panel = DesignPanel(observed, observed, capacity, censored, metadata, calendar,
                        prices, day_to_week, static_codes, _calendar_values(calendar))
    return panel


def risk_features(panel, forecasts, target_days, *, forecast_start=FORECAST_START):
    """Exactly the 21 origin-aligned columns in run_selective_study.construct."""
    days = np.asarray(target_days, dtype=np.int32)
    origins = origin(days)
    indices = np.maximum(origins - (forecast_start - 1), 0)
    select = days - forecast_start
    allp = np.asarray(forecasts["proposal"], dtype=np.float32)
    obs = panel.observed.astype(np.float32, copy=False)
    cen = panel.censored.astype(np.float32, copy=False)
    if select.min() < 0 or select.max() >= allp.shape[1] or indices.max() > allp.shape[1]:
        raise ValueError("Risk feature request outside the fixed forecast history")
    allo = obs[:, forecast_start - 1:forecast_start - 1 + allp.shape[1]]
    ce = np.pad(np.cumsum(np.abs(allo.astype(np.float64) - allp), axis=1), ((0, 0), (1, 0)))
    cd = np.pad(np.cumsum(allo.astype(np.float64), axis=1), ((0, 0), (1, 0)))
    gp = (ce.sum(axis=0) + 30 * .75) / (cd.sum(axis=0) + 30)
    prior = (ce[:, indices] + 30 * gp[indices]) / (cd[:, indices] + 30)
    oc = np.pad(np.cumsum(obs, dtype=np.float64, axis=1), ((0, 0), (1, 0)))
    oc2 = np.pad(np.cumsum(obs * obs, dtype=np.float64, axis=1), ((0, 0), (1, 0)))
    zc = np.pad(np.cumsum(obs == 0, axis=1), ((0, 0), (1, 0)))
    cc = np.pad(np.cumsum(cen, axis=1), ((0, 0), (1, 0)))
    values = {name: np.asarray(value)[:, select] for name, value in forecasts.items()}
    cols = [np.log1p(values[name]) for name in ("baseline", "proposal", "raw_poisson_histgb", "censored_poisson_em")]
    cols += [values["censor_probability"], np.maximum(values["proposal"] - values["baseline"], 0),
             prior, np.broadcast_to((days - origins)[None, :], prior.shape)]
    means = {w: (oc[:, origins] - oc[:, origins - w]) / w for w in (7, 28, 56)}
    cols += [means[7], means[28], means[56],
             np.sqrt(np.maximum((oc2[:, origins] - oc2[:, origins - 28]) / 28 - means[28] ** 2, 0)),
             (zc[:, origins] - zc[:, origins - 28]) / 28,
             (cc[:, origins] - cc[:, origins - 7]) / 7,
             (cc[:, origins] - cc[:, origins - 28]) / 28,
             obs[:, days - 8], obs[:, days - 15]]
    cols += [np.broadcast_to(panel.static_codes[name][:, None], prior.shape) for name in RISK_STATIC]
    features = np.stack([np.asarray(column, np.float32) for column in cols], axis=-1)
    if features.shape[-1] != 21 or not np.isfinite(features).all():
        raise ValueError("Invalid origin-aligned risk features")
    return features


def _assert_gate(gate_receipt, freeze_sha256):
    if not isinstance(gate_receipt, dict) or gate_receipt.get("status") != "AUTHORIZED_EXTERNAL_PREDICTION":
        raise PermissionError("The persisted parent opening gate must authorize prediction")
    if not isinstance(freeze_sha256, str) or len(freeze_sha256) != 64 or gate_receipt.get("freeze_sha256") != freeze_sha256:
        raise PermissionError("Opening receipt does not match the frozen protocol")
    if gate_receipt.get("external_use_count") != 1:
        raise PermissionError("Only the declared first external opening is supported")
    if gate_receipt.get("original_external_gate_pass") is not False or gate_receipt.get("certificate_issued") is not False:
        raise PermissionError("This comparative study cannot issue the original operational certificate")


def predict_external(context, outcomes, calendar_csv, prices_csv, design_metadata,
                     gate_receipt, freeze_sha256, output_dir=None, threads=4):
    """Return ``(data, metadata, raw_head_scores, receipt)`` after the gate.

    ``data`` contains final 28-day truth/proposal/baseline/target_days arrays;
    head scores are raw float32 predictions named error/demand. The parent
    applies its already frozen clipping, mixture weights, reference mean, and thresholds.
    """
    _assert_gate(gate_receipt, freeze_sha256)
    runtime = {name: version(name) for name in POINT_RUNTIME}
    if runtime != POINT_RUNTIME:
        raise RuntimeError(f"Frozen point bundle requires recorded runtime {POINT_RUNTIME}; found {runtime}")
    import joblib
    import lightgbm as lgb

    model_paths = {name: ROOT / rel for name, rel in MODEL_PATHS.items()}
    model_hashes = {name: sha(path) for name, path in model_paths.items()}
    expected = gate_receipt.get("model_sha256")
    if expected is not None and expected != model_hashes:
        raise PermissionError("Frozen model hashes differ from the opening receipt")
    panel = build_extended_panel(context, outcomes, calendar_csv, prices_csv, design_metadata)
    builder = FrozenFeatureBuilder(panel)
    bundle = joblib.load(model_paths["bundle"])
    point = lgb.Booster(model_file=str(model_paths["point"]))
    forecasts = {name: np.empty((panel.n_series, LAST_DAY - FORECAST_START + 1), np.float32)
                 for name in ("baseline", "proposal", "raw_poisson_histgb", "censored_poisson_em", "censor_probability")}
    days = np.arange(FORECAST_START, LAST_DAY + 1, dtype=np.int32)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        for start in range(0, len(days), 7):
            block = days[start:start + 7]
            series = np.repeat(np.arange(panel.n_series), len(block))
            targets = np.tile(block, panel.n_series)
            common = builder.make_features(series, targets, include_censor=False)
            censor = builder.make_features(series, targets, include_censor=True)
            baseline = np.maximum(point.predict(common, num_threads=threads), 0).reshape(panel.n_series, -1).astype(np.float32)
            legacy = bundle.predict(common, censor)
            forecasts["baseline"][:, start:start + len(block)] = baseline
            for name, array in legacy.items():
                forecasts[name][:, start:start + len(block)] = array.reshape(panel.n_series, -1)
    forecasts["proposal"] = (forecasts["baseline"] + 2 * forecasts["censor_probability"] ** 2
                             * np.maximum(forecasts["censored_poisson_em"] - forecasts["baseline"], 0))
    features = risk_features(panel, forecasts, TARGET_DAYS)
    flat = features.reshape(-1, features.shape[-1])
    scores = {}
    for name in ("error", "demand"):
        model = lgb.Booster(model_file=str(model_paths[name]))
        if model.num_trees() != 220 or model.num_feature() != 21:
            raise ValueError(f"Unexpected frozen {name} architecture")
        scores[name] = model.predict(flat, num_iteration=220, num_threads=threads).reshape(features.shape[:2]).astype(np.float32)
    index = TARGET_DAYS - FORECAST_START
    data = {"truth": np.asarray(outcomes["truth"], dtype=np.float32).copy(),
            "proposal": forecasts["proposal"][:, index], "baseline": forecasts["baseline"][:, index],
            "target_days": TARGET_DAYS.copy()}
    receipt = {"status": "FIXED_EXTERNAL_PREDICTIONS_READY", "training_calls": 0,
               "threshold_search_calls": 0, "model_sha256": model_hashes,
               "runtime": runtime, "freeze_sha256": freeze_sha256,
               "prediction_days": [FORECAST_START, LAST_DAY], "evaluation_days": [1914, LAST_DAY],
               "forecast_origins": np.unique(origin(TARGET_DAYS)).tolist(),
               "time_fraction_denominator": LEGACY_LAST_DAY, "feature_names": list(FEATURE_NAMES),
               "context_truth_available": False, "truth_used_in_features": False,
               "censor_flag_semantics": "strict benchmark truth > capacity, inherited unchanged",
               "event_type_vocabulary": "calendar days 1 through 1913 only",
               "static_vocabulary": "model-development metadata, sorted labels",
               "historical_price_fill_unchanged": True, "original_external_gate_pass": False,
               "certificate_issued": False}
    if output_dir is not None:
        output_dir = Path(output_dir)
        for filename, values in (("external_predictions.npz", data), ("external_head_scores.npz", scores),
                                 ("external_metadata.npz", panel.metadata)):
            stream = io.BytesIO()
            np.savez_compressed(stream, **values)
            write(output_dir / filename, stream.getvalue())
        dump(output_dir / "PREDICTION_RECEIPT.json", receipt)
    return data, panel.metadata, scores, receipt
