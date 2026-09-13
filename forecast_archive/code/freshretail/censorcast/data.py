from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]

KNOWN_FUTURE_COLUMNS = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


@dataclass(frozen=True)
class SplitDates:
    fit_end: pd.Timestamp
    selection: tuple[pd.Timestamp, ...]
    risk_train: tuple[pd.Timestamp, ...]
    calibration_a: tuple[pd.Timestamp, ...]
    calibration_b: tuple[pd.Timestamp, ...]
    shadow: tuple[pd.Timestamp, ...]
    test: tuple[pd.Timestamp, ...]

    def to_dict(self) -> dict:
        return {
            "fit_end": self.fit_end.strftime("%Y-%m-%d"),
            "selection": [x.strftime("%Y-%m-%d") for x in self.selection],
            "risk_train": [x.strftime("%Y-%m-%d") for x in self.risk_train],
            "calibration_a": [x.strftime("%Y-%m-%d") for x in self.calibration_a],
            "calibration_b": [x.strftime("%Y-%m-%d") for x in self.calibration_b],
            "shadow": [x.strftime("%Y-%m-%d") for x in self.shadow],
            "test": [x.strftime("%Y-%m-%d") for x in self.test],
        }


def _series_tuple(row: dict) -> tuple[int, int]:
    return int(row["store_id"]), int(row["product_id"])


def _stream_series_slice(
    split: str,
    dataset_name: str,
    max_series: int,
    series_offset: int = 0,
    dataset_revision: str | None = None,
) -> pd.DataFrame:
    """Stream one deterministic contiguous series tranche.

    FreshRetailNet is grouped by store-product.  ``series_offset=0`` exactly
    reproduces the v0.2 pilot tranche; a positive offset reserves unseen train
    series for a fresh development guardian without touching ``eval``.
    """
    from datasets import load_dataset

    if series_offset < 0:
        raise ValueError("series_offset must be non-negative")
    encountered: list[tuple[int, int]] = []
    encountered_set: set[tuple[int, int]] = set()
    stream = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        revision=dataset_revision,
    )
    wanted: list[tuple[int, int]] = []
    wanted_set: set[tuple[int, int]] = set()
    records: list[dict] = []
    closed = False
    for row in stream:
        key = _series_tuple(row)
        if key not in encountered_set:
            position = len(encountered)
            encountered.append(key)
            encountered_set.add(key)
            if position >= series_offset and len(wanted) < max_series:
                wanted.append(key)
                wanted_set.add(key)
            elif position >= series_offset and len(wanted) >= max_series:
                # The first group after the requested tranche closes the slice.
                closed = True
                break
        if key in wanted_set:
            records.append(dict(row))
    if not records or not closed:
        if not records:
            raise RuntimeError(f"No rows loaded from {dataset_name}:{split}")
    frame = pd.DataFrame.from_records(records)
    frame.attrs["selected_series"] = wanted
    frame.attrs["series_offset"] = int(series_offset)
    frame.attrs["series_slice_closed"] = bool(closed)
    return frame


def _stream_selected_series(
    split: str,
    dataset_name: str,
    selected: set[tuple[int, int]],
    dataset_revision: str | None = None,
) -> pd.DataFrame:
    from datasets import load_dataset

    records: list[dict] = []
    for row in load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        revision=dataset_revision,
    ):
        if _series_tuple(row) in selected:
            records.append(dict(row))
    if not records:
        raise RuntimeError(f"No selected rows loaded from {dataset_name}:{split}")
    return pd.DataFrame.from_records(records)


def load_freshretail(
    dataset_name: str,
    max_series: int | None,
    cache_dir: str | Path | None = None,
    include_eval: bool = False,
    series_offset: int = 0,
    dataset_revision: str | None = None,
) -> pd.DataFrame:
    """Load FreshRetailNet without silently opening an evaluation metric.

    The public ``eval`` rows are loaded as features/labels, but the experiment
    runner will refuse to compute their metrics unless ``confirm_open_test`` is
    true. For a strict sealed workflow, keep the eval split in a separate
    credentialed location and replace this loader with a feature-only export.
    """

    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        offset_tag = "" if int(series_offset) == 0 else f"_offset{int(series_offset)}"
        cache_file = cache_path / f"freshretail_{max_series or 'full'}{offset_tag}_{'with_eval' if include_eval else 'train_only'}.pkl.gz"
        if cache_file.exists():
            cached = pd.read_pickle(cache_file, compression="gzip")
            cached.attrs["cache_hit"] = True
            cached.attrs["requested_dataset_revision"] = dataset_revision
            return cached

    if max_series:
        train = _stream_series_slice(
            "train",
            dataset_name,
            int(max_series),
            int(series_offset),
            dataset_revision,
        )
        selected = set(train.attrs.get("selected_series", []))
        eval_frame = (
            _stream_selected_series(
                "eval", dataset_name, selected, dataset_revision
            )
            if include_eval
            else None
        )
    else:
        if int(series_offset) != 0:
            raise ValueError("series_offset requires max_series")
        from datasets import load_dataset

        train = load_dataset(
            dataset_name,
            split="train",
            revision=dataset_revision,
        ).to_pandas()
        eval_frame = (
            load_dataset(
                dataset_name,
                split="eval",
                revision=dataset_revision,
            ).to_pandas()
            if include_eval
            else None
        )

    train["_source_split"] = "train"
    pieces = [train]
    if eval_frame is not None:
        eval_frame["_source_split"] = "eval"
        pieces.append(eval_frame)
    frame = pd.concat(pieces, ignore_index=True)
    frame = normalize_frame(frame)
    frame.attrs["cache_hit"] = False
    frame.attrs["requested_dataset_revision"] = dataset_revision

    if max_series:
        counts = frame.loc[frame["_source_split"] == "train"].groupby("series_id")["dt"].nunique()
        if counts.min() < 80:
            raise RuntimeError(
                "Pilot stream did not contain complete grouped series; use full loading "
                "or prepare a deterministic series manifest."
            )
    if cache_path:
        temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
        frame.to_pickle(temporary, compression="gzip")
        temporary.replace(cache_file)
    return frame


def generate_synthetic(
    n_series: int = 240,
    n_train_days: int = 90,
    n_test_days: int = 7,
    seed: int = 20260904,
) -> pd.DataFrame:
    """Generate a known-truth panel with MNAR stockouts.

    Demand peaks, discounts, and bad weather increase both demand and stockout
    probability. The observed label is therefore censored not at random.
    """

    rng = np.random.default_rng(seed)
    total_days = n_train_days + n_test_days
    dates = pd.date_range("2025-01-01", periods=total_days, freq="D")
    rows: list[dict] = []
    for sid in range(n_series):
        store_count = max(8, min(40, n_series // 4))
        store = sid % store_count
        product = (sid // store_count) * 1000 + (sid % store_count)
        city = store % 6
        category = product % 8
        base = rng.normal(2.1, 0.55)
        trend = rng.normal(0.0, 0.0025)
        phase = rng.uniform(0, 2 * np.pi)
        inventory_anchor = np.exp(base + rng.normal(-0.05, 0.18))
        for t, date in enumerate(dates):
            dow = date.dayofweek
            holiday = int(dow >= 5)
            activity = int(rng.random() < 0.09)
            discount = float(np.clip(1.0 - activity * rng.uniform(0.08, 0.30), 0.65, 1.0))
            rain = float(max(0.0, rng.gamma(1.1, 2.0) - 1.3))
            temperature = float(18 + 8 * np.sin(2 * np.pi * t / 97 + phase) + rng.normal(0, 1.2))
            humidity = float(np.clip(58 + rain * 3 + rng.normal(0, 7), 20, 98))
            wind = float(np.clip(rng.normal(2.2, 0.8), 0, 7))
            log_mu = (
                base
                + trend * t
                + 0.20 * holiday
                + 0.55 * activity
                + 0.08 * np.log1p(rain)
                + 0.13 * np.sin(2 * np.pi * dow / 7 + phase)
            )
            latent = float(rng.lognormal(log_mu, 0.32))
            inventory = float(
                max(
                    0.25,
                    inventory_anchor
                    * (1 + 0.05 * np.sin(2 * np.pi * dow / 7))
                    * rng.lognormal(0, 0.12),
                )
            )
            observed = min(latent, inventory)
            censored = latent > inventory
            stockout_hours = int(np.clip(np.ceil(17 * (latent - inventory) / max(latent, 1e-8)), 0, 17))
            rows.append(
                {
                    "city_id": city,
                    "store_id": store,
                    "management_group_id": category // 4,
                    "first_category_id": category // 2,
                    "second_category_id": category,
                    "third_category_id": product % 16,
                    "product_id": product,
                    "dt": date,
                    "sale_amount": observed,
                    "latent_demand": latent,
                    "inventory_cap": inventory,
                    "stock_hour6_22_cnt": stockout_hours,
                    "discount": discount,
                    "holiday_flag": holiday,
                    "activity_flag": activity,
                    "precpt": rain,
                    "avg_temperature": temperature,
                    "avg_humidity": humidity,
                    "avg_wind_level": wind,
                    "_source_split": "train" if t < n_train_days else "eval",
                    "is_censored": int(censored),
                }
            )
    return normalize_frame(pd.DataFrame(rows))


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(ID_COLUMNS + KNOWN_FUTURE_COLUMNS + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = frame.copy()
    out["dt"] = pd.to_datetime(out["dt"], utc=False)
    out["series_id"] = out["store_id"].astype(str) + "::" + out["product_id"].astype(str)
    out["is_censored"] = (out["stock_hour6_22_cnt"].fillna(0).astype(float) > 0).astype(int)
    out["stockout_fraction"] = np.clip(out["stock_hour6_22_cnt"].astype(float) / 17.0, 0.0, 1.0)
    out["sale_amount"] = np.clip(pd.to_numeric(out["sale_amount"], errors="coerce"), 0.0, None)
    out = out.sort_values(["series_id", "dt"]).reset_index(drop=True)
    return out


def apply_controlled_recensoring(
    frame: pd.DataFrame,
    *,
    seed: int,
    target_rate: float = 0.20,
    minimum_loss_fraction: float = 0.15,
    maximum_loss_fraction: float = 0.55,
) -> pd.DataFrame:
    """Create a known-truth MNAR diagnostic without touching the eval split.

    Only rows that were naturally uncensored have an observable demand proxy.
    A deterministic, demand-dependent subset of those rows is clipped to a
    lower sales value. The original value is retained in ``latent_demand`` and
    evaluated only where ``latent_demand_known`` is true. Naturally censored
    rows remain censored and are never assigned fabricated ground truth.
    """

    if not 0.0 < float(target_rate) < 0.8:
        raise ValueError("target_rate must be between 0 and 0.8")
    if not 0.0 < minimum_loss_fraction <= maximum_loss_fraction < 1.0:
        raise ValueError("Invalid controlled re-censoring loss fractions")

    out = frame.copy()
    rng = np.random.default_rng(int(seed))
    original_sale = out["sale_amount"].to_numpy(dtype=float).copy()
    natural = out["is_censored"].astype(bool).to_numpy()
    train = out["_source_split"].eq("train").to_numpy()
    known = (~natural) & np.isfinite(original_sale)
    eligible = known & train & (original_sale > 0.0)

    log_sale = np.log1p(np.clip(original_sale, 0.0, None))
    work = pd.DataFrame({"series_id": out["series_id"], "log_sale": log_sale})
    center = work.groupby("series_id", sort=False)["log_sale"].transform("median").to_numpy()
    spread = work.groupby("series_id", sort=False)["log_sale"].transform("std").to_numpy()
    spread = np.where(np.isfinite(spread) & (spread > 0.05), spread, 0.25)
    # Demand pressure makes masking not-at-random; logistic noise prevents a
    # degenerate top-sales deletion exercise.
    pressure = (log_sale - center) / spread
    priority = pressure + rng.logistic(0.0, 0.85, size=len(out))
    candidates = np.flatnonzero(eligible)
    count = int(round(float(target_rate) * len(candidates)))
    selected = np.zeros(len(out), dtype=bool)
    if count > 0:
        chosen = candidates[np.argsort(priority[candidates], kind="mergesort")[-count:]]
        selected[chosen] = True

    loss_fraction = rng.uniform(
        float(minimum_loss_fraction),
        float(maximum_loss_fraction),
        size=len(out),
    )
    retained = 1.0 - loss_fraction
    out.loc[selected, "sale_amount"] = original_sale[selected] * retained[selected]
    simulated_hours = np.ceil(17.0 * loss_fraction[selected]).astype(int)
    out.loc[selected, "stock_hour6_22_cnt"] = np.clip(simulated_hours, 1, 17)
    out["natural_is_censored"] = natural.astype(int)
    out["artificially_censored"] = selected.astype(int)
    out["is_censored"] = (natural | selected).astype(int)
    out["stockout_fraction"] = np.clip(
        out["stock_hour6_22_cnt"].astype(float) / 17.0,
        0.0,
        1.0,
    )
    out["latent_demand"] = np.where(known, original_sale, np.nan)
    out["latent_demand_known"] = known.astype(int)
    out.attrs["controlled_recensoring"] = {
        "seed": int(seed),
        "target_rate": float(target_rate),
        "eligible_rows": int(np.sum(eligible)),
        "artificially_censored_rows": int(np.sum(selected)),
        "minimum_loss_fraction": float(minimum_loss_fraction),
        "maximum_loss_fraction": float(maximum_loss_fraction),
    }
    return out


def infer_split_dates(frame: pd.DataFrame) -> SplitDates:
    train_dates = sorted(pd.Timestamp(x) for x in frame.loc[frame["_source_split"] == "train", "dt"].unique())
    test_dates = sorted(pd.Timestamp(x) for x in frame.loc[frame["_source_split"] == "eval", "dt"].unique())
    if len(train_dates) < 63:
        raise ValueError(f"Need >=63 train dates for v0.3 blocked development; got {len(train_dates)}")
    if test_dates and len(test_dates) < 7:
        raise ValueError(f"If eval is loaded it must contain >=7 dates; got {len(test_dates)}")
    selection = tuple(train_dates[-35:-28])
    risk_train = tuple(train_dates[-28:-21])
    calibration_a = tuple(train_dates[-21:-14])
    calibration_b = tuple(train_dates[-14:-7])
    shadow = tuple(train_dates[-7:])
    fit_end = train_dates[-36]
    return SplitDates(
        fit_end=fit_end,
        selection=selection,
        risk_train=risk_train,
        calibration_a=calibration_a,
        calibration_b=calibration_b,
        shadow=shadow,
        test=tuple(test_dates[:7]) if test_dates else tuple(),
    )


def _rolling_feature(series: pd.Series, shift: int, window: int, stat: str) -> pd.Series:
    shifted = series.shift(shift)
    roller = shifted.rolling(window, min_periods=max(2, window // 3))
    if stat == "mean":
        return roller.mean()
    if stat == "std":
        return roller.std().fillna(0.0)
    raise ValueError(stat)


def make_supervised(
    frame: pd.DataFrame,
    horizons: Iterable[int] = range(1, 8),
    max_train_rows: int | None = None,
    seed: int = 20260904,
    future_covariates: Iterable[str] = ("discount", "holiday_flag", "activity_flag"),
    development_holdout_days: int = 35,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create direct multi-horizon rows without using post-origin outcomes.

    For a target at time ``t`` and horizon ``h``, every historical feature is
    shifted by at least ``h``. Therefore it is available at origin ``t-h``.
    Future covariates are copied from the target date under the explicit
    assumption that they are scheduled or forecast at the origin.
    """

    df = frame.sort_values(["series_id", "dt"]).copy()
    group = df.groupby("series_id", sort=False, group_keys=False)
    horizons = [int(h) for h in horizons]
    future_covariates = list(future_covariates)
    unknown_future = sorted(set(future_covariates) - set(KNOWN_FUTURE_COLUMNS))
    if unknown_future:
        raise ValueError(f"Unknown future covariates: {unknown_future}")
    train_dates = sorted(pd.Timestamp(x) for x in df.loc[df["_source_split"] == "train", "dt"].unique())
    reserve = int(development_holdout_days)
    fit_end_for_sampling = (
        train_dates[-(reserve + 1)] if len(train_dates) > reserve else train_dates[-1]
    )
    per_horizon_fit_budget = (
        int(np.ceil(max_train_rows / max(len(horizons), 1))) if max_train_rows else None
    )
    pieces: list[pd.DataFrame] = []
    for horizon in horizons:
        h = int(horizon)
        part = df[
            [
                "series_id",
                "dt",
                "_source_split",
                "sale_amount",
                "is_censored",
                "stockout_fraction",
                *ID_COLUMNS,
                *KNOWN_FUTURE_COLUMNS,
                *(["latent_demand"] if "latent_demand" in df.columns else []),
                *(["latent_demand_known"] if "latent_demand_known" in df.columns else []),
                *(["natural_is_censored"] if "natural_is_censored" in df.columns else []),
                *(["artificially_censored"] if "artificially_censored" in df.columns else []),
            ]
        ].copy()
        part["horizon"] = h
        part["origin_sale"] = group["sale_amount"].shift(h)
        part["same_weekday_sale"] = group["sale_amount"].shift(7)
        part["origin_lag7"] = group["sale_amount"].shift(h + 7)
        part["origin_lag14"] = group["sale_amount"].shift(h + 14)
        part["origin_lag28"] = group["sale_amount"].shift(h + 28)
        part["origin_stockout"] = group["is_censored"].shift(h)
        for weather_column in ("precpt", "avg_temperature", "avg_humidity", "avg_wind_level"):
            part[f"origin_{weather_column}"] = group[weather_column].shift(h)
        for window in (7, 14, 28):
            part[f"origin_roll_mean_{window}"] = group["sale_amount"].transform(
                lambda s, hh=h, ww=window: _rolling_feature(s, hh, ww, "mean")
            )
        part["origin_roll_std_14"] = group["sale_amount"].transform(
            lambda s, hh=h: _rolling_feature(s, hh, 14, "std")
        )
        part["origin_stockout_rate_7"] = group["is_censored"].transform(
            lambda s, hh=h: _rolling_feature(s.astype(float), hh, 7, "mean")
        )
        part["origin_stockout_rate_28"] = group["is_censored"].transform(
            lambda s, hh=h: _rolling_feature(s.astype(float), hh, 28, "mean")
        )
        part["dow_sin"] = np.sin(2 * np.pi * part["dt"].dt.dayofweek / 7)
        part["dow_cos"] = np.cos(2 * np.pi * part["dt"].dt.dayofweek / 7)
        part["day_index"] = (part["dt"] - df["dt"].min()).dt.days.astype(float)
        part = part.dropna(subset=["origin_sale", "same_weekday_sale", "origin_lag28"])
        if per_horizon_fit_budget:
            early = (part["_source_split"] == "train") & (part["dt"] <= fit_end_for_sampling)
            fit_part = part[early]
            retained = part[~early]
            if len(fit_part) > per_horizon_fit_budget:
                fit_part = fit_part.sample(per_horizon_fit_budget, random_state=seed + h)
            part = pd.concat([fit_part, retained], ignore_index=True)
        pieces.append(part)
    out = pd.concat(pieces, ignore_index=True)
    out = out.reset_index(drop=True)

    numeric_features = [
        "horizon",
        "origin_sale",
        "same_weekday_sale",
        "origin_lag7",
        "origin_lag14",
        "origin_lag28",
        "origin_stockout",
        "origin_roll_mean_7",
        "origin_roll_mean_14",
        "origin_roll_mean_28",
        "origin_roll_std_14",
        "origin_stockout_rate_7",
        "origin_stockout_rate_28",
        "dow_sin",
        "dow_cos",
        "day_index",
        "origin_precpt",
        "origin_avg_temperature",
        "origin_avg_humidity",
        "origin_avg_wind_level",
        *future_covariates,
    ]
    categorical_features = ID_COLUMNS.copy()
    if max_train_rows:
        out.attrs["max_train_rows"] = int(max_train_rows)
        out.attrs["sampling_seed"] = int(seed)
    return out, numeric_features, categorical_features


def fixed_origin_mask(rows: pd.DataFrame, block_dates: tuple[pd.Timestamp, ...]) -> pd.Series:
    if not block_dates:
        return pd.Series(False, index=rows.index)
    order = {pd.Timestamp(date): i + 1 for i, date in enumerate(block_dates)}
    required_horizon = rows["dt"].map(order)
    return required_horizon.notna() & (rows["horizon"].astype(int) == required_horizon.astype("Int64"))


def split_supervised(rows: pd.DataFrame, dates: SplitDates) -> dict[str, pd.DataFrame]:
    fit = rows[(rows["_source_split"] == "train") & (rows["dt"] <= dates.fit_end)].copy()
    return {
        "fit": fit,
        "selection": rows[fixed_origin_mask(rows, dates.selection)].copy(),
        "risk_train": rows[fixed_origin_mask(rows, dates.risk_train)].copy(),
        "calibration_a": rows[fixed_origin_mask(rows, dates.calibration_a)].copy(),
        "calibration_b": rows[fixed_origin_mask(rows, dates.calibration_b)].copy(),
        "shadow": rows[fixed_origin_mask(rows, dates.shadow)].copy(),
        "test": rows[fixed_origin_mask(rows, dates.test)].copy(),
    }
