from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


META_NAMES = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")


def _stable_codes(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    strings = np.asarray(values).astype(str)
    levels = sorted(set(strings.tolist()))
    mapping = {value: index for index, value in enumerate(levels)}
    return np.asarray([mapping[value] for value in strings], dtype=np.int16), mapping


def _fill_price_matrix(matrix: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill prices without leaking demand outcomes."""
    valid = np.isfinite(matrix)
    columns = np.arange(matrix.shape[1], dtype=np.int32)[None, :]
    previous = np.maximum.accumulate(np.where(valid, columns, -1), axis=1)
    rows = np.arange(matrix.shape[0], dtype=np.int32)[:, None]
    row_grid = np.broadcast_to(rows, matrix.shape)
    forward = matrix.copy()
    has_previous = previous >= 0
    forward[has_previous] = matrix[row_grid[has_previous], previous[has_previous]]
    valid_forward = np.isfinite(forward)
    following = np.minimum.accumulate(
        np.where(valid_forward, columns, matrix.shape[1])[:, ::-1], axis=1
    )[:, ::-1]
    has_following = following < matrix.shape[1]
    forward[~valid_forward & has_following] = forward[
        row_grid[~valid_forward & has_following],
        following[~valid_forward & has_following],
    ]
    return np.nan_to_num(forward, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass
class DesignPanel:
    truth: np.ndarray
    observed: np.ndarray
    capacity: np.ndarray
    censored: np.ndarray
    metadata: dict[str, np.ndarray]
    calendar: pd.DataFrame
    price_matrix: np.ndarray
    day_to_week_index: np.ndarray
    static_codes: dict[str, np.ndarray]
    calendar_arrays: dict[str, np.ndarray]

    @classmethod
    def load(
        cls,
        design_shard: str | Path,
        calendar_csv: str | Path,
        prices_csv: str | Path,
    ) -> "DesignPanel":
        design_shard = Path(design_shard)
        with np.load(design_shard, allow_pickle=False) as loaded:
            required = set(META_NAMES) | {
                "truth",
                "observed",
                "capacity",
                "censored",
                "day_start",
                "day_end",
            }
            missing = sorted(required - set(loaded.files))
            if missing:
                raise RuntimeError(f"Design shard is missing arrays: {missing}")
            if int(loaded["day_start"][0]) != 1 or int(loaded["day_end"][0]) != 1913:
                raise RuntimeError("The design shard must cover exactly d_1 through d_1913")
            truth = loaded["truth"].astype(np.float32)
            observed = loaded["observed"].astype(np.float32)
            capacity = loaded["capacity"].astype(np.float32)
            censored = loaded["censored"].astype(np.uint8)
            metadata = {name: loaded[name].astype(str) for name in META_NAMES}
        if not (truth.shape == observed.shape == capacity.shape == censored.shape):
            raise RuntimeError("Design outcome array shapes differ")
        if truth.shape[1] != 1913 or truth.shape[0] < 100:
            raise RuntimeError(f"Unexpected design panel shape: {truth.shape}")
        if np.any(~np.isfinite(truth)) or np.any(truth < 0):
            raise RuntimeError("Design truth contains invalid values")
        if np.any(observed > truth) or np.any(observed > capacity):
            raise RuntimeError("Controlled-censoring invariants failed")

        calendar = pd.read_csv(calendar_csv)
        expected_days = [f"d_{day}" for day in range(1, 1914)]
        if calendar["d"].astype(str).tolist()[:1913] != expected_days:
            raise RuntimeError("M5 calendar day ordering changed")
        calendar = calendar.iloc[:1913].copy().reset_index(drop=True)
        week_values = sorted(calendar["wm_yr_wk"].astype(int).unique().tolist())
        week_map = {week: index for index, week in enumerate(week_values)}
        day_to_week_index = np.asarray(
            [week_map[int(value)] for value in calendar["wm_yr_wk"]], dtype=np.int16
        )

        series_keys = np.char.add(
            np.char.add(metadata["store_id"].astype(str), "\0"), metadata["item_id"].astype(str)
        )
        if len(set(series_keys.tolist())) != len(series_keys):
            raise RuntimeError("Design store-item keys are not unique")
        series_map = {value: index for index, value in enumerate(series_keys.tolist())}
        price_matrix = np.full((len(series_keys), len(week_values)), np.nan, dtype=np.float32)
        usecols = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
        for chunk in pd.read_csv(
            prices_csv,
            usecols=usecols,
            dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
            chunksize=750_000,
        ):
            keys = chunk["store_id"].astype(str) + "\0" + chunk["item_id"].astype(str)
            series_index = keys.map(series_map)
            keep = series_index.notna() & chunk["wm_yr_wk"].isin(week_map)
            if not keep.any():
                continue
            rows = series_index.loc[keep].astype(np.int32).to_numpy()
            cols = chunk.loc[keep, "wm_yr_wk"].map(week_map).astype(np.int16).to_numpy()
            price_matrix[rows, cols] = chunk.loc[keep, "sell_price"].to_numpy(dtype=np.float32)
        price_matrix = _fill_price_matrix(price_matrix)

        static_codes: dict[str, np.ndarray] = {}
        for name in ("dept_id", "cat_id", "store_id", "state_id"):
            static_codes[name], _ = _stable_codes(metadata[name])

        event_1 = calendar.get("event_name_1", pd.Series([None] * len(calendar))).notna().to_numpy(dtype=np.float32)
        event_2 = calendar.get("event_name_2", pd.Series([None] * len(calendar))).notna().to_numpy(dtype=np.float32)
        event_type = calendar.get("event_type_1", pd.Series([None] * len(calendar))).fillna("NONE").astype(str)
        event_levels = {value: index for index, value in enumerate(sorted(event_type.unique().tolist()))}
        calendar_arrays = {
            "wday": pd.to_numeric(calendar["wday"], errors="coerce").fillna(1).to_numpy(dtype=np.float32),
            "month": pd.to_numeric(calendar["month"], errors="coerce").fillna(1).to_numpy(dtype=np.float32),
            "year": pd.to_numeric(calendar["year"], errors="coerce").fillna(2011).to_numpy(dtype=np.float32),
            "event_any": np.maximum(event_1, event_2),
            "event_type": event_type.map(event_levels).to_numpy(dtype=np.float32),
            "snap_CA": pd.to_numeric(calendar.get("snap_CA", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32),
            "snap_TX": pd.to_numeric(calendar.get("snap_TX", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32),
            "snap_WI": pd.to_numeric(calendar.get("snap_WI", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32),
        }
        return cls(
            truth=truth,
            observed=observed,
            capacity=capacity,
            censored=censored,
            metadata=metadata,
            calendar=calendar,
            price_matrix=price_matrix,
            day_to_week_index=day_to_week_index,
            static_codes=static_codes,
            calendar_arrays=calendar_arrays,
        )

    @property
    def n_series(self) -> int:
        return int(self.truth.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.truth.shape[1])


class FeatureBuilder:
    COMMON_NAMES = (
        "horizon",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "lag_56",
        "mean_7",
        "mean_28",
        "mean_56",
        "std_7",
        "std_28",
        "zero_fraction_28",
        "trend_7_vs_28",
        "price",
        "price_ratio_1w",
        "price_ratio_4w",
        "wday_sin",
        "wday_cos",
        "month_sin",
        "month_cos",
        "event_any",
        "event_type",
        "snap",
        "dept_code",
        "cat_code",
        "store_code",
        "state_code",
        "time_fraction",
    )
    CENSOR_NAMES = (
        "censor_rate_7",
        "censor_rate_28",
        "origin_capacity",
        "origin_fill_ratio",
        "capacity_vs_mean_28",
    )

    def __init__(self, panel: DesignPanel):
        self.panel = panel
        observed = panel.observed.astype(np.float64, copy=False)
        self.obs_cumsum = np.pad(np.cumsum(observed, axis=1), ((0, 0), (1, 0)))
        self.obs2_cumsum = np.pad(np.cumsum(observed * observed, axis=1), ((0, 0), (1, 0)))
        self.zero_cumsum = np.pad(
            np.cumsum((panel.observed <= 0).astype(np.int32), axis=1), ((0, 0), (1, 0))
        )
        self.censor_cumsum = np.pad(
            np.cumsum(panel.censored.astype(np.int32), axis=1), ((0, 0), (1, 0))
        )

    @staticmethod
    def horizon_for_day(target_day: np.ndarray) -> np.ndarray:
        values = np.asarray(target_day, dtype=np.int32)
        return 1 + ((values - 1) % 7)

    def _rolling(self, cumsum: np.ndarray, series: np.ndarray, origin: np.ndarray, window: int) -> np.ndarray:
        return cumsum[series, origin] - cumsum[series, origin - int(window)]

    def make_features(
        self,
        series: np.ndarray,
        target_day: np.ndarray,
        *,
        include_censor: bool,
    ) -> np.ndarray:
        series = np.asarray(series, dtype=np.int32)
        target_day = np.asarray(target_day, dtype=np.int32)
        if series.shape != target_day.shape:
            raise ValueError("series and target_day must have the same shape")
        horizon = self.horizon_for_day(target_day)
        origin = target_day - horizon
        if np.any(origin < 56) or np.any(target_day > self.panel.n_days):
            raise ValueError("Feature request is outside the deployable day range")
        obs = self.panel.observed
        lag_values = [obs[series, origin - lag] for lag in (1, 7, 14, 28, 56)]
        sums = {window: self._rolling(self.obs_cumsum, series, origin, window) for window in (7, 28, 56)}
        means = {window: sums[window] / float(window) for window in sums}
        sq7 = self._rolling(self.obs2_cumsum, series, origin, 7) / 7.0
        sq28 = self._rolling(self.obs2_cumsum, series, origin, 28) / 28.0
        std7 = np.sqrt(np.maximum(sq7 - means[7] ** 2, 0.0))
        std28 = np.sqrt(np.maximum(sq28 - means[28] ** 2, 0.0))
        zero28 = self._rolling(self.zero_cumsum, series, origin, 28) / 28.0

        week = self.panel.day_to_week_index[target_day - 1].astype(np.int32)
        week1 = np.maximum(week - 1, 0)
        week4 = np.maximum(week - 4, 0)
        price = self.panel.price_matrix[series, week]
        price1 = self.panel.price_matrix[series, week1]
        price4 = self.panel.price_matrix[series, week4]
        price_ratio1 = price / np.maximum(price1, 1e-3) - 1.0
        price_ratio4 = price / np.maximum(price4, 1e-3) - 1.0

        day_index = target_day - 1
        wday = self.panel.calendar_arrays["wday"][day_index]
        month = self.panel.calendar_arrays["month"][day_index]
        state = self.panel.metadata["state_id"][series].astype(str)
        snap = np.zeros(len(series), dtype=np.float32)
        for state_name in ("CA", "TX", "WI"):
            mask = np.char.startswith(state, state_name)
            snap[mask] = self.panel.calendar_arrays[f"snap_{state_name}"][day_index[mask]]

        columns: list[np.ndarray] = [
            horizon,
            *lag_values,
            means[7],
            means[28],
            means[56],
            std7,
            std28,
            zero28,
            means[7] - means[28],
            price,
            price_ratio1,
            price_ratio4,
            np.sin(2.0 * np.pi * (wday - 1.0) / 7.0),
            np.cos(2.0 * np.pi * (wday - 1.0) / 7.0),
            np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            self.panel.calendar_arrays["event_any"][day_index],
            self.panel.calendar_arrays["event_type"][day_index],
            snap,
            self.panel.static_codes["dept_id"][series],
            self.panel.static_codes["cat_id"][series],
            self.panel.static_codes["store_id"][series],
            self.panel.static_codes["state_id"][series],
            target_day / float(self.panel.n_days),
        ]
        if include_censor:
            censor7 = self._rolling(self.censor_cumsum, series, origin, 7) / 7.0
            censor28 = self._rolling(self.censor_cumsum, series, origin, 28) / 28.0
            origin_capacity = self.panel.capacity[series, origin - 1]
            origin_observed = self.panel.observed[series, origin - 1]
            fill_ratio = origin_observed / np.maximum(origin_capacity, 1.0)
            columns.extend(
                [
                    censor7,
                    censor28,
                    origin_capacity,
                    fill_ratio,
                    origin_capacity / np.maximum(means[28], 0.25),
                ]
            )
        result = np.column_stack(columns).astype(np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=10_000.0, neginf=-10_000.0)

    def seasonal_predictions(self, series: np.ndarray, target_day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        series = np.asarray(series, dtype=np.int32)
        target_day = np.asarray(target_day, dtype=np.int32)
        lag7 = self.panel.observed[series, target_day - 8]
        seasonal_stack = np.column_stack(
            [self.panel.observed[series, target_day - 7 * k - 1] for k in (1, 2, 3, 4)]
        )
        return lag7.astype(np.float32), np.median(seasonal_stack, axis=1).astype(np.float32)

    def labels(self, series: np.ndarray, target_day: np.ndarray) -> dict[str, np.ndarray]:
        series = np.asarray(series, dtype=np.int32)
        target_day = np.asarray(target_day, dtype=np.int32)
        index = target_day - 1
        return {
            "truth": self.panel.truth[series, index].astype(np.float32),
            "observed": self.panel.observed[series, index].astype(np.float32),
            "capacity": self.panel.capacity[series, index].astype(np.float32),
            "censored": self.panel.censored[series, index].astype(np.uint8),
        }

    def sample_pairs(self, start_day: int, end_day: int, rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        total = self.panel.n_series * (int(end_day) - int(start_day) + 1)
        take = min(int(rows), int(total))
        rng = np.random.default_rng(int(seed))
        if take == total:
            flat = np.arange(total, dtype=np.int64)
        else:
            flat = rng.choice(total, size=take, replace=False)
        days = int(end_day) - int(start_day) + 1
        series = (flat // days).astype(np.int32)
        target_day = (int(start_day) + flat % days).astype(np.int32)
        return series, target_day

    def block_pairs(self, start_day: int, end_day: int) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
        series = np.arange(self.panel.n_series, dtype=np.int32)
        for target_day in range(int(start_day), int(end_day) + 1):
            yield target_day, series, np.full(self.panel.n_series, target_day, dtype=np.int32)
