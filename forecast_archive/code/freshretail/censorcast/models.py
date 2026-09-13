from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class SmoothedTargetEncoder:
    """Leakage-contained fit-period encoder for high-cardinality IDs."""

    def __init__(self, categorical: list[str], numeric: list[str], smoothing: float = 30.0):
        self.categorical = list(categorical)
        self.numeric = list(numeric)
        self.smoothing = float(smoothing)
        self.global_mean_: float | None = None
        self.maps_: dict[str, pd.DataFrame] = {}
        self.medians_: dict[str, float] = {}
        self.feature_names_: list[str] = []

    def fit(self, frame: pd.DataFrame, y: np.ndarray, uncensored: np.ndarray | None = None):
        log_y = np.log1p(np.asarray(y, dtype=float))
        mask = np.isfinite(log_y)
        if uncensored is not None and np.sum(uncensored) >= 100:
            mask &= np.asarray(uncensored, dtype=bool)
        self.global_mean_ = float(np.nanmean(log_y[mask]))
        work = frame.loc[mask, self.categorical].copy()
        work["__target"] = log_y[mask]
        for column in self.categorical:
            stats = work.groupby(column, dropna=False)["__target"].agg(["mean", "count"])
            stats["encoded"] = (
                stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing
            ) / (stats["count"] + self.smoothing)
            stats["frequency"] = stats["count"] / max(len(work), 1)
            self.maps_[column] = stats[["encoded", "frequency"]]
        numeric = frame[self.numeric].apply(pd.to_numeric, errors="coerce")
        self.medians_ = {col: float(numeric[col].median()) for col in self.numeric}
        self.feature_names_ = list(self.numeric)
        for column in self.categorical:
            self.feature_names_.extend([f"{column}__te", f"{column}__freq"])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("Encoder is not fitted")
        blocks: list[np.ndarray] = []
        numeric = frame[self.numeric].apply(pd.to_numeric, errors="coerce").copy()
        for col in self.numeric:
            numeric[col] = numeric[col].replace([np.inf, -np.inf], np.nan).fillna(self.medians_[col])
        blocks.append(numeric.to_numpy(dtype=np.float32))
        for column in self.categorical:
            mapping = self.maps_[column]
            enc = frame[column].map(mapping["encoded"]).fillna(self.global_mean_).to_numpy(dtype=np.float32)
            freq = frame[column].map(mapping["frequency"]).fillna(0.0).to_numpy(dtype=np.float32)
            blocks.append(enc[:, None])
            blocks.append(freq[:, None])
        return np.concatenate(blocks, axis=1)


def make_histgb(
    params: dict,
    seed: int,
    *,
    loss: str = "squared_error",
    quantile: float | None = None,
) -> HistGradientBoostingRegressor:
    kwargs = {}
    if loss == "quantile":
        if quantile is None:
            raise ValueError("quantile must be supplied for quantile loss")
        kwargs["quantile"] = float(quantile)
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=float(params.get("learning_rate", 0.06)),
        max_iter=int(params.get("max_iter", 220)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(params.get("min_samples_leaf", 30)),
        l2_regularization=float(params.get("l2_regularization", 1.0)),
        max_bins=int(params.get("max_bins", 255)),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=int(params.get("n_iter_no_change", 20)),
        random_state=int(seed),
        **kwargs,
    )


def make_histgb_classifier(params: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(params.get("learning_rate", 0.06)),
        max_iter=int(params.get("max_iter", 180)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(params.get("min_samples_leaf", 40)),
        l2_regularization=float(params.get("l2_regularization", 2.0)),
        max_bins=int(params.get("max_bins", 255)),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=int(params.get("n_iter_no_change", 20)),
        random_state=int(seed),
    )


class RawHistGB:
    def __init__(self, params: dict, seed: int):
        self.params = dict(params)
        self.seed = int(seed)
        self.model = make_histgb(self.params, self.seed)
        self.sigma_: float = 0.3

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        censored: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ):
        z = np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))
        self.model.fit(x, z, sample_weight=sample_weight)
        residual = z - self.model.predict(x)
        if censored is not None and np.sum(~np.asarray(censored, dtype=bool)) >= 50:
            residual = residual[~np.asarray(censored, dtype=bool)]
        self.sigma_ = float(max(np.std(residual), 0.05))
        return self

    def predict_log(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(self.predict_log(x)), 0.0, None)


class CensoredEMHistGB:
    """Auditable Censored-EM development model.

    It alternates between (a) fitting a nonlinear conditional mean in log space
    and (b) replacing stockout labels with the conditional expectation of a
    Gaussian log-demand truncated below log(observed sales + 1).

    This is deliberately simpler than the final distributional neural adapter.
    It tests whether respecting the lower-bound semantics is useful at all.
    """

    def __init__(self, params: dict, seed: int, em_iterations: int = 3):
        self.params = dict(params)
        self.seed = int(seed)
        self.em_iterations = int(em_iterations)
        self.model: HistGradientBoostingRegressor | None = None
        self.sigma_: float = 0.3
        self.history_: list[dict] = []

    @staticmethod
    def _truncated_normal_mean(mu: np.ndarray, lower: np.ndarray, sigma: float) -> np.ndarray:
        alpha = np.clip((lower - mu) / sigma, -12.0, 12.0)
        phi = np.exp(-0.5 * alpha**2) / np.sqrt(2 * np.pi)
        survival = np.clip(ndtr(-alpha), 1e-12, 1.0)
        return mu + sigma * phi / survival

    def fit(self, x: np.ndarray, y: np.ndarray, censored: np.ndarray):
        lower = np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))
        censored = np.asarray(censored, dtype=bool)
        target = lower.copy()
        upper_guard = float(np.quantile(lower[~censored] if np.any(~censored) else lower, 0.999) + 2.0)
        for iteration in range(self.em_iterations + 1):
            model = make_histgb(self.params, self.seed + iteration)
            model.fit(x, target)
            mu = model.predict(x)
            residual = lower[~censored] - mu[~censored]
            sigma = float(max(np.std(residual), 0.05)) if residual.size else 0.3
            violations = int(np.sum(mu[censored] < lower[censored]))
            self.history_.append(
                {
                    "iteration": iteration,
                    "sigma": sigma,
                    "censored_lower_bound_violations": violations,
                }
            )
            self.model = model
            self.sigma_ = sigma
            if iteration == self.em_iterations:
                break
            target = lower.copy()
            imputed = self._truncated_normal_mean(mu[censored], lower[censored], sigma)
            target[censored] = np.clip(np.maximum(imputed, lower[censored]), lower[censored], upper_guard)
        return self

    def predict_log(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return self.model.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(self.predict_log(x)), 0.0, None)


class QuantileHistGB:
    """Direct log-demand quantile used only for the replenishment decision."""

    def __init__(self, params: dict, seed: int, quantile: float):
        self.params = dict(params)
        self.seed = int(seed)
        self.quantile = float(quantile)
        self.model = make_histgb(
            self.params,
            self.seed,
            loss="quantile",
            quantile=self.quantile,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        z = np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))
        self.model.fit(x, z, sample_weight=sample_weight)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(self.model.predict(x)), 0.0, None)


class CensorRiskClassifier:
    """Predict future stockout/censoring risk from origin-available features."""

    def __init__(self, params: dict, seed: int):
        self.params = dict(params)
        self.seed = int(seed)
        self.model = make_histgb_classifier(self.params, self.seed)

    def fit(self, x: np.ndarray, censored: np.ndarray):
        target = np.asarray(censored, dtype=int)
        if np.unique(target).size < 2:
            raise ValueError("Censor-risk classifier requires both classes")
        self.model.fit(x, target)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict_proba(x)[:, 1], 0.0, 1.0)


def seasonal_naive(frame: pd.DataFrame) -> np.ndarray:
    return np.clip(frame["same_weekday_sale"].to_numpy(dtype=float), 0.0, None)


def select_recenter_factor(
    y_true: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray | None = None,
    grid: np.ndarray | None = None,
) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if eligible is not None:
        mask &= np.asarray(eligible, dtype=bool)
    if not np.any(mask):
        return 1.0
    if grid is None:
        grid = np.linspace(0.75, 1.40, 131)
    denominator = max(float(np.sum(y[mask])), 1e-12)
    scores = [float(np.sum(np.abs(y[mask] - pred[mask] * f)) / denominator) for f in grid]
    return float(grid[int(np.argmin(scores))])


@dataclass
class HorizonRecenterer:
    """Selection-only multiplicative calibration with partial pooling by horizon."""

    shrinkage: float = 350.0
    min_rows: int = 100
    global_factor_: float | None = None
    factors_: dict[int, float] | None = None

    def fit(
        self,
        y_true: np.ndarray,
        prediction: np.ndarray,
        horizons: np.ndarray,
        eligible: np.ndarray | None = None,
    ):
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(prediction, dtype=float)
        h = np.asarray(horizons, dtype=int)
        mask = np.isfinite(y) & np.isfinite(p)
        if eligible is not None:
            mask &= np.asarray(eligible, dtype=bool)
        self.global_factor_ = select_recenter_factor(y, p, mask)
        self.factors_ = {}
        for value in sorted(np.unique(h[mask])):
            local = mask & (h == value)
            n = int(np.sum(local))
            if n < self.min_rows:
                self.factors_[int(value)] = float(self.global_factor_)
                continue
            local_factor = select_recenter_factor(y, p, local)
            weight = n / (n + float(self.shrinkage))
            pooled = np.exp(
                (1.0 - weight) * np.log(self.global_factor_)
                + weight * np.log(local_factor)
            )
            self.factors_[int(value)] = float(pooled)
        return self

    def predict(self, prediction: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        if self.global_factor_ is None or self.factors_ is None:
            raise RuntimeError("HorizonRecenterer is not fitted")
        p = np.asarray(prediction, dtype=float)
        h = np.asarray(horizons, dtype=int)
        factors = np.asarray(
            [self.factors_.get(int(value), self.global_factor_) for value in h],
            dtype=float,
        )
        return np.clip(p * factors, 0.0, None)

    def to_dict(self) -> dict:
        return {
            "global": self.global_factor_,
            "by_horizon": {str(k): v for k, v in (self.factors_ or {}).items()},
            "shrinkage": float(self.shrinkage),
        }


@dataclass
class CensorGatedBlend:
    """Use the censor-aware expert only where predicted censoring risk supports it.

    The correction is one-sided in log space: missing demand may raise the
    uncensored anchor, but the censoring component is not allowed to lower it.
    Strength and gate power are selected on the selection split only. Strength
    zero is an explicit safe fallback to the anchor.
    """

    strengths: tuple[float, ...] = tuple(np.linspace(0.0, 1.5, 31))
    powers: tuple[float, ...] = (0.5, 1.0, 2.0)
    strength_: float | None = None
    power_: float | None = None
    selection_wape_: float | None = None
    top_candidates_: list[dict] | None = None

    @staticmethod
    def _apply(
        anchor: np.ndarray,
        expert: np.ndarray,
        censor_probability: np.ndarray,
        strength: float,
        power: float,
    ) -> np.ndarray:
        a = np.log1p(np.clip(np.asarray(anchor, dtype=float), 0.0, None))
        e = np.log1p(np.clip(np.asarray(expert, dtype=float), 0.0, None))
        gate = np.clip(np.asarray(censor_probability, dtype=float), 0.0, 1.0) ** float(power)
        positive_gap = np.maximum(e - a, 0.0)
        return np.clip(np.expm1(a + float(strength) * gate * positive_gap), 0.0, None)

    def fit(
        self,
        y_true: np.ndarray,
        anchor: np.ndarray,
        expert: np.ndarray,
        censor_probability: np.ndarray,
        eligible: np.ndarray | None = None,
    ):
        y = np.asarray(y_true, dtype=float)
        mask = np.isfinite(y)
        if eligible is not None:
            mask &= np.asarray(eligible, dtype=bool)
        denominator = max(float(np.sum(np.abs(y[mask]))), 1e-12)
        candidates: list[dict] = []
        for power in self.powers:
            for strength in self.strengths:
                prediction = self._apply(anchor, expert, censor_probability, strength, power)
                score = float(np.sum(np.abs(y[mask] - prediction[mask])) / denominator)
                candidates.append(
                    {"strength": float(strength), "power": float(power), "wape": score}
                )
        candidates.sort(key=lambda row: (row["wape"], row["strength"], row["power"]))
        selected = candidates[0]
        self.strength_ = selected["strength"]
        self.power_ = selected["power"]
        self.selection_wape_ = selected["wape"]
        self.top_candidates_ = candidates[:10]
        return self

    def predict(
        self,
        anchor: np.ndarray,
        expert: np.ndarray,
        censor_probability: np.ndarray,
    ) -> np.ndarray:
        if self.strength_ is None or self.power_ is None:
            raise RuntimeError("CensorGatedBlend is not fitted")
        return self._apply(
            anchor,
            expert,
            censor_probability,
            self.strength_,
            self.power_,
        )

    def to_dict(self) -> dict:
        return {
            "strength": self.strength_,
            "power": self.power_,
            "selection_wape": self.selection_wape_,
            "uses_censor_signal": bool(self.strength_ is not None and self.strength_ > 0.0),
            "top_candidates": self.top_candidates_ or [],
        }


class ResidualRiskRegressor:
    """Selection-trained heteroscedastic scale and deployable error score."""

    def __init__(self, params: dict, seed: int):
        compact = dict(params)
        compact["max_iter"] = min(int(compact.get("max_iter", 180)), 180)
        compact["max_leaf_nodes"] = min(int(compact.get("max_leaf_nodes", 31)), 31)
        compact["min_samples_leaf"] = max(int(compact.get("min_samples_leaf", 30)), 30)
        self.scale_model = make_histgb(compact, seed)
        self.risk_model = make_histgb(compact, seed + 1)
        self.scale_floor_: float | None = None
        self.scale_ceiling_: float | None = None

    def fit(
        self,
        features: np.ndarray,
        y_true: np.ndarray,
        prediction: np.ndarray,
        eligible: np.ndarray | None = None,
    ):
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(prediction, dtype=float)
        mask = np.isfinite(y) & np.isfinite(p) & np.all(np.isfinite(x), axis=1)
        if eligible is not None:
            mask &= np.asarray(eligible, dtype=bool)
        log_error = np.abs(np.log1p(y[mask]) - np.log1p(np.clip(p[mask], 0.0, None)))
        relative_error = np.abs(y[mask] - p[mask]) / (np.abs(y[mask]) + 1.0)
        self.scale_floor_ = float(max(np.quantile(log_error, 0.05), 1e-3))
        self.scale_ceiling_ = float(max(np.quantile(log_error, 0.995), self.scale_floor_ * 2))
        self.scale_model.fit(x[mask], np.log(np.clip(log_error, 1e-4, None)))
        self.risk_model.fit(x[mask], np.log1p(np.clip(relative_error, 0.0, 20.0)))
        return self

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.scale_floor_ is None or self.scale_ceiling_ is None:
            raise RuntimeError("ResidualRiskRegressor is not fitted")
        x = np.asarray(features, dtype=np.float32)
        scale = np.exp(self.scale_model.predict(x))
        scale = np.clip(scale, self.scale_floor_, self.scale_ceiling_)
        risk = np.expm1(self.risk_model.predict(x))
        return scale, np.clip(risk, 0.0, None)


@dataclass
class SplitConformalLogInterval:
    alpha: float = 0.2
    radius_: float | None = None

    def fit(self, y_true: np.ndarray, prediction: np.ndarray, eligible: np.ndarray | None = None):
        y = np.asarray(y_true, dtype=float)
        pred = np.asarray(prediction, dtype=float)
        mask = np.isfinite(y) & np.isfinite(pred)
        if eligible is not None:
            mask &= np.asarray(eligible, dtype=bool)
        residual = np.abs(np.log1p(y[mask]) - np.log1p(np.clip(pred[mask], 0.0, None)))
        n = len(residual)
        if n < 20:
            raise ValueError(f"Need at least 20 eligible calibration rows, got {n}")
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.radius_ = float(np.quantile(residual, level, method="higher"))
        return self

    def predict(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.radius_ is None:
            raise RuntimeError("Interval is not calibrated")
        center = np.log1p(np.clip(np.asarray(prediction, dtype=float), 0.0, None))
        lo = np.clip(np.expm1(center - self.radius_), 0.0, None)
        hi = np.clip(np.expm1(center + self.radius_), 0.0, None)
        return lo, hi


@dataclass
class ScaledMondrianConformalLogInterval:
    """Horizon-conditional conformal interval with learned local scale.

    ``coverage_margin`` deliberately calibrates above the nominal target to
    absorb the one-week temporal shift observed in v0.1. The guarantee remains
    conditional on exchangeability within each horizon; the shadow audit is
    reported separately and is not described as a formal guarantee.
    """

    alpha: float = 0.2
    coverage_margin: float = 0.04
    min_group: int = 100
    global_quantile_: float | None = None
    quantiles_: dict[int, float] | None = None

    @staticmethod
    def _finite_sample_quantile(values: np.ndarray, target: float) -> float:
        n = len(values)
        if n < 20:
            raise ValueError(f"Need at least 20 calibration rows, got {n}")
        level = min(np.ceil((n + 1) * target) / n, 1.0)
        return float(np.quantile(values, level, method="higher"))

    def fit(
        self,
        y_true: np.ndarray,
        prediction: np.ndarray,
        scale: np.ndarray,
        horizons: np.ndarray,
        eligible: np.ndarray | None = None,
    ):
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(prediction, dtype=float)
        s = np.clip(np.asarray(scale, dtype=float), 1e-4, None)
        h = np.asarray(horizons, dtype=int)
        mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(s)
        if eligible is not None:
            mask &= np.asarray(eligible, dtype=bool)
        score = np.abs(np.log1p(y) - np.log1p(np.clip(p, 0.0, None))) / s
        target = min(1.0 - self.alpha + self.coverage_margin, 0.995)
        self.global_quantile_ = self._finite_sample_quantile(score[mask], target)
        self.quantiles_ = {}
        for value in sorted(np.unique(h[mask])):
            local = mask & (h == value)
            if int(np.sum(local)) < self.min_group:
                self.quantiles_[int(value)] = float(self.global_quantile_)
            else:
                self.quantiles_[int(value)] = self._finite_sample_quantile(score[local], target)
        return self

    def predict(
        self,
        prediction: np.ndarray,
        scale: np.ndarray,
        horizons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.global_quantile_ is None or self.quantiles_ is None:
            raise RuntimeError("Scaled conformal interval is not calibrated")
        p = np.asarray(prediction, dtype=float)
        s = np.asarray(scale, dtype=float)
        h = np.asarray(horizons, dtype=int)
        q = np.asarray(
            [self.quantiles_.get(int(value), self.global_quantile_) for value in h],
            dtype=float,
        )
        center = np.log1p(np.clip(p, 0.0, None))
        radius = np.clip(q * s, 1e-4, 5.0)
        lo = np.clip(np.expm1(center - radius), 0.0, None)
        hi = np.clip(np.expm1(center + radius), 0.0, None)
        return lo, hi

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "coverage_margin": float(self.coverage_margin),
            "effective_target": float(min(1.0 - self.alpha + self.coverage_margin, 0.995)),
            "global_quantile": self.global_quantile_,
            "by_horizon": {str(k): v for k, v in (self.quantiles_ or {}).items()},
        }


class WAPEAlignedRiskRegressor:
    """Predict deployable WAPE contribution and a local log-error scale.

    v0.2 learned row-wise relative error using the unavailable future outcome
    in the denominator, while policy quality was judged by aggregate WAPE.
    v0.3 instead learns the absolute-error numerator and divides it by the
    deployable point forecast.  Demand-weighted fitting keeps the ranking
    aligned with the mass-weighted operational contract.
    """

    def __init__(self, params: dict, seed: int):
        compact = dict(params)
        compact["max_iter"] = min(int(compact.get("max_iter", 180)), 180)
        compact["max_leaf_nodes"] = min(int(compact.get("max_leaf_nodes", 31)), 31)
        compact["min_samples_leaf"] = max(int(compact.get("min_samples_leaf", 30)), 30)
        self.scale_model = make_histgb(compact, seed)
        self.absolute_error_model = make_histgb(compact, seed + 1)
        self.scale_floor_: float | None = None
        self.scale_ceiling_: float | None = None
        self.mass_floor_: float | None = None

    def fit(
        self,
        features: np.ndarray,
        y_true: np.ndarray,
        prediction: np.ndarray,
        eligible: np.ndarray | None = None,
    ):
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(prediction, dtype=float)
        mask = np.isfinite(y) & np.isfinite(p) & np.all(np.isfinite(x), axis=1)
        if eligible is not None:
            mask &= np.asarray(eligible, dtype=bool)
        if int(np.sum(mask)) < 100:
            raise ValueError("WAPE risk model needs at least 100 eligible rows")
        absolute_error = np.abs(y[mask] - p[mask])
        log_error = np.abs(
            np.log1p(np.clip(y[mask], 0.0, None))
            - np.log1p(np.clip(p[mask], 0.0, None))
        )
        demand_mass = np.clip(np.abs(y[mask]), 0.0, None) + 1.0
        weights = np.sqrt(demand_mass)
        weights = np.clip(weights, 0.25, np.quantile(weights, 0.995))
        weights = weights / max(float(np.mean(weights)), 1e-12)
        positive_prediction = np.clip(p[mask], 0.0, None)
        self.mass_floor_ = float(max(np.quantile(positive_prediction, 0.10), 0.25))
        self.scale_floor_ = float(max(np.quantile(log_error, 0.05), 1e-3))
        self.scale_ceiling_ = float(
            max(np.quantile(log_error, 0.995), self.scale_floor_ * 2)
        )
        self.scale_model.fit(
            x[mask],
            np.log(np.clip(log_error, 1e-4, None)),
            sample_weight=weights,
        )
        self.absolute_error_model.fit(
            x[mask],
            np.log1p(np.clip(absolute_error, 0.0, None)),
            sample_weight=weights,
        )
        return self

    def predict(
        self,
        features: np.ndarray,
        prediction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.scale_floor_ is None
            or self.scale_ceiling_ is None
            or self.mass_floor_ is None
        ):
            raise RuntimeError("WAPE-aligned risk model is not fitted")
        x = np.asarray(features, dtype=np.float32)
        point = np.clip(np.asarray(prediction, dtype=float), 0.0, None)
        scale = np.exp(self.scale_model.predict(x))
        scale = np.clip(scale, self.scale_floor_, self.scale_ceiling_)
        predicted_absolute_error = np.clip(
            np.expm1(self.absolute_error_model.predict(x)), 0.0, None
        )
        predicted_demand_mass = np.maximum(point, self.mass_floor_)
        risk = predicted_absolute_error / predicted_demand_mass
        return scale, np.clip(risk, 0.0, None), predicted_absolute_error

    def to_dict(self) -> dict:
        return {
            "target": "absolute_error_numerator",
            "deployable_denominator": "max(point_forecast, mass_floor)",
            "demand_weighting": "sqrt(abs(y)+1), clipped_at_q0.995",
            "mass_floor": self.mass_floor_,
            "scale_floor": self.scale_floor_,
            "scale_ceiling": self.scale_ceiling_,
        }


@dataclass
class RollingBlockScaledConformalLogInterval:
    """Conservative conformal interval across two ordered calibration blocks.

    Each weekly block receives its own finite-sample quantile.  The deployed
    radius is the maximum across blocks, globally and by horizon, so one easy
    calibration week cannot hide failure on the adjacent week.  This is a
    development-time robustness rule, not a claim of coverage under arbitrary
    temporal shift.
    """

    alpha: float = 0.2
    coverage_margin: float = 0.04
    min_group: int = 100
    global_quantile_: float | None = None
    quantiles_: dict[int, float] | None = None
    block_quantiles_: dict[str, dict] | None = None

    @staticmethod
    def _finite_sample_quantile(values: np.ndarray, target: float) -> float:
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n < 20:
            raise ValueError(f"Need at least 20 calibration rows, got {n}")
        level = min(np.ceil((n + 1) * target) / n, 1.0)
        return float(np.quantile(values, level, method="higher"))

    def fit(self, blocks: dict[str, dict]):
        target = min(1.0 - self.alpha + self.coverage_margin, 0.995)
        block_results: dict[str, dict] = {}
        all_horizons: set[int] = set()
        for name, block in blocks.items():
            y = np.asarray(block["y_true"], dtype=float)
            p = np.asarray(block["prediction"], dtype=float)
            s = np.clip(np.asarray(block["scale"], dtype=float), 1e-4, None)
            h = np.asarray(block["horizons"], dtype=int)
            mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(s)
            if block.get("eligible") is not None:
                mask &= np.asarray(block["eligible"], dtype=bool)
            score = np.abs(
                np.log1p(np.clip(y, 0.0, None))
                - np.log1p(np.clip(p, 0.0, None))
            ) / s
            global_q = self._finite_sample_quantile(score[mask], target)
            by_horizon: dict[int, float] = {}
            for value in sorted(np.unique(h[mask])):
                all_horizons.add(int(value))
                local = mask & (h == value)
                by_horizon[int(value)] = (
                    self._finite_sample_quantile(score[local], target)
                    if int(np.sum(local)) >= self.min_group
                    else global_q
                )
            block_results[str(name)] = {
                "n": int(np.sum(mask)),
                "global": float(global_q),
                "by_horizon": by_horizon,
            }
        if len(block_results) < 2:
            raise ValueError("Rolling-block conformal requires at least two blocks")
        self.global_quantile_ = float(
            max(block["global"] for block in block_results.values())
        )
        self.quantiles_ = {
            value: float(
                max(
                    block["by_horizon"].get(value, block["global"])
                    for block in block_results.values()
                )
            )
            for value in sorted(all_horizons)
        }
        self.block_quantiles_ = block_results
        return self

    def predict(
        self,
        prediction: np.ndarray,
        scale: np.ndarray,
        horizons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.global_quantile_ is None or self.quantiles_ is None:
            raise RuntimeError("Rolling-block conformal interval is not calibrated")
        p = np.clip(np.asarray(prediction, dtype=float), 0.0, None)
        s = np.clip(np.asarray(scale, dtype=float), 1e-4, None)
        h = np.asarray(horizons, dtype=int)
        q = np.asarray(
            [self.quantiles_.get(int(value), self.global_quantile_) for value in h],
            dtype=float,
        )
        center = np.log1p(p)
        radius = np.clip(q * s, 1e-4, 5.0)
        return (
            np.clip(np.expm1(center - radius), 0.0, None),
            np.clip(np.expm1(center + radius), 0.0, None),
        )

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "coverage_margin": float(self.coverage_margin),
            "effective_target": float(
                min(1.0 - self.alpha + self.coverage_margin, 0.995)
            ),
            "aggregation": "maximum_across_ordered_calibration_blocks",
            "global_quantile": self.global_quantile_,
            "by_horizon": {str(k): v for k, v in (self.quantiles_ or {}).items()},
            "blocks": self.block_quantiles_ or {},
        }


def select_cost_scale_factor(
    y_true: np.ndarray,
    order: np.ndarray,
    eligible: np.ndarray | None,
    underage_cost: float,
    overage_cost: float,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(order, dtype=float)
    mask = np.isfinite(y) & np.isfinite(q)
    if eligible is not None:
        mask &= np.asarray(eligible, dtype=bool)
    if grid is None:
        grid = np.linspace(0.70, 1.60, 181)
    best_factor = 1.0
    best_cost = float("inf")
    for factor in grid:
        candidate = np.clip(q[mask] * float(factor), 0.0, None)
        cost = np.mean(
            float(underage_cost) * np.maximum(y[mask] - candidate, 0.0)
            + float(overage_cost) * np.maximum(candidate - y[mask], 0.0)
        )
        if cost < best_cost:
            best_factor = float(factor)
            best_cost = float(cost)
    return best_factor, best_cost
