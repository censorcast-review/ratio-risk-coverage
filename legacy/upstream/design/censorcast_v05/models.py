from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _regressor(parameters: dict, seed: int, *, loss: str, quantile: float | None = None):
    extra = {}
    if loss == "quantile":
        if quantile is None:
            raise ValueError("quantile is required for quantile loss")
        extra["quantile"] = float(quantile)
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=float(parameters["learning_rate"]),
        max_iter=int(parameters["max_iter"]),
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        l2_regularization=float(parameters["l2_regularization"]),
        n_iter_no_change=int(parameters["n_iter_no_change"]),
        early_stopping=True,
        validation_fraction=0.10,
        random_state=int(seed),
        **extra,
    )


def _classifier(parameters: dict, seed: int):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(parameters["learning_rate"]),
        max_iter=int(parameters["max_iter"]),
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        l2_regularization=float(parameters["l2_regularization"]),
        n_iter_no_change=int(parameters["n_iter_no_change"]),
        early_stopping=True,
        validation_fraction=0.10,
        random_state=int(seed),
    )


def censored_poisson_conditional_mean(
    mean: np.ndarray,
    capacity: np.ndarray,
    censored: np.ndarray,
    *,
    cap_multiplier: float,
) -> np.ndarray:
    """E[Y | Y > capacity] under Poisson(mean), with stable tail handling."""
    lam = np.clip(np.asarray(mean, dtype=np.float64), 1e-6, 1e7)
    cap = np.maximum(np.rint(np.asarray(capacity, dtype=np.float64)), 0.0)
    flag = np.asarray(censored, dtype=bool)
    result = np.maximum(lam, 0.0)
    if not np.any(flag):
        return result.astype(np.float32)
    c = cap[flag]
    l = lam[flag]
    log_numerator = np.log(l) + poisson.logsf(c - 1.0, l)
    log_denominator = poisson.logsf(c, l)
    conditional = np.exp(np.clip(log_numerator - log_denominator, -30.0, 30.0))
    fallback = np.maximum(c + 1.0, l)
    conditional = np.where(np.isfinite(conditional), conditional, fallback)
    upper = np.maximum(c + 1.0, float(cap_multiplier) * np.maximum(c, 1.0) + 10.0)
    result[flag] = np.clip(conditional, c + 1.0, upper)
    return result.astype(np.float32)


@dataclass
class ForecastBundle:
    raw_model: HistGradientBoostingRegressor
    em_model: HistGradientBoostingRegressor
    censor_model: HistGradientBoostingClassifier
    em_iterations: int
    cap_multiplier: float

    @classmethod
    def fit(
        cls,
        common_x: np.ndarray,
        censor_x: np.ndarray,
        observed: np.ndarray,
        capacity: np.ndarray,
        censored: np.ndarray,
        config: dict,
        seed: int,
    ) -> "ForecastBundle":
        observed = np.clip(np.asarray(observed, dtype=np.float64), 0.0, None)
        censored_bool = np.asarray(censored, dtype=bool)
        raw = _regressor(config["histgb"], seed, loss="poisson")
        raw.fit(common_x, observed)
        current = np.clip(raw.predict(common_x), 0.0, None)
        iterations = int(config["censored_poisson_em_iterations"])
        cap_multiplier = float(config["em_conditional_mean_cap_multiplier"])
        em = None
        for iteration in range(iterations):
            pseudo = observed.astype(np.float32, copy=True)
            conditional = censored_poisson_conditional_mean(
                current,
                capacity,
                censored_bool,
                cap_multiplier=cap_multiplier,
            )
            pseudo[censored_bool] = conditional[censored_bool]
            em = _regressor(config["histgb"], seed + 101 + iteration, loss="poisson")
            em.fit(censor_x, np.clip(pseudo, 0.0, None))
            current = np.clip(em.predict(censor_x), 0.0, None)
        if em is None:
            raise AssertionError("At least one EM iteration is required")
        censor_model = _classifier(config["censor_classifier"], seed + 701)
        censor_model.fit(censor_x, censored_bool.astype(np.uint8))
        return cls(raw, em, censor_model, iterations, cap_multiplier)

    def predict(self, common_x: np.ndarray, censor_x: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "raw_poisson_histgb": np.clip(self.raw_model.predict(common_x), 0.0, None).astype(np.float32),
            "censored_poisson_em": np.clip(self.em_model.predict(censor_x), 0.0, None).astype(np.float32),
            "censor_probability": np.clip(self.censor_model.predict_proba(censor_x)[:, 1], 0.0, 1.0).astype(np.float32),
        }


@dataclass
class SupportProfile:
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "SupportProfile":
        values = np.asarray(x, dtype=np.float32)
        median = np.median(values, axis=0).astype(np.float32)
        q25 = np.quantile(values, 0.25, axis=0).astype(np.float32)
        q75 = np.quantile(values, 0.75, axis=0).astype(np.float32)
        scale = np.maximum(q75 - q25, 1e-3).astype(np.float32)
        return cls(median=median, scale=scale)

    def distance(self, x: np.ndarray) -> np.ndarray:
        z = np.abs((np.asarray(x, dtype=np.float32) - self.median) / self.scale)
        return np.mean(np.minimum(z, 20.0), axis=1).astype(np.float32)


def select_point_policy(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    censor_probability: np.ndarray,
    em_prediction: np.ndarray,
    strengths: list[float],
    powers: list[float],
) -> dict:
    y = np.asarray(truth, dtype=np.float64).ravel()

    def wape(prediction: np.ndarray) -> float:
        p = np.asarray(prediction, dtype=np.float64).ravel()
        return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-12))

    baseline_rows = [
        {"name": name, "selection_wape": wape(prediction)}
        for name, prediction in predictions.items()
    ]
    baseline_rows.sort(key=lambda row: (row["selection_wape"], row["name"]))
    baseline_name = baseline_rows[0]["name"]
    baseline = np.asarray(predictions[baseline_name], dtype=np.float32)
    correction = np.maximum(np.asarray(em_prediction, dtype=np.float32) - baseline, 0.0)
    probability = np.asarray(censor_probability, dtype=np.float32)
    adapter_rows = []
    for strength in strengths:
        for power in powers:
            proposal = baseline + float(strength) * np.power(probability, float(power)) * correction
            adapter_rows.append(
                {
                    "strength": float(strength),
                    "power": float(power),
                    "selection_wape": wape(proposal),
                }
            )
    adapter_rows.sort(key=lambda row: (row["selection_wape"], row["strength"], row["power"]))
    chosen = adapter_rows[0]
    return {
        "selected_baseline": baseline_name,
        "baseline_candidates": baseline_rows,
        "baseline_selection_wape": float(baseline_rows[0]["selection_wape"]),
        "adapter": chosen,
        "adapter_candidates": adapter_rows,
        "uses_censor_signal": bool(chosen["strength"] > 0.0),
    }


def apply_point_policy(
    policy: dict,
    predictions: dict[str, np.ndarray],
    em_prediction: np.ndarray,
    censor_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(predictions[policy["selected_baseline"]], dtype=np.float32)
    adapter = policy["adapter"]
    correction = np.maximum(np.asarray(em_prediction, dtype=np.float32) - baseline, 0.0)
    proposal = baseline + float(adapter["strength"]) * np.power(
        np.asarray(censor_probability, dtype=np.float32), float(adapter["power"])
    ) * correction
    return baseline.astype(np.float32), np.clip(proposal, 0.0, None).astype(np.float32)


def risk_feature_matrix(
    *,
    baseline: np.ndarray,
    proposal: np.ndarray,
    seasonal_naive: np.ndarray,
    seasonal_median: np.ndarray,
    raw: np.ndarray,
    em: np.ndarray,
    censor_probability: np.ndarray,
    common_features: np.ndarray,
    censor_features: np.ndarray,
    prior_observed_wape: np.ndarray,
    support_distance: np.ndarray,
) -> np.ndarray:
    common = np.asarray(common_features, dtype=np.float32)
    censor = np.asarray(censor_features, dtype=np.float32)
    # Named columns from FeatureBuilder: horizon, mean_28, std_28,
    # zero_fraction_28, price ratios, event; censor-rate columns trail common.
    derived = np.column_stack(
        [
            np.log1p(np.clip(baseline, 0.0, None)),
            np.log1p(np.clip(proposal, 0.0, None)),
            np.log1p(np.clip(raw, 0.0, None)),
            np.log1p(np.clip(em, 0.0, None)),
            np.log1p(np.clip(seasonal_naive, 0.0, None)),
            np.log1p(np.clip(seasonal_median, 0.0, None)),
            censor_probability,
            np.abs(np.log1p(np.clip(em, 0.0, None)) - np.log1p(np.clip(raw, 0.0, None))),
            np.std(
                np.column_stack(
                    [
                        np.log1p(np.clip(seasonal_naive, 0.0, None)),
                        np.log1p(np.clip(seasonal_median, 0.0, None)),
                        np.log1p(np.clip(raw, 0.0, None)),
                        np.log1p(np.clip(em, 0.0, None)),
                    ]
                ),
                axis=1,
            ),
            prior_observed_wape,
            support_distance,
            common[:, 0],
            common[:, 7],
            common[:, 10],
            common[:, 11],
            common[:, 14],
            common[:, 15],
            common[:, 20],
            common[:, 22],
            censor[:, -5],
            censor[:, -4],
            censor[:, -2],
        ]
    ).astype(np.float32)
    return np.nan_to_num(derived, nan=0.0, posinf=10_000.0, neginf=-10_000.0)


@dataclass
class RiskModel:
    model: HistGradientBoostingRegressor
    quantile: float

    @classmethod
    def fit(cls, x: np.ndarray, absolute_error: np.ndarray, config: dict, seed: int):
        target = np.asarray(absolute_error, dtype=np.float64)
        finite = np.isfinite(target)
        if finite.sum() < 100:
            raise RuntimeError("Insufficient finite risk-training targets")
        cap = float(np.quantile(target[finite], 0.9995))
        target = np.clip(target, 0.0, max(cap, 1.0))
        model = _regressor(
            config["histgb"],
            seed,
            loss="quantile",
            quantile=float(config["quantile"]),
        )
        model.fit(np.asarray(x, dtype=np.float32), target)
        return cls(model=model, quantile=float(config["quantile"]))

    def predict_absolute_error(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(np.asarray(x, dtype=np.float32)), 0.0, None).astype(np.float32)

