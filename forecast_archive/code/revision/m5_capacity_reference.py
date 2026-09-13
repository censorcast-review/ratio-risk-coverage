"""Verbatim capacity function from the historical M5 preparation source."""
import numpy as np

def simulate_controlled_censoring(
    truth: np.ndarray,
    *,
    warmup_days: int = 56,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen recursive inventory-cap mechanism.

    The rule is prequential: capacity at t depends only on observed values no
    later than t-1.  The first warmup window is deliberately uncensored.
    """
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[1] <= warmup_days:
        raise ValueError("truth must be a series-by-day matrix longer than warmup")
    if np.any(~np.isfinite(truth)) or np.any(truth < 0):
        raise ValueError("M5 unit sales must be finite and non-negative")
    truth_i = truth.astype(np.int32, copy=False)
    observed = np.empty_like(truth_i, dtype=np.int32)
    capacity = np.empty_like(truth_i, dtype=np.int32)
    censored = np.zeros(truth_i.shape, dtype=np.uint8)
    observed[:, :warmup_days] = truth_i[:, :warmup_days]
    capacity[:, :warmup_days] = np.maximum(truth_i[:, :warmup_days], 1)
    level = observed[:, :warmup_days].mean(axis=1, dtype=np.float64)
    for day in range(warmup_days, truth_i.shape[1]):
        level = 0.90 * level + 0.10 * observed[:, day - 1]
        cap = np.maximum(
            1,
            np.ceil(1.15 * level + 0.25 * np.sqrt(level + 1.0)).astype(np.int32),
        )
        capacity[:, day] = cap
        current = truth_i[:, day]
        observed[:, day] = np.minimum(current, cap)
        censored[:, day] = (current > cap).astype(np.uint8)
    return observed, capacity, censored

