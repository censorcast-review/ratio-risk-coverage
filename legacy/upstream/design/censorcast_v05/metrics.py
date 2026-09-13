from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score


def wape(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-12))


def wpe(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    return float((p - y).sum() / max(np.abs(y).sum(), 1e-12))


def metric_bundle(y: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    return {
        "n": int(len(y)),
        "wape": wape(y, p),
        "wpe": wpe(y, p),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


def _prior_within_cluster(values: np.ndarray, clusters_in_score_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly prior cumulative value and row count within cluster."""
    n = len(values)
    group_order = np.argsort(clusters_in_score_order, kind="stable")
    group_cluster = clusters_in_score_order[group_order]
    grouped = np.asarray(values, dtype=np.float64)[group_order]
    starts = np.r_[0, np.flatnonzero(group_cluster[1:] != group_cluster[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    cumulative = np.cumsum(grouped, dtype=np.float64)
    offsets = np.zeros(len(starts), dtype=np.float64)
    if len(starts) > 1:
        offsets[1:] = cumulative[starts[1:] - 1]
    repeated_offsets = np.repeat(offsets, ends - starts)
    prior_grouped = cumulative - grouped - repeated_offsets
    prior_count_grouped = np.arange(n, dtype=np.int64) - np.repeat(starts, ends - starts)
    prior = np.empty(n, dtype=np.float64)
    prior_count = np.empty(n, dtype=np.int64)
    prior[group_order] = prior_grouped
    prior_count[group_order] = prior_count_grouped
    return prior, prior_count


@dataclass
class ExactCurve:
    threshold: np.ndarray
    coverage: np.ndarray
    coverage_lcb95: np.ndarray
    accepted_wape: np.ndarray
    accepted_wape_ratio: np.ndarray
    accepted_wape_ratio_ucb95: np.ndarray
    accepted_n: np.ndarray
    cluster_count: int

    def at(self, thresholds: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(thresholds, dtype=np.float64)
        index = np.searchsorted(self.threshold, values, side="right") - 1
        present = index >= 0
        safe = np.maximum(index, 0)
        result = {}
        for name in (
            "coverage",
            "coverage_lcb95",
            "accepted_wape",
            "accepted_wape_ratio",
            "accepted_wape_ratio_ucb95",
            "accepted_n",
        ):
            source = np.asarray(getattr(self, name))
            fill = 0.0 if name in {"coverage", "coverage_lcb95", "accepted_n"} else np.inf
            output = np.full(len(values), fill, dtype=np.float64)
            output[present] = source[safe[present]]
            result[name] = output
        return result


def exact_item_cluster_delta_curve(
    *,
    score: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    item_cluster: np.ndarray,
    baseline_selection_wape: float,
) -> ExactCurve:
    score = np.asarray(score, dtype=np.float64).ravel()
    y = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    cluster = np.asarray(item_cluster, dtype=np.int32).ravel()
    valid = np.isfinite(score) & np.isfinite(y) & np.isfinite(p) & (y >= 0)
    score, y, p, cluster = score[valid], y[valid], p[valid], cluster[valid]
    if len(score) < 100 or len(np.unique(cluster)) < 10:
        raise RuntimeError("Insufficient rows or item clusters for exact frontier")
    _, cluster = np.unique(cluster, return_inverse=True)
    cluster = cluster.astype(np.int32)
    cluster_count = int(cluster.max()) + 1
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    yy = np.abs(y[order])
    error = np.abs(y[order] - p[order])
    cc = cluster[order]
    total_by_cluster = np.bincount(cluster, minlength=cluster_count).astype(np.float64)
    total = float(total_by_cluster.sum())

    prior_error, prior_count = _prior_within_cluster(error, cc)
    prior_demand, _ = _prior_within_cluster(yy, cc)
    delta_a2 = 2.0 * prior_count + 1.0
    sum_a2 = np.cumsum(delta_a2, dtype=np.float64)
    sum_at = np.cumsum(total_by_cluster[cc], dtype=np.float64)
    sum_t2 = float(np.square(total_by_cluster).sum())

    delta_e2 = 2.0 * prior_error * error + error * error
    delta_ed = prior_error * yy + prior_demand * error + error * yy
    delta_d2 = 2.0 * prior_demand * yy + yy * yy
    sum_e2 = np.cumsum(delta_e2, dtype=np.float64)
    sum_ed = np.cumsum(delta_ed, dtype=np.float64)
    sum_d2 = np.cumsum(delta_d2, dtype=np.float64)
    cumulative_error = np.cumsum(error, dtype=np.float64)
    cumulative_demand = np.cumsum(yy, dtype=np.float64)
    accepted = np.arange(1, len(s) + 1, dtype=np.float64)

    coverage = accepted / total
    coverage_u2 = np.maximum(sum_a2 - 2.0 * coverage * sum_at + coverage * coverage * sum_t2, 0.0)
    correction = cluster_count / max(cluster_count - 1.0, 1.0)
    coverage_se = np.sqrt(correction * coverage_u2) / total
    coverage_lcb = np.clip(coverage - norm.ppf(0.95) * coverage_se, 0.0, 1.0)

    accepted_wape = cumulative_error / np.maximum(cumulative_demand, 1e-12)
    wape_u2 = np.maximum(
        sum_e2 - 2.0 * accepted_wape * sum_ed + accepted_wape * accepted_wape * sum_d2,
        0.0,
    )
    wape_se = np.sqrt(correction * wape_u2) / np.maximum(cumulative_demand, 1e-12)
    ratio = accepted_wape / max(float(baseline_selection_wape), 1e-12)
    ratio_ucb = (accepted_wape + norm.ppf(0.95) * wape_se) / max(
        float(baseline_selection_wape), 1e-12
    )

    tie_end = np.r_[s[1:] != s[:-1], True]
    return ExactCurve(
        threshold=s[tie_end],
        coverage=coverage[tie_end],
        coverage_lcb95=coverage_lcb[tie_end],
        accepted_wape=accepted_wape[tie_end],
        accepted_wape_ratio=ratio[tie_end],
        accepted_wape_ratio_ucb95=ratio_ucb[tie_end],
        accepted_n=accepted[tie_end].astype(np.int64),
        cluster_count=cluster_count,
    )


def search_two_block_exact_threshold(
    curve_a: ExactCurve,
    curve_b: ExactCurve,
    *,
    maximum_ratio_ucb: float,
    minimum_coverage_lcb: float,
    downsample_points: int = 401,
) -> tuple[dict, dict[str, np.ndarray]]:
    thresholds = np.unique(np.concatenate([curve_a.threshold, curve_b.threshold]))
    a = curve_a.at(thresholds)
    b = curve_b.at(thresholds)
    minimum_lcb = np.minimum(a["coverage_lcb95"], b["coverage_lcb95"])
    minimum_coverage = np.minimum(a["coverage"], b["coverage"])
    maximum_ucb = np.maximum(a["accepted_wape_ratio_ucb95"], b["accepted_wape_ratio_ucb95"])
    feasible = (minimum_lcb >= float(minimum_coverage_lcb)) & (
        maximum_ucb <= float(maximum_ratio_ucb)
    )
    violation = np.maximum(float(minimum_coverage_lcb) - minimum_lcb, 0.0) + np.maximum(
        maximum_ucb - float(maximum_ratio_ucb), 0.0
    )
    if np.any(feasible):
        eligible = np.flatnonzero(feasible)
        ranking = np.lexsort((thresholds[eligible], maximum_ucb[eligible], -minimum_coverage[eligible]))
        chosen_index = int(eligible[ranking[0]])
        status = "EXACT_MARGIN_FEASIBLE_THRESHOLD_FOUND"
    else:
        finite = np.isfinite(violation)
        if not np.any(finite):
            raise RuntimeError("No finite exact-threshold diagnostics")
        eligible = np.flatnonzero(finite)
        ranking = np.lexsort((thresholds[eligible], -minimum_coverage[eligible], violation[eligible]))
        chosen_index = int(eligible[ranking[0]])
        status = "NO_EXACT_MARGIN_FEASIBLE_THRESHOLD"
    selected = {
        "status": status,
        "threshold": float(thresholds[chosen_index]) if np.any(feasible) else None,
        "best_diagnostic_threshold": float(thresholds[chosen_index]),
        "thresholds_examined": int(len(thresholds)),
        "threshold_vector_sha256": hashlib.sha256(thresholds.astype("<f8").tobytes()).hexdigest(),
        "calibration_a": {name: float(values[chosen_index]) for name, values in a.items()},
        "calibration_b": {name: float(values[chosen_index]) for name, values in b.items()},
        "minimum_block_coverage": float(minimum_coverage[chosen_index]),
        "minimum_block_coverage_lcb95": float(minimum_lcb[chosen_index]),
        "maximum_block_wape_ratio_ucb95": float(maximum_ucb[chosen_index]),
        "margin_violation": float(violation[chosen_index]),
    }
    positions = np.unique(
        np.r_[
            np.linspace(0, len(thresholds) - 1, min(int(downsample_points), len(thresholds))).round().astype(int),
            chosen_index,
        ]
    )
    frontier = {
        "threshold": thresholds[positions],
        "calibration_a_coverage": a["coverage"][positions],
        "calibration_a_coverage_lcb95": a["coverage_lcb95"][positions],
        "calibration_a_wape_ratio": a["accepted_wape_ratio"][positions],
        "calibration_a_wape_ratio_ucb95": a["accepted_wape_ratio_ucb95"][positions],
        "calibration_b_coverage": b["coverage"][positions],
        "calibration_b_coverage_lcb95": b["coverage_lcb95"][positions],
        "calibration_b_wape_ratio": b["accepted_wape_ratio"][positions],
        "calibration_b_wape_ratio_ucb95": b["accepted_wape_ratio_ucb95"][positions],
    }
    return selected, frontier


def risk_rank_diagnostics(truth: np.ndarray, prediction: np.ndarray, risk: np.ndarray) -> dict:
    y = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    r = np.asarray(risk, dtype=np.float64).ravel()
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(r)
    y, p, r = y[valid], p[valid], r[valid]
    realized = np.abs(y - p)
    cutoff = float(np.quantile(realized, 0.80))
    high = realized >= cutoff
    return {
        "n": int(len(y)),
        "unique_scores": int(len(np.unique(r))),
        "spearman": float(spearmanr(r, realized).statistic),
        "auc_top20pct_absolute_error": float(roc_auc_score(high.astype(np.uint8), r)),
        "high_error_cutoff": cutoff,
    }


def cluster_aggregate(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
    threshold: float,
    cluster: np.ndarray,
    cluster_count: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(truth, dtype=np.float64).ravel()
    p = np.asarray(prediction, dtype=np.float64).ravel()
    s = np.asarray(score, dtype=np.float64).ravel()
    c = np.asarray(cluster, dtype=np.int32).ravel()
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(s)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).ravel()
    accepted = valid & (s <= float(threshold))
    result = np.column_stack(
        [
            np.bincount(c, weights=valid.astype(np.float64), minlength=cluster_count),
            np.bincount(c, weights=accepted.astype(np.float64), minlength=cluster_count),
            np.bincount(c, weights=np.abs(y) * accepted, minlength=cluster_count),
            np.bincount(c, weights=np.abs(y - p) * accepted, minlength=cluster_count),
        ]
    )
    return result.astype(np.float64)


def fixed_policy_familywise_bootstrap(
    aggregates: dict[str, np.ndarray],
    *,
    baseline_selection_wape: float,
    reps: int,
    seed: int,
    familywise_alpha: float,
    minimum_coverage_lcb: float,
    maximum_ratio_ucb: float,
) -> dict:
    if not aggregates:
        raise ValueError("No bootstrap units")
    cluster_count = next(iter(aggregates.values())).shape[0]
    if any(value.shape != (cluster_count, 4) for value in aggregates.values()):
        raise ValueError("Bootstrap aggregate shapes differ")
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=int(reps),
    ).astype(np.float32)
    endpoint_count = 2 * len(aggregates)
    endpoint_alpha = float(familywise_alpha) / endpoint_count
    units = {}
    for name, values in aggregates.items():
        point = values.sum(axis=0)
        boot = weights @ values.astype(np.float32)
        coverage_point = float(point[1] / max(point[0], 1.0))
        wape_point = float(point[3] / max(point[2], 1e-12))
        boot_coverage = boot[:, 1] / np.maximum(boot[:, 0], 1.0)
        boot_ratio = (boot[:, 3] / np.maximum(boot[:, 2], 1e-12)) / max(
            float(baseline_selection_wape), 1e-12
        )
        coverage_lcb = float(np.quantile(boot_coverage, endpoint_alpha, method="linear"))
        ratio_ucb = float(np.quantile(boot_ratio, 1.0 - endpoint_alpha, method="linear"))
        units[name] = {
            "eligible_n": int(point[0]),
            "accepted_n": int(point[1]),
            "coverage": coverage_point,
            "coverage_lcb_familywise": coverage_lcb,
            "accepted_wape": wape_point,
            "accepted_wape_ratio": wape_point / max(float(baseline_selection_wape), 1e-12),
            "accepted_wape_ratio_ucb_familywise": ratio_ucb,
            "coverage_pass": coverage_lcb >= float(minimum_coverage_lcb),
            "wape_ratio_pass": ratio_ucb <= float(maximum_ratio_ucb),
        }
    passed = all(row["coverage_pass"] and row["wape_ratio_pass"] for row in units.values())
    return {
        "status": "FIXED_POLICY_FAMILYWISE_PASS" if passed else "FIXED_POLICY_FAMILYWISE_NO_GO",
        "cluster": "item_id",
        "paired_across_units": True,
        "cluster_count": int(cluster_count),
        "bootstrap_reps": int(reps),
        "familywise_alpha": float(familywise_alpha),
        "endpoint_count": int(endpoint_count),
        "per_endpoint_alpha_bonferroni": endpoint_alpha,
        "units": units,
    }


def paired_point_cluster_bootstrap(
    *,
    truth: np.ndarray,
    baseline: np.ndarray,
    proposal: np.ndarray,
    cluster: np.ndarray,
    reps: int,
    seed: int,
) -> dict:
    y = np.asarray(truth, dtype=np.float64).ravel()
    b = np.asarray(baseline, dtype=np.float64).ravel()
    p = np.asarray(proposal, dtype=np.float64).ravel()
    c = np.asarray(cluster, dtype=np.int32).ravel()
    _, c = np.unique(c, return_inverse=True)
    count = int(c.max()) + 1
    demand = np.bincount(c, weights=np.abs(y), minlength=count)
    b_error = np.bincount(c, weights=np.abs(y - b), minlength=count)
    p_error = np.bincount(c, weights=np.abs(y - p), minlength=count)
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(count, np.full(count, 1.0 / count), size=int(reps)).astype(np.float32)
    boot_demand = weights @ demand.astype(np.float32)
    boot_b = (weights @ b_error.astype(np.float32)) / np.maximum(boot_demand, 1e-12)
    boot_p = (weights @ p_error.astype(np.float32)) / np.maximum(boot_demand, 1e-12)
    ratio = boot_p / np.maximum(boot_b, 1e-12)
    delta = boot_p - boot_b
    point_b = float(b_error.sum() / max(demand.sum(), 1e-12))
    point_p = float(p_error.sum() / max(demand.sum(), 1e-12))
    return {
        "status": "PAIRED_ITEM_CLUSTER_BOOTSTRAP_COMPLETED",
        "clusters": count,
        "reps": int(reps),
        "baseline_wape": point_b,
        "proposal_wape": point_p,
        "wape_ratio_proposal_to_baseline": point_p / max(point_b, 1e-12),
        "wape_ratio_ci95": [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))],
        "delta_wape": point_p - point_b,
        "delta_wape_ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
    }

