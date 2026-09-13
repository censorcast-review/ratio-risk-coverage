"""Exactly two fixed policies; no threshold selection or fitting.

The item-cluster percentile bootstrap follows the recorded guardian estimator.
All accumulation is float64; categories never mutate the shared policy masks.
"""
from __future__ import annotations
import numpy as np

CAP = 0.85 * 0.7530939208313988
ROW_FLOOR = 0.35
REFERENCE_MEAN = 1.3048948713314772
THRESHOLDS = {"composed_excess_lambda_1": 0.2888724531967165,
              "mixed_utility_lambda_0_25": 0.37635288579749315}
LAMBDAS = {"composed_excess_lambda_1": 1.0, "mixed_utility_lambda_0_25": 0.25}
PERIODS = {"primary_23": [1919, 1941], "first_11": [1919, 1929],
           "second_12": [1930, 1941]}
GROUPS = ["ALL", "FOODS", "HOBBIES", "HOUSEHOLD"]


def mixed_scores(scores):
    # Match development construction: clip float32 model outputs, then promote
    # to float64 before taking a ratio. T is fixed on development A/B only.
    error = np.maximum(np.asarray(scores["error"], dtype=np.float32), 0).astype(np.float64)
    demand = np.maximum(np.asarray(scores["demand"], dtype=np.float32), 0).astype(np.float64)
    if error.shape != demand.shape:
        raise ValueError("The two frozen score arrays must have identical shapes")
    return {name: (error - CAP * demand) / (lam + (1-lam) * demand / REFERENCE_MEAN)
            for name, lam in LAMBDAS.items()}


def masks_from_scores(scores):
    return {name: np.isfinite(value) & (value <= THRESHOLDS[name])
            for name, value in mixed_scores(scores).items()}


def metric(truth, proposal, mask):
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(proposal, dtype=np.float64)
    a = np.asarray(mask, dtype=bool)
    if y.shape != p.shape or a.shape != y.shape:
        raise ValueError("Aligned arrays required")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any():
        raise ValueError("Invalid demand or forecasts")
    n = y.size
    accepted = int(a.sum())
    mass = float(y[a].sum(dtype=np.float64))
    total = float(y.sum(dtype=np.float64))
    error = float(np.abs(y - p)[a].sum(dtype=np.float64))
    return {"rows": int(n), "accepted_rows": accepted,
            "row_coverage": accepted / n if n else None,
            "demand_coverage": mass / total if total > 0 else None,
            "wape": error / mass if mass > 0 else None,
            "accepted_demand": mass, "total_demand": total, "accepted_absolute_error": error,
            "point_operating_check_pass": bool(n and mass > 0 and error / mass <= CAP and accepted / n >= ROW_FLOOR)}


def components(truth, proposal, mask, item_ids):
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(proposal, dtype=np.float64)
    a = np.asarray(mask, dtype=bool)
    labels, codes = np.unique(np.asarray(item_ids).astype(str), return_inverse=True)
    arrays = [a.sum(axis=1), np.full(len(a), a.shape[1]),
              (y * a).sum(axis=1), (np.abs(y-p) * a).sum(axis=1), y.sum(axis=1)]
    return np.column_stack([np.bincount(codes, weights=v, minlength=len(labels)) for v in arrays])


def summarize_draws(values):
    return np.column_stack([values[:, 0] / values[:, 1],
        np.divide(values[:, 2], values[:, 4], out=np.full(len(values), np.nan), where=values[:, 4] > 0),
        np.divide(values[:, 3], values[:, 2], out=np.full(len(values), np.inf), where=values[:, 2] > 0)])


def bootstrap_matrices(matrices, reps=10000, seed=20260906):
    n = len(matrices[0])
    if n < 1 or any(len(m) != n for m in matrices):
        raise ValueError("Paired nonempty item clusters required")
    rng = np.random.default_rng(seed)
    all_values = [[] for _ in matrices]
    for offset in range(0, reps, 200):
        weights = rng.multinomial(n, np.full(n, 1/n), size=min(200, reps-offset))
        for j, mat in enumerate(matrices):
            all_values[j].append(summarize_draws(weights @ mat))
    return [np.concatenate(v) for v in all_values]


def finite(value):
    return float(value) if np.isfinite(value) else None


def evaluate(data, meta, scores, reps=10000):
    days = np.asarray(data["target_days"], dtype=int)
    if not np.array_equal(days, np.arange(1914, 1942)):
        raise ValueError("The preregistered target window is exactly d1914..1941")
    y, p = data["truth"], data["proposal"]
    items = np.asarray(meta["item_id"]).astype(str)
    categories = np.asarray(meta["cat_id"]).astype(str)
    masks = masks_from_scores(scores)
    names = list(THRESHOLDS)
    primary_cols = days >= 1919
    primary_metrics = {n: metric(y[:,primary_cols], p[:,primary_cols], masks[n][:,primary_cols]) for n in names}
    primary_parts = [components(y[:,primary_cols], p[:,primary_cols], masks[n][:,primary_cols], items) for n in names]
    draws = bootstrap_matrices(primary_parts, reps=reps)
    # Mixed utility minus composed excess, paired across the identical item draws.
    delta = draws[1] - draws[0]
    valid = np.isfinite(delta[:, :2]).all(axis=1)
    lower_d = float(np.quantile(delta[valid, 1], .025)) if valid.all() else float("nan")
    upper_c = float(np.quantile(delta[valid, 0], .975)) if valid.all() else float("nan")
    primary = {"contrast": "mixed_utility_lambda_0_25 minus composed_excess_lambda_1", "target_days":[1919,1941],
        "row_coverage_difference": primary_metrics[names[1]]["row_coverage"] - primary_metrics[names[0]]["row_coverage"],
        "demand_coverage_difference": (primary_metrics[names[1]]["demand_coverage"] - primary_metrics[names[0]]["demand_coverage"])
            if all(primary_metrics[n]["demand_coverage"] is not None for n in names) else None,
        "row_difference_upper_one_sided_97_5": finite(upper_c),
        "demand_difference_lower_one_sided_97_5": finite(lower_d),
        "directional_confirmation_pass": bool(valid.all() and lower_d > 0 and upper_c < 0),
        "familywise_alpha": .05, "one_sided_endpoint_alpha": .025,
        "invalid_bootstrap_demand_draws": int((~valid).sum()), "metrics": primary_metrics}
    # Two policies x three periods x four strata x two one-sided bounds = 48.
    endpoint_alpha = .05 / 48
    secondary = []
    for period, (start, end) in PERIODS.items():
        cols = (days >= start) & (days <= end)
        for group in GROUPS:
            rows = np.ones(len(items), dtype=bool) if group == "ALL" else categories == group
            if not rows.any():
                for name in names:
                    secondary.append({"period":period,"group":group,"policy":name,
                        "item_clusters":0,"metrics":None,"row_coverage_lcb":None,"wape_ucb":None,
                        "adjusted_operating_check_pass":False,"reason":"prespecified_stratum_absent"})
                continue
            yy, pp = y[rows][:, cols], p[rows][:, cols]
            aa = [masks[name][rows][:, cols].copy() for name in names]
            parts = [components(yy, pp, a, items[rows]) for a in aa]
            values = bootstrap_matrices(parts, reps=reps)
            for name, a, z in zip(names, aa, values):
                cl = float(np.quantile(z[:, 0], endpoint_alpha))
                # Infinities conservatively fail instead of being discarded.
                wu = float(np.quantile(z[:, 2], 1-endpoint_alpha, method="higher"))
                secondary.append({"period":period,"group":group,"policy":name,
                    "item_clusters":len(np.unique(items[rows])),"metrics":metric(yy,pp,a),
                    "row_coverage_lcb":finite(cl),"wape_ucb":finite(wu),
                    "undefined_wape_draws":int((~np.isfinite(z[:,2])).sum()),
                    "adjusted_operating_check_pass":bool(cl >= ROW_FLOOR and wu <= CAP)})
    return {"primary":primary,"secondary":secondary,
        "descriptive_full_28_sensitivity":{"target_days":[1914,1941],
            "metrics":{n:metric(y,p,masks[n]) for n in names},
            "scope":"Prespecified point-estimate sensitivity only. Targets d1914..1918 have origin1911, preceding the last consumed development day1913; excluded from confirmation."},
        "joint_secondary_operating_check_pass":all(x["adjusted_operating_check_pass"] for x in secondary),
        "bootstrap_reps":reps,"bootstrap_seed":20260906,"secondary_endpoint_alpha":endpoint_alpha,
        "secondary_endpoint_count":48,"policy_count":2,"fit_calls":0,"threshold_selection_calls":0,
        "statistical_scope":"Prespecified item-cluster percentile bootstrap; shared store/calendar shocks are not removed; not an exact distribution-free certificate."}
