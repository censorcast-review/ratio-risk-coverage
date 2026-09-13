"""Descriptive accounting of the two already evaluated external acceptance sets.

This reads only the saved prediction cache from the completed comparison. It
does not retrieve external data, fit a model, select a threshold, or evaluate
another policy. Freeze the descriptive purpose with --prepare, then use --run.
The default destination is separate from the archived scientific evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np

from r2_io import dump, sha, write

ROOT = Path(__file__).resolve().parents[1]
PLAN = "code/review5_external/EXTERNAL_PLAN.json"
COMPARISON = "results/external_confirmation/EXTERNAL_COMPARISON.json"
EVALUATOR = "code/review5_external/evaluate.py"
PREDICTIONS = "results/external_confirmation/prediction_cache/external_predictions.npz"
HEADS = "results/external_confirmation/prediction_cache/external_head_scores.npz"
EXPECTED_PLAN = "8173b957c2c1d4fdb3a962862a4435a1531e378569f8a6602d02af97a0dd7f1d"
EXPECTED_COMPARISON = "61fd5159b8b152626f2d14eaea6074985575140da6c8db89440321fd0ab195b9"
EXPECTED_EVALUATOR = "b9b3b0a93ab2f636f583c5e56faa720872c9e896169dbf8f10596d05f4ee4928"
POLICIES = ["composed_excess_lambda_1", "mixed_utility_lambda_0_25"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare(destination):
    path = destination / "PROTOCOL.json"
    if path.exists():
        raise FileExistsError("A descriptive protocol already exists; preserve it unchanged.")
    for name, expected in [(PLAN, EXPECTED_PLAN), (COMPARISON, EXPECTED_COMPARISON),
                           (EVALUATOR, EXPECTED_EVALUATOR)]:
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    # Hashing cache bytes records provenance without loading array contents.
    inputs = {name: sha(ROOT / name) for name in [PLAN, COMPARISON, EVALUATOR, PREDICTIONS, HEADS]}
    protocol = {
        "schema_version": "external-descriptive-budget-1",
        "prepared_utc": utc(),
        "purpose": "Retrospective Eq. (6) budget decomposition of exactly the two completed frozen external policies.",
        "inferential_status": "Descriptive only; specified after the registered external result was known, not a new confirmatory endpoint.",
        "data_scope": "Previously saved prediction cache only; no retrieval or reopening of source external outcomes.",
        "period": [1919, 1941],
        "expected_rows": 140300,
        "policies": POLICIES,
        "set_definitions": {"K": "Accepted by both policies", "U": "Accepted only by composed excess (lambda=1)",
                            "V": "Accepted only by mixed utility (lambda=0.25)", "N": "Accepted by neither"},
        "summaries": ["row count", "demand mass and coverage", "absolute-error mass", "WAPE", "forecast sum and mean",
                      "signed excess E-rM", "common-set slack", "incremental slack consumption", "Eq. (6) identity"],
        "excluded": ["new policy", "superseded pair", "new threshold or scale", "new fit", "new subgroup or period analysis",
                     "raw external inputs", "new hypothesis test or confidence interval"],
        "input_sha256": inputs,
        "script_sha256": sha(Path(__file__)),
        "cache_arrays_loaded_at_preparation": False,
    }
    dump(path, protocol)
    print(json.dumps({"status": "DESCRIPTIVE_PURPOSE_FROZEN", "protocol": str(path), "sha256": sha(path)}))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def summarize(mask, y, p, cap, total):
    n = int(mask.sum())
    mass = float(y[mask].sum(dtype=np.float64))
    error = float(np.abs(y-p)[mask].sum(dtype=np.float64))
    forecasts = float(p[mask].sum(dtype=np.float64))
    return {"rows": n, "row_share": n/y.size,
            "demand_mass": mass, "demand_share": mass/total,
            "absolute_error_mass": error, "wape": error/mass if mass else None,
            "forecast_sum": forecasts, "mean_forecast": forecasts/n if n else None,
            "mean_demand": mass/n if n else None,
            "signed_excess": error-cap*mass,
            "signed_excess_per_original_row": (error-cap*mass)/y.size}


def run(destination, table):
    protocol_path = destination / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if (destination / "EXTERNAL_BUDGET.json").exists():
        raise FileExistsError("Archived analysis already exists; use aggregate verification instead of overwriting it.")
    if protocol["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Analysis source changed after purpose freeze")
    for name, expected in protocol["input_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Recorded input changed: {name}")
    plan = json.loads((ROOT / PLAN).read_text())
    comparison = json.loads((ROOT / COMPARISON).read_text())
    assert comparison["external_use_count"] == 1
    assert [p["name"] for p in plan["policies"]] == POLICIES
    assert plan["primary_target_days"] == protocol["period"] == [1919, 1941]
    spec = importlib.util.spec_from_file_location("frozen_external_evaluator", ROOT / EVALUATOR)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    first_cache_array_read_utc = utc()
    data, scores = load_npz(ROOT / PREDICTIONS), load_npz(ROOT / HEADS)
    days = np.asarray(data["target_days"], dtype=int)
    assert np.array_equal(days, np.arange(1914, 1942))
    cols = (days >= 1919) & (days <= 1941)
    assert np.array_equal(days[cols], np.arange(1919, 1942))
    y = np.asarray(data["truth"][:, cols], dtype=np.float64)
    p = np.asarray(data["proposal"][:, cols], dtype=np.float64)
    assert y.shape == p.shape == (6100, 23)
    assert y.size == protocol["expected_rows"]
    masks = {name: mask[:, cols] for name, mask in evaluator.masks_from_scores(scores).items()}
    assert list(masks) == POLICIES
    registered_checks = []
    for name in POLICIES:
        actual = evaluator.metric(y, p, masks[name])
        expected = comparison["primary"]["metrics"][name]
        for key, value in expected.items():
            if isinstance(value, float):
                assert np.isclose(actual[key], value, rtol=1e-13, atol=1e-10), (name, key, actual[key], value)
            else:
                assert actual[key] == value, (name, key)
            registered_checks.append({"policy": name, "field": key, "actual": actual[key], "registered": value})
    ar, am = (masks[name] for name in POLICIES)
    parts = {"K": ar & am, "U": ar & ~am, "V": ~ar & am, "N": ~ar & ~am}
    total = float(y.sum(dtype=np.float64))
    cap = plan["wape_cap"]
    sets = {name: summarize(mask, y, p, cap, total) for name, mask in parts.items()}
    union = {name: summarize(mask, y, p, cap, total) for name, mask in masks.items()}
    slack = -sets["K"]["signed_excess"]
    budget = {"common_slack": slack, "identity": "E(K+J)-rM(K+J)=(E(K)-rM(K))+(E(J)-rM(J)), J=U,V",
              "additions": {}}
    for key, name in [("U", POLICIES[0]), ("V", POLICIES[1])]:
        cost = sets[key]["signed_excess"]
        combined = union[name]["signed_excess"]
        budget["additions"][key] = {
            "policy": name, "cost": cost, "fraction_of_common_slack": cost/slack if slack else None,
            "remaining_slack": slack-cost, "combined_signed_excess": combined,
            "identity_residual": combined-(-slack+cost), "point_cap_pass": bool(combined <= 0)}
    diff = {"row_count_mixed_minus_row": sets["V"]["rows"]-sets["U"]["rows"],
            "row_coverage_mixed_minus_row": (sets["V"]["rows"]-sets["U"]["rows"])/y.size,
            "demand_mass_mixed_minus_row": sets["V"]["demand_mass"]-sets["U"]["demand_mass"],
            "demand_coverage_mixed_minus_row": (sets["V"]["demand_mass"]-sets["U"]["demand_mass"])/total}
    result = {"schema_version": "external-descriptive-budget-1", "inferential_status": protocol["inferential_status"],
              "protocol_sha256": sha(protocol_path), "completed_utc": utc(),
              "primary_target_days": [1919, 1941], "total_rows": y.size, "total_demand": total,
              "empirical_mean_demand": total/y.size, "fixed_score_reference_mean": plan["reference_mean"],
              "wape_cap": cap, "sets": sets, "policies": union, "budget": budget, "differences": diff,
              "registered_primary_reproduction": registered_checks,
              "joint_secondary_operating_check_pass": comparison["joint_secondary_operating_check_pass"],
              "note": "Realized sample accounting for the two unchanged learned policies; not a conditional-moment or optimality claim."}
    for name, expected in protocol["input_sha256"].items():
        assert sha(ROOT / name) == expected, f"Input mutated during accounting: {name}"
    dump(destination / "EXTERNAL_BUDGET.json", result)
    dump(destination / "EXECUTION_AUDIT.json", {
        "status": "DESCRIPTIVE_CACHED_SET_ACCOUNTING_COMPLETED", "protocol_sha256": sha(protocol_path),
        "protocol_prepared_utc": protocol["prepared_utc"], "first_cache_array_read_utc": first_cache_array_read_utc,
        "completed_utc": result["completed_utc"], "input_sha256_before_and_after": protocol["input_sha256"],
        "input_hashes_unchanged": True, "external_source_openings_this_analysis": 0,
        "previous_registered_external_openings": 1, "saved_cache_reads": 2,
        "fits": 0, "threshold_searches": 0, "additional_policies": 0,
        "evaluated_existing_policy_count": 2, "evaluated_periods": [[1919, 1941]],
        "registered_primary_metric_checks": len(registered_checks),
        "output_sha256": sha(destination / "EXTERNAL_BUDGET.json")})
    if table:
        lines = [r"\begin{table}[t]", r"\centering\small",
                 r"\caption{Descriptive budget accounting for the two frozen external policies, days 1919--1941. $K$ is their intersection; $U$ is accepted only at $\lambda=1$, and $V$ only at $\lambda=0.25$. Demand shares use the total 209,893 units. Signed excess is $E_B-rM_B$; a negative value supplies slack. This analysis reuses saved predictions after the confirmation result.}",
                 r"\label{tab:externalbudget}", r"\begin{tabular}{lrrrrr}", r"\toprule",
                 r"Set & Rows & Demand (\%) & WAPE & Mean $f$ & Excess \\", r"\midrule"]
        for key, label in [("K", r"Common $K$"), ("U", r"Row-only $U$"), ("V", r"Mixed-only $V$")]:
            s = sets[key]
            lines.append(f"{label} & {s['rows']:,} & {100*s['demand_share']:.2f} & {s['wape']:.4f} & {s['mean_forecast']:.3f} & {s['signed_excess']:,.2f} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        write(table, "\n".join(lines).encode())
    print(json.dumps({"status": "PASS", "rows": y.size, "sets": sets, "budget": budget, "differences": diff}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction_outputs/external_budget")
    parser.add_argument("--table", type=Path, help="Optional new LaTeX table; omitted by default.")
    args = parser.parse_args()
    prepare(args.output_dir) if args.prepare else run(args.output_dir, args.table)


if __name__ == "__main__":
    main()
