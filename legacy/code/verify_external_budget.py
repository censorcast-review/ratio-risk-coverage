"""Replay the external set-accounting identities from aggregate sums only.

No individual outcomes, prediction cache, model, or policy evaluation is read.
Outputs are separate from the packaged scientific evidence by default.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from r2_io import dump, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/external_budget"
COMPARISON = ROOT / "results/external_confirmation/EXTERNAL_COMPARISON.json"


def verify(source, comparison_path):
    source = Path(source)
    result = json.loads((source / "EXTERNAL_BUDGET.json").read_text())
    protocol = json.loads((source / "PROTOCOL.json").read_text())
    audit = json.loads((source / "EXECUTION_AUDIT.json").read_text())
    comparison = json.loads(comparison_path.read_text())
    checks = []

    def check(name, actual, expected):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)):
            ok = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8)
        else:
            ok = actual == expected
        if not ok:
            raise AssertionError(f"{name}: {actual!r} != {expected!r}")
        checks.append(name)

    check("protocol_hash", result["protocol_sha256"], sha(source / "PROTOCOL.json"))
    check("audit_protocol_hash", audit["protocol_sha256"], sha(source / "PROTOCOL.json"))
    check("audit_result_hash", audit["output_sha256"], sha(source / "EXTERNAL_BUDGET.json"))
    check("comparison_hash", protocol["input_sha256"]["results/external_confirmation/EXTERNAL_COMPARISON.json"], sha(comparison_path))
    check("primary_period", result["primary_target_days"], comparison["primary"]["target_days"])
    check("protocol_period", protocol["period"], result["primary_target_days"])
    check("purpose_precedes_cached_read", protocol["prepared_utc"] < audit["first_cache_array_read_utc"], True)
    check("cached_read_precedes_completion", audit["first_cache_array_read_utc"] < audit["completed_utc"], True)
    check("descriptive_status", result["inferential_status"], protocol["inferential_status"])
    check("unchanged_inputs", audit["input_hashes_unchanged"], True)
    check("original_hash_inventory", audit["input_sha256_before_and_after"], protocol["input_sha256"])
    for key in ["external_source_openings_this_analysis", "fits", "threshold_searches", "additional_policies"]:
        check(key, audit[key], 0)
    check("original_opening_count", audit["previous_registered_external_openings"], comparison["external_use_count"])
    check("registered_opening_count", comparison["external_use_count"], 1)
    check("registered_joint_check", result["joint_secondary_operating_check_pass"], comparison["joint_secondary_operating_check_pass"])
    check("registered_joint_check_still_fails", result["joint_secondary_operating_check_pass"], False)
    check("recorded_registered_metric_count", audit["registered_primary_metric_checks"], len(result["registered_primary_reproduction"]))
    for i, record in enumerate(result["registered_primary_reproduction"]):
        registered = comparison["primary"]["metrics"][record["policy"]][record["field"]]
        check(f"registered_field_{i}", record["actual"], registered)
        check(f"registered_field_archive_{i}", record["registered"], registered)

    sets = result["sets"]
    count = result["total_rows"]
    total = result["total_demand"]
    cap = result["wape_cap"]
    check("recorded_row_count", count, 140300)
    check("row_partition", sum(s["rows"] for s in sets.values()), count)
    check("demand_partition", sum(s["demand_mass"] for s in sets.values()), total)
    check("row_share_partition", sum(s["row_share"] for s in sets.values()), 1)
    check("demand_share_partition", sum(s["demand_share"] for s in sets.values()), 1)
    check("realized_mean_is_separate_from_frozen_scale", result["empirical_mean_demand"], total/count)
    check("score_reference_mean_unchanged", result["fixed_score_reference_mean"], 1.3048948713314772)
    for name, s in list(sets.items()) + list(result["policies"].items()):
        check(f"{name}_row_share", s["row_share"], s["rows"]/count)
        check(f"{name}_demand_share", s["demand_share"], s["demand_mass"]/total)
        check(f"{name}_wape", s["wape"], s["absolute_error_mass"]/s["demand_mass"])
        check(f"{name}_mean_forecast", s["mean_forecast"], s["forecast_sum"]/s["rows"])
        check(f"{name}_mean_demand", s["mean_demand"], s["demand_mass"]/s["rows"])
        check(f"{name}_excess", s["signed_excess"], s["absolute_error_mass"]-cap*s["demand_mass"])
        check(f"{name}_excess_per_row", s["signed_excess_per_original_row"], s["signed_excess"]/count)

    slack = cap*sets["K"]["demand_mass"]-sets["K"]["absolute_error_mass"]
    check("common_slack", result["budget"]["common_slack"], slack)
    for addition, name in [("U", "composed_excess_lambda_1"), ("V", "mixed_utility_lambda_0_25")]:
        policy = result["policies"][name]
        registered = comparison["primary"]["metrics"][name]
        for field, target in [("rows", "accepted_rows"), ("demand_mass", "accepted_demand"),
                              ("absolute_error_mass", "accepted_absolute_error")]:
            summed = sets["K"][field]+sets[addition][field]
            check(f"{name}_{field}_set_addition", policy[field], summed)
            check(f"{name}_{field}_registered", summed, registered[target])
        check(f"{name}_forecast_sum_addition", policy["forecast_sum"], sets["K"]["forecast_sum"]+sets[addition]["forecast_sum"])
        check(f"{name}_row_coverage_registered", policy["row_share"], registered["row_coverage"])
        check(f"{name}_demand_coverage_registered", policy["demand_share"], registered["demand_coverage"])
        check(f"{name}_wape_registered", policy["wape"], registered["wape"])
        cost = sets[addition]["absolute_error_mass"]-cap*sets[addition]["demand_mass"]
        b = result["budget"]["additions"][addition]
        check(f"{addition}_cost", b["cost"], cost)
        check(f"{addition}_fraction_of_slack", b["fraction_of_common_slack"], cost/slack)
        check(f"{addition}_remaining_slack", b["remaining_slack"], slack-cost)
        check(f"{addition}_eq6_excess_additivity", policy["signed_excess"], cost-slack)
        check(f"{addition}_reported_residual", b["identity_residual"], policy["signed_excess"]-(cost-slack))
        check(f"{addition}_point_feasibility", b["point_cap_pass"], cost <= slack)
        check(f"{addition}_eq6_R_form", sets[addition]["demand_mass"]*(sets[addition]["wape"]-cap), cost)
    check("common_slack_R_form", sets["K"]["demand_mass"]*(cap-sets["K"]["wape"]), slack)
    d = result["differences"]
    check("row_difference_count", d["row_count_mixed_minus_row"], sets["V"]["rows"]-sets["U"]["rows"])
    check("row_difference_share", d["row_coverage_mixed_minus_row"], d["row_count_mixed_minus_row"]/count)
    check("demand_difference_mass", d["demand_mass_mixed_minus_row"], sets["V"]["demand_mass"]-sets["U"]["demand_mass"])
    check("demand_difference_share", d["demand_coverage_mixed_minus_row"], d["demand_mass_mixed_minus_row"]/total)
    check("row_difference_registered", d["row_coverage_mixed_minus_row"], comparison["primary"]["row_coverage_difference"])
    check("demand_difference_registered", d["demand_coverage_mixed_minus_row"], comparison["primary"]["demand_coverage_difference"])
    return {"status": "PASS", "check_count": len(checks), "checks": checks,
            "scope": "Aggregate arithmetic and registered-result consistency; no cached individual observations read.",
            "result_sha256": sha(source / "EXTERNAL_BUDGET.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--comparison", type=Path, default=COMPARISON)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction_outputs/external_budget_verification.json")
    args = parser.parse_args()
    output = verify(args.source_dir, args.comparison)
    dump(args.output, output)
    print(json.dumps({"status": output["status"], "check_count": output["check_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
