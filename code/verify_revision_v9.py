#!/usr/bin/env python3
"""Independently verify the saved v9 C1--C4 arithmetic and frozen-input hashes.

Run from any working directory. Only the requested report is written; model
fitting, threshold design, new resampling and row-array evaluation are absent.
The manuscript's scientific interpretation also received a separate human
review, recorded in docs/revision_v9/INDEPENDENT_REVIEW.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["error", "row", "mixed", "demand", "weight_descending"]


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval(cases, exposures, eligible, target=3):
    """Derive the half-open interval without importing a production selector."""
    ids = [i for i, active in enumerate(eligible) if active]
    if target not in ids:
        return None, None, True, []
    maximum = max(exposures[i] for i in ids)
    lower = maximum - exposures[target]
    priority = (cases[target], exposures[target], -target)
    superior = [i for i in ids if (cases[i], exposures[i], -i) > priority]
    upper = min((maximum - exposures[i] for i in superior), default=None)
    empty = upper is not None and upper <= lower
    endings = [i for i in superior if maximum - exposures[i] == upper]
    return lower, upper, empty, endings


class Checks:
    def __init__(self):
        self.numeric = 0
        self.hashes = 0
        self.verified_hashes = {}

    def require(self, condition, message):
        self.numeric += 1
        if not condition:
            raise AssertionError(message)

    def near(self, actual, expected, message):
        self.require(abs(float(actual) - float(expected)) < 1e-10, message)

    def check_hash(self, relative, expected):
        self.hashes += 1
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise AssertionError(f"Hash mismatch: {relative}")
        self.verified_hashes[relative] = actual


def verify_hashes(checks):
    specs = [
        ("evidence/revision_v9_epsilon/PROTOCOL.json", "code/analyze_epsilon_v9.py", "input_sha256"),
        ("evidence/revision_v9_screen_tolerance/PROTOCOL.json", "code/analyze_screen_tolerance_v9.py", "input_sha256"),
        ("docs/revision_v9/DESCENDING_COVERAGE_PROTOCOL.json", "code/verify_descending_coverage_v9.py", "inputs_sha256"),
    ]
    for relative, code, input_key in specs:
        protocol = read_json(relative)
        checks.check_hash(code, protocol["code_sha256"])
        for path, expected in protocol[input_key].items():
            checks.check_hash(path, expected)
    for directory in ["evidence/revision_v9_epsilon", "evidence/revision_v9_screen_tolerance"]:
        result = read_json(f"{directory}/RESULTS.json")
        checks.check_hash(f"{directory}/PROTOCOL.json", result["protocol_sha256"])
    coverage = read_json("docs/revision_v9/DESCENDING_COVERAGE_CHECK.json")
    checks.check_hash("docs/revision_v9/DESCENDING_COVERAGE_PROTOCOL.json", coverage["protocol_sha256"])
    for relative, expected in read_json("evidence/revision_v9_epsilon/HASHES.json").items():
        checks.check_hash(f"evidence/revision_v9_epsilon/{relative}", expected)


def verify_points(checks, result):
    intervals = []
    for record in result["points"]:
        rows = {row["method"]: row for row in record["candidates"]}
        cases = [F(rows[method]["case_percent"]) for method in METHODS]
        exposures = [F(rows[method]["exposure_percent"]) for method in METHODS]
        eligible = [rows[method]["eligible"] for method in METHODS]
        lower, upper, empty, endings = interval(cases, exposures, eligible)
        checks.near(lower, record["lower_pp"], f"{record['cell']} lower")
        checks.near(upper, record["upper_pp"], f"{record['cell']} upper")
        checks.require([METHODS[i] for i in endings] == record["upper_competitors"], "Point upper competitors")
        checks.require(not empty, "Point interval must be nonempty")
        intervals.append((float(lower), float(upper)))
    lower, upper = max(x[0] for x in intervals), min(x[1] for x in intervals)
    checks.near(lower, result["intersection"]["lower_pp"], "Original intersection lower")
    checks.near(upper, result["intersection"]["upper_pp"], "Original intersection upper")
    checks.require(lower <= .39 and 1.63 < upper, "Main closed subset [.39,1.63]")
    checks.require(1.64 > upper, "Review's proposed 1.64 upper is outside exact interval")
    return [lower, upper]


def verify_draws(checks):
    summaries = []
    for setting in ["complete_220_220", "bike"]:
        saved = read_json(f"evidence/revision_v9_epsilon/{setting}_INTERVALS.json")
        lowers, uppers, empty_count, half_count = [], [], 0, 0
        with np.load(ROOT / f"evidence/revision_v7_stability/{setting}/FULL_DRAWS.npz") as archive:
            for draw, record in enumerate(saved["draws"]):
                # Statistics columns are accepted rows, exposure, absolute loss.
                statistics, totals = archive["statistics"][draw], archive["totals"][draw]
                utility = np.stack([
                    (statistics[:, :, 0] / totals[None, :, 0]).min(axis=1),
                    (statistics[:, :, 1] / totals[None, :, 1]).min(axis=1),
                ], axis=1)
                checks.require(np.allclose(utility, archive["utilities"][draw], rtol=0, atol=1e-14), "Raw draw utility reconstruction")
                cases = [F(float(value)) for value in utility[:, 0]]
                exposures = [F(float(value)) for value in utility[:, 1]]
                lower, upper, empty, endings = interval(cases, exposures, archive["eligible"][draw].tolist())
                checks.require(empty == record["empty"], "Draw empty interval")
                checks.near(lower * 100, record["lower_pp"], "Draw lower")
                checks.near(upper * 100, record["upper_pp"], "Draw upper")
                checks.require([METHODS[i] for i in endings] == record["upper_competitors"], "Draw upper competitor")
                contains_half = not empty and lower <= F(.005) < upper
                checks.require(contains_half == record["includes_half_pp"], "Draw .5 pp membership")
                half_count += int(contains_half)
                if empty:
                    empty_count += 1
                else:
                    lowers.append(float(lower) * 100)
                    uppers.append(float(upper) * 100)
        summary = saved["summary"]
        checks.require(empty_count == summary["empty"], "Draw empty count")
        checks.require(half_count == summary["half_pp_ratio_count"], "Draw .5 pp count")
        for values, key in [(lowers, "lower_quantiles_pp"), (uppers, "finite_upper_quantiles_pp")]:
            quantiles = np.quantile(values, [0, .025, .25, .5, .75, .975, 1])
            for name, value in zip(["min", "q025", "q25", "median", "q75", "q975", "max"], quantiles):
                checks.near(value, summary[key][name], f"{setting} {key} {name}")
        summaries.append({
            "setting": setting, "draws": len(saved["draws"]), "nonempty": len(lowers),
            "half_pp_count": half_count,
            "lower_q025_q50_q975": np.quantile(lowers, [.025, .5, .975]).tolist(),
            "upper_q025_q50_q975": np.quantile(uppers, [.025, .5, .975]).tolist(),
        })
    return summaries


def verify_screens(checks, result):
    all_intervals = {}
    for plan in result["plans"]:
        candidates = plan["candidate_utilities"]
        cases, exposures, eligible = [], [], []
        for method in METHODS:
            if method not in candidates:
                cases.append(F(0)); exposures.append(F(0)); eligible.append(False)
            else:
                candidate = candidates[method]
                cases.append(F(candidate["case"]["numerator"], candidate["case"]["denominator"]))
                exposures.append(F(candidate["exposure"]["numerator"], candidate["exposure"]["denominator"]))
                eligible.append(True)
        for target, method in enumerate(METHODS):
            if not eligible[target]:
                continue
            lower, upper, empty, endings = interval(cases, exposures, eligible, target)
            saved = plan["intervals"][method]
            checks.require(lower == F(saved["lower"]["numerator"], saved["lower"]["denominator"]), "Screen exact lower")
            checks.require(
                (upper is None and saved["upper"] is None)
                or upper == F(saved["upper"]["numerator"], saved["upper"]["denominator"]),
                "Screen exact upper",
            )
            checks.require(empty == saved["empty"], "Screen empty interval")
            checks.require([METHODS[i] for i in endings] == saved["ending_candidates"], "Screen upper competitors")
            if method == "demand":
                all_intervals.setdefault((plan["floor"], plan["family"]), []).append((lower, upper))
        for epsilon, choice in plan["choices_at_epsilon_pp"].items():
            allowance = F(epsilon) / 100
            maximum = max(exposures[i] for i, active in enumerate(eligible) if active)
            admitted = [i for i, active in enumerate(eligible) if active and exposures[i] >= maximum - allowance]
            winner = max(admitted, key=lambda i: (cases[i], exposures[i], -i))
            checks.require(METHODS[winner] == choice, "Screen grid choice")
        if plan["floor"] == .35:
            checks.near((exposures[4] - exposures[3]) * 100, plan["descending_minus_ratio_exposure"]["percentage_points"], "Strict design gap")
    for record in result["ratio_intersections"]:
        intervals = all_intervals[(record["floor"], record["family"])]
        lower, upper = max(x[0] for x in intervals), min(x[1] for x in intervals)
        checks.require(lower == F(record["lower"]["numerator"], record["lower"]["denominator"]), "Screen intersection lower")
        checks.require(upper == F(record["upper"]["numerator"], record["upper"]["denominator"]), "Screen intersection upper")


def verify_coverage(checks, result):
    # This checks the dedicated row-level audit's recorded counts, not row masks.
    for record in result["original_AB_records"]:
        for window in record["windows"]:
            checks.near(window["rows"] / window["total_rows"], window["c"], "C4 window case coverage")
        checks.near(min(window["c"] for window in record["windows"]) * 100, record["min_case_percent"], "C4 minimum case coverage")
        checks.require(f"{record['min_case_percent']:.2f}" == "48.67", "C4 two-decimal rounding")
    for record in result["original_A_only_records"]:
        window = record["windows"][0]
        checks.near(window["rows"] / window["total_rows"] * 100, record["A_case_percent"], "C4 stricter design coverage")
    for record in result["exposure_head_membership_comparison"]:
        checks.require(record["only_w660_rows"] - record["only_w220_rows"] == record["net_row_difference_w660_minus_w220"], "C4 net row difference")
        checks.require(record["only_w660_rows"] + record["only_w220_rows"] == record["changed_rows"], "C4 changed row count")
    for record in result["eq7_complete_half_pp_contrasts"]:
        dc = (record["exposure_metrics"]["c"] - record["case_metrics"]["c"]) * 100
        dd = (record["exposure_metrics"]["d"] - record["case_metrics"]["d"]) * 100
        checks.near(dc, record["delta_case_pp"], "C3 case contrast")
        checks.near(dd, record["delta_exposure_pp"], "C3 exposure contrast")
        checks.near(dd / -dc, record["exchange_ratio"], "C3 exchange ratio denominator")
        checks.require(record["case_choice"] == "Excess" and record["exposure_choice"] == "Ratio", "C3 comparators")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction_outputs/revision_v9_verification.json", help="Report file; default is relative to the source root, independent of current directory.")
    args = parser.parse_args()
    destination = args.output.resolve()
    # Protect the input/archive/source directories from accidental report overwrite.
    for protected in ["code", "paper", "evidence", "docs"]:
        if destination.is_relative_to(ROOT / protected):
            parser.error("--output must not overwrite an input or source directory")
    checks = Checks()
    verify_hashes(checks)
    epsilon = read_json("evidence/revision_v9_epsilon/RESULTS.json")
    screen = read_json("evidence/revision_v9_screen_tolerance/RESULTS.json")
    coverage = read_json("docs/revision_v9/DESCENDING_COVERAGE_CHECK.json")
    common_interval = verify_points(checks, epsilon)
    draws = verify_draws(checks)
    verify_screens(checks, screen)
    verify_coverage(checks, coverage)
    report = {
        "status": "PASS", "utc": datetime.now(timezone.utc).isoformat(),
        "numeric_assertions": checks.numeric, "hash_assertions": checks.hashes,
        "code_sha256": sha256(Path(__file__).resolve()),
        "original_common_interval_pp": common_interval,
        "main_closed_subset_pp": [.39, 1.63], "draw_summaries": draws,
        "screen_plans": len(screen["plans"]),
        "screen_intersections": screen["ratio_intersections"],
        "eq7_complete_half_pp_ranges": coverage["eq7_complete_half_pp_ranges"],
        "verified_input_sha256": checks.verified_hashes,
        "scope": "Independent deterministic arithmetic and hash replay. No production selector imports; no fits, threshold designs, new draws or row-array evaluation. Raw archived redesign sufficient statistics reconstruct utilities. C4 counts and membership arithmetic are checked against the dedicated row-level audit record; underlying files are hashed, not reevaluated.",
        "limitations": [
            "The saved endpoint distributions are descriptive conditional summaries, not confidence intervals or a joint four-head redesign analysis.",
            "The manuscript's primary-versus-alternative framing and causal/generalization qualifications received a separate human review in docs/revision_v9/INDEPENDENT_REVIEW.json.",
            "This replay does not establish population risk, prospective generalization, or an operational epsilon default.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "numeric_assertions": checks.numeric, "hash_assertions": checks.hashes, "report": str(destination)}))


if __name__ == "__main__":
    main()
