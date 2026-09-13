"""Verify the attribution controls and frozen FreshRetailNet evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(actual: float, expected: float, tol: float = 1e-10) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tol):
        raise AssertionError(f"{actual} != {expected}")


def verify_m5() -> int:
    result = json.loads(
        (ROOT / "evidence/risk_calibration/attribution_controls/RESULTS.json").read_text()
    )
    corrected = result["corrected_score_ablation"]
    expected = {
        "base_forecast": 0.6704509610803776,
        "historical_hit_rate_28": 0.6707037965051693,
        "origin_fill_ratio": 0.6707340318421272,
        "capacity_pressure": 0.6710384478132642,
        "learned_hit_risk": 0.6685093103012937,
    }
    checks = 0
    close(corrected["base"]["wape"], 0.6713924901583933)
    checks += 1
    for key, value in expected.items():
        close(corrected["scores"][key]["later"]["wape"], value)
        checks += 1
    cells = [row for row in result["cells"] if row["population"] == "later"]
    if len(cells) != 24:
        raise AssertionError("Later cell count is not 24")
    close(sum(row["row_share"] for row in cells), 1.0)
    close(sum(row["demand_share"] for row in cells), 1.0)
    checks += 3
    integration = result["integration_and_oracle"]
    close(integration["q_as_feature"]["wape"], 0.6736429976175518)
    close(integration["proposed_posthoc_calibration"]["wape"], 0.6685093103012937)
    close(integration["truth_label_oracle"]["wape"], 0.6574316225649043)
    recovered = (
        integration["hierarchy_base"]["wape"]
        - integration["proposed_posthoc_calibration"]["wape"]
    ) / (
        integration["hierarchy_base"]["wape"]
        - integration["truth_label_oracle"]["wape"]
    )
    close(recovered, 0.20651867355609513)
    checks += 4
    return checks


def score(y: np.ndarray, f: np.ndarray) -> tuple[float, float, float, float]:
    error = f - y
    return (
        float(np.abs(error).sum() / y.sum()),
        float(np.abs(error).mean()),
        float(np.sqrt(np.mean(error**2))),
        float(error.sum() / y.sum()),
    )


def verify_fresh() -> int:
    directory = ROOT / "evidence/freshretail/frozen_eval"
    result = json.loads((directory / "RESULTS.json").read_text())
    receipt = json.loads((directory / "RUN_RECEIPT.json").read_text())
    if receipt["post_decode_fit_calls"] != 0 or receipt["post_decode_selection_calls"] != 0:
        raise AssertionError("Post-decode learning occurred")
    if receipt["base_sha256"] != sha(directory / "base_l1.ubj"):
        raise AssertionError("Base hash mismatch")
    if receipt["risk_sha256"] != sha(directory / "stockout_risk.ubj"):
        raise AssertionError("Risk hash mismatch")
    if receipt["policy_sha256"] != sha(directory / "FINAL_POLICY.json"):
        raise AssertionError("Policy hash mismatch")
    prediction = pd.read_csv(directory / "EVAL_PREDICTIONS.csv.gz")
    if len(prediction) != 14000 or prediction["series_id"].nunique() != 2000:
        raise AssertionError("Frozen evaluation population mismatch")
    available = prediction["is_censored"].to_numpy(int) == 0
    if int(available.sum()) != 8020:
        raise AssertionError("Available row count mismatch")
    y = prediction.loc[available, "sale_amount"].to_numpy(float)
    base = prediction.loc[available, "base"].to_numpy(float)
    rc = prediction.loc[available, "risk_conditioned"].to_numpy(float)
    base_score = score(y, base)
    rc_score = score(y, rc)
    expected_base = result["primary_fully_available"]["base"]
    expected_rc = result["primary_fully_available"]["risk_conditioned"]
    for actual, key in zip(base_score, ["wape", "mae", "rmse", "signed_percentage_bias"], strict=True):
        close(actual, expected_base[key])
    for actual, key in zip(rc_score, ["wape", "mae", "rmse", "signed_percentage_bias"], strict=True):
        close(actual, expected_rc[key])
    close(rc_score[0] - base_score[0], -0.001868427167062215)

    # Independently reconstruct the exact paired multinomial-series interval.
    ids, inverse = np.unique(prediction["series_id"].astype(str), return_inverse=True)
    yy = prediction["sale_amount"].to_numpy(float)
    f0 = prediction["base"].to_numpy(float)
    f1 = prediction["risk_conditioned"].to_numpy(float)
    mass = np.bincount(inverse, weights=np.where(available, yy, 0.0))
    e0 = np.bincount(inverse, weights=np.where(available, np.abs(yy - f0), 0.0))
    e1 = np.bincount(inverse, weights=np.where(available, np.abs(yy - f1), 0.0))
    rng = np.random.default_rng(20260907)
    draws = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=4000)
    samples = (draws @ (e1 - e0)) / (draws @ mass)
    interval = np.quantile(samples, [0.025, 0.975])
    stored = result["primary_fully_available"]["paired_series_bootstrap"]
    close(interval[0], stored["ci95"][0])
    close(interval[1], stored["ci95"][1])
    if interval[1] >= 0:
        raise AssertionError("Frozen interval does not exclude zero")
    return 3 + 2 + 8 + 1 + 3


def verify_manuscript() -> int:
    main = "\n".join(path.read_text() for path in [
        ROOT / "paper/main.tex", ROOT / "paper/risk_integration_table.tex"
    ])
    required = [
        "0.32928 to 0.32741",
        "0.67364",
        "20.7\\%",
        "[-0.00354,-0.00016]",
        "candidate-set instability",
    ]
    for text in required:
        if text not in main:
            raise AssertionError(f"Missing manuscript claim: {text}")
    return len(required)


if __name__ == "__main__":
    total = verify_m5() + verify_fresh() + verify_manuscript()
    print(f"REVISION3_VERIFY_PASS checks={total}")
