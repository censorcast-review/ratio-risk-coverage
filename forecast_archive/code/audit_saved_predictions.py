"""Independent post-hoc audit of the already scored FreshRetailNet predictions.

Reads saved predictions and frozen evidence only. No model fitting, policy
selection, raw evaluation decoding, or external requests are performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(frame, prediction):
    residual = frame[prediction].to_numpy(float) - frame.sale_amount.to_numpy(float)
    mass = float(frame.sale_amount.sum())
    return dict(rows=len(frame), demand_mass=mass,
                wape=float(np.abs(residual).sum() / mass),
                mae=float(np.mean(np.abs(residual))),
                rmse=float(np.sqrt(np.mean(residual * residual))),
                signed_percentage_bias=float(residual.sum() / mass))


def main(package, output):
    evidence = package / "evidence/freshretail/frozen_eval"
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((evidence / "RUN_RECEIPT.json").read_text())
    frozen = json.loads((evidence / "FROZEN_RC_EVAL_RECEIPT.json").read_text())
    reported = json.loads((evidence / "RESULTS.json").read_text())
    policy = json.loads((evidence / "FINAL_POLICY.json").read_text())
    assertions = []

    def check(name, condition):
        assertions.append(dict(check=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    expected = {
        "EVAL_PREDICTIONS.csv.gz": receipt["predictions_sha256"],
        "RESULTS.json": receipt["results_sha256"],
        "FINAL_POLICY.json": receipt["policy_sha256"],
        "base_l1.ubj": receipt["base_sha256"],
        "stockout_risk.ubj": receipt["risk_sha256"],
        "FROZEN_RC_EVAL_PROTOCOL.md": receipt["protocol_sha256"],
    }
    for name, expected_hash in expected.items():
        check("sha256:" + name, digest(evidence / name) == expected_hash)
    check("frozen_protocol_sha", frozen["protocol_sha256"] == receipt["protocol_sha256"])
    check("original_runner_sha", digest(package / "code/freshretail/run_frozen_rc_eval.py") == receipt["original_runner_sha256"] == frozen["runner_sha256"])
    check("recovery_runner_sha", digest(package / "code/freshretail/resume_frozen_rc_eval_after_engine_failure.py") == receipt["recovery_runner_sha256"])
    check("train_sha", digest(package / "inputs/freshretail/freshretail_2000_train_only.pkl.gz") == frozen["train_cache_sha256"])
    check("eval_bytes_sha", digest(package / "inputs/freshretail/official_eval.parquet") == receipt["eval_sha256"] == frozen["eval_expected_sha256"])
    check("freeze_precedes_successful_decode", pd.Timestamp(frozen["frozen_utc"]) < pd.Timestamp(reported["eval_first_successful_decode_utc"]))
    check("reported_one_decode_no_post_fit_selection", receipt["successful_eval_decodes"] == 1 and receipt["post_decode_fit_calls"] == receipt["post_decode_selection_calls"] == 0)

    data = pd.read_csv(evidence / "EVAL_PREDICTIONS.csv.gz", dtype={"series_id": str})
    check("14000_rows_2000_series", len(data) == 14000 and data.series_id.nunique() == 2000)
    check("no_duplicate_target", not data.duplicated(["series_id", "dt"]).any())
    check("complete_seven_horizons", data.groupby("series_id").horizon.apply(lambda x: sorted(x.tolist()) == list(range(1, 8))).all())
    check("finite_values", np.isfinite(data[["sale_amount", "base", "risk_conditioned", "predicted_stockout_risk"]].to_numpy()).all())
    check("nonnegative_sales_predictions", (data[["sale_amount", "base", "risk_conditioned"]] >= 0).all().all())
    check("risk_unit_interval", data.predicted_stockout_risk.between(0, 1).all())
    origin = pd.to_datetime(data.dt) - pd.to_timedelta(data.horizon, unit="D")
    check("single_forecast_origin", origin.nunique() == 1 and origin.iloc[0] == pd.Timestamp("2024-06-25"))
    check("binary_availability", set(data.is_censored.unique()) == {0, 1})
    primary = data.loc[data.is_censored == 0].copy()
    check("8020_primary_rows", len(primary) == 8020)

    measures = {}
    for key, frame in [("primary_fully_available", primary), ("secondary_all_recorded_sales", data)]:
        measures[key] = {method: metrics(frame, method) for method in ("base", "risk_conditioned")}
        for method, values in measures[key].items():
            for metric, value in values.items():
                check(f"reported:{key}:{method}:{metric}", np.isclose(value, reported[key][method][metric], atol=2e-14, rtol=2e-14))

    # Independently build sufficient statistics with pandas aggregation; then
    # reproduce the exact preregistered multinomial paired-cluster scheme.
    working = data.assign(
        mass=np.where(data.is_censored == 0, data.sale_amount, 0.0),
        e0=np.where(data.is_censored == 0, np.abs(data.base-data.sale_amount), 0.0),
        e1=np.where(data.is_censored == 0, np.abs(data.risk_conditioned-data.sale_amount), 0.0),
        eligible=(data.is_censored == 0).astype(int),
    )
    series = working.groupby("series_id", sort=True)[["mass", "e0", "e1", "eligible"]].sum()
    rng = np.random.default_rng(20260907)
    counts = rng.multinomial(len(series), np.ones(len(series)) / len(series), size=4000)
    deltas = (counts @ (series.e1-series.e0).to_numpy()) / (counts @ series.mass.to_numpy())
    interval = np.quantile(deltas, [0.025, 0.975])
    point = float((series.e1.sum()-series.e0.sum())/series.mass.sum())
    saved_boot = reported["primary_fully_available"]["paired_series_bootstrap"]
    check("paired_bootstrap_point", np.isclose(point, saved_boot["estimate"], atol=2e-14, rtol=0))
    check("paired_bootstrap_interval", np.allclose(interval, saved_boot["ci95"], atol=2e-14, rtol=0))
    check("paired_bootstrap_negative_share", float(np.mean(deltas < 0)) == saved_boot["share_negative"])
    series["delta_wape"] = (series.e1-series.e0)/series.mass.replace(0, np.nan)
    series.to_csv(output / "series_sufficient_statistics.csv")

    decomposition = {}
    for key in ["management_group_id", "horizon"]:
        rows = []
        saved = {str(row[key]): row for row in reported["by_management_group" if key == "management_group_id" else "by_horizon"]}
        for value, group in primary.groupby(key, sort=True):
            pair = {method: metrics(group, method) for method in ("base", "risk_conditioned")}
            for method, values in pair.items():
                for metric, measured in values.items():
                    check(f"group:{key}:{value}:{method}:{metric}", np.isclose(measured, saved[str(value)][method][metric], atol=2e-14, rtol=2e-14))
            rows.append({key: int(value), "rows": len(group),
                         "demand_mass": float(group.sale_amount.sum()),
                         "base_wape": pair["base"]["wape"],
                         "rc_wape": pair["risk_conditioned"]["wape"],
                         "delta_wape": pair["risk_conditioned"]["wape"]-pair["base"]["wape"],
                         "error_mass_change": float(np.abs(group.risk_conditioned-group.sale_amount).sum()-np.abs(group.base-group.sale_amount).sum())})
        decomposition[key] = rows
        pd.DataFrame(rows).to_csv(output / f"by_{key}.csv", index=False)

    # Verify every saved prediction follows the same already-frozen policy.
    categories = data.management_group_id.astype(str).to_numpy()
    bins = np.searchsorted(policy["edges"], data.predicted_stockout_risk.to_numpy(), side="right")
    category_factor = np.array([policy["category_scales"][c] for c in categories])
    factor = np.array([policy["scales"][f"{c}|{b}"] for c,b in zip(categories,bins)])
    recreated = data.base.to_numpy()/category_factor*factor
    check("saved_predictions_follow_frozen_policy", np.allclose(recreated, data.risk_conditioned, atol=1e-12, rtol=1e-12))
    eligible_series = series.loc[series.eligible > 0]
    positive_mass = series.loc[series.mass > 0]
    change = eligible_series.e1-eligible_series.e0
    result = dict(
        status="PASS", audit_type="post_hoc_saved_prediction_recalculation",
        new_model_fits=0, new_policy_selection=0, external_requests=0,
        raw_eval_decodes=0, assertions=len(assertions), checks=assertions,
        metrics=measures,
        paired_bootstrap=dict(clusters=len(series), draws=4000, estimate=point, ci95=interval.tolist(), share_negative=float(np.mean(deltas<0))),
        per_series=dict(total=len(series), with_primary_rows=len(eligible_series), with_positive_primary_mass=len(positive_mass),
                        improved=int((change < -1e-12).sum()), worsened=int((change > 1e-12).sum()), tied=int((np.abs(change) <= 1e-12).sum()),
                        improvement_measure="within-series absolute-error mass on fixed eligible rows",
                        defined_wape_improved=int((positive_mass.e1 < positive_mass.e0 - 1e-12).sum()),
                        defined_wape_worsened=int((positive_mass.e1 > positive_mass.e0 + 1e-12).sum()),
                        no_primary_rows=int((series.eligible == 0).sum())),
        decomposition=decomposition,
        population=dict(stores=int(data.series_id.str.split("::").str[0].nunique()), products=int(data.series_id.str.split("::").str[1].nunique()),
                        eval_dates=sorted(data.dt.unique().tolist()), forecast_origin=str(origin.iloc[0].date())),
        limitations=["Conditional on recorded zero-stockout rows; no latent-demand claim for stockout rows.",
                     "Bootstrap resamples store-product series in one fixed future week; shared store/product/date shocks are not separately resampled.",
                     "Protocol/run receipts are local records, not an externally timestamped immutable access audit.",
                     "Deterministic first-2000 cohort is not a random sample of all 50000 series."])
    (output / "INDEPENDENT_NATURAL_AUDIT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ["status","assertions","metrics","paired_bootstrap","per_series","population"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.package, args.output)
