#!/usr/bin/env python3
"""Design-only development runner for the independent CENSORCAST v0.5 M5 study.

The executable has no command that can open the fresh guardian or external
outcome shards.  It reads only the allowlisted design shard and shared
covariates, checkpoints long phases, and keeps confirmatory alpha at zero.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from censorcast_v05.features import DesignPanel, FeatureBuilder
from censorcast_v05.metrics import (
    cluster_aggregate,
    exact_item_cluster_delta_curve,
    fixed_policy_familywise_bootstrap,
    metric_bundle,
    paired_point_cluster_bootstrap,
    risk_rank_diagnostics,
    search_two_block_exact_threshold,
)
from censorcast_v05.models import (
    ForecastBundle,
    RiskModel,
    SupportProfile,
    apply_point_policy,
    risk_feature_matrix,
    select_point_policy,
)


VERSION = "0.5.1"
DESIGN_AUTH_PHRASE = "RUN_CENSORCAST_V0_5_M5_DESIGN_ONLY_AFTER_DATA_AUDIT"
FREEZE_AUTH_PHRASE = "FREEZE_CENSORCAST_V0_5_M5_DESIGN_POLICY_AFTER_DESIGN_AUDIT"
CONFIG_RELATIVE = Path("configs") / "DESIGN_PROTOCOL_v0_5.json"
SOURCE_MANIFEST = "SOURCE_MANIFEST_v0_5_DESIGN.json"
DATA_AUDIT = Path("audit") / "DATA_PREPARATION_AUDIT_v0_5.json"
DATA_RECEIPT = Path("audit") / "DATA_PREPARATION_RECEIPT_v0_5.json"
DATA_HASH_INDEX = Path("audit") / "DATA_ARTIFACT_SHA256_v0_5.json"
DATA_LEDGER = Path("state") / "DATA_ACCESS_LEDGER_v0_5.json"
DESIGN_SHARD = Path("prepared") / "design_outcomes_v0_5.npz"
DESIGN_IDS = Path("prepared") / "design_item_ids_v0_5.txt"
CALENDAR = Path("source") / "calendar.csv"
PRICES = Path("source") / "sell_prices.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".npz", prefix=path.name + ".", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Missing/non-regular {label}: {path}")


def verify_source(project: Path, expected_runner: str, expected_config: str, expected_manifest: str) -> dict:
    runner = Path(__file__).resolve()
    if sha256_file(runner) != expected_runner:
        raise RuntimeError("Design runner source hash mismatch")
    config_path = project / CONFIG_RELATIVE
    regular_file(config_path, "design protocol")
    if sha256_file(config_path) != expected_config:
        raise RuntimeError("Design protocol hash mismatch")
    manifest_path = project / SOURCE_MANIFEST
    regular_file(manifest_path, "source manifest")
    if sha256_file(manifest_path) != expected_manifest:
        raise RuntimeError("Source manifest hash mismatch")
    manifest = json_load(manifest_path)
    for relative, specification in manifest["files"].items():
        path = project / relative
        regular_file(path, f"source {relative}")
        if path.stat().st_size != int(specification["size_bytes"]) or sha256_file(path) != specification["sha256"]:
            raise RuntimeError(f"Source package changed: {relative}")
    return {
        "runner_sha256": expected_runner,
        "design_protocol_sha256": expected_config,
        "source_manifest_sha256": expected_manifest,
        "verified_source_files": len(manifest["files"]),
    }


def verify_data_boundary(data_root: Path, config: dict) -> dict:
    audit_path = data_root / DATA_AUDIT
    receipt_path = data_root / DATA_RECEIPT
    index_path = data_root / DATA_HASH_INDEX
    ledger_path = data_root / DATA_LEDGER
    for path, label in (
        (audit_path, "M5 preparation audit"),
        (receipt_path, "M5 preparation receipt"),
        (index_path, "M5 artifact index"),
        (ledger_path, "M5 access ledger"),
    ):
        regular_file(path, label)
    audit = json_load(audit_path)
    receipt = json_load(receipt_path)
    ledger = json_load(ledger_path)
    expected = config["upstream"]
    checks = {
        "audit_pass": audit.get("status") == expected["required_data_audit_status"],
        "preparation_ready": receipt.get("status") == "V0_5_M5_INDEPENDENT_DATA_READY",
        "data_protocol_exact": audit.get("protocol_sha256") == expected["data_protocol_sha256"],
        "data_runner_exact": audit.get("runner_sha256") == expected["data_runner_sha256"],
        "fresh_guardian_closed": ledger.get("fresh_guardian_opened") is False,
        "fresh_guardian_unused": int(ledger.get("fresh_guardian_use_count", -1)) == 0,
        "external_closed": ledger.get("external_opened") is False,
        "alpha_unspent": float(ledger.get("confirmatory_alpha_spent", -1.0)) == 0.0,
        "certificate_absent": ledger.get("certificate_issued") is False,
        "no_guardian_metrics": int(audit.get("guardian_or_external_metrics_computed", -1)) == 0,
    }
    forbidden_receipts = [
        data_root / "state" / "FRESH_GUARDIAN_OPEN_RECEIPT_v0_5.json",
        data_root / "state" / "EXTERNAL_OPEN_RECEIPT_v0_5.json",
        data_root / "audit" / "V0_5_GUARDIAN_DECISION.json",
        data_root / "audit" / "V0_5_EXTERNAL_DECISION.json",
    ]
    checks["no_open_receipts"] = not any(path.exists() for path in forbidden_receipts)
    if not all(checks.values()):
        raise RuntimeError(f"M5 sealed-data boundary failed: {checks}")
    index = json_load(index_path)["artifacts"]
    allowlist = [DESIGN_SHARD, DESIGN_IDS, CALENDAR, PRICES]
    verified = {}
    for relative in allowlist:
        path = data_root / relative
        regular_file(path, f"allowlisted input {relative}")
        specification = index.get(relative.as_posix())
        if not specification:
            raise RuntimeError(f"Allowlisted input absent from upstream hash index: {relative}")
        observed = sha256_file(path)
        if observed != specification["sha256"] or path.stat().st_size != int(specification["size_bytes"]):
            raise RuntimeError(f"Allowlisted upstream input changed: {relative}")
        verified[relative.as_posix()] = observed
    # Deliberately do not open, hash, list, or deserialize either sealed shard.
    return {
        "checks": checks,
        "allowlisted_input_hashes": verified,
        "data_audit_sha256": sha256_file(audit_path),
        "data_receipt_sha256": sha256_file(receipt_path),
        "data_ledger_sha256": sha256_file(ledger_path),
        "sealed_outcome_file_reads": 0,
    }


def load_config(project: Path) -> dict:
    config = json_load(project / CONFIG_RELATIVE)
    if config.get("schema_version") != "0.5-design-1":
        raise RuntimeError("Unexpected design protocol schema")
    return config


def common_preflight(args: argparse.Namespace) -> tuple[Path, Path, Path, dict, dict, dict]:
    project = Path(args.project).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    source = verify_source(
        project,
        args.source_sha256,
        args.design_protocol_sha256,
        args.source_manifest_sha256,
    )
    config = load_config(project)
    boundary = verify_data_boundary(data_root, config)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state" / "RUN_STATE_v0_5.json"
    if state_path.exists():
        state = json_load(state_path)
        expected = {
            "runner_sha256": args.source_sha256,
            "design_protocol_sha256": args.design_protocol_sha256,
            "data_audit_sha256": boundary["data_audit_sha256"],
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Existing development output belongs to different immutable inputs")
    return project, data_root, output, config, source, boundary


def update_state(output: Path, phase: str, args: argparse.Namespace, boundary: dict, **extra: Any) -> None:
    path = output / "state" / "RUN_STATE_v0_5.json"
    previous = json_load(path) if path.exists() else {}
    previous.update(
        {
            "schema_version": "0.5-design-1",
            "runner_version": VERSION,
            "phase": phase,
            "updated_utc": utc_now(),
            "runner_sha256": args.source_sha256,
            "design_protocol_sha256": args.design_protocol_sha256,
            "source_manifest_sha256": args.source_manifest_sha256,
            "data_audit_sha256": boundary["data_audit_sha256"],
            "fresh_guardian_opened": False,
            "external_opened": False,
            "confirmatory_alpha_spent": 0.0,
            "certificate_issued": False,
            **extra,
        }
    )
    atomic_json(path, previous)
    print(f"[CENSORCAST v0.5 design] {phase}", flush=True)


def preflight(args: argparse.Namespace) -> dict:
    project, data_root, output, config, source, boundary = common_preflight(args)
    payload = {
        "schema_version": "0.5-design-1",
        "created_utc": utc_now(),
        "status": "V0_5_M5_DESIGN_ONLY_PREFLIGHT_PASS",
        **source,
        "data_root": str(data_root),
        "output": str(output),
        "upstream_data_audit_status": "V0_5_M5_INDEPENDENT_DATA_AUDIT_PASS",
        "data_boundary": boundary,
        "permitted_outcome_shard": config["upstream"]["permitted_outcome_shard"],
        "forbidden_outcome_shards": config["upstream"]["forbidden_outcome_shards"],
        "fresh_guardian_opened": False,
        "external_opened": False,
        "confirmatory_alpha_spent": 0.0,
        "next": "Manual authorization may start resumable design-only development.",
    }
    atomic_json(output / "audit" / "DESIGN_PREFLIGHT_v0_5.json", payload)
    update_state(output, "PREFLIGHT_PASS", args, boundary)
    return payload


def block_arrays(path: Path) -> dict[str, np.ndarray]:
    regular_file(path, f"block checkpoint {path.name}")
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def prediction_candidates(block: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "seasonal_naive_7": block["seasonal_naive_7"],
        "seasonal_median_4": block["seasonal_median_4"],
        "raw_poisson_histgb": block["raw_poisson_histgb"],
    }


def generate_base_block(
    builder: FeatureBuilder,
    bundle: ForecastBundle,
    support: SupportProfile,
    start_day: int,
    end_day: int,
) -> dict[str, np.ndarray]:
    series_count = builder.panel.n_series
    days = np.arange(int(start_day), int(end_day) + 1, dtype=np.int32)
    shape = (series_count, len(days))
    names = (
        "truth",
        "observed",
        "censored",
        "seasonal_naive_7",
        "seasonal_median_4",
        "raw_poisson_histgb",
        "censored_poisson_em",
        "censor_probability",
        "support_distance",
    )
    arrays = {name: np.empty(shape, dtype=np.float32) for name in names}
    for column, (target_day, series, target) in enumerate(builder.block_pairs(start_day, end_day)):
        common_x = builder.make_features(series, target, include_censor=False)
        censor_x = builder.make_features(series, target, include_censor=True)
        seasonal, median = builder.seasonal_predictions(series, target)
        labels = builder.labels(series, target)
        predicted = bundle.predict(common_x, censor_x)
        arrays["truth"][:, column] = labels["truth"]
        arrays["observed"][:, column] = labels["observed"]
        arrays["censored"][:, column] = labels["censored"]
        arrays["seasonal_naive_7"][:, column] = seasonal
        arrays["seasonal_median_4"][:, column] = median
        arrays["raw_poisson_histgb"][:, column] = predicted["raw_poisson_histgb"]
        arrays["censored_poisson_em"][:, column] = predicted["censored_poisson_em"]
        arrays["censor_probability"][:, column] = predicted["censor_probability"]
        arrays["support_distance"][:, column] = support.distance(common_x)
        if (column + 1) % 20 == 0 or column + 1 == len(days):
            print(f"  predicted days: {column + 1}/{len(days)}", flush=True)
    arrays["target_days"] = days
    return arrays


def initialize_history(
    block: dict[str, np.ndarray], policy: dict, prior_mass: float
) -> tuple[np.ndarray, np.ndarray, float]:
    candidates = prediction_candidates(block)
    _, proposal = apply_point_policy(
        policy,
        candidates,
        block["censored_poisson_em"],
        block["censor_probability"],
    )
    n_series, n_days = proposal.shape
    cumulative_error = np.zeros(n_series, dtype=np.float64)
    cumulative_demand = np.zeros(n_series, dtype=np.float64)
    global_error = 0.0
    global_demand = 0.0
    initial = float(policy["baseline_selection_wape"])
    for day in range(n_days):
        observed = block["observed"][:, day].astype(np.float64)
        predicted = proposal[:, day].astype(np.float64)
        error = np.abs(observed - predicted)
        cumulative_error += error
        cumulative_demand += np.abs(observed)
        global_error += float(error.sum())
        global_demand += float(np.abs(observed).sum())
    global_prior = (global_error + prior_mass * initial) / max(global_demand + prior_mass, 1e-12)
    return cumulative_error, cumulative_demand, float(global_prior)


def generate_risk_block(
    builder: FeatureBuilder,
    bundle: ForecastBundle,
    support: SupportProfile,
    point_policy: dict,
    start_day: int,
    end_day: int,
    cumulative_error: np.ndarray,
    cumulative_demand: np.ndarray,
    global_prior: float,
    prior_mass: float,
    risk_model: RiskModel | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    series_count = builder.panel.n_series
    days = np.arange(int(start_day), int(end_day) + 1, dtype=np.int32)
    shape = (series_count, len(days))
    arrays = {
        name: np.empty(shape, dtype=np.float32)
        for name in (
            "truth",
            "observed",
            "censored",
            "baseline",
            "proposal",
            "raw_poisson_histgb",
            "censored_poisson_em",
            "censor_probability",
            "prior_observed_wape",
            "instantaneous_absolute_error",
        )
    }
    risk_feature_blocks: list[np.ndarray] = []
    global_error = float(global_prior * max(cumulative_demand.sum() + prior_mass, 1.0))
    global_demand = float(cumulative_demand.sum() + prior_mass)
    for column, (target_day, series, target) in enumerate(builder.block_pairs(start_day, end_day)):
        prior = (cumulative_error + prior_mass * global_prior) / np.maximum(
            cumulative_demand + prior_mass, 1e-12
        )
        common_x = builder.make_features(series, target, include_censor=False)
        censor_x = builder.make_features(series, target, include_censor=True)
        seasonal, median = builder.seasonal_predictions(series, target)
        labels = builder.labels(series, target)
        predicted = bundle.predict(common_x, censor_x)
        candidates = {
            "seasonal_naive_7": seasonal,
            "seasonal_median_4": median,
            "raw_poisson_histgb": predicted["raw_poisson_histgb"],
        }
        baseline, proposal = apply_point_policy(
            point_policy,
            candidates,
            predicted["censored_poisson_em"],
            predicted["censor_probability"],
        )
        features = risk_feature_matrix(
            baseline=baseline,
            proposal=proposal,
            seasonal_naive=seasonal,
            seasonal_median=median,
            raw=predicted["raw_poisson_histgb"],
            em=predicted["censored_poisson_em"],
            censor_probability=predicted["censor_probability"],
            common_features=common_x,
            censor_features=censor_x,
            prior_observed_wape=prior.astype(np.float32),
            support_distance=support.distance(common_x),
        )
        arrays["truth"][:, column] = labels["truth"]
        arrays["observed"][:, column] = labels["observed"]
        arrays["censored"][:, column] = labels["censored"]
        arrays["baseline"][:, column] = baseline
        arrays["proposal"][:, column] = proposal
        arrays["raw_poisson_histgb"][:, column] = predicted["raw_poisson_histgb"]
        arrays["censored_poisson_em"][:, column] = predicted["censored_poisson_em"]
        arrays["censor_probability"][:, column] = predicted["censor_probability"]
        arrays["prior_observed_wape"][:, column] = prior
        if risk_model is None:
            arrays["instantaneous_absolute_error"][:, column] = np.nan
            risk_feature_blocks.append(features.astype(np.float32))
        else:
            arrays["instantaneous_absolute_error"][:, column] = risk_model.predict_absolute_error(features)

        observed = labels["observed"].astype(np.float64)
        error = np.abs(observed - proposal.astype(np.float64))
        cumulative_error += error
        cumulative_demand += np.abs(observed)
        global_error += float(error.sum())
        global_demand += float(np.abs(observed).sum())
        global_prior = global_error / max(global_demand, 1e-12)
        if (column + 1) % 20 == 0 or column + 1 == len(days):
            print(f"  risk-feature days: {column + 1}/{len(days)}", flush=True)
    arrays["target_days"] = days
    if risk_feature_blocks:
        # series-major ordering to align with all flattened block arrays.
        arrays["risk_features"] = np.stack(risk_feature_blocks, axis=1).astype(np.float32)
    return arrays, cumulative_error, cumulative_demand, float(global_prior)


def history_save(path: Path, error: np.ndarray, demand: np.ndarray, global_prior: float) -> None:
    atomic_npz(
        path,
        cumulative_error=np.asarray(error, dtype=np.float64),
        cumulative_demand=np.asarray(demand, dtype=np.float64),
        global_prior=np.asarray([global_prior], dtype=np.float64),
    )


def history_load(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    values = block_arrays(path)
    return values["cumulative_error"], values["cumulative_demand"], float(values["global_prior"][0])


def item_codes(panel: DesignPanel) -> tuple[np.ndarray, list[str]]:
    levels = sorted(set(panel.metadata["item_id"].astype(str).tolist()))
    mapping = {value: index for index, value in enumerate(levels)}
    return np.asarray([mapping[value] for value in panel.metadata["item_id"].astype(str)], dtype=np.int32), levels


def score_for_weight(block: dict[str, np.ndarray], weight: float, accepted_wape_cap: float) -> np.ndarray:
    mass = np.maximum(block["proposal"].astype(np.float64), 1.0)
    instantaneous_ratio = block["instantaneous_absolute_error"].astype(np.float64) / mass
    historical = block["prior_observed_wape"].astype(np.float64)
    return (mass * (float(weight) * instantaneous_ratio + (1.0 - float(weight)) * historical - float(accepted_wape_cap))).astype(np.float64)


def write_frontier_csv(path: Path, frontier: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(frontier).to_csv(temporary, index=False)
    temporary.replace(path)


def output_artifacts(output: Path) -> list[Path]:
    fixed = [
        output / "audit" / "DESIGN_PREFLIGHT_v0_5.json",
        output / "state" / "RUN_STATE_v0_5.json",
        output / "models" / "forecast_bundle_v0_5.joblib",
        output / "models" / "support_profile_v0_5.joblib",
        output / "models" / "risk_model_v0_5.joblib",
        output / "policy" / "POINT_POLICY_v0_5.json",
        output / "policy" / "SELECTIVE_POLICY_v0_5.json",
        output / "diagnostics" / "RISK_DIAGNOSTICS_v0_5.json",
        output / "diagnostics" / "FIXED_POLICY_BOOTSTRAP_v0_5.json",
        output / "diagnostics" / "POINT_COMPARISON_v0_5.json",
        output / "results" / "DEVELOPMENT_DECISION_v0_5.json",
    ]
    fixed.extend(sorted((output / "frontiers").glob("*.csv")))
    return [path for path in fixed if path.is_file()]


def create_artifact_index(output: Path) -> dict:
    artifacts = {}
    for path in output_artifacts(output):
        relative = path.relative_to(output).as_posix()
        artifacts[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    payload = {
        "schema_version": "0.5-design-1",
        "created_utc": utc_now(),
        "status": "V0_5_DESIGN_ARTIFACT_INDEX_CREATED",
        "artifacts": artifacts,
    }
    atomic_json(output / "audit" / "DESIGN_ARTIFACT_INDEX_v0_5.json", payload)
    return payload


def develop(args: argparse.Namespace) -> dict:
    if args.phrase != DESIGN_AUTH_PHRASE:
        raise RuntimeError("Exact design-only authorization phrase required")
    project, data_root, output, config, source, boundary = common_preflight(args)
    decision_path = output / "results" / "DEVELOPMENT_DECISION_v0_5.json"
    if decision_path.exists():
        existing = json_load(decision_path)
        print("[CENSORCAST v0.5 design] COMPLETED_DESIGN_RUN_REUSED", flush=True)
        return existing
    preflight(args)
    checkpoint = output / "checkpoints"
    model_dir = output / "models"
    policy_dir = output / "policy"
    diagnostics_dir = output / "diagnostics"
    for path in (checkpoint, model_dir, policy_dir, diagnostics_dir, output / "frontiers", output / "results"):
        path.mkdir(parents=True, exist_ok=True)

    update_state(output, "LOADING_ALLOWLISTED_DESIGN_DATA", args, boundary)
    panel = DesignPanel.load(data_root / DESIGN_SHARD, data_root / CALENDAR, data_root / PRICES)
    builder = FeatureBuilder(panel)
    update_state(
        output,
        "DESIGN_DATA_READY",
        args,
        boundary,
        design_series=panel.n_series,
        design_days=panel.n_days,
        raw_dataset_rows_loaded=0,
        sealed_outcome_file_reads=0,
    )

    forecast_path = model_dir / "forecast_bundle_v0_5.joblib"
    support_path = model_dir / "support_profile_v0_5.joblib"
    if forecast_path.exists() and support_path.exists():
        bundle = joblib.load(forecast_path)
        support = joblib.load(support_path)
        print("[CENSORCAST v0.5 design] BASE_MODELS_REUSED", flush=True)
    else:
        base_cfg = config["base_models"]
        start, end = config["time_partition"]["base_fit_target_days"]
        series, days = builder.sample_pairs(
            start,
            end,
            int(base_cfg["training_sample_rows"]),
            int(base_cfg["random_seed"]),
        )
        print(f"[CENSORCAST v0.5 design] fitting base models on {len(series):,} sampled rows", flush=True)
        common_x = builder.make_features(series, days, include_censor=False)
        censor_x = builder.make_features(series, days, include_censor=True)
        labels = builder.labels(series, days)
        support = SupportProfile.fit(common_x)
        bundle = ForecastBundle.fit(
            common_x,
            censor_x,
            labels["observed"],
            labels["capacity"],
            labels["censored"],
            base_cfg,
            int(base_cfg["random_seed"]),
        )
        atomic_joblib(forecast_path, bundle)
        atomic_joblib(support_path, support)
        del common_x, censor_x, labels, series, days
        gc.collect()
    update_state(output, "BASE_MODELS_CHECKPOINTED", args, boundary)

    selection_path = checkpoint / "selection_base_v0_5.npz"
    if not selection_path.exists():
        selection = generate_base_block(
            builder, bundle, support, *config["time_partition"]["selection"]
        )
        atomic_npz(selection_path, **selection)
    selection = block_arrays(selection_path)
    point_policy_path = policy_dir / "POINT_POLICY_v0_5.json"
    if point_policy_path.exists():
        point_policy = json_load(point_policy_path)
    else:
        point_policy = select_point_policy(
            selection["truth"],
            prediction_candidates(selection),
            selection["censor_probability"],
            selection["censored_poisson_em"],
            config["base_models"]["adapter_strengths"],
            config["base_models"]["adapter_powers"],
        )
        point_policy.update(
            {
                "schema_version": "0.5-design-1",
                "created_utc": utc_now(),
                "status": "V0_5_POINT_POLICY_SELECTED_ON_DESIGN_SELECTION",
                "statistical_use": "DEVELOPMENT_ONLY",
                "item_id_feature_used": False,
            }
        )
        atomic_json(point_policy_path, point_policy)
    update_state(output, "POINT_POLICY_CHECKPOINTED", args, boundary)

    history_selection = checkpoint / "history_after_selection_v0_5.npz"
    prior_mass = float(config["risk_layer"]["historical_prior_demand_mass"])
    if not history_selection.exists():
        ce, cd, gp = initialize_history(selection, point_policy, prior_mass)
        history_save(history_selection, ce, cd, gp)
    ce, cd, gp = history_load(history_selection)

    risk_train_path = checkpoint / "risk_train_v0_5.npz"
    history_risk = checkpoint / "history_after_risk_train_v0_5.npz"
    if not risk_train_path.exists() or not history_risk.exists():
        risk_train, ce, cd, gp = generate_risk_block(
            builder,
            bundle,
            support,
            point_policy,
            *config["time_partition"]["risk_train"],
            ce,
            cd,
            gp,
            prior_mass,
            None,
        )
        atomic_npz(risk_train_path, **risk_train)
        history_save(history_risk, ce, cd, gp)
    risk_train = block_arrays(risk_train_path)
    risk_model_path = model_dir / "risk_model_v0_5.joblib"
    risk_cfg = config["risk_layer"]
    if risk_model_path.exists():
        risk_model = joblib.load(risk_model_path)
    else:
        x = risk_train["risk_features"].reshape(-1, risk_train["risk_features"].shape[-1])
        error = np.abs(risk_train["truth"] - risk_train["proposal"]).ravel()
        max_rows = int(risk_cfg["training_sample_rows"])
        if len(error) > max_rows:
            rng = np.random.default_rng(int(config["base_models"]["random_seed"]) + 4001)
            chosen = rng.choice(len(error), size=max_rows, replace=False)
            x_fit, error_fit = x[chosen], error[chosen]
        else:
            x_fit, error_fit = x, error
        risk_model = RiskModel.fit(
            x_fit,
            error_fit,
            risk_cfg,
            int(config["base_models"]["random_seed"]) + 5001,
        )
        atomic_joblib(risk_model_path, risk_model)
        del x_fit, error_fit, x, error
        gc.collect()
    update_state(output, "RISK_MODEL_CHECKPOINTED", args, boundary)

    # Forward-only diagnostic: fit on early risk-train days, predict later days.
    risk_diagnostic_path = diagnostics_dir / "RISK_DIAGNOSTICS_v0_5.json"
    if not risk_diagnostic_path.exists():
        rf = risk_train["risk_features"]
        initial_days = int(risk_cfg["forward_diagnostic_initial_days"])
        fold_days = int(risk_cfg["forward_diagnostic_fold_days"])
        forward_rows = []
        for fold_start in range(initial_days, rf.shape[1], fold_days):
            fold_end = min(fold_start + fold_days, rf.shape[1])
            train_x = rf[:, :fold_start, :].reshape(-1, rf.shape[-1])
            train_y = np.abs(risk_train["truth"][:, :fold_start] - risk_train["proposal"][:, :fold_start]).ravel()
            diagnostic_cfg = dict(risk_cfg)
            diagnostic_cfg["histgb"] = dict(risk_cfg["histgb"])
            diagnostic_cfg["histgb"]["max_iter"] = min(140, int(risk_cfg["histgb"]["max_iter"]))
            limit = min(len(train_y), 600_000)
            if len(train_y) > limit:
                rng = np.random.default_rng(7001 + fold_start)
                take = rng.choice(len(train_y), size=limit, replace=False)
                train_x, train_y = train_x[take], train_y[take]
            diagnostic_model = RiskModel.fit(train_x, train_y, diagnostic_cfg, 8001 + fold_start)
            test_x = rf[:, fold_start:fold_end, :].reshape(-1, rf.shape[-1])
            predicted_risk = diagnostic_model.predict_absolute_error(test_x)
            forward_rows.append(
                {
                    "train_days": [1, fold_start],
                    "validation_day_offsets": [fold_start + 1, fold_end],
                    **risk_rank_diagnostics(
                        risk_train["truth"][:, fold_start:fold_end],
                        risk_train["proposal"][:, fold_start:fold_end],
                        predicted_risk,
                    ),
                }
            )
            del train_x, train_y, test_x, predicted_risk, diagnostic_model
            gc.collect()
        full_risk = risk_model.predict_absolute_error(rf.reshape(-1, rf.shape[-1])).reshape(risk_train["truth"].shape)
        diagnostics = {
            "schema_version": "0.5-design-1",
            "status": "V0_5_FORWARD_RISK_DIAGNOSTIC_COMPLETED",
            "target": risk_cfg["target"],
            "forward_folds": forward_rows,
            "in_sample_reference": risk_rank_diagnostics(
                risk_train["truth"], risk_train["proposal"], full_risk
            ),
            "truth_used_as_deployment_feature": False,
            "prior_history_uses_observed_sales_only": True,
        }
        atomic_json(risk_diagnostic_path, diagnostics)
    diagnostics = json_load(risk_diagnostic_path)

    def ensure_scored_block(name: str, previous_history: Path, next_history: Path) -> dict[str, np.ndarray]:
        path = checkpoint / f"{name}_v0_5.npz"
        if path.exists() and next_history.exists():
            return block_arrays(path)
        local_ce, local_cd, local_gp = history_load(previous_history)
        block, local_ce, local_cd, local_gp = generate_risk_block(
            builder,
            bundle,
            support,
            point_policy,
            *config["time_partition"][name],
            local_ce,
            local_cd,
            local_gp,
            prior_mass,
            risk_model,
        )
        atomic_npz(path, **block)
        history_save(next_history, local_ce, local_cd, local_gp)
        return block

    history_cal_a = checkpoint / "history_after_calibration_a_v0_5.npz"
    cal_a = ensure_scored_block("calibration_a", history_risk, history_cal_a)
    history_cal_b = checkpoint / "history_after_calibration_b_v0_5.npz"
    cal_b = ensure_scored_block("calibration_b", history_cal_a, history_cal_b)
    update_state(output, "CALIBRATION_BLOCKS_CHECKPOINTED", args, boundary)

    item_code, item_levels = item_codes(panel)
    item_a = np.repeat(item_code, cal_a["truth"].shape[1])
    item_b = np.repeat(item_code, cal_b["truth"].shape[1])
    baseline_selection_wape = float(point_policy["baseline_selection_wape"])
    accepted_cap = float(config["selective_contract"]["accepted_wape_cap_multiplier_of_selection_baseline"]) * baseline_selection_wape
    margin = config["selective_contract"]["development_margin"]
    exact_candidates = []
    best_policy = None
    for weight in config["risk_layer"]["instantaneous_weights"]:
        score_a = score_for_weight(cal_a, float(weight), accepted_cap)
        score_b = score_for_weight(cal_b, float(weight), accepted_cap)
        curve_a = exact_item_cluster_delta_curve(
            score=score_a,
            truth=cal_a["truth"],
            prediction=cal_a["proposal"],
            item_cluster=item_a,
            baseline_selection_wape=baseline_selection_wape,
        )
        curve_b = exact_item_cluster_delta_curve(
            score=score_b,
            truth=cal_b["truth"],
            prediction=cal_b["proposal"],
            item_cluster=item_b,
            baseline_selection_wape=baseline_selection_wape,
        )
        result, frontier = search_two_block_exact_threshold(
            curve_a,
            curve_b,
            maximum_ratio_ucb=float(margin["maximum_accepted_wape_ratio_ucb"]),
            minimum_coverage_lcb=float(margin["minimum_coverage_lcb"]),
        )
        result["instantaneous_weight"] = float(weight)
        result["historical_weight"] = 1.0 - float(weight)
        exact_candidates.append(result)
        write_frontier_csv(output / "frontiers" / f"exact_frontier_w{int(round(float(weight)*100)):03d}.csv", frontier)
        if result["status"] == "EXACT_MARGIN_FEASIBLE_THRESHOLD_FOUND":
            if best_policy is None:
                best_policy = result
            else:
                current_key = (
                    result["minimum_block_coverage"],
                    -result["maximum_block_wape_ratio_ucb95"],
                    -float(weight),
                )
                best_key = (
                    best_policy["minimum_block_coverage"],
                    -best_policy["maximum_block_wape_ratio_ucb95"],
                    -best_policy["instantaneous_weight"],
                )
                if current_key > best_key:
                    best_policy = result
        del score_a, score_b, curve_a, curve_b
        gc.collect()

    selective_path = policy_dir / "SELECTIVE_POLICY_v0_5.json"
    if best_policy is None:
        closest = min(exact_candidates, key=lambda row: (row["margin_violation"], -row["minimum_block_coverage"]))
        selective_policy = {
            "schema_version": "0.5-design-1",
            "created_utc": utc_now(),
            "status": "NO_EXACT_MARGIN_FEASIBLE_POLICY",
            "selected": None,
            "best_near_miss": closest,
            "candidates": exact_candidates,
            "accepted_wape_cap": accepted_cap,
            "baseline_selection_wape": baseline_selection_wape,
            "shadow_opened_for_policy_selection": False,
        }
        atomic_json(selective_path, selective_policy)
        fixed_bootstrap = {
            "status": "NOT_RUN_NO_POLICY",
            "fresh_guardian_opened": False,
            "external_opened": False,
        }
        point_comparison = {"status": "NOT_RUN_NO_POLICY"}
        atomic_json(diagnostics_dir / "FIXED_POLICY_BOOTSTRAP_v0_5.json", fixed_bootstrap)
        atomic_json(diagnostics_dir / "POINT_COMPARISON_v0_5.json", point_comparison)
        decision = {
            "schema_version": "0.5-design-1",
            "created_utc": utc_now(),
            "status": "V0_5_DESIGN_NO_GO_KEEP_GUARDIAN_SEALED",
            "reason": "No calibration A/B threshold met the prespecified development margin.",
            "point_policy": point_policy,
            "selective_policy": selective_policy,
            "risk_diagnostics": diagnostics,
            "shadow_policy_metrics_computed": False,
            "fresh_guardian_opened": False,
            "external_opened": False,
            "confirmatory_alpha_spent": 0.0,
            "certificate_issued": False,
        }
        atomic_json(decision_path, decision)
        update_state(output, "DESIGN_NO_GO_BEFORE_SHADOW", args, boundary)
        create_artifact_index(output)
        return decision

    selective_policy = {
        "schema_version": "0.5-design-1",
        "created_utc": utc_now(),
        "status": "EXACT_CALIBRATION_POLICY_FROZEN_BEFORE_SHADOW",
        "selected": best_policy,
        "candidates": exact_candidates,
        "accepted_wape_cap": accepted_cap,
        "baseline_selection_wape": baseline_selection_wape,
        "shadow_opened_for_policy_selection": False,
        "statistical_use": "DEVELOPMENT_ONLY_NOT_A_CERTIFICATE",
    }
    atomic_json(selective_path, selective_policy)
    update_state(output, "SELECTIVE_POLICY_FROZEN_BEFORE_SHADOW", args, boundary)

    history_shadow = checkpoint / "history_after_shadow_v0_5.npz"
    shadow = ensure_scored_block("shadow", history_cal_b, history_shadow)
    selected_weight = float(best_policy["instantaneous_weight"])
    threshold = float(best_policy["threshold"])
    score_blocks = {
        "calibration_a": score_for_weight(cal_a, selected_weight, accepted_cap),
        "calibration_b": score_for_weight(cal_b, selected_weight, accepted_cap),
        "shadow": score_for_weight(shadow, selected_weight, accepted_cap),
    }
    blocks = {"calibration_a": cal_a, "calibration_b": cal_b, "shadow": shadow}
    cluster_count = len(item_levels)
    aggregates = {}
    for name, block in blocks.items():
        clusters = np.repeat(item_code, block["truth"].shape[1])
        aggregates[name] = cluster_aggregate(
            truth=block["truth"],
            prediction=block["proposal"],
            score=score_blocks[name],
            threshold=threshold,
            cluster=clusters,
            cluster_count=cluster_count,
        )
    categories = panel.metadata["cat_id"].astype(str)
    for category in sorted(set(categories.tolist())):
        pooled = np.zeros((cluster_count, 4), dtype=np.float64)
        series_mask = categories == category
        for name, block in blocks.items():
            days_in_block = block["truth"].shape[1]
            pooled += cluster_aggregate(
                truth=block["truth"],
                prediction=block["proposal"],
                score=score_blocks[name],
                threshold=threshold,
                cluster=np.repeat(item_code, days_in_block),
                cluster_count=cluster_count,
                mask=np.repeat(series_mask, days_in_block),
            )
        aggregates[f"pooled_cat_id:{category}"] = pooled
    fixed_bootstrap = fixed_policy_familywise_bootstrap(
        aggregates,
        baseline_selection_wape=baseline_selection_wape,
        reps=int(config["selective_contract"]["fixed_policy_bootstrap_reps"]),
        seed=int(config["base_models"]["random_seed"]) + 9001,
        familywise_alpha=float(config["selective_contract"]["fixed_policy_familywise_alpha"]),
        minimum_coverage_lcb=float(margin["minimum_coverage_lcb"]),
        maximum_ratio_ucb=float(margin["maximum_accepted_wape_ratio_ucb"]),
    )
    fixed_bootstrap["selected_weight"] = selected_weight
    fixed_bootstrap["selected_threshold"] = threshold
    atomic_json(diagnostics_dir / "FIXED_POLICY_BOOTSTRAP_v0_5.json", fixed_bootstrap)

    shadow_cluster = np.repeat(item_code, shadow["truth"].shape[1])
    point_comparison = paired_point_cluster_bootstrap(
        truth=shadow["truth"],
        baseline=shadow["baseline"],
        proposal=shadow["proposal"],
        cluster=shadow_cluster,
        reps=int(config["point_forecast_gate"]["paired_item_cluster_bootstrap_reps"]),
        seed=int(config["base_models"]["random_seed"]) + 10001,
    )
    atomic_json(diagnostics_dir / "POINT_COMPARISON_v0_5.json", point_comparison)
    shadow_metrics = {
        "baseline": metric_bundle(shadow["truth"], shadow["baseline"]),
        "proposal": metric_bundle(shadow["truth"], shadow["proposal"]),
        "risk_rank": risk_rank_diagnostics(
            shadow["truth"], shadow["proposal"], shadow["instantaneous_absolute_error"]
        ),
    }
    point_gate = config["point_forecast_gate"]
    checks = {
        "upstream_data_audit_pass": True,
        "sealed_outcome_file_reads_zero": boundary["sealed_outcome_file_reads"] == 0,
        "fresh_guardian_remains_closed": True,
        "external_remains_closed": True,
        "censor_signal_selected": bool(point_policy["uses_censor_signal"]),
        "exact_calibration_margin_reachable": True,
        "fixed_policy_familywise_margin_pass": fixed_bootstrap["status"] == "FIXED_POLICY_FAMILYWISE_PASS",
        "shadow_point_wape_ratio_pass": point_comparison["wape_ratio_proposal_to_baseline"] <= float(point_gate["shadow_wape_ratio_point_max"]),
        "shadow_point_wape_ratio_ci_pass": point_comparison["wape_ratio_ci95"][1] <= float(point_gate["shadow_wape_ratio_ci95_upper_max"]),
    }
    status = (
        "V0_5_DESIGN_GO_PENDING_MANUAL_FREEZE"
        if all(checks.values())
        else "V0_5_DESIGN_NO_GO_KEEP_GUARDIAN_SEALED"
    )
    decision = {
        "schema_version": "0.5-design-1",
        "program_version": VERSION,
        "created_utc": utc_now(),
        "status": status,
        "statistical_status": "DESIGN_ONLY_NOT_A_CERTIFICATE",
        "checks": checks,
        "point_policy": point_policy,
        "selective_policy": selective_policy,
        "shadow_metrics": shadow_metrics,
        "point_comparison": point_comparison,
        "fixed_policy_bootstrap": fixed_bootstrap,
        "risk_diagnostics": diagnostics,
        "data_scope": {
            "design_series": panel.n_series,
            "design_days": panel.n_days,
            "sealed_outcome_file_reads": 0,
            "raw_dataset_rows_loaded": 0,
            "fresh_guardian_opened": False,
            "external_opened": False,
            "confirmatory_alpha_spent": 0.0,
            "certificate_issued": False,
        },
        "next": (
            "Manual audit, then administrative freeze; do not open the guardian from this notebook."
            if status == "V0_5_DESIGN_GO_PENDING_MANUAL_FREEZE"
            else "STOP. Diagnose design failures without opening the fresh guardian."
        ),
    }
    atomic_json(decision_path, decision)
    update_state(output, "DESIGN_COMPLETE", args, boundary, decision_status=status)
    create_artifact_index(output)
    return decision


def audit(args: argparse.Namespace) -> dict:
    project, data_root, output, config, source, boundary = common_preflight(args)
    decision_path = output / "results" / "DEVELOPMENT_DECISION_v0_5.json"
    index_path = output / "audit" / "DESIGN_ARTIFACT_INDEX_v0_5.json"
    regular_file(decision_path, "development decision")
    regular_file(index_path, "development artifact index")
    decision = json_load(decision_path)
    index = json_load(index_path)
    verified = 0
    for relative, specification in index["artifacts"].items():
        path = output / relative
        regular_file(path, f"development artifact {relative}")
        if path.stat().st_size != int(specification["size_bytes"]) or sha256_file(path) != specification["sha256"]:
            raise RuntimeError(f"Development artifact changed: {relative}")
        verified += 1
    boundary_after = verify_data_boundary(data_root, config)
    if boundary_after["data_ledger_sha256"] != boundary["data_ledger_sha256"]:
        raise RuntimeError("M5 access ledger changed during design development")
    go = decision.get("status") == "V0_5_DESIGN_GO_PENDING_MANUAL_FREEZE"
    status = (
        "V0_5_DESIGN_DEVELOPMENT_AUDIT_PASS_GO_PENDING_FREEZE"
        if go
        else "V0_5_DESIGN_DEVELOPMENT_AUDIT_PASS_NO_GO"
    )
    payload = {
        "schema_version": "0.5-design-1",
        "created_utc": utc_now(),
        "status": status,
        "decision": decision.get("status"),
        "verified_artifacts": verified,
        "runner_sha256": args.source_sha256,
        "design_protocol_sha256": args.design_protocol_sha256,
        "data_audit_sha256": boundary["data_audit_sha256"],
        "development_decision_sha256": sha256_file(decision_path),
        "artifact_index_sha256": sha256_file(index_path),
        "sealed_outcome_file_reads": 0,
        "fresh_guardian_opened": False,
        "external_opened": False,
        "confirmatory_alpha_spent": 0.0,
        "certificate_issued": False,
        "instruction": (
            "STOP and return this complete audit before the manual freeze cell."
            if go
            else "STOP. The guardian remains sealed; do not run the freeze cell."
        ),
    }
    audit_path = output / "audit" / "DESIGN_DEVELOPMENT_AUDIT_v0_5.json"
    atomic_json(audit_path, payload)
    zip_path = output.parent / "CENSORCAST_v0_5_M5_DESIGN_RESULTS.zip"
    temporary = zip_path.with_suffix(".zip.tmp")
    report_paths = output_artifacts(output) + [index_path, audit_path]
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(set(report_paths)):
            if "models" in path.relative_to(output).parts or "checkpoints" in path.relative_to(output).parts:
                continue
            archive.write(path, path.relative_to(output).as_posix())
    temporary.replace(zip_path)
    payload["result_zip"] = str(zip_path)
    payload["result_zip_sha256"] = sha256_file(zip_path)
    atomic_json(audit_path, payload)
    return payload


def freeze(args: argparse.Namespace) -> dict:
    if args.phrase != FREEZE_AUTH_PHRASE:
        raise RuntimeError("Exact administrative-freeze phrase required")
    project, data_root, output, config, source, boundary = common_preflight(args)
    audit_path = output / "audit" / "DESIGN_DEVELOPMENT_AUDIT_v0_5.json"
    decision_path = output / "results" / "DEVELOPMENT_DECISION_v0_5.json"
    regular_file(audit_path, "design audit")
    regular_file(decision_path, "design decision")
    design_audit = json_load(audit_path)
    decision = json_load(decision_path)
    if design_audit.get("status") != "V0_5_DESIGN_DEVELOPMENT_AUDIT_PASS_GO_PENDING_FREEZE":
        raise RuntimeError("Administrative freeze requires a GO design audit")
    if decision.get("status") != "V0_5_DESIGN_GO_PENDING_MANUAL_FREEZE":
        raise RuntimeError("Administrative freeze requires a GO decision")
    freeze_dir = Path(args.freeze_output).resolve()
    receipt_path = freeze_dir / "FROZEN_V0_5_M5_DESIGN_POLICY.json"
    if receipt_path.exists():
        existing = json_load(receipt_path)
        if existing.get("development_decision_sha256") == sha256_file(decision_path):
            print("[CENSORCAST v0.5 design] EXISTING_FREEZE_REUSED", flush=True)
            return existing
        raise RuntimeError("Existing freeze belongs to a different decision")
    freeze_dir.mkdir(parents=True, exist_ok=True)
    copy_specs = {
        "forecast_bundle_v0_5.joblib": output / "models" / "forecast_bundle_v0_5.joblib",
        "support_profile_v0_5.joblib": output / "models" / "support_profile_v0_5.joblib",
        "risk_model_v0_5.joblib": output / "models" / "risk_model_v0_5.joblib",
        "POINT_POLICY_v0_5.json": output / "policy" / "POINT_POLICY_v0_5.json",
        "SELECTIVE_POLICY_v0_5.json": output / "policy" / "SELECTIVE_POLICY_v0_5.json",
        "DESIGN_PROTOCOL_v0_5.json": project / CONFIG_RELATIVE,
        "DESIGN_DEVELOPMENT_AUDIT_v0_5.json": audit_path,
    }
    hashes = {}
    for name, source_path in copy_specs.items():
        regular_file(source_path, f"freeze source {name}")
        destination = freeze_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source_path, temporary)
        temporary.replace(destination)
        hashes[name] = sha256_file(destination)
    receipt = {
        "schema_version": "0.5-design-1",
        "created_utc": utc_now(),
        "status": "V0_5_M5_DESIGN_POLICY_ADMINISTRATIVELY_FROZEN",
        "statistical_status": "AWAITING_SINGLE_USE_FRESH_GUARDIAN",
        "runner_sha256": args.source_sha256,
        "design_protocol_sha256": args.design_protocol_sha256,
        "development_decision_sha256": sha256_file(decision_path),
        "design_audit_sha256": sha256_file(audit_path),
        "frozen_artifacts": hashes,
        "point_policy": decision["point_policy"],
        "selective_policy": decision["selective_policy"]["selected"],
        "fresh_guardian_opened": False,
        "external_opened": False,
        "confirmatory_alpha_spent": 0.0,
        "certificate_issued": False,
        "authorization_phrase_sha256": hashlib.sha256(FREEZE_AUTH_PHRASE.encode()).hexdigest(),
        "next": "STOP. A separate audited notebook is required for the single-use fresh guardian.",
    }
    atomic_json(receipt_path, receipt)
    atomic_json(
        freeze_dir / "FREEZE_AUDIT_v0_5.json",
        {
            "schema_version": "0.5-design-1",
            "created_utc": utc_now(),
            "status": "V0_5_M5_DESIGN_FREEZE_AUDIT_PASS",
            "frozen_policy_sha256": sha256_file(receipt_path),
            "fresh_guardian_opened": False,
            "external_opened": False,
            "confirmatory_alpha_spent": 0.0,
            "certificate_issued": False,
            "instruction": "STOP. Do not open the fresh guardian from this notebook.",
        },
    )
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("preflight", "develop", "audit", "freeze"):
        command = sub.add_parser(name)
        command.add_argument("--project", required=True)
        command.add_argument("--data-root", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--source-sha256", required=True)
        command.add_argument("--design-protocol-sha256", required=True)
        command.add_argument("--source-manifest-sha256", required=True)
        if name in {"develop", "freeze"}:
            command.add_argument("--phrase", required=True)
        if name == "freeze":
            command.add_argument("--freeze-output", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "preflight":
            payload = preflight(args)
        elif args.command == "develop":
            payload = develop(args)
        elif args.command == "audit":
            payload = audit(args)
        elif args.command == "freeze":
            payload = freeze(args)
        else:
            raise AssertionError(args.command)
        print(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

