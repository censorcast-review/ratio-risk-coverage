#!/usr/bin/env python3
"""Prepare a new, independently partitioned M5 data boundary for CENSORCAST v0.5.

This program is deliberately a data custodian, not an experiment runner.  It
may mechanically transform outcome bytes into sealed shards, but it never fits
a model, selects a policy, computes guardian/external metrics, or opens either
sealed split.  The v0.4 FreshRetailNet guardian is not an input.
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
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "0.5.0"
AUTH_PHRASE = "PREPARE_CENSORCAST_V0_5_M5_BLIND_SHARDS_AFTER_PROTOCOL_AUDIT"
PROTOCOL_RELATIVE = Path("configs") / "PROTOCOL_v0_5.json"
PREFLIGHT_NAME = "PREPARE_PREFLIGHT_v0_5.json"
STATE_NAME = "DATA_PREPARATION_STATE_v0_5.json"
LEDGER_NAME = "DATA_ACCESS_LEDGER_v0_5.json"
SOURCE_RECEIPT_NAME = "DATASET_SOURCE_RECEIPT_v0_5.json"
PARTITION_NAME = "DATA_PARTITION_MANIFEST_v0_5.json"
PREPARATION_NAME = "DATA_PREPARATION_RECEIPT_v0_5.json"
HASH_INDEX_NAME = "DATA_ARTIFACT_SHA256_v0_5.json"
DESIGN_SHARD = Path("prepared") / "design_outcomes_v0_5.npz"
GUARDIAN_CONTEXT = Path("prepared") / "fresh_guardian_context_v0_5.npz"
GUARDIAN_OUTCOMES = Path("sealed") / "fresh_guardian_outcomes_v0_5.npz"
EXTERNAL_CONTEXT = Path("prepared") / "external_context_v0_5.npz"
EXTERNAL_OUTCOMES = Path("sealed") / "external_final_outcomes_v0_5.npz"
DESIGN_IDS = Path("prepared") / "design_item_ids_v0_5.txt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Missing/non-regular {label}: {path}")


def verify_source(expected: str | None) -> str:
    observed = file_digest(Path(__file__).resolve())
    if expected and observed != expected:
        raise RuntimeError(
            "Source hash mismatch; refusing an unreviewed data preparation. "
            f"expected={expected} observed={observed}"
        )
    return observed


def load_protocol(project: Path, expected_hash: str | None) -> tuple[dict[str, Any], str]:
    path = project / PROTOCOL_RELATIVE
    regular_file(path, "v0.5 protocol")
    observed = file_digest(path)
    if expected_hash and observed != expected_hash:
        raise RuntimeError(
            "Protocol hash mismatch. "
            f"expected={expected_hash} observed={observed}"
        )
    protocol = json_load(path)
    if protocol.get("schema_version") != "0.5":
        raise RuntimeError("Unexpected protocol schema")
    return protocol, observed


def verify_ledger_sealed(data_root: Path) -> dict[str, Any] | None:
    path = data_root / "state" / LEDGER_NAME
    if not path.exists():
        return None
    ledger = json_load(path)
    checks = {
        "fresh_guardian_opened_false": ledger.get("fresh_guardian_opened") is False,
        "external_opened_false": ledger.get("external_opened") is False,
        "guardian_use_count_zero": int(ledger.get("fresh_guardian_use_count", -1)) == 0,
        "alpha_unspent": float(ledger.get("confirmatory_alpha_spent", -1.0)) == 0.0,
        "certificate_absent": ledger.get("certificate_issued") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A sealed-data ledger check failed: {checks}")
    return ledger


def compatible_state(data_root: Path, runner_sha: str, protocol_sha: str) -> dict[str, Any] | None:
    path = data_root / "state" / STATE_NAME
    if not path.exists():
        return None
    state = json_load(path)
    if state.get("runner_sha256") != runner_sha or state.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Existing v0.5 preparation state belongs to different source/protocol hashes")
    return state


def disk_free_bytes(path: Path) -> int:
    anchor = path if path.exists() else path.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    return int(shutil.disk_usage(anchor).free)


def make_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    project = Path(args.project).resolve()
    data_root = Path(args.data_root).resolve()
    runner_sha = verify_source(args.source_sha256)
    protocol, protocol_sha = load_protocol(project, args.protocol_sha256)
    data_root.mkdir(parents=True, exist_ok=True)
    ledger = verify_ledger_sealed(data_root)
    state = compatible_state(data_root, runner_sha, protocol_sha)
    receipt_path = data_root / "audit" / PREPARATION_NAME
    already_ready = receipt_path.is_file() and json_load(receipt_path).get("status") in {
        "V0_5_M5_INDEPENDENT_DATA_READY",
        "V0_5_M5_INDEPENDENT_DATA_READY_DESIGN_CENSORING_NO_GO",
    }
    free_bytes = disk_free_bytes(data_root)
    if not already_ready and free_bytes < 3_000_000_000:
        raise RuntimeError(f"At least 3 GB free space is required; observed={free_bytes}")
    forbidden = [
        data_root / "state" / "FRESH_GUARDIAN_OPEN_RECEIPT_v0_5.json",
        data_root / "state" / "EXTERNAL_OPEN_RECEIPT_v0_5.json",
        data_root / "audit" / "V0_5_GUARDIAN_DECISION.json",
        data_root / "audit" / "V0_5_EXTERNAL_DECISION.json",
    ]
    present_forbidden = [str(path) for path in forbidden if path.exists()]
    if present_forbidden:
        raise RuntimeError(f"Unexpected downstream/open artifacts already exist: {present_forbidden}")
    payload = {
        "schema_version": "0.5",
        "created_utc": utc_now(),
        "status": "V0_5_M5_PREPARE_PREFLIGHT_PASS",
        "runner_sha256": runner_sha,
        "protocol_sha256": protocol_sha,
        "data_root": str(data_root),
        "new_dataset": protocol["dataset"]["name"],
        "dataset_record": protocol["dataset"]["record_url"],
        "freshretail_v04_guardian_reused": False,
        "freshretail_eval_or_test_authorized": False,
        "partition": protocol["item_cluster_partition"],
        "time_partition": protocol["time_partition"],
        "controlled_censoring": protocol["controlled_censoring"],
        "state": state.get("phase") if state else "EMPTY_SEALED_DATA_ROOT",
        "already_prepared": bool(already_ready),
        "fresh_guardian_opened": False if ledger is None else ledger["fresh_guardian_opened"],
        "external_opened": False if ledger is None else ledger["external_opened"],
        "confirmatory_alpha_spent": 0.0 if ledger is None else ledger["confirmatory_alpha_spent"],
        "free_bytes": free_bytes,
        "next": "Manual authorization may download and blindly shard M5; no guardian/external metrics will be computed.",
    }
    atomic_json(data_root / "audit" / PREFLIGHT_NAME, payload)
    return payload, protocol, runner_sha, protocol_sha


def item_set_hash(items: list[str]) -> str:
    canonical = "\n".join(sorted(items)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def partition_items(items: list[str], specification: dict[str, Any]) -> dict[str, list[str]]:
    if len(items) != len(set(items)):
        raise ValueError("Item IDs must be unique")
    salt = str(specification["salt"])
    ordered = sorted(
        items,
        key=lambda item: hashlib.sha256(
            salt.encode("utf-8") + b"\0" + item.encode("utf-8")
        ).hexdigest(),
    )
    n_design = int(specification["design_items"])
    n_guardian = int(specification["fresh_guardian_items"])
    n_external = int(specification["external_items"])
    if n_design + n_guardian + n_external != len(ordered):
        raise RuntimeError("Prespecified item counts do not cover the observed items exactly")
    result = {
        "design": ordered[:n_design],
        "fresh_guardian": ordered[n_design : n_design + n_guardian],
        "external": ordered[n_design + n_guardian :],
    }
    sets = {name: set(values) for name, values in result.items()}
    if sets["design"] & sets["fresh_guardian"] or sets["design"] & sets["external"] or sets["fresh_guardian"] & sets["external"]:
        raise AssertionError("Item-cluster overlap detected")
    return result


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


def metadata_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: frame[name].astype(str).to_numpy(dtype=str)
        for name in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    }


def shard_payload(
    metadata: pd.DataFrame,
    truth: np.ndarray,
    observed: np.ndarray,
    capacity: np.ndarray,
    censored: np.ndarray,
    start_day: int,
    end_day: int,
    include_truth: bool,
) -> dict[str, np.ndarray]:
    start = start_day - 1
    end = end_day
    payload = metadata_arrays(metadata)
    payload.update(
        {
            "day_start": np.asarray([start_day], dtype=np.int32),
            "day_end": np.asarray([end_day], dtype=np.int32),
            "observed": observed[:, start:end].astype(np.int32, copy=False),
            "capacity": capacity[:, start:end].astype(np.int32, copy=False),
            "censored": censored[:, start:end].astype(np.uint8, copy=False),
        }
    )
    if include_truth:
        payload["truth"] = truth[:, start:end].astype(np.int32, copy=False)
    return payload


def resolve_zenodo_links(protocol: dict[str, Any]) -> dict[str, str]:
    request = urllib.request.Request(
        protocol["dataset"]["api_url"],
        headers={"User-Agent": "CENSORCAST-v0.5-data-custodian/0.5.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.loads(response.read().decode("utf-8"))
    if str(record.get("id")) != str(protocol["dataset"]["record_id"]):
        raise RuntimeError("Zenodo record ID mismatch")
    declared = protocol["dataset"]["files"]
    links: dict[str, str] = {}
    observed_md5: dict[str, str] = {}
    for entry in record.get("files", []):
        key = entry.get("key")
        if key not in declared:
            continue
        checksum = str(entry.get("checksum", ""))
        if checksum.startswith("md5:"):
            observed_md5[key] = checksum.split(":", 1)[1]
        link = entry.get("links", {}).get("content") or entry.get("links", {}).get("self")
        if link:
            links[key] = str(link)
    for name, specification in declared.items():
        if name not in links:
            raise RuntimeError(f"Zenodo did not return a content link for {name}")
        if observed_md5.get(name) != specification["md5"]:
            raise RuntimeError(
                f"Zenodo checksum changed for {name}: expected={specification['md5']} observed={observed_md5.get(name)}"
            )
    return links


def download_atomic(url: str, destination: Path, expected_md5: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_digest(destination, "md5") == expected_md5:
        return {
            "path": str(destination),
            "md5": expected_md5,
            "sha256": file_digest(destination),
            "size_bytes": destination.stat().st_size,
            "download_reused": True,
        }
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, 6):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            headers["User-Agent"] = "CENSORCAST-v0.5-data-custodian/0.5.0"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(getattr(response, "status", response.getcode()))
                if existing and status == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        if chunk:
                            handle.write(chunk)
            if file_digest(partial, "md5") != expected_md5:
                if attempt == 5:
                    raise RuntimeError(f"MD5 mismatch after download: {destination.name}")
                partial.unlink(missing_ok=True)
                continue
            partial.replace(destination)
            return {
                "path": str(destination),
                "md5": expected_md5,
                "sha256": file_digest(destination),
                "size_bytes": destination.stat().st_size,
                "download_reused": False,
            }
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def stage_local_source(source: Path, destination: Path, expected_md5: str) -> dict[str, Any]:
    regular_file(source, f"local source {source.name}")
    if file_digest(source, "md5") != expected_md5:
        raise RuntimeError(f"Local source MD5 mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    return {
        "path": str(destination),
        "md5": expected_md5,
        "sha256": file_digest(destination),
        "size_bytes": destination.stat().st_size,
        "download_reused": True,
        "local_source_used": True,
    }


def design_censoring_diagnostic(
    truth: np.ndarray,
    observed: np.ndarray,
    censored: np.ndarray,
    start_day: int,
    end_day: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    sl = slice(start_day - 1, end_day)
    y = truth[:, sl].astype(np.float64, copy=False)
    obs = observed[:, sl].astype(np.float64, copy=False)
    flag = censored[:, sl].astype(bool, copy=False)
    row_rate = float(flag.mean())
    truth_mass = float(y.sum())
    hidden_fraction = float(np.maximum(y - obs, 0.0).sum() / max(truth_mass, 1.0))
    checks = {
        "row_censor_rate_min": row_rate >= float(gate["minimum_row_censor_rate"]),
        "row_censor_rate_max": row_rate <= float(gate["maximum_row_censor_rate"]),
        "hidden_mass_min": hidden_fraction >= float(gate["minimum_hidden_demand_mass_fraction"]),
        "hidden_mass_max": hidden_fraction <= float(gate["maximum_hidden_demand_mass_fraction"]),
    }
    return {
        "statistical_scope": "DESIGN_ITEMS_ONLY",
        "days": [start_day, end_day],
        "row_censor_rate": row_rate,
        "hidden_demand_mass_fraction": hidden_fraction,
        "checks": checks,
        "status": "DESIGN_CENSORING_VALIDITY_PASS" if all(checks.values()) else "DESIGN_CENSORING_VALIDITY_NO_GO",
    }


def artifact_hashes(data_root: Path, relative_paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = data_root / relative
        regular_file(path, str(relative))
        result[relative.as_posix()] = {
            "sha256": file_digest(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.phrase != AUTH_PHRASE:
        raise RuntimeError("Exact blind-preparation authorization phrase required")
    preflight, protocol, runner_sha, protocol_sha = make_preflight(args)
    data_root = Path(args.data_root).resolve()
    completed = data_root / "audit" / PREPARATION_NAME
    if completed.is_file():
        existing = json_load(completed)
        if existing.get("runner_sha256") == runner_sha and existing.get("protocol_sha256") == protocol_sha:
            print("[CENSORCAST v0.5] COMPLETED_DATA_PREPARATION_REUSED", flush=True)
            return existing
        raise RuntimeError("Existing preparation receipt belongs to different hashes")

    ledger_path = data_root / "state" / LEDGER_NAME
    if not ledger_path.exists():
        atomic_json(
            ledger_path,
            {
                "schema_version": "0.5",
                "created_utc": utc_now(),
                "fresh_guardian_opened": False,
                "fresh_guardian_use_count": 0,
                "external_opened": False,
                "confirmatory_alpha_spent": 0.0,
                "certificate_issued": False,
                "note": "Data bytes may exist in sealed shards; statistical outcomes have not been opened.",
            },
        )
    verify_ledger_sealed(data_root)
    state_path = data_root / "state" / STATE_NAME
    atomic_json(
        state_path,
        {
            "schema_version": "0.5",
            "updated_utc": utc_now(),
            "phase": "ACQUIRING_PINNED_M5_SOURCE",
            "runner_sha256": runner_sha,
            "protocol_sha256": protocol_sha,
            "fresh_guardian_opened": False,
            "external_opened": False,
        },
    )

    declared = protocol["dataset"]["files"]
    source_dir = data_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_receipts: dict[str, Any] = {}
    if args.source_dir:
        local = Path(args.source_dir).resolve()
        for name, specification in declared.items():
            source_receipts[name] = stage_local_source(
                local / name, source_dir / name, specification["md5"]
            )
    else:
        links = resolve_zenodo_links(protocol)
        for name, specification in declared.items():
            print(f"[CENSORCAST v0.5] ACQUIRING {name}", flush=True)
            source_receipts[name] = download_atomic(
                links[name], source_dir / name, specification["md5"]
            )
    source_receipt = {
        "schema_version": "0.5",
        "created_utc": utc_now(),
        "status": "PINNED_M5_SOURCE_VERIFIED",
        "record_id": protocol["dataset"]["record_id"],
        "doi": protocol["dataset"]["doi"],
        "files": source_receipts,
        "outcome_metrics_computed": False,
    }
    atomic_json(data_root / "audit" / SOURCE_RECEIPT_NAME, source_receipt)
    atomic_json(
        state_path,
        {
            "schema_version": "0.5",
            "updated_utc": utc_now(),
            "phase": "BLINDLY_SHARDING_M5",
            "runner_sha256": runner_sha,
            "protocol_sha256": protocol_sha,
            "fresh_guardian_opened": False,
            "external_opened": False,
        },
    )

    sales_path = source_dir / "sales_train_evaluation.csv"
    header = pd.read_csv(sales_path, nrows=0)
    id_columns = list(protocol["dataset"]["id_columns"])
    day_columns = [f"d_{day}" for day in range(1, int(protocol["dataset"]["expected_days"]) + 1)]
    missing = [name for name in id_columns + day_columns if name not in header.columns]
    if missing:
        raise RuntimeError(f"M5 sales schema missing columns: {missing[:10]}")
    dtype = {name: "int32" for name in day_columns}
    sales = pd.read_csv(sales_path, dtype=dtype)
    if len(sales) != int(protocol["dataset"]["expected_bottom_series"]):
        raise RuntimeError(f"Unexpected bottom-series count: {len(sales)}")
    unique_items = sorted(sales["item_id"].astype(str).unique().tolist())
    if len(unique_items) != int(protocol["dataset"]["expected_unique_items"]):
        raise RuntimeError(f"Unexpected item count: {len(unique_items)}")
    partitions = partition_items(unique_items, protocol["item_cluster_partition"])
    metadata = sales[id_columns].astype(str).copy()
    truth_all = sales[day_columns].to_numpy(dtype=np.int32, copy=True)
    del sales
    gc.collect()

    masks = {
        name: metadata["item_id"].isin(values).to_numpy()
        for name, values in partitions.items()
    }
    expected_series = {
        "design": int(protocol["item_cluster_partition"]["expected_design_series"]),
        "fresh_guardian": int(protocol["item_cluster_partition"]["expected_fresh_guardian_series"]),
        "external": int(protocol["item_cluster_partition"]["expected_external_series"]),
    }
    observed_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if observed_counts != expected_series:
        raise RuntimeError(
            f"Unexpected series counts after item split: expected={expected_series} observed={observed_counts}"
        )
    membership_count = sum(mask.astype(np.int8) for mask in masks.values())
    if membership_count.min() != 1 or membership_count.max() != 1:
        raise RuntimeError("Every series must belong to exactly one item partition")

    time_spec = protocol["time_partition"]
    diagnostic: dict[str, Any] | None = None
    artifact_paths: list[Path] = []
    print("[CENSORCAST v0.5] WRITING DESIGN SHARD", flush=True)
    for role in ["design", "fresh_guardian", "external"]:
        indices = np.flatnonzero(masks[role])
        role_meta = metadata.iloc[indices].reset_index(drop=True)
        role_truth = truth_all[indices]
        role_observed, role_capacity, role_censored = simulate_controlled_censoring(
            role_truth,
            warmup_days=int(protocol["controlled_censoring"]["warmup_days_uncensored"]),
        )
        if role == "design":
            diagnostic = design_censoring_diagnostic(
                role_truth,
                role_observed,
                role_censored,
                int(time_spec["selection"][0]),
                int(time_spec["shadow"][1]),
                protocol["controlled_censoring"]["design_validity_gate"],
            )
            atomic_npz(
                data_root / DESIGN_SHARD,
                **shard_payload(
                    role_meta,
                    role_truth,
                    role_observed,
                    role_capacity,
                    role_censored,
                    1,
                    int(time_spec["shadow"][1]),
                    True,
                ),
            )
            atomic_text(data_root / DESIGN_IDS, "\n".join(partitions["design"]) + "\n")
            artifact_paths.extend([DESIGN_SHARD, DESIGN_IDS])
        elif role == "fresh_guardian":
            print("[CENSORCAST v0.5] WRITING SEALED FRESH-GUARDIAN SHARDS WITHOUT METRICS", flush=True)
            atomic_npz(
                data_root / GUARDIAN_CONTEXT,
                **shard_payload(
                    role_meta,
                    role_truth,
                    role_observed,
                    role_capacity,
                    role_censored,
                    1,
                    int(time_spec["risk_train"][1]),
                    False,
                ),
            )
            atomic_npz(
                data_root / GUARDIAN_OUTCOMES,
                **shard_payload(
                    role_meta,
                    role_truth,
                    role_observed,
                    role_capacity,
                    role_censored,
                    int(time_spec["calibration_a"][0]),
                    int(time_spec["shadow"][1]),
                    True,
                ),
            )
            artifact_paths.extend([GUARDIAN_CONTEXT, GUARDIAN_OUTCOMES])
        else:
            print("[CENSORCAST v0.5] WRITING SEALED EXTERNAL SHARDS WITHOUT METRICS", flush=True)
            atomic_npz(
                data_root / EXTERNAL_CONTEXT,
                **shard_payload(
                    role_meta,
                    role_truth,
                    role_observed,
                    role_capacity,
                    role_censored,
                    1,
                    int(time_spec["shadow"][1]),
                    False,
                ),
            )
            atomic_npz(
                data_root / EXTERNAL_OUTCOMES,
                **shard_payload(
                    role_meta,
                    role_truth,
                    role_observed,
                    role_capacity,
                    role_censored,
                    int(time_spec["external_final"][0]),
                    int(time_spec["external_final"][1]),
                    True,
                ),
            )
            artifact_paths.extend([EXTERNAL_CONTEXT, EXTERNAL_OUTCOMES])
        del role_truth, role_observed, role_capacity, role_censored, role_meta
        gc.collect()

    if diagnostic is None:
        raise AssertionError("Design diagnostic was not created")
    partition_manifest = {
        "schema_version": "0.5",
        "created_utc": utc_now(),
        "status": "M5_ITEM_AND_TIME_PARTITIONS_FIXED",
        "partition_algorithm": protocol["item_cluster_partition"],
        "time_partition": time_spec,
        "item_counts": {name: len(values) for name, values in partitions.items()},
        "series_counts": observed_counts,
        "item_set_sha256": {name: item_set_hash(values) for name, values in partitions.items()},
        "design_item_ids_released": True,
        "fresh_guardian_item_ids_released": False,
        "external_item_ids_released": False,
        "overlap_count": 0,
        "freshretail_series_included": 0,
    }
    atomic_json(data_root / "audit" / PARTITION_NAME, partition_manifest)
    artifact_paths.extend(
        [
            Path("source") / "calendar.csv",
            Path("source") / "sell_prices.csv",
            Path("audit") / SOURCE_RECEIPT_NAME,
            Path("audit") / PARTITION_NAME,
            Path("state") / LEDGER_NAME,
        ]
    )

    # The exact raw sales copy is now redundant and includes the final labels.
    # Delete only this verified, newly staged file; it remains recoverable from
    # the pinned Zenodo record and its digests are retained in the receipt.
    expected_sales_md5 = protocol["dataset"]["files"]["sales_train_evaluation.csv"]["md5"]
    if file_digest(sales_path, "md5") != expected_sales_md5:
        raise RuntimeError("Refusing to remove an unexpected raw sales file")
    sales_path.unlink()

    hashes = artifact_hashes(data_root, artifact_paths)
    hash_index = {
        "schema_version": "0.5",
        "created_utc": utc_now(),
        "status": "V0_5_DATA_ARTIFACT_HASH_INDEX_CREATED",
        "artifacts": hashes,
    }
    atomic_json(data_root / "audit" / HASH_INDEX_NAME, hash_index)
    preparation_status = (
        "V0_5_M5_INDEPENDENT_DATA_READY"
        if diagnostic["status"] == "DESIGN_CENSORING_VALIDITY_PASS"
        else "V0_5_M5_INDEPENDENT_DATA_READY_DESIGN_CENSORING_NO_GO"
    )
    receipt = {
        "schema_version": "0.5",
        "program_version": VERSION,
        "created_utc": utc_now(),
        "status": preparation_status,
        "runner_sha256": runner_sha,
        "protocol_sha256": protocol_sha,
        "dataset": {
            "name": protocol["dataset"]["name"],
            "doi": protocol["dataset"]["doi"],
            "source_receipt_sha256": file_digest(data_root / "audit" / SOURCE_RECEIPT_NAME),
        },
        "partition_manifest_sha256": file_digest(data_root / "audit" / PARTITION_NAME),
        "artifact_index_sha256": file_digest(data_root / "audit" / HASH_INDEX_NAME),
        "design_censoring_diagnostic": diagnostic,
        "scope": {
            "model_training_calls": 0,
            "policy_selection_calls": 0,
            "guardian_metrics_computed": 0,
            "external_metrics_computed": 0,
            "freshretail_v04_guardian_rows_read": 0,
            "freshretail_eval_or_test_rows_read": 0,
            "fresh_guardian_opened": False,
            "external_opened": False,
            "confirmatory_alpha_spent": 0.0,
            "certificate_issued": False,
        },
        "raw_sales_plaintext_retained": False,
        "raw_sales_recoverable_from_pinned_source": True,
        "next": "STOP for audit. Development may use only the design shard after a separate authorization.",
    }
    atomic_json(completed, receipt)
    atomic_json(
        state_path,
        {
            "schema_version": "0.5",
            "updated_utc": utc_now(),
            "phase": "INDEPENDENT_DATA_READY_GUARDIAN_AND_EXTERNAL_SEALED",
            "runner_sha256": runner_sha,
            "protocol_sha256": protocol_sha,
            "preparation_status": preparation_status,
            "fresh_guardian_opened": False,
            "external_opened": False,
            "confirmatory_alpha_spent": 0.0,
        },
    )
    return receipt


def audit(args: argparse.Namespace) -> dict[str, Any]:
    preflight, protocol, runner_sha, protocol_sha = make_preflight(args)
    data_root = Path(args.data_root).resolve()
    receipt_path = data_root / "audit" / PREPARATION_NAME
    regular_file(receipt_path, "preparation receipt")
    receipt = json_load(receipt_path)
    if receipt.get("runner_sha256") != runner_sha or receipt.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Preparation receipt hashes do not match")
    hash_index_path = data_root / "audit" / HASH_INDEX_NAME
    regular_file(hash_index_path, "artifact hash index")
    if file_digest(hash_index_path) != receipt.get("artifact_index_sha256"):
        raise RuntimeError("Artifact index changed after preparation")
    index = json_load(hash_index_path)
    verified = 0
    for relative, specification in index.get("artifacts", {}).items():
        path = data_root / relative
        regular_file(path, relative)
        if file_digest(path) != specification.get("sha256") or path.stat().st_size != int(specification.get("size_bytes")):
            raise RuntimeError(f"Prepared artifact changed: {relative}")
        verified += 1
    raw_sales = data_root / "source" / "sales_train_evaluation.csv"
    if raw_sales.exists():
        raise RuntimeError("Raw final-label-bearing sales file was unexpectedly retained")
    ledger = verify_ledger_sealed(data_root)
    output = {
        "schema_version": "0.5",
        "created_utc": utc_now(),
        "status": "V0_5_M5_INDEPENDENT_DATA_AUDIT_PASS",
        "preparation_status": receipt["status"],
        "runner_sha256": runner_sha,
        "protocol_sha256": protocol_sha,
        "verified_artifacts": verified,
        "new_dataset": protocol["dataset"]["name"],
        "freshretail_v04_guardian_reused": False,
        "fresh_guardian_opened": ledger["fresh_guardian_opened"],
        "fresh_guardian_use_count": ledger["fresh_guardian_use_count"],
        "external_opened": ledger["external_opened"],
        "confirmatory_alpha_spent": ledger["confirmatory_alpha_spent"],
        "certificate_issued": ledger["certificate_issued"],
        "guardian_or_external_metrics_computed": 0,
        "raw_sales_plaintext_retained": False,
        "next": "STOP and return this audit before starting any design-only model development.",
    }
    atomic_json(data_root / "audit" / "DATA_PREPARATION_AUDIT_v0_5.json", output)
    return output


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    for command in ["preflight", "prepare", "audit"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--project", required=True)
        sub.add_argument("--data-root", required=True)
        sub.add_argument("--source-sha256")
        sub.add_argument("--protocol-sha256")
        if command == "prepare":
            sub.add_argument("--phrase", required=True)
            sub.add_argument("--source-dir")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "preflight":
            result = make_preflight(args)[0]
        elif args.command == "prepare":
            result = prepare(args)
        else:
            result = audit(args)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
