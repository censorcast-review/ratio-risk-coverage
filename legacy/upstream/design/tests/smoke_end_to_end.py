#!/usr/bin/env python3
"""Local tiny-panel end-to-end smoke for the resumable design runner."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def manifest(project: Path) -> Path:
    files = {}
    for path in sorted(project.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(project).as_posix()
        if relative == "SOURCE_MANIFEST_v0_5_DESIGN.json":
            continue
        files[relative] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    target = project / "SOURCE_MANIFEST_v0_5_DESIGN.json"
    write_json(target, {"schema_version": "0.5-design-1", "status": "SOURCE_MANIFEST", "files": files})
    return target


def make_data(root: Path) -> None:
    rng = np.random.default_rng(77)
    stores = [f"CA_{i}" for i in range(1, 5)] + [f"TX_{i}" for i in range(1, 4)] + [f"WI_{i}" for i in range(1, 4)]
    items = [f"item_{i:03d}" for i in range(12)]
    metadata = []
    for item_index, item in enumerate(items):
        for store in stores:
            state = store.split("_")[0]
            cat = ["FOODS", "HOUSEHOLD", "HOBBIES"][item_index % 3]
            metadata.append((f"{item}_{store}_evaluation", item, f"dept_{item_index % 6}", cat, store, state))
    days = np.arange(1, 1914)
    truth = np.empty((len(metadata), len(days)), dtype=np.int32)
    for index in range(len(metadata)):
        level = 3.0 + (index % 12) * 0.35 + (index % 10) * 0.08
        mean = level * (1.0 + 0.25 * np.sin(2 * np.pi * days / 7 + index / 5))
        mean *= 1.0 + 0.00025 * days
        truth[index] = rng.poisson(np.maximum(mean, 0.1))
    capacity = np.maximum(1, np.ceil(0.90 * np.maximum(truth.mean(axis=1, keepdims=True), 1))).astype(np.int32)
    capacity = np.broadcast_to(capacity, truth.shape).copy()
    observed = np.minimum(truth, capacity)
    censored = (truth > capacity).astype(np.uint8)
    arrays = {
        "id": np.asarray([row[0] for row in metadata]),
        "item_id": np.asarray([row[1] for row in metadata]),
        "dept_id": np.asarray([row[2] for row in metadata]),
        "cat_id": np.asarray([row[3] for row in metadata]),
        "store_id": np.asarray([row[4] for row in metadata]),
        "state_id": np.asarray([row[5] for row in metadata]),
        "day_start": np.asarray([1], dtype=np.int32),
        "day_end": np.asarray([1913], dtype=np.int32),
        "truth": truth,
        "observed": observed,
        "capacity": capacity,
        "censored": censored,
    }
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    np.savez_compressed(prepared / "design_outcomes_v0_5.npz", **arrays)
    (prepared / "design_item_ids_v0_5.txt").write_text("\n".join(items) + "\n")

    calendar = pd.DataFrame(
        {
            "date": pd.date_range("2011-01-29", periods=1913),
            "wm_yr_wk": 1000 + (days - 1) // 7,
            "weekday": ["x"] * len(days),
            "wday": 1 + (days - 1) % 7,
            "month": 1 + ((days - 1) // 30) % 12,
            "year": 2011 + (days - 1) // 365,
            "d": [f"d_{day}" for day in days],
            "event_name_1": [None] * len(days),
            "event_type_1": [None] * len(days),
            "event_name_2": [None] * len(days),
            "event_type_2": [None] * len(days),
            "snap_CA": (days % 10 == 0).astype(int),
            "snap_TX": (days % 11 == 0).astype(int),
            "snap_WI": (days % 12 == 0).astype(int),
        }
    )
    source = root / "source"
    source.mkdir(parents=True)
    calendar.to_csv(source / "calendar.csv", index=False)
    price_rows = []
    for _, item, _, _, store, _ in metadata:
        for week in sorted(calendar["wm_yr_wk"].unique()):
            price_rows.append((store, item, int(week), 1.0 + (hash(item) % 7) / 10))
    pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        source / "sell_prices.csv", index=False
    )
    ledger = {
        "schema_version": "0.5",
        "fresh_guardian_opened": False,
        "fresh_guardian_use_count": 0,
        "external_opened": False,
        "confirmatory_alpha_spent": 0.0,
        "certificate_issued": False,
    }
    write_json(root / "state" / "DATA_ACCESS_LEDGER_v0_5.json", ledger)
    write_json(
        root / "audit" / "DATA_PREPARATION_AUDIT_v0_5.json",
        {
            "status": "V0_5_M5_INDEPENDENT_DATA_AUDIT_PASS",
            "protocol_sha256": "a1629cf0565f7fa661f9b71f125b97d3575d72178837e230c0e4386b6a659f06",
            "runner_sha256": "fa339e9537283bf6c5b47a4096656a8375ac3e79341facc904d05f51b75499bb",
            "guardian_or_external_metrics_computed": 0,
        },
    )
    write_json(root / "audit" / "DATA_PREPARATION_RECEIPT_v0_5.json", {"status": "V0_5_M5_INDEPENDENT_DATA_READY"})
    artifacts = {}
    for relative in (
        "prepared/design_outcomes_v0_5.npz",
        "prepared/design_item_ids_v0_5.txt",
        "source/calendar.csv",
        "source/sell_prices.csv",
    ):
        path = root / relative
        artifacts[relative] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    write_json(root / "audit" / "DATA_ARTIFACT_SHA256_v0_5.json", {"artifacts": artifacts})


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        project = work / "project"
        shutil.copytree(ROOT, project, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "SOURCE_MANIFEST_v0_5_DESIGN.json"))
        config_path = project / "configs" / "DESIGN_PROTOCOL_v0_5.json"
        config = json.loads(config_path.read_text())
        config["base_models"]["training_sample_rows"] = 6000
        config["base_models"]["histgb"].update({"max_iter": 8, "min_samples_leaf": 20, "n_iter_no_change": 4})
        config["base_models"]["censor_classifier"].update({"max_iter": 8, "min_samples_leaf": 20, "n_iter_no_change": 4})
        config["base_models"]["censored_poisson_em_iterations"] = 1
        config["risk_layer"]["training_sample_rows"] = 6000
        config["risk_layer"]["histgb"].update({"max_iter": 8, "min_samples_leaf": 20, "n_iter_no_change": 4})
        config["risk_layer"]["instantaneous_weights"] = [0.25]
        config["risk_layer"]["forward_diagnostic_initial_days"] = 40
        config["risk_layer"]["forward_diagnostic_fold_days"] = 80
        config["selective_contract"]["fixed_policy_bootstrap_reps"] = 100
        config["point_forecast_gate"]["paired_item_cluster_bootstrap_reps"] = 100
        write_json(config_path, config)
        manifest_path = manifest(project)
        data = work / "data"
        output = work / "output"
        make_data(data)
        common = [
            "--project", str(project),
            "--data-root", str(data),
            "--output", str(output),
            "--source-sha256", sha(project / "run_v0_5_design.py"),
            "--design-protocol-sha256", sha(config_path),
            "--source-manifest-sha256", sha(manifest_path),
        ]
        runner = project / "run_v0_5_design.py"
        subprocess.run([sys.executable, str(runner), "preflight", *common], check=True, cwd=project)
        subprocess.run(
            [sys.executable, str(runner), "develop", *common, "--phrase", "RUN_CENSORCAST_V0_5_M5_DESIGN_ONLY_AFTER_DATA_AUDIT"],
            check=True,
            cwd=project,
        )
        subprocess.run([sys.executable, str(runner), "audit", *common], check=True, cwd=project)
        decision = json.loads((output / "results" / "DEVELOPMENT_DECISION_v0_5.json").read_text())
        assert decision["status"] in {
            "V0_5_DESIGN_GO_PENDING_MANUAL_FREEZE",
            "V0_5_DESIGN_NO_GO_KEEP_GUARDIAN_SEALED",
        }
        assert decision["fresh_guardian_opened"] is False if "fresh_guardian_opened" in decision else True
        print(json.dumps({"status": "V0_5_DESIGN_END_TO_END_SMOKE_PASS", "decision": decision["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
