#!/usr/bin/env python3
"""Build the public-package SHA-256 manifest from distributable files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"

EXCLUDED_PARTS = {"tmp", "qa", "__pycache__"}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pyc",
}
EXCLUDED_NAMES = {
    "MANIFEST.json",
    "CENSORCAST_ICLR_2027_Accuracy_Manuscript.pdf",
    "build_transcript.log",
}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
files = {
    path.relative_to(ROOT).as_posix(): sha256(path)
    for path in sorted(ROOT.rglob("*"))
    if included(path)
}

manifest = {
    "schema": "censorcast-risk-calibration-revision-3",
    "files": files,
    "historical_frozen_evidence_modified": False,
    "raw_uci_source": old.get(
        "raw_uci_source", "public download; checksum in STUDY_PROTOCOL.json"
    ),
    "m5_foundation_forecast_arrays_included": False,
    "risk_calibration_three_seed_models_included": True,
    "m5_observable_first_seed_head_arrays_included": True,
    "m5_development_inputs_included": True,
    "freshretail_official_inputs_included": True,
    "freshretail_frozen_models_and_predictions_included": True,
    "separate_selection_companion_included": True,
}

MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
print(json.dumps({"status": "PASS", "files": len(files)}))
