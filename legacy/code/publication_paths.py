"""Resolve historical artifact hashes without rewriting original records."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def recorded_path(original_path):
    mapping=json.loads((ROOT/'provenance/PUBLIC_PATHS.json').read_text())
    return ROOT/mapping['original_artifact_paths'].get(str(original_path),str(original_path))
