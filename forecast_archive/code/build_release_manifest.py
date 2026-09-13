"""Build a release manifest after completed verification and visual inspection."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
directories = {'code', 'paper', 'companion', 'evidence', 'inputs', 'verification', 'provenance'}
top_files = {'README.md', 'requirements.txt', 'iclr-style.zip'}
skip_parts = {'__pycache__', 'tmp', 'qa', 'audit_work'}
skip_suffixes = {'.aux', '.blg', '.fdb_latexmk', '.fls', '.log', '.out', '.pyc', '.synctex.gz'}
files = {}
for p in sorted(root.rglob('*')):
    if not p.is_file():
        continue
    rel = p.relative_to(root)
    if rel.parts[0] not in directories and rel.as_posix() not in top_files:
        continue
    if any(part in skip_parts for part in rel.parts) or p.suffix in skip_suffixes:
        continue
    with p.open('rb') as f:
        files[rel.as_posix()] = hashlib.file_digest(f, 'sha256').hexdigest()
manifest = dict(schema='censorcast-forward-controls-v7-2026-09-08', files=files,
                historical_frozen_evidence_modified=False,
                original_manifest='provenance/PRIOR_PACKAGE_MANIFEST.json',
                prior_release_manifest='provenance/PRE_FORWARD_CONTROL_MANIFEST.json',
                forward_controls='evidence/forward_controls/',
                coordinate_tests='evidence/coordinate_tests/',
                post_hoc_replay_and_refit='evidence/reproducibility/',
                cell_and_fixed_control_audit='evidence/cell_audit/',
                independent_historical_custody_established=False,
                separate_selection_companion_included=True,
                internal_revision_notes_included=False)
(root/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(status='CREATED', files=len(files))))
