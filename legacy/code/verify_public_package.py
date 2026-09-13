"""Verify publication hashes and unchanged scientific freeze dependencies."""
from pathlib import Path
import hashlib,json
from publication_paths import recorded_path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
x=json.loads((ROOT/'ANONYMIZATION_MANIFEST.json').read_text())
checked=0
for item in x['files']:
    p=ROOT/item['path']
    assert p.is_file(),item['path']
    assert sha(p)==item['public_sha256'],item['path']
    checked+=1
f=json.loads((ROOT/'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json').read_text())
for name,h in f['immutable_files'].items():
    assert sha(recorded_path(name))==h,name
mapping=json.loads((ROOT/'provenance/external_confirmation/EXTERNAL_TRANSFER.json').read_text())
external_checked=0
for item in mapping['records']:
    if item['public_path'] is not None:
        assert sha(ROOT/item['public_path'])==item['original_sha256'],item['public_path']
        external_checked+=1
assert external_checked==26 and mapping['frozen_dependency_count']==27
assert sha(ROOT/'provenance/external_confirmation/FROZEN_PLAN_ORIGINAL.json')==mapping['original_plan_sha256']
evolution=json.loads((ROOT/'provenance/external_confirmation/PLAN_EVOLUTION.json').read_text())
from datetime import datetime
times=[evolution['old_pair']['frozen_utc'],evolution['executed_pair']['frozen_utc'],evolution['opened_utc'],evolution['completed_utc']]
assert all(datetime.fromisoformat(a)<datetime.fromisoformat(b) for a,b in zip(times,times[1:]))
assert evolution['old_plan_external_opened'] is False and evolution['executed_plan_external_opened_at_freeze'] is False
for key in ['old_pair','executed_pair']:
    assert sha(ROOT/'provenance/external_confirmation'/evolution[key]['public_original_path'])==evolution[key]['sha256']
assert evolution['external_use_count']==1 and evolution['fit_calls']==0 and evolution['threshold_selection_calls']==0
assert evolution['no_external_access_during_replacement'] and evolution['old_plan_preserved_byte_for_byte']
print(json.dumps(dict(status='PUBLIC_DISTRIBUTION_HASHES_PASS',files=checked,guardian_frozen_files=len(f['immutable_files']),external_frozen_files=external_checked,external_administrative_dependency_omitted=1)))
