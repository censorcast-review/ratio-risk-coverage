"""Verify earlier frozen scientific files; write any report outside the package."""
from pathlib import Path
import argparse
import hashlib
import json

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest_path = root/'provenance/PRE_FORWARD_CONTROL_MANIFEST.json'
prior = json.loads(manifest_path.read_text())
checked = {}
for name, expected in prior['files'].items():
    if not name.startswith(('evidence/', 'inputs/')):
        continue
    p = root/name
    assert p.is_file(), f'Missing frozen scientific file: {name}'
    actual = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
    assert actual == expected, f'Changed frozen scientific file: {name}'
    checked[name] = actual
result = dict(status='PASS', prior_scientific_files_preserved=len(checked),
              original_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              original_manifest_path=str(manifest_path.relative_to(root)),
              scope='Earlier inputs and evidence are byte-identical; presentation and verification code may be revised.',
              files=checked)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))
