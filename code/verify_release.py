"""Check the delivered file manifest; never modifies scientific evidence."""
from pathlib import Path
import argparse,json,hashlib
R=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--source-only',action='store_true',help='Permit missing declared numerical assets in the smaller source attachment');x=a.parse_args();m=json.loads((R/'MANIFEST.json').read_text());missing=[];checked=0
for rel,rec in m['files'].items():
 p=R/rel
 if not p.exists():
  if x.source_only and rec['numerical_asset']:missing.append(rel);continue
  raise SystemExit('Missing required file: '+rel)
 if hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=rec['sha256']:raise SystemExit('Hash mismatch: '+rel)
 checked+=1
print(json.dumps({'status':'PASS','checked':checked,'declared_numerical_assets_absent':len(missing),'mode':'source_only' if x.source_only else 'complete'}))
