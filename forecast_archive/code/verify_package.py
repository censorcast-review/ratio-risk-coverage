"""Check original package bytes before rebuilding presentation artifacts."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text())
for name,digest in manifest['files'].items():
    p=root/name
    assert p.is_file(),name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
print(json.dumps({'status':'PASS','files':len(manifest['files']),'scope':'supplied package snapshot; rebuilds can change presentation hashes'}))
