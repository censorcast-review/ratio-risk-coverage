"""Restore packaged numerical assets and verify their original manifest hashes."""
from pathlib import Path
import hashlib
import json
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore(spec_path):
    spec = json.loads((ROOT / spec_path).read_text())
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())['files']
    missing = []
    for name in spec['files']:
        target = ROOT / name
        if name not in manifest or not target.resolve().is_relative_to(ROOT):
            raise ValueError('Unexpected asset path: ' + name)
        if target.exists():
            if digest(target) != manifest[name]['sha256']:
                raise ValueError('Existing asset differs; refusing to overwrite: ' + name)
        else:
            missing.append(name)
    if not missing:
        print('PASS: all numerical assets already present and hash-verified')
        return
    with tempfile.TemporaryDirectory(prefix='censorcast-assets-') as temp:
        archive = Path(temp) / 'assets.zip'
        with archive.open('wb') as out:
            for part in spec['parts']:
                source = ROOT / part['path']
                if source.stat().st_size != part['bytes'] or digest(source) != part['sha256']:
                    raise ValueError('Invalid archive part: ' + part['path'])
                with source.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        out.write(block)
        if archive.stat().st_size != spec['bytes'] or digest(archive) != spec['sha256']:
            raise ValueError('Assembled archive hash mismatch')
        with zipfile.ZipFile(archive) as z:
            if sorted(z.namelist()) != sorted(spec['files']):
                raise ValueError('Archive file list differs from declared assets')
            for name in missing:
                target = ROOT / name
                data = z.read(name)
                if len(data) != manifest[name]['bytes'] or hashlib.sha256(data).hexdigest() != manifest[name]['sha256']:
                    raise ValueError('Original asset hash mismatch: ' + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    print('PASS: restored', len(missing), 'assets with original hashes')


if __name__ == '__main__':
    restore('data_parts/ASSETS.json')
    restore('data_parts/ADDITIONAL_ASSETS.json')
    restore('data_parts/MATCHED_ASSETS.json')
