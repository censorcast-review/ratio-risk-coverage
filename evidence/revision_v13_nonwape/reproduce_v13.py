"""Replay the existing frozen study without altering original evidence."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from run_nonwape import HERE, sha, utc, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--raw-cache')
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.exists():
        raise RuntimeError('output must not exist; preserve every previous execution')
    target.mkdir(parents=True)
    (target/'evidence').mkdir()
    (target/'raw').mkdir()
    protocol = json.loads((HERE/'evidence'/'PROTOCOL.json').read_text())
    for name, expected in protocol['code_sha256'].items():
        if sha(HERE/name) != expected:
            raise RuntimeError('distributed frozen code mismatch: '+name)
        shutil.copy2(HERE/name, target/name)
    shutil.copy2(HERE/'evidence'/'PROTOCOL.json', target/'evidence'/'PROTOCOL.json')
    shutil.copy2(HERE/'evidence'/'FREEZE_RECEIPT.json', target/'evidence'/'FREEZE_RECEIPT.json')
    write_json(target/'REPRODUCTION_START.json', {'utc': utc(),
               'scope': 'Replay of an existing protocol, not a new prospective registration.',
               'original_protocol_sha256': sha(HERE/'evidence'/'PROTOCOL.json')})
    if args.raw_cache:
        for dataset in protocol['datasets']:
            path = Path(args.raw_cache)/(dataset+'.rar')
            original = json.loads((HERE/'evidence'/('DOWNLOAD_'+dataset+'.json')).read_text())
            if sha(path) != original['sha256']:
                raise RuntimeError('raw cache hash mismatch: '+dataset)
            shutil.copy2(path, target/'raw'/path.name)
    else:
        subprocess.run([sys.executable, str(target/'fetch_nonwape.py')], check=True)
    for dataset in protocol['datasets']:
        original = json.loads((HERE/'evidence'/('DOWNLOAD_'+dataset+'.json')).read_text())
        path = target/'raw'/(dataset+'.rar')
        if not path.is_file() or sha(path) != original['sha256']:
            raise RuntimeError('download absent or differs from original raw: '+dataset)
        subprocess.run([sys.executable, str(target/'run_nonwape.py'), '--dataset', dataset,
                        '--raw-root', str(path)], check=True)
    differences = {}
    for dataset in protocol['datasets']:
        differences[dataset] = {}
        for filename in ['SPLITS.npz','A_ARRAYS.npz','B_ARRAYS.npz','E_ARRAYS.npz',
                         'B_BOOTSTRAP.npz','E_BOOTSTRAP.npz']:
            with np.load(HERE/'evidence'/dataset/filename, allow_pickle=False) as old, \
                 np.load(target/'evidence'/dataset/filename, allow_pickle=False) as new:
                if set(old.files) != set(new.files):
                    raise RuntimeError('member mismatch '+dataset+'/'+filename)
                differences[dataset][filename] = {}
                for key in old.files:
                    a,b=old[key],new[key]
                    same = bool(np.array_equal(a,b,equal_nan=True))
                    maximum = 0.0
                    if not same and a.shape == b.shape and np.issubdtype(a.dtype,np.number):
                        good=np.isfinite(a)&np.isfinite(b)
                        maximum=float(np.max(np.abs(a[good].astype(float)-b[good].astype(float)))) if good.any() else None
                    differences[dataset][filename][key]={'exact_match':same,'max_finite_difference':maximum}
    write_json(target/'REPRODUCTION_COMPARISON.json', {'utc':utc(),'scientific_arrays':differences,
               'note':'Differences are reported; this script never tunes or reruns in response.'})
    print(target/'REPRODUCTION_COMPARISON.json')


if __name__ == '__main__':
    main()
