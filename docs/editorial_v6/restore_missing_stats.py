"""Reconstruct only missing derived statistics from already frozen policies.

No fitting, threshold design, policy selection, or external access. An output is
copied to its missing evidence path only if it matches the original manifest.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from policy_audit import scores, mask, sufficient_statistics


def load(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


protocol = load(ROOT / 'evidence/policy_choice/PROTOCOL.json')
frozen = load(ROOT / 'evidence/policy_choice/FROZEN.json')
manifest = load(ROOT / 'MANIFEST.json')['files']
old = ROOT / 'evidence/accuracy_selection'
report = {'scope': 'Rebuild derived statistics from frozen thresholds only.',
          'fit_calls': 0, 'external_accesses': 0,
          'new_thresholds_or_policy_choices': 0, 'numpy_version': np.__version__,
          'inputs': {}, 'outputs': {}}


def verified_input(name):
    path = ROOT / name
    actual = sha(path)
    assert actual == manifest[name]['sha256'], name
    if name in frozen['inputs']:
        assert actual == frozen['inputs'][name], name
    report['inputs'][name] = actual
    return path


for name in ['code/policy_audit.py', 'code/run_policy_choice.py',
             'evidence/policy_choice/FROZEN.json',
             'evidence/policy_choice/PROTOCOL.json']:
    verified_input(name)

with np.load(verified_input('legacy/results/objectives/cache_design/metadata.npz')) as z:
    items, item_index = np.unique(z['item_id'], return_inverse=True)
ni = len(items)
screening_statistics = {}
for model in protocol['predictors']:
    plan = next(p for p in frozen['screens'] if p['model'] == model)
    with np.load(verified_input(f'evidence/additional/sensitivity/{model}_CALIBRATION.npz')) as z:
        e, w, L, W, blocks = z['e'], z['mu'], abs(z['y']-z['f']), z['y'], z['blocks']
    B = blocks == 1
    unitB = np.repeat(item_index, int(B.sum() / len(item_index)))
    assert len(unitB) == B.sum()
    for method, rule in plan['policies'].items():
        s = scores(e, w, plan['cap'], method, plan['T'])
        a = mask(s[B], rule['threshold'])
        screening_statistics[model + '__' + method] = sufficient_statistics(L[B], W[B], a, unitB, ni)
    print('SCREEN statistics reconstructed:', model, flush=True)
np.savez_compressed(OUT / 'SCREEN_STATISTICS.npz', items=items, **screening_statistics)

saved_stats = {}
with np.load(verified_input('evidence/accuracy_selection/evaluation_forecasts.npz')) as z:
    y = z['y'].ravel()
    n_days = len(z['days'])
    predictions = {k: z[k].ravel() for k in protocol['predictors']}
units = np.repeat(item_index, n_days)
total = np.column_stack([np.bincount(units), np.bincount(units, weights=y)])
with np.load(verified_input('evidence/accuracy_selection/ITEM_STATISTICS.npz')) as z:
    assert np.array_equal(items, z['items']) and np.array_equal(total, z['total'])
saved_stats['M5__total'] = total
for model in protocol['predictors'] + ['bike']:
    if model == 'bike':
        with np.load(verified_input('evidence/nonretail/bike/TEST_default.npz')) as z:
            L, W, e, w, units = [z[k] for k in ('L', 'W', 'e', 'w', 'units')]
        ni = int(units.max()) + 1
        total = np.column_stack([np.bincount(units), np.bincount(units, weights=W)])
        saved_stats['Bike__total'] = total
    else:
        with np.load(verified_input(f'evidence/accuracy_selection/{model}_EVAL_SELECTORS.npz')) as z:
            e, w = z['e'], z['mu']
        L, W = abs(y - predictions[model]), y
    for section in ['menus', 'screens']:
        for plan in (p for p in frozen[section] if p['model'] == model):
            primary = ((model == 'bike' and plan['factor'] == .85) or
                       (model != 'bike' and plan['cap'] == protocol['m5_primary_cap']))
            if not primary and section != 'screens':
                continue
            for method, rule in plan['policies'].items():
                a = mask(scores(e, w, plan['cap'], method, plan['T']), rule['threshold'])
                saved_stats[f'{section}__{model}__{method}'] = sufficient_statistics(L, W, a, units, ni)
    print('TEST statistics reconstructed:', model, flush=True)
np.savez_compressed(OUT / 'TEST_STATISTICS.npz', **saved_stats)

for name in ['SCREEN_STATISTICS.npz', 'TEST_STATISTICS.npz']:
    source = OUT / name
    relative = 'evidence/policy_choice/' + name
    expected = manifest[relative]
    actual_hash = sha(source)
    actual_bytes = source.stat().st_size
    matches = actual_hash == expected['sha256'] and actual_bytes == expected['bytes']
    result = {'sha256': actual_hash, 'bytes': actual_bytes,
              'expected_sha256': expected['sha256'], 'expected_bytes': expected['bytes'],
              'matches_original_manifest': matches, 'restored_missing_evidence_path': False}
    target = ROOT / relative
    if matches:
        if target.exists():
            assert sha(target) == expected['sha256'], relative
            result['already_present_and_identical'] = True
        else:
            # Exclusive creation refuses any unexpectedly appearing target.
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert sha(target) == expected['sha256'], relative
            result['restored_missing_evidence_path'] = True
    report['outputs'][name] = result
    print(name, json.dumps(result), flush=True)
report['status'] = 'PASS' if all(r['matches_original_manifest'] for r in report['outputs'].values()) else 'HASH_MISMATCH'
(OUT / 'RESTORATION_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
print(report['status'], flush=True)
