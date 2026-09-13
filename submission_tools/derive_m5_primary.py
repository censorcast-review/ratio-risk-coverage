"""Derive compact replay statistics from immutable M5 prediction arrays.

No fitting, re-selection, type narrowing, or modification of source records.
Run against the complete archive, never against the compact supplement alone.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

METHODS = ('error', 'row', 'mixed', 'demand', 'weight_descending')
SEEDS = (20260910, 20260913, 20260914)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def score(error, exposure, cap, T, method):
    if method == 'error':
        return error
    if method == 'weight_descending':
        return -exposure
    excess = error - cap * exposure
    if method == 'row':
        return excess
    denominator = exposure / T if method == 'demand' else .25 + .75 * exposure / T
    return np.divide(excess, denominator,
                     out=np.where(excess > 0, np.inf, np.where(excess < 0, -np.inf, 0.)),
                     where=denominator > 0)


def derive(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, provenance, checks = [], {}, 0
    for seed in SEEDS:
        relbase = ('evidence/matched_censoring' if seed == SEEDS[0] else
                   f'evidence/revision_v12_seeds/seed_{seed}')
        for arm in ('censored', 'complete'):
            rel = f'{relbase}/{arm}'
            base = source / rel
            frozen, result = read(base / 'FROZEN.json'), read(base / 'RESULTS.json')
            for name in ('FROZEN.json', 'RESULTS.json', 'EVALUATION.npz'):
                q = base / name
                provenance[f'{rel}/{name}'] = {'sha256': digest(q), 'bytes': q.stat().st_size}
            with np.load(base / 'EVALUATION.npz', allow_pickle=False) as archive:
                a = {key: archive[key] for key in archive.files}
            y, loss = a['y'], np.abs(a['y'] - a['f'])
            item_index, days = a['series_item_index'], a['days']
            unit = np.repeat(item_index, len(days))
            n, nrows = len(a['items']), len(y)
            assert len(unit) == nrows
            total = np.column_stack((np.bincount(unit, minlength=n),
                                     np.bincount(unit, weights=y, minlength=n),
                                     np.bincount(unit, weights=loss, minlength=n)))
            arrays = {'total': total, 'series_item_index': item_index, 'days': days,
                      'items': a['items']}
            plans = [p for p in result['menus'] if p['kind'] == 'relative' and p['value'] == .95]
            assert len(plans) == 4
            for p in plans:
                cell = f'e{p["e_trees"]}_w{p["w_trees"]}'
                rec = {'seed': seed, 'arm': arm, 'heads': [p['e_trees'], p['w_trees']],
                       'source_directory': rel, 'array_file': f'{seed}_{arm}.npz',
                       'cell_key': cell, 'rows': nrows, 'items': n, 'cap': p['cap'],
                       'selected': p['selected'], 'mask_sha256': {}}
                for method in METHODS:
                    s = score(a[f'e{p["e_trees"]}'], a[f'w{p["w_trees"]}'],
                              p['cap'], frozen['T'], method)
                    threshold = p['policies'][method]['threshold']
                    mask = (np.ones(nrows, dtype=bool) if threshold == 'all' else
                            np.zeros(nrows, dtype=bool) if threshold is None else s <= threshold)
                    stats = np.column_stack((np.bincount(unit, weights=mask, minlength=n),
                                             np.bincount(unit, weights=mask*y, minlength=n),
                                             np.bincount(unit, weights=mask*loss, minlength=n)))
                    # Do not round or cast floats. The summation order can differ from
                    # the saved row-level sum; this is checked at disclosed tolerance.
                    expected = p['test'][method]
                    direct = np.array([mask.sum(), y[mask].sum(), loss[mask].sum()])
                    assert np.array_equal(stats[:, 0].sum(), direct[0])
                    assert np.allclose(stats.sum(axis=0), direct, rtol=2e-12, atol=2e-7)
                    assert np.allclose(direct, [expected['rows'], expected['weight'], expected['loss']],
                                       rtol=2e-12, atol=2e-7), (seed, arm, cell, method)
                    packed = np.packbits(mask, bitorder='little')
                    assert np.array_equal(np.unpackbits(packed, bitorder='little')[:nrows], mask)
                    arrays[f'{cell}__{method}__statistics'] = stats
                    arrays[f'{cell}__{method}__mask'] = packed
                    rec['mask_sha256'][method] = hashlib.sha256(mask.tobytes()).hexdigest()
                    checks += 4
                records.append(rec)
            target = output / f'{seed}_{arm}.npz'
            np.savez_compressed(target, **arrays)
            with np.load(target, allow_pickle=False) as back:
                assert set(back.files) == set(arrays)
                for key in arrays:
                    assert np.array_equal(back[key], arrays[key]), (target.name, key)
                    checks += 1
            print(f'Derived {seed} {arm}: {target.stat().st_size} bytes', flush=True)
    manifest = {
        'schema': 'compact-m5-primary-v1',
        'status': 'PASS',
        'scope': '24 primary cells; all five fixed candidates; item statistics and exact bit-packed row masks',
        'excluded_scope': ['training replay', 'calibration score/threshold reconstruction',
                           'non-primary cap grids and secondary original-seed analyses'],
        'bootstrap': {'seed': 20260910, 'draws': 2000, 'unit': 'item',
                      'quantiles': [.025, .975], 'paired_across_policies': True},
        'statistics_columns': ['accepted_rows', 'accepted_exposure', 'accepted_absolute_loss'],
        'mask_bitorder': 'little',
        'summation_check': {'rtol': 2e-12, 'atol': 2e-7,
                            'note': 'counts and masks exact; exposure/loss sums permit floating-point order error'},
        'dtype_policy': 'original arrays are not modified; float64 statistics are saved without type narrowing',
        'derivation_code_sha256': digest(__file__),
        'checks': checks,
        'source_files': provenance,
        'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                  for p in sorted(output.glob('*.npz'))},
        'records': records,
    }
    (output / 'DERIVATION.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    derive(args.source, args.output)
