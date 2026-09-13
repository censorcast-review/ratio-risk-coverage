"""Replay all 24 M5 primary cells, CIs, and Table 1 from compact item statistics.

This verifies arithmetic and fixed-mask counts. It does not independently recover
the omitted row-level loss/forecast arrays or reselect calibration thresholds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

METHODS = ('error', 'row', 'mixed', 'demand', 'weight_descending')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(root, output):
    root, output = Path(root), Path(output)
    derived = root / 'submission_derived/m5_primary'
    manifest = read(derived / 'DERIVATION.json')
    checks, replay = 0, []

    def close(a, b, tol=2e-9):
        nonlocal checks
        assert np.allclose(a, b, atol=tol, rtol=tol), (a, b)
        checks += 1

    for name, rec in manifest['files'].items():
        assert sha(derived / name) == rec['sha256'], name
        checks += 1
    # All fixed decisions and declared outcomes are included byte-for-byte. Large
    # prediction-array hashes remain provenance links, not independently checked
    # source data in this compact archive.
    for rel, rec in manifest['source_files'].items():
        p = root / rel
        if p.is_file():
            assert sha(p) == rec['sha256'], rel
            checks += 1
    cached, bootstrap = {}, {}
    for rec in manifest['records']:
        rel, cell, n = rec['source_directory'], rec['cell_key'], rec['items']
        if rec['array_file'] not in cached:
            with np.load(derived / rec['array_file'], allow_pickle=False) as z:
                cached[rec['array_file']] = {k: z[k] for k in z.files}
        a = cached[rec['array_file']]
        if n not in bootstrap:
            bootstrap[n] = np.random.default_rng(20260910).multinomial(n, np.full(n, 1/n), size=2000)
        counts, total = bootstrap[n], a['total']
        result, frozen = read(root / rel / 'RESULTS.json'), read(root / rel / 'FROZEN.json')
        plans = [p for p in result['menus'] if p['kind'] == 'relative' and p['value'] == .95
                 and [p['e_trees'], p['w_trees']] == rec['heads']]
        assert len(plans) == 1
        p = plans[0]
        assert rec['selected'] == p['selected']
        close(p['cap'], .95*frozen['full_calibration_reference'])
        eligible = [m for m in METHODS if p['policies'][m]['threshold'] is not None]
        selected = {objective: max(eligible, key=lambda m: min(x[objective] for x in p['policies'][m]['calibration']))
                    for objective in ('c', 'd')}
        assert selected == rec['selected']
        checks += 3
        stats = {}
        for m in METHODS:
            v = a[f'{cell}__{m}__statistics']
            mask = np.unpackbits(a[f'{cell}__{m}__mask'], bitorder='little')[:rec['rows']].astype(bool)
            assert hashlib.sha256(mask.tobytes()).hexdigest() == rec['mask_sha256'][m]
            # Independent count aggregation: first sum each series over days,
            # then add its count to the item's total.
            row_counts = np.zeros(n, dtype=np.int64)
            for item, count in zip(a['series_item_index'], mask.reshape(-1, len(a['days'])).sum(axis=1)):
                row_counts[item] += count
            assert np.array_equal(v[:, 0], row_counts)
            checks += 2
            mass = v.sum(axis=0)
            expected = p['test'][m]
            close(mass, [expected['rows'], expected['weight'], expected['loss']])
            close(mass[:2] / total[:, :2].sum(axis=0), [expected['c'], expected['d']])
            if mass[1] > 0:
                close(mass[2] / mass[1], expected['risk'])
            else:
                assert expected['risk'] is None
                checks += 1
            stats[m] = v
        c, d = selected['c'], selected['d']
        numerator = (stats[d] - stats[c])[:, :2]
        delta = numerator.sum(axis=0) / total[:, :2].sum(axis=0)
        draws = (counts @ numerator) / (counts @ total[:, :2])
        ci = np.quantile(draws, [.025, .975], axis=0).T
        close(delta, [p['contrast']['dc'], p['contrast']['dd']])
        close(ci, p['contrast']['ci95'])
        replay.append({'seed': rec['seed'], 'arm': rec['arm'], 'heads': rec['heads'],
                       'dc': float(delta[0]), 'dd': float(delta[1]), 'ci95': ci.tolist(),
                       'exposure_choice': d})
    summary = read(root / 'evidence/revision_v12_seeds/SUMMARY.json')
    for row in summary['summary']:
        group = sorted([r for r in replay if r['arm'] == row['arm'] and r['heads'] == row['heads']],
                       key=lambda r: r['seed'])
        assert len(group) == 3
        effects = 100*np.array([[r['dc'], r['dd']] for r in group])
        close(effects.mean(axis=0), row['mean_pp'])
        close(effects.std(axis=0, ddof=1), row['sample_sd_pp'])
        close([effects[:, 0].min(), effects[:, 0].max()], row['dc_range_pp'])
        close([effects[:, 1].min(), effects[:, 1].max()], row['dd_range_pp'])
        assert [r['exposure_choice'] for r in group] == row['exposure_choices']
        checks += 2
    report = {'status': 'PASS', 'checks': checks, 'primary_cells': len(replay),
              'scope': 'fixed-mask item counts, all candidate point metrics, 2,000 paired item-bootstrap intervals, Table 1 mean/sample SD and rule switches',
              'limitation': 'calibration metrics are preserved records; omitted row-level forecasts/losses are not reconstructed by this compact replay',
              'records': replay, 'verifier_sha256': sha(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'checks': checks, 'primary_cells': len(replay)}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('reproduction_outputs/compact_m5_check.json'))
    args = parser.parse_args()
    verify(args.root, args.output)
