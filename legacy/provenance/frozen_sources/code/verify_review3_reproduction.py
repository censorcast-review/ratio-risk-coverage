"""Repeat the review3 audit using saved, already-consumed prediction caches.

Run from the project directory:
    python code/verify_review3_reproduction.py

This wrapper fits no models and never invokes a raw guardian/external loader.
It writes only to reproduction_outputs/cached_replay (or --output), apart from Python bytecode and
the invariant tests' temporary synthetic fixtures. Historical reports are
checked unchanged. The six demand-loss models are historical stored models,
not fits performed by this invocation. This is post-hoc reproduction, not a
new independent holdout evaluation or an attestation of human verification.
"""

from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import re
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
R2 = ROOT / 'results/review2'
GUARDIAN = ROOT.parent / 'evaluation_checkpoints'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def without_run_metadata(value):
    if isinstance(value, dict):
        return {key: without_run_metadata(item) for key, item in value.items()
                if key not in {'created_utc', 'seconds'}}
    if isinstance(value, list):
        return [without_run_metadata(item) for item in value]
    return value


def run(script, *args):
    subprocess.run([sys.executable, str(CODE / script), *map(str, args)],
                   cwd=ROOT, check=True)


def table_numbers(name):
    rows = []
    for line in (ROOT / 'paper/tables' / name).read_text().splitlines():
        if ' & ' not in line or 'midrule' in line:
            continue
        values = []
        for token in line.replace('\\\\', '').strip().split(' & '):
            try:
                values.append(float(token))
            except ValueError:
                pass
        if values:
            rows.append(values)
    return rows


def verify_tables(d, loss):
    def average(rows, value):
        return sum(value(x) for x in rows) / len(rows)

    count = 0

    def check(name, expected):
        nonlocal count
        require(table_numbers(name) == expected, 'Printed table mismatch: ' + name)
        count += len(expected)

    expected = []
    for objective in ['row', 'demand']:
        rows = [x for x in d['family_choices'] if x['objective'] == objective]
        for cohort in ['design', 'guardian_posthoc']:
            expected.append([
                round(100 * average(rows, lambda x: x[cohort]['shadow']['ALL'][m]), 2)
                for m in ['coverage', 'demand_coverage']
            ] + [round(average(rows, lambda x: x[cohort]['shadow']['ALL']['wape']), 4)])
    check('objective_table.tex', expected)

    expected = []
    for ne, nm, name in [(110, 110, 'composite_excess_budget220'),
                         (110, 220, 'error110_demand220'),
                         (220, 110, 'error220_demand110'),
                         (220, 220, 'composite_excess')]:
        for cohort in ['design', 'guardian_posthoc']:
            rows = [x for x in d['records'] if x['score'] == name]
            heads = [x for x in d['head_diagnostics']
                     if x['cohort'] == cohort and x['block'] == 'shadow'
                     and x['group'] == 'ALL'
                     and (x['error_trees'], x['demand_trees']) == (ne, nm)]
            expected.append([
                round(100 * average(rows, lambda x: x[cohort]['shadow']['ALL']['coverage']), 2)
            ] + [round(average(heads, lambda x: x[m]), 4)
                 for m in ['error_mse', 'demand_mse', 'cross_error_moment', 'excess_mse']])
    check('head_factorial_table.tex', expected)

    expected = []
    for cohort in ['design', 'guardian_posthoc']:
        for name in ['composite_excess_budget220', 'error110_demand220',
                     'error220_demand110', 'composite_excess']:
            rows = [x for x in d['records'] if x['score'] == name]
            expected.append([
                round(100 * average(rows, lambda x: x[cohort]['shadow']['ALL'][m]), 2)
                for m in ['coverage', 'demand_coverage']
            ] + [round(average(rows, lambda x: x[cohort]['shadow']['ALL']['wape']), 4)])
    check('head_risk_table.tex', expected)

    expected = []
    for objective in ['squared', 'poisson', 'tweedie']:
        for name in ['composite_excess', 'relative_error_mu']:
            records = d['records'] if objective == 'squared' else loss['records']
            rows = [x for x in records if x['score'] == name
                    and (objective == 'squared' or x['demand_loss'] == objective)]
            expected.append([
                round(100 * average(rows, lambda x: x[c]['shadow']['ALL'][m]), 2)
                for m in ['coverage', 'demand_coverage']
                for c in ['design', 'guardian_posthoc']
            ] + [round(average(rows, lambda x: x['guardian_posthoc']['shadow']['ALL']['wape']), 4)])
    check('loss_controls_table.tex', expected)

    expected = []
    for x in d['floor_sensitivity']:
        v, dv = x['guardian_posthoc']['shadow']['ALL'], x['design']['shadow']['ALL']
        expected.append([x['floor'], round(100 * dv['demand_coverage'], 2),
                         round(100 * v['coverage'], 2), round(100 * v['demand_coverage'], 2),
                         round(v['wape'], 4)])
    check('floor_sensitivity_table.tex', expected)

    expected = []
    for cohort in ['design', 'guardian_posthoc']:
        for x in d['partial_oracles']:
            if x['cohort'] == cohort and x['group'] == 'HOBBIES' and x['ranking'] == 'excess':
                expected.append([round(100 * x[m], 2) for m in ['row_coverage', 'demand_coverage']]
                                + ([round(x['wape'], 4)] if x['wape'] is not None else []))
    check('partial_oracle_table.tex', expected)
    return count


def verify_static_reports(d, loss):
    count = 0

    def walk(x):
        nonlocal count
        if isinstance(x, dict):
            if {'n', 'accepted_n', 'coverage', 'wape'} <= set(x):
                require(0 <= x['accepted_n'] <= x['n'], 'Invalid accepted count')
                require(math.isclose(x['accepted_n'] / x['n'], x['coverage'], abs_tol=1e-13),
                        'Coverage/count mismatch')
                if x.get('demand_coverage') is not None:
                    require(0 <= x['demand_coverage'] <= 1.0000001, 'Invalid demand coverage')
                if x.get('baseline_same_subset_wape') not in [None, 0] and x['wape'] is not None:
                    require(math.isclose(x['wape'] / x['baseline_same_subset_wape'],
                                         x['paired_error_ratio'], abs_tol=1e-12),
                            'Paired-error ratio mismatch')
                count += 1
            if set(x) == {'ALL', 'FOODS', 'HOBBIES', 'HOUSEHOLD'}:
                for key in ['n', 'accepted_n']:
                    require(x['ALL'][key] == sum(x[g][key] for g in ['FOODS', 'HOBBIES', 'HOUSEHOLD']),
                            'Category count mismatch')
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)

    walk(d)
    walk(loss)
    aggregates = read(R2 / 'PAPER_AGGREGATES.json')
    for f in d['family_choices']:
        candidates = [x for x in d['records'] if x['seed'] == f['seed']
                      and x['score'] in d['protocol']['primary_families']
                      and x['row_threshold'] is not None]
        best = sorted(candidates, key=lambda x: (-x[f['objective'] + '_objective'], x['score']))[0]
        require(f['selected_score'] == best['score'] and f['threshold'] == best['row_threshold'],
                'Family selection is not the documented development argmax')
        require(f['minimum_calibration_objective'] == best[f['objective'] + '_objective'],
                'Family objective mismatch')
    for obj in ['row', 'demand']:
        rows = [x for x in d['family_choices'] if x['objective'] == obj]
        for cohort in ['design', 'guardian_posthoc']:
            for metric in ['coverage', 'demand_coverage', 'wape']:
                value = sum(x[cohort]['shadow']['ALL'][metric] for x in rows) / len(rows)
                require(math.isclose(value, aggregates['objectives'][obj][cohort][metric], abs_tol=1e-14),
                        'Paper aggregate mismatch')
    reference = read(R2 / 'CAP_REFERENCE.json')
    require(math.isclose(reference['value'] * reference['multiplier'], reference['cap'], abs_tol=1e-12),
            'Legacy cap multiplication mismatch')
    require(reference['source_sha256'] == read(ROOT / 'provenance/input_manifest.json')
            ['selection_base_v0_5.npz']['sha256'], 'Legacy cap source provenance mismatch')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduction_outputs/cached_replay')
    args = parser.parse_args()
    out = __import__('reproduction_io').safe_output(args.output)
    require(out != R2.resolve() and R2.resolve() not in out.parents,
            'Output must not overwrite historical review2 results')
    required = [R2 / 'cache_design/metadata.npz', GUARDIAN / 'cache/metadata.npz',
                GUARDIAN / 'GUARDIAN_COMPARISON.json']
    for seed in [20260906, 20260907, 20260908]:
        required += [R2 / f'design_predictions_s{seed}.npz', GUARDIAN / f'predictions_s{seed}.npz']
        required += [R2 / f'loss_controls/predictions_{loss}_s{seed}.npz'
                     for loss in ['poisson', 'tweedie']]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, 'Restore the seven saved consumed-cache archives first; missing: ' + ', '.join(missing))
    out.mkdir(parents=True, exist_ok=True)
    frozen_path = ROOT / 'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json'
    frozen = read(frozen_path)
    preserved = set(R2.rglob('*.json')) | {frozen_path, GUARDIAN / 'GUARDIAN_COMPARISON.json'}
    preserved |= {ROOT / path for path in frozen['immutable_files']}
    before = {path: sha(path) for path in preserved}

    for script in ['test_core_invariants.py', 'test_review2_objectives.py', 'test_review_invariants.py']:
        run(script)

    # Redirect only the historical verifier's report writer, never historical input files.
    sys.path.insert(0, str(CODE))
    import r2_io
    original_dump = r2_io.dump

    def redirected_dump(path, data):
        require(Path(path).resolve() == (R2 / 'VERIFICATION.json').resolve(),
                'Unexpected historical verifier write: ' + str(path))
        original_dump(out / 'VERIFICATION.json', data)

    r2_io.dump = redirected_dump
    try:
        runpy.run_path(str(CODE / 'verify_review2.py'), run_name='__main__')
    finally:
        r2_io.dump = original_dump

    run('run_review2_loss_cached.py', '--design', R2, '--guardian', GUARDIAN,
        '--loss', R2 / 'loss_controls', '--output', out / 'LOSS_CACHE_VERIFICATION.json')
    run('run_review2_objectives.py', '--design', R2, '--guardian', GUARDIAN,
        '--output', out / 'objectives')

    d, rerun = read(R2 / 'objectives/REVIEW2_RESULTS.json'), read(out / 'objectives/REVIEW2_RESULTS.json')
    require(without_run_metadata(d) == without_run_metadata(rerun),
            'Full objective report differs beyond run timestamp/elapsed time')
    require(without_run_metadata(read(R2 / 'objectives/PROTOCOL.json'))
            == without_run_metadata(read(out / 'objectives/PROTOCOL.json')),
            'Objective protocol differs beyond run timestamp')
    counts = {}
    for key in ['records', 'family_choices', 'floor_sensitivity', 'head_diagnostics', 'partial_oracles']:
        require(d[key] == rerun[key], 'Objective rerun mismatch: ' + key)
        counts[key] = len(d[key])
    for name in ['FRONTIERS.json', 'OBJECTIVE_CONTRAST.json', 'DEVELOPMENT_SELECTED_POLICIES.json']:
        require(read(R2 / 'objectives' / name) == read(out / 'objectives' / name), 'Rerun mismatch: ' + name)
    counts['frontier_records'] = len(read(out / 'objectives/FRONTIERS.json'))
    loss = read(R2 / 'loss_controls/LOSS_RESULTS.json')
    require(loss['records'] == read(out / 'LOSS_CACHE_VERIFICATION.json')['records'], 'Loss rerun mismatch')
    metric_count = verify_static_reports(d, loss)
    table_count = verify_tables(d, loss)
    models = sorted((R2 / 'loss_controls').glob('demand_*.txt'))
    require(len(models) == 6, 'Expected six stored historical loss models')
    for path in models:
        text = path.read_text()
        objective = re.search(r'^objective=(.*)$', text, re.M).group(1)
        require(len(re.findall(r'^Tree=\d+$', text, re.M)) == 220, 'Loss model tree count mismatch')
        for label in ['poisson', 'tweedie']:
            require((label in objective) == (label in path.name), 'Loss model objective mismatch')
    for path, expected in before.items():
        require(sha(path) == expected, 'Historical file changed: ' + str(path))

    historical = read(out / 'VERIFICATION.json')
    audit = dict(
        status='CURRENT_REVISION_REPRODUCIBILITY_PASS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_calls_this_revision=0,
        stored_historical_demand_loss_models=6,
        raw_guardian_opened_this_revision=False,
        external_shard_opened=False,
        consumed_aligned_guardian_caches_reused=True,
        new_independent_holdout_evaluations=0,
        historical_guardian_result_sha256=sha(GUARDIAN / 'GUARDIAN_COMPARISON.json'),
        historical_files_preserved=len(before),
        frozen_files_verified=historical['frozen_files_unchanged'],
        historical_policy_block_group_metrics_recomputed=historical['historical_cohort_block_group_metric_records_reproduced'],
        cached_loss_reports_exactly_recomputed=len(loss['records']),
        objective_rerun_exact_record_counts=counts,
        objective_first_seed_bootstrap_exactly_reproduced=True,
        objective_design_selected_policies_exactly_reproduced=True,
        cache_free_core_tests=6,
        cache_free_objective_tests=4,
        historical_policy_replay_check='PASS',
        new_table_numeric_rows_verified=table_count,
        static_nested_metric_reports_verified=metric_count,
        loss_models_metadata_verified=len(models),
        legacy_reference_cap_check='stored source hash agrees with input manifest; scalar multiplication checked; original selection archive not included in compact caches',
        human_author_verification='NOT_ATTESTED_BY_AUTOMATED_CHECKS',
        scope='Rerun of saved post-hoc diagnostics; no new confirmation or human attestation.'
    )
    original_dump(out / 'CURRENT_REPRO_AUDIT.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
