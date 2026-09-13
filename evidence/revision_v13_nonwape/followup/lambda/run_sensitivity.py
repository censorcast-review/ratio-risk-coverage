"""Retrospective coefficient sensitivity; never modifies primary evidence.

All threshold fitting is A-only, but the analysis was requested after A/B/E
results were seen. Its intervals and screen decisions are descriptive only.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MENU = ['Error', 'Excess', 'Mixed', 'Ratio', 'Descending']
METHODS = MENU + ['GateCase', 'GateExposure']
WEIGHTS = [.25, .5, .75]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def measure(arr, accept):
    w, loss = arr['exposure'], arr['loss']
    count, W, L = int(accept.sum()), float(w[accept].sum()), float(loss[accept].sum())
    return {'n_total': len(w), 'n_accepted': count, 'accepted_exposure': W,
            'accepted_loss': L, 'case_coverage': count/len(w),
            'exposure_coverage': W/float(w.sum()), 'risk': L/W if W else None}


def mask(score, tie, policy):
    if not policy['eligible']:
        return np.zeros(len(score), dtype=bool)
    return (score < policy['threshold']) | ((score == policy['threshold']) &
                                           (tie <= policy['boundary_hash']))


def design(arr, score, cap, floor):
    order = np.lexsort((arr['tie'], score))
    W, L = arr['exposure'][order].cumsum(), arr['loss'][order].cumsum()
    feasible = (np.arange(1, len(score)+1)/len(score) >= floor) & (W > 0) & (L <= cap*W)
    positions = np.flatnonzero(feasible)
    if not len(positions):
        return {'eligible': False, 'threshold': None, 'boundary_hash': None, 'A': None}
    row = order[positions[-1]]
    policy = {'eligible': True, 'threshold': float(score[row]), 'boundary_hash': int(arr['tie'][row])}
    policy['A'] = measure(arr, mask(score, arr['tie'], policy))
    return policy


def bootstrap(arr, masks, cap, seed):
    n = len(arr['exposure'])
    w, loss = arr['exposure'], arr['loss']
    columns = [w]
    for take in masks.values():
        columns.extend([take.astype(float), take*w, take*loss])
    matrix = np.column_stack(columns)
    result = {key: {field: [] for field in ['c', 'd', 'risk', 'g']} for key in masks}
    rng = np.random.default_rng(seed)
    for first in range(0, 20000, 128):
        counts = rng.multinomial(n, np.full(n, 1/n), size=min(128, 20000-first))
        sums = counts @ matrix
        for j, key in enumerate(masks):
            N, W, L = (sums[:, 1+3*j+k] for k in range(3))
            values = [N/n, W/sums[:, 0], np.divide(L, W, out=np.full_like(L, np.nan), where=W > 0), (L-cap*W)/n]
            for field, values in zip(['c', 'd', 'risk', 'g'], values):
                result[key][field].append(values)
    return {key: {field: np.concatenate(parts) for field, parts in rec.items()} for key, rec in result.items()}


def select(policies, utility):
    valid = [key for key in MENU if policies[key]['eligible']]
    field = utility+'_coverage'
    return max(valid, key=lambda name: policies[name]['A'][field]) if valid else None


def descending_diagnostic(arr, cap, design_cap, floor):
    order = np.lexsort((arr['tie'], -arr['exposure']))
    counts = np.arange(1, len(order)+1)
    W, L = arr['exposure'][order].cumsum(), arr['loss'][order].cumsum()
    risk = L/W
    eligible = counts/len(order) >= floor
    minimum_index = int(np.flatnonzero(eligible)[np.argmin(risk[eligible])])
    return {'n_floor_eligible_prefixes': int(eligible.sum()),
            'minimum_A_risk_among_floor_eligible_prefixes': float(risk[minimum_index]),
            'minimum_at_case_coverage': float(counts[minimum_index]/len(order)),
            'minimum_at_exposure_coverage': float(W[minimum_index]/W[-1]),
            'minimum_accepted_count': int(counts[minimum_index]),
            'minimum_accepted_loss': float(L[minimum_index]),
            'minimum_accepted_exposure': float(W[minimum_index]),
            'full_A_risk': float(risk[-1]), 'cap': cap, 'design_cap': design_cap,
            'number_feasible_at_cap': int((eligible & (L <= cap*W)).sum()),
            'number_feasible_at_design_cap': int((eligible & (L <= design_cap*W)).sum()),
            'interpretation': 'Observed prefix feasibility only. This does not identify why this classifier assigns false positives to particular exposures.'}


def run_dataset(dataset, protocol):
    base = ROOT/'evidence'/dataset
    fixed = json.loads((base/'A_FIXED.json').read_text())
    primary = json.loads((base/'RESULT.json').read_text())
    arrays = {role: dict(np.load(base/(role+'_ARRAYS.npz'), allow_pickle=False)) for role in ['A', 'B', 'E']}
    original_boot = {role: dict(np.load(base/(role+'_BOOTSTRAP.npz'), allow_pickle=False)) for role in ['B', 'E']}
    scores, policies, masks = {}, {}, {role: {} for role in ['A', 'B', 'E']}
    check_count = 0
    for weight in WEIGHTS:
        key = f'Mixed_exposure_weight_{weight:.2f}'
        scores[key] = {role: (arr['Error']-fixed['r']*arr['exposure'])/
                       ((1-weight)+weight*arr['exposure']/fixed['mean_H_w']) for role, arr in arrays.items()}
        policies[key] = design(arrays['A'], scores[key]['A'], fixed['design_cap'], fixed['floor'])
        for role, arr in arrays.items():
            masks[role][key] = mask(scores[key][role], arr['tie'], policies[key])
        if weight == .5:
            assert policies[key]['eligible'] == fixed['policies']['Mixed']['eligible']
            assert policies[key]['threshold'] == fixed['policies']['Mixed']['threshold']
            assert policies[key]['boundary_hash'] == fixed['policies']['Mixed']['boundary_hash']
            check_count += 3
            for role, arr in arrays.items():
                assert np.array_equal(scores[key][role], arr['Mixed'])
                if role != 'A':
                    assert np.array_equal(masks[role][key], arr['mask_Mixed'])
                    check_count += 1
                check_count += 1
    for name in METHODS:
        for role, arr in arrays.items():
            original_mask = mask(arr[name], arr['tie'], fixed['policies'][name])
            masks[role]['Original_'+name] = original_mask
            if role != 'A':
                assert np.array_equal(original_mask, arr['mask_'+name])
                check_count += 1
    # Only changed Mixed policies need new bootstrap columns. Include .5 to
    # independently validate the draw sequence against the frozen arrays.
    boot = {}
    max_boot_difference = 0.0
    for role, seed_offset in [('B', 1), ('E', 2)]:
        subset = {key: value for key, value in masks[role].items() if not key.startswith('Original_')}
        boot[role] = bootstrap(arrays[role], subset, fixed['r'], protocol['datasets'][dataset]['seed']*100+seed_offset)
        for field in ['c', 'd', 'risk', 'g']:
            recreated = boot[role]['Mixed_exposure_weight_0.50'][field]
            original = original_boot[role]['Mixed_'+field]
            difference = float(np.max(np.abs(recreated-original)))
            max_boot_difference = max(max_boot_difference, difference)
            assert np.allclose(recreated, original, rtol=0, atol=2e-14, equal_nan=True)
            check_count += 1
        for name in METHODS:
            boot[role]['Original_'+name] = {field: original_boot[role][name+'_'+field] for field in ['c', 'd', 'risk', 'g']}

    sensitivity = {}
    for weight in WEIGHTS:
        key = f'Mixed_exposure_weight_{weight:.2f}'
        menu = dict(fixed['policies'])
        menu['Mixed'] = policies[key]
        choices = {u: select(menu, u) for u in ['case', 'exposure']}
        B = measure(arrays['B'], masks['B'][key])
        B['signed_excess_upper'] = float(np.quantile(boot['B'][key]['g'], 1-.05/14, method='linear'))
        B['risk_percentile_upper'] = float(np.quantile(boot['B'][key]['risk'], 1-.05/14, method='linear'))
        B['passes_original_numerical_rule_descriptively'] = bool(policies[key]['eligible'] and np.isfinite(boot['B'][key]['risk']).all() and B['signed_excess_upper'] < 0)
        E = measure(arrays['E'], masks['E'][key])
        chosen_keys = {u: key if name == 'Mixed' else 'Original_'+name for u, name in choices.items()}
        chosen_metrics = {u: measure(arrays['E'], masks['E'][name]) for u, name in chosen_keys.items()}
        comparison = {'case_choice': choices['case'], 'exposure_choice': choices['exposure'],
                      'interval_level': .9875, 'interval_status': 'post hoc descriptive; not added to the original confirmatory family'}
        for short, long in [('c', 'case'), ('d', 'exposure')]:
            contrast = 100*(boot['E'][chosen_keys['exposure']][short]-boot['E'][chosen_keys['case']][short])
            comparison['delta_'+short+'_pp'] = 100*(chosen_metrics['exposure'][long+'_coverage']-chosen_metrics['case'][long+'_coverage'])
            comparison['interval_'+short+'_pp'] = [float(v) for v in np.quantile(contrast, [.00625, .99375], method='linear')]
        comparison['both_selected_pass_original_screen_numerically'] = all(
            B['passes_original_numerical_rule_descriptively'] if name == 'Mixed' else primary['B'][name]['passed'] for name in choices.values())
        comparison['satisfies_original_joint_inequalities_descriptively'] = bool(
            comparison['both_selected_pass_original_screen_numerically'] and comparison['interval_c_pp'][1] < 0 and comparison['interval_d_pp'][0] > 0)
        if weight == .5:
            assert choices == fixed['menu_selected']
            for field in ['delta_c_pp', 'delta_d_pp']:
                assert abs(comparison[field]-primary['primary'][field]) < 1e-12
                check_count += 1
            for field in ['interval_c_pp', 'interval_d_pp']:
                assert np.allclose(comparison[field], primary['primary'][field], atol=1e-12, rtol=0)
                check_count += 1
            check_count += 1
        sensitivity[str(weight)] = {'mixed_denominator': f'{1-weight:g}+{weight:g}*w/mean_H(w)',
            'exposure_utility_weight': weight, 'analysis_status': 'frozen reference' if weight == .5 else 'post hoc sensitivity',
            'Mixed_policy': policies[key], 'Mixed_B': B, 'Mixed_E': E,
            'selected_comparison_E': comparison,
            'changed_masks_vs_frozen_Mixed': {role: int(np.count_nonzero(masks[role][key] != masks[role]['Original_Mixed'])) for role in arrays}}

    all_candidates = {}
    for name in METHODS:
        all_candidates[name] = {'A_eligible': fixed['policies'][name]['eligible'],
            'A': fixed['policies'][name].get('A'),
            'B_point_risk': primary['B'][name]['risk'] if fixed['policies'][name]['eligible'] else None,
            'B_signed_excess_upper': primary['B'][name]['signed_excess_upper'] if fixed['policies'][name]['eligible'] else None,
            'B_pass': primary['B'][name]['passed'],
            'E': primary['E'][name] if fixed['policies'][name]['eligible'] else None}
    return {'dataset': dataset, 'cap': fixed['r'], 'design_cap': fixed['design_cap'], 'floor': fixed['floor'],
            'mean_H_w': fixed['mean_H_w'], 'original_joint_success_unchanged': primary['joint_success'],
            'frozen_candidates': all_candidates, 'sensitivity': sensitivity,
            'Descending_A_prefix_diagnostic': descending_diagnostic(arrays['A'], fixed['r'], fixed['design_cap'], fixed['floor']),
            'verification': {'assertions_passed': check_count, 'maximum_frozen_bootstrap_absolute_difference': max_boot_difference}}


def main():
    spec = json.loads((HERE/'ANALYSIS_SPEC.json').read_text())
    assert spec['analysis_status'] == 'retrospective; original results already known'
    before = {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT/'evidence').rglob('*')) if path.is_file()}
    protocol = json.loads((ROOT/'evidence/PROTOCOL.json').read_text())
    for rel, expected in protocol['code_sha256'].items():
        assert sha(ROOT/rel) == expected
    receipt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'spec_sha256': sha(HERE/'ANALYSIS_SPEC.json'),
               'script_sha256': sha(Path(__file__)), 'primary_input_hashes': before}
    dump(HERE/'STARTED.json', receipt)
    results = {dataset: run_dataset(dataset, protocol) for dataset in ['bibtex', 'mediamill']}
    after = {rel: sha(ROOT/rel) for rel in before}
    assert before == after, 'Primary evidence changed'
    for rel, expected in protocol['code_sha256'].items():
        assert sha(ROOT/rel) == expected
    payload = {'analysis_status': spec['analysis_status'], 'v12_equivalence_claimed': False,
               'interpretation': spec['interpretation'], 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'original_evidence_unchanged': True, 'datasets': results}
    dump(HERE/'RESULTS.json', payload)
    dump(HERE/'COMPLETE.json', {'utc': datetime.now(timezone.utc).isoformat(), 'spec_sha256': sha(HERE/'ANALYSIS_SPEC.json'),
        'script_sha256': sha(Path(__file__)), 'results_sha256': sha(HERE/'RESULTS.json'),
        'primary_evidence_files_unchanged': len(before), 'frozen_code_files_unchanged': len(protocol['code_sha256'])})
    print(json.dumps({'original_evidence_unchanged': True, 'dataset_checks': {k: v['verification'] for k, v in results.items()}}))


if __name__ == '__main__':
    main()
