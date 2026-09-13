"""Check the completed, fixed guardian audit against frozen objects and masks.

This verification does not fit models, choose thresholds, or read external data.
"""
from pathlib import Path
import json, hashlib, datetime, sys, platform
import numpy as np
from run_selective_study import load, metric, origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/guardian_comparison'
EV = OUT / 'evaluation'
FREEZE = ROOT / 'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
f = json.loads(FREEZE.read_text())
assert sha(FREEZE) == '3d94aec80d993b92684a61f058ed4e3ceb2351233f07fb8eea3f9f542fe2b156'
for rel, wanted in f['immutable_files'].items():
    assert sha(ROOT / rel) == wanted, rel
result = json.loads((EV / 'GUARDIAN_COMPARISON.json').read_text())
assert result['freeze_sha256'] == sha(FREEZE)
assert result['status'] == 'HELD_OUT_ITEM_COMPARISON_COMPLETED'
assert result['training_calls'] == result['threshold_selection_calls'] == 0
assert result['fresh_guardian_use_count'] == 1
assert not result['external_opened'] and not result['certificate_issued']
assert len(result['records']) == len(f['policies']) == 42
meta = load(EV / 'cache/metadata.npz')
design = load(ROOT / 'inputs/design_outcomes_v0_5.npz')
assert len(meta['id']) == 6100 and len(set(meta['item_id'])) == 610
assert not (set(meta['item_id']) & set(design['item_id']))
del design
data = {k: load(EV / ('cache/' + k + '_aligned.npz')) for k in f['evaluation']['blocks']}
truth = load(ROOT / 'inputs/guardian/fresh_guardian_outcomes_v0_5.npz')
assert sha(ROOT / 'inputs/guardian/fresh_guardian_outcomes_v0_5.npz') == f['guardian_outcomes_sha256']
assert sha(ROOT / 'inputs/guardian/fresh_guardian_context_v0_5.npz') == f['guardian_context_sha256']
for block, (start, end) in f['evaluation']['blocks'].items():
    v = data[block]
    np.testing.assert_array_equal(v['target_days'], np.arange(start, end + 1))
    np.testing.assert_array_equal(v['truth'], truth['truth'][:, start-1554:end-1554+1])
    assert v['truth'].shape == (6100, end-start+1)
    assert np.isfinite(v['proposal']).all() and (v['proposal'] >= 0).all()
pred = {s: load(EV / f'predictions_s{s}.npz') for s in f['seeds']}
groups = ['ALL'] + sorted(set(meta['cat_id']))
checked = 0
for report, frozen in zip(result['records'], f['policies']):
    for key in ['seed', 'score', 'threshold_mode', 'thresholds']:
        assert report[key] == frozen[key], key
    stored = load(EV / f"masks_{report['score']}_{report['threshold_mode']}_s{report['seed']}.npz")
    p = pred[report['seed']]
    for block, v in data.items():
        e = np.maximum(p['error_'+block], 0)
        mu = np.maximum(p['demand_'+block], 0)
        family = {
            'mean_error': e,
            'relative_error_f': e / np.maximum(v['proposal'], .25),
            'relative_error_mu': e / np.maximum(mu, .25),
            'excess_f': e - f['wape_cap'] * v['proposal'],
            'composite_excess': e - f['wape_cap'] * mu,
            'composite_excess_budget220': np.maximum(p['error110_'+block], 0) - f['wape_cap'] * np.maximum(p['demand110_'+block], 0),
            'direct_excess': p['direct_'+block],
        }
        score = family[report['score']]
        mask = np.zeros(score.shape, bool)
        for g, t in frozen['thresholds'].items():
            if t is None:
                continue
            rows = np.ones(len(meta['id']), bool) if g == 'ALL' else meta['cat_id'] == g
            mask[rows] = np.isfinite(score[rows]) & (score[rows] <= t)
        np.testing.assert_array_equal(mask, stored[block])
        for g in groups:
            rows = np.ones(len(meta['id']), bool) if g == 'ALL' else meta['cat_id'] == g
            actual = metric(v['truth'][rows], v['proposal'][rows], v['baseline'][rows], mask[rows])
            assert actual == report['metrics'][block][g], (report['score'], block, g)
            checked += 1

summary = {'freeze_sha256': sha(FREEZE), 'items': 610, 'series': 6100,
           'shadow_rows': int(data['shadow']['truth'].size),
           'wape_cap': f['wape_cap'], 'coverage_floor': f['coverage_floor'],
           'three_seed_pooled_shadow': {}, 'primary_seed_pooled': {},
           'primary_contrast': result['primary_direct_minus_composite']}
for name in ['mean_error','relative_error_f','relative_error_mu','excess_f','composite_excess','composite_excess_budget220','direct_excess']:
    records = [r for r in result['records'] if r['score'] == name and r['threshold_mode'] == 'pooled']
    summary['three_seed_pooled_shadow'][name] = {
        k: None if any(r['metrics']['shadow']['ALL'][k] is None for r in records) else float(np.mean([r['metrics']['shadow']['ALL'][k] for r in records]))
        for k in ['coverage','demand_coverage','wape']}
    first = next(r for r in records if r['seed'] == f['primary_seed'])
    summary['primary_seed_pooled'][name] = {k: first[k] for k in ['metrics','thresholds']}
    if 'adjusted_48_endpoint_bounds' in first:
        summary['primary_seed_pooled'][name]['adjusted_bounds'] = first['adjusted_48_endpoint_bounds']
        summary['primary_seed_pooled'][name]['all_adjusted_bounds_pass'] = all(b['operating_bounds_pass'] for v in first['adjusted_48_endpoint_bounds'].values() for b in v.values())
receipt = {'status':'FIXED_GUARDIAN_NUMERICAL_AUDIT_PASS',
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'freeze_sha256':sha(FREEZE),'comparison_sha256':sha(EV/'GUARDIAN_COMPARISON.json'),
    'immutable_files_verified':len(f['immutable_files']), 'fixed_policy_masks_verified':126,
    'group_metric_reports_recomputed':checked,'independent_item_ids_verified':610,
    'evaluation_days_match_prespecified_blocks':True,'fresh_guardian_use_count':1,
    'training_calls':0,'threshold_selection_calls':0,'external_opened':False,'certificate_issued':False,
    'prediction_software':{'python':'3.12.13','numpy':'2.1.3','scipy':'1.16.3','scikit_learn':'1.6.1','joblib':'1.5.3','lightgbm':'4.6.0'},
    'evaluation_python':platform.python_version(),'evaluation_numpy':np.__version__,
    'scope':'Previously held-out items, same calendar and stores; approximate item-bootstrap inference.'}
for path, obj in [(OUT/'GUARDIAN_REPORT_SUMMARY.json',summary),(ROOT/'provenance/GUARDIAN_VERIFICATION_RECEIPT.json',receipt)]:
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
print(json.dumps({'verification':receipt,'pooled':summary['three_seed_pooled_shadow'],'primary_contrast':summary['primary_contrast']},indent=2))
