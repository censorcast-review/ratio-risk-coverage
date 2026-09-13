"""Recompute the common/unique-set budget from consumed development caches.

No training, no threshold selection, and no guardian or external reads.
"""
from pathlib import Path
import json
import numpy as np
from run_selective_study import load, R
from r2_io import dump, sha

ROOT = Path(__file__).resolve().parents[1]
cache = ROOT / 'results/review2/cache_design/shadow_aligned.npz'
pred_file = ROOT / 'results/review2/design_predictions_s20260906.npz'
v, pred = load(cache), load(pred_file)
y = v['truth'].astype(float)
f = v['proposal']
err = np.abs(y-f)
masks = {}
for name, score in [
    ('direct', pred['direct_shadow']),
    ('relative', np.maximum(pred['error_shadow'], 0) / np.maximum(f, .25)),
]:
    report = ROOT / f'results/review_revision/report_{"direct_excess" if name == "direct" else "relative_error_f"}_pooled_s20260906.json'
    t = json.loads(report.read_text())['thresholds']['ALL']
    assert t is not None
    masks[name] = score <= t
a, b = masks['direct'], masks['relative']
sets = {'both': a & b, 'direct_only': a & ~b, 'relative_only': b & ~a, 'neither': ~a & ~b}
old = json.loads((ROOT / 'results/review_revision/SELECTION_MECHANISM.json').read_text())['acceptance_overlap']
rows = {}
for name, mask in sets.items():
    mass, error = float(y[mask].sum()), float(err[mask].sum())
    excess = float((err[mask]-R*y[mask]).sum())
    rows[name] = dict(rows=int(mask.sum()), demand_mass=mass, absolute_error_mass=error,
                      excess=excess, wape=error/mass, demand_coverage=mass/float(y.sum()))
    assert rows[name]['rows'] == old[name]['rows']
    assert np.isclose(excess, old[name]['total_contract_excess'], rtol=0, atol=1e-8)
    assert np.isclose(excess, error-R*mass, rtol=0, atol=1e-8)
slack = -rows['both']['excess']
for name in ['direct', 'relative']:
    added = rows[name+'_only']['excess']
    whole = float((err[masks[name]]-R*y[masks[name]]).sum())
    assert np.isclose(whole, added-slack, rtol=0, atol=1e-8)
    rows[name+'_policy'] = dict(excess=whole, fraction_of_common_slack_used=added/slack)
result = dict(status='BUDGET_ACCOUNTING_VERIFIED', scope='consumed_development_first_seed_original_thresholds',
              cap=R, input_hashes={str(p.relative_to(ROOT)): sha(p) for p in [cache, pred_file]},
              common_slack=slack, sets=rows, new_fits=0, guardian_read=False, external_read=False)
dump(ROOT / 'results/review3/BUDGET_ACCOUNTING.json', result)
print(json.dumps(result, indent=2))
