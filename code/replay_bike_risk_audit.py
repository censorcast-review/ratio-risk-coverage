"""Post-hoc review audit of already-published predictions; no fitting or threshold search."""
from pathlib import Path
import json, hashlib
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('reproduction_outputs/bike_risk_review.json'))
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / 'evidence/nonretail/bike'
data = np.load(root / 'TEST_default.npz')
results = json.loads((root / 'RESULTS.json').read_text())
cap = next(p['cap'] for p in results['policies'] if p['factor'] == .85)
loss, weight, units = data['L'], data['W'], data['units']
row, demand = data['0.85_row'], data['0.85_demand']
groups = int(units.max()) + 1
sample = np.random.default_rng(20260910).integers(0, groups, (4000, groups))
out = {'scope': 'post-hoc descriptive audit, conditional on published fit and fixed cap',
       'source_commit': 'dd8ed9a5230d39148731b5a684b7bad0c6b58ff5',
       'source_sha256': hashlib.sha256((root / 'TEST_default.npz').read_bytes()).hexdigest(),
       'seed': 20260910, 'replicates': 4000, 'clusters': groups, 'cap': cap,
       'risk_intervals': {}, 'new_fits': 0, 'threshold_searches': 0}
for name, mask in {'row': row, 'demand': demand, 'union': row | demand}.items():
    numerator = np.bincount(units, weights=loss * mask, minlength=groups)
    denominator = np.bincount(units, weights=weight * mask, minlength=groups)
    boot = numerator[sample].sum(1) / denominator[sample].sum(1)
    out['risk_intervals'][name] = {
        'point': float(numerator.sum() / denominator.sum()),
        'ci95': np.quantile(boot, [.025, .975]).tolist(),
        'ci_bonferroni_three': np.quantile(boot, [.05/6, 1-.05/6]).tolist(),
        'excess_total': float(numerator.sum() - cap * denominator.sum())}
# Verify already-reported coverage differences using exactly the original resampling draws.
delta = demand.astype(int) - row.astype(int)
dn = np.bincount(units, weights=delta, minlength=groups)
dd = np.bincount(units, weights=delta * weight, minlength=groups)
nn = np.bincount(units, minlength=groups)
ww = np.bincount(units, weights=weight, minlength=groups)
values = np.column_stack([dn[sample].sum(1)/nn[sample].sum(1), dd[sample].sum(1)/ww[sample].sum(1)])
ci = np.quantile(values, [.00625, .99375], axis=0).T
saved = next(x for x in results['contrasts'] if x['factor'] == .85)
assert np.allclose(ci, saved['ci_family98_75'], atol=1e-12, rtol=0)
out['coverage_interval_replay'] = 'PASS'
out['coverage_ci98_75'] = ci.tolist()
out['limitations'] = 'Day-resampling ignores dependence across days and fitting uncertainty; intervals are approximate and this is not a new confirmatory test.'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
