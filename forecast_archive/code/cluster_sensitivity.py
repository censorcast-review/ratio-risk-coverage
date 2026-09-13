"""Post-hoc cluster sensitivity on existing predictions; no model/policy changes."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--package', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.package
out = args.output
out.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(root/'evidence/freshretail/frozen_eval/EVAL_PREDICTIONS.csv.gz')
data['store'] = data.series_id.str.split('::').str[0]
data['product'] = data.series_id.str.split('::').str[1]
keep = data.is_censored.eq(0)
data['mass'] = np.where(keep, data.sale_amount, 0)
data['error_change'] = np.where(keep, np.abs(data.risk_conditioned-data.sale_amount)-np.abs(data.base-data.sale_amount), 0)
results = {}
for key in ['store', 'product', 'dt']:
    sums = data.groupby(key, sort=True)[['mass', 'error_change']].sum()
    n = len(sums)
    weights = np.random.default_rng(20260908).multinomial(n, np.full(n,1/n), size=4000)
    sample = (weights@sums.error_change)/(weights@sums.mass)
    results[key] = dict(clusters=n, draws=4000, seed=20260908,
                        estimate=float(sums.error_change.sum()/sums.mass.sum()),
                        ci95=np.quantile(sample,[.025,.975]).tolist(),
                        share_negative=float(np.mean(sample<0)))
result = dict(status='COMPLETED_POST_HOC_SENSITIVITY',
              interpretation='Alternative one-way cluster resampling assumptions; not replacement for prespecified series CI. Day results use seven clusters and are especially unstable.',
              eligible_rule='is_censored == 0', new_model_fits=0, new_policy_selections=0,
              secondary_cluster_sensitivity=results)
(out/'CLUSTER_SENSITIVITY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
