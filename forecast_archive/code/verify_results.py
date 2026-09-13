"""Recompute the completed, fixed contrast; never fit or select a policy."""
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
import numpy as np

root=Path(sys.argv[1])
def read(name):return json.loads((root/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
result=read('CONFIRMATION_RESULTS.json');freeze=read('FINAL_FREEZE.json');access=read('FIRST_TEST_ACCESS.json')
assert result['status']=='COMPLETED'
assert result['freeze_sha256']==sha(root/'FINAL_FREEZE.json')==access['freeze_sha256']
assert access['prediction_sha256']==sha(root/'FROZEN_PREDICTIONS.npz')
assert freeze['test_sha256']==sha(root/'sealed_targets.npz')
assert freeze['code_sha256']==sha(root.parent/'run_study.py')
assert freeze['protocol_sha256']==sha(root/'STUDY_PROTOCOL.json')
for name,h in freeze['model_hashes'].items():assert sha(root/'models'/name)==h
times=[read(x)['created_utc'] for x in ['STUDY_PROTOCOL.json','TRAINING_START.json','FINAL_FREEZE.json','FIRST_TEST_ACCESS.json','CONFIRMATION_RESULTS.json']]
assert sorted(map(datetime.fromisoformat,times))==list(map(datetime.fromisoformat,times))
y=np.load(root/'sealed_targets.npz')['truth'].astype(float)
pred=np.load(root/'FROZEN_PREDICTIONS.npz');b=pred['baseline'];p=pred['proposal']
assert y.shape==b.shape==p.shape
assert np.isfinite(b).all() and np.isfinite(p).all() and (b>=0).all() and (p>=0).all()
for row,f in zip(result['rows_results'],[b,p]):
    assert abs(row['wape']-np.abs(y-f).sum()/y.sum())<1e-12
    assert abs(row['mae']-np.abs(y-f).mean())<1e-12
assert result['rows']==y.size
assert abs(result['relative_error_reduction']-(1-np.abs(y-p).sum()/np.abs(y-b).sum()))<1e-12
stats=np.load(root/'RECOMPUTABLE_TEST_STATISTICS.npz');diff=stats['error_proposal_by_item']-stats['error_baseline_by_item'];mass=stats['mass_by_item']
assert abs(diff.sum()/mass.sum()-result['delta_wape_proposal_minus_baseline'])<1e-12
rng=np.random.default_rng(20260907);draw=[]
for _ in range(2000):
    idx=rng.integers(0,len(mass),len(mass));draw.append(diff[idx].sum()/mass[idx].sum())
np.testing.assert_allclose(np.quantile(draw,[.025,.975]),result['paired_item_bootstrap_ci95'],atol=1e-12,rtol=0)
audit={'status':'PASS','scope':'post-test recomputation of exactly the frozen predictions; no fit or policy selection',
       'checks':['receipt hashes','model/code hashes','timestamp order','full coverage','WAPE/MAE','relative difference','paired item interval'],
       'no_independent_claim_about_all_prior_user_data_access':True}
audit_path=root/'RECOMPUTATION_AUDIT.json'
if audit_path.exists():
    assert json.loads(audit_path.read_text())==audit
else:
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
