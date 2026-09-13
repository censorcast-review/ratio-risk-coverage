"""Recompute fixed evidence without fitting, choosing policies, or editing receipts."""
from pathlib import Path
from datetime import datetime
import hashlib,json,re
import numpy as np
import fitz

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'verification';OUT.mkdir(exist_ok=True)
checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(a,b,tol=1e-10):assert np.isclose(a,b,rtol=0,atol=tol),(a,b)

u=ROOT/'evidence/uci/run_001'
r=read(u/'CONFIRMATION_RESULTS.json');f=read(u/'FINAL_FREEZE.json');access=read(u/'FIRST_TEST_ACCESS.json')
assert sha(u/'FINAL_FREEZE.json')==r['freeze_sha256']==access['freeze_sha256']
for key,name in [('protocol_sha256','STUDY_PROTOCOL.json'),('cohort_sha256','COHORT.json'),('development_sha256','DEVELOPMENT_RESULTS.json'),('test_sha256','sealed_targets.npz')]:assert f[key]==sha(u/name)
assert f['code_sha256']==sha(u.parent/'run_study.py')==sha(ROOT/'code/run_study.py')
assert access['prediction_sha256']==sha(u/'FROZEN_PREDICTIONS.npz')
for name,h in f['model_hashes'].items():assert h==sha(u/'models'/name)
times=[datetime.fromisoformat(read(u/n)['created_utc']) for n in ['STUDY_PROTOCOL.json','TRAINING_START.json','FINAL_FREEZE.json','FIRST_TEST_ACCESS.json','CONFIRMATION_RESULTS.json']]
assert times==sorted(times)
assert r['test_evaluations']==1
y=np.load(u/'sealed_targets.npz')['truth'].astype(float);pr=np.load(u/'FROZEN_PREDICTIONS.npz')
assert y.size==298000 and y.shape==(2000,149)
errors=[]
for record,key in zip(r['rows_results'],['baseline','proposal']):
    pred=pr[key];assert pred.shape==y.shape and np.isfinite(pred).all() and (pred>=0).all()
    error=np.abs(y-pred);errors.append(error)
    close(record['wape'],error.sum()/y.sum());close(record['mae'],error.mean());close(record['volume_bias'],(pred-y).sum()/y.sum())
    assert record['row_coverage']==1
close(r['delta_wape_proposal_minus_baseline'],(errors[1]-errors[0]).sum()/y.sum())
close(r['relative_error_reduction'],1-errors[1].sum()/errors[0].sum())
st=np.load(u/'RECOMPUTABLE_TEST_STATISTICS.npz')
np.testing.assert_allclose(st['mass_by_item'],y.sum(axis=1),atol=1e-10)
for key,error in zip(['error_baseline_by_item','error_proposal_by_item'],errors):np.testing.assert_allclose(st[key],error.sum(axis=1),atol=1e-10)
rng=np.random.default_rng(20260907)
def boot(mass,diff):
    vals=[]
    for _ in range(2000):
        idx=rng.integers(0,len(mass),len(mass));vals.append(diff[idx].sum()/mass[idx].sum())
    return np.quantile(vals,[.025,.975])
np.testing.assert_allclose(boot(st['mass_by_item'],st['error_proposal_by_item']-st['error_baseline_by_item']),r['paired_item_bootstrap_ci95'],atol=1e-12)
origins=(pr['days']//7)*7-1
wm=np.array([y[:,origins==o].sum() for o in np.unique(origins)])
wd=np.array([(errors[1]-errors[0])[:,origins==o].sum() for o in np.unique(origins)])
np.testing.assert_allclose(wm,st['mass_by_week'],atol=1e-10);np.testing.assert_allclose(wd,st['error_difference_by_week'],atol=1e-10)
np.testing.assert_allclose(boot(wm,wd),r['calendar_week_sensitivity_ci95'],atol=1e-12)
checks.append('UCI: frozen hashes, chronology, 298000 paired forecasts, WAPE/MAE/bias and both bootstrap intervals')

m=ROOT/'evidence/m5';d=read(m/'FOUNDATION_RESULTS.json');p=read(m/'bundle/FOUNDATION_PROTOCOL.json');rr=read(m/'bundle/PREDICTION_RECEIPTS.json')
assert d==read(m/'bundle/FOUNDATION_RESULTS.json')
assert d['protocol_sha256']==sha(m/'bundle/FOUNDATION_PROTOCOL.json')
assert p['runner_sha256']==sha(ROOT/'code/m5/run_foundation_comparison.py')
assert p['feature_loader_sha256']==sha(ROOT/'code/m5/legacy_features/features.py')
assert len(rr)==156 and len({(z['model'],z['origin']) for z in rr})==156
assert all(z['protocol_sha256']==d['protocol_sha256'] for z in rr)
assert {z['origin'] for z in rr}==set(p['origins'])
assert d['finetuning_calls']==0 and not d['external_opened'] and not d['fresh_guardian_opened']
rows=d['rows']
for model in {z['model'] for z in rows}:
    for block,n in [('calibration_a',2176510),('calibration_b',2194800),('shadow',2085060)]:
        allrow=next(z for z in rows if z['model']==model and z['block']==block and z['category']=='ALL' and 'horizon' not in z)
        assert allrow['n']==n
        for subset in [[z for z in rows if z['model']==model and z['block']==block and z['category']!='ALL'],[z for z in rows if z['model']==model and z['block']==block and 'horizon' in z]]:
            assert sum(z['n'] for z in subset)==n
            close(sum(z['mae']*z['n'] for z in subset)/n,allrow['mae'])
            mass=sum(z['mae']*z['n']/z['wape'] for z in subset)
            close(sum(z['mae']*z['n'] for z in subset)/mass,allrow['wape'])
for c in d['paired_comparisons']:
    b='observed_l1' if c['reference']=='baseline' else 'original_censor_adapter'
    def find(name):return next(z['wape'] for z in rows if z['model']==name and z['block']==c['block'] and z['category']=='ALL' and 'horizon' not in z)
    close(c['delta_wape'],find(c['model'])-find(b))
checks.append('M5: protocol and runner hashes; 156 unique receipts; block/category/horizon aggregate consistency; difference arithmetic (not raw prediction replay)')

s=ROOT/'evidence/selection';ex=read(s/'EXTERNAL_COMPARISON.json');stats=read(s/'EXTERNAL_ITEM_STATISTICS.json');budget=read(s/'EXTERNAL_BUDGET.json')
assert stats['freeze_sha256']==sha(s/'FINAL_PLAN_ORIGINAL.json')
final=read(s/'FINAL_PLAN_ORIGINAL.json');prior=read(s/'PRIOR_PLAN_ORIGINAL.json')
assert final['supersedes_sha256']==sha(s/'PRIOR_PLAN_ORIGINAL.json')
evo=read(s/'PLAN_EVOLUTION_AUDIT.json')
assert datetime.fromisoformat(prior['created_utc'])<datetime.fromisoformat(final['created_utc'])<datetime.fromisoformat(evo['opening_utc'])
assert not evo['external_access_before_replacement'] and not evo['frozen_originals_modified']
primary=next(v for v in stats['records'] if v['period']=='primary_23' and v['group']=='ALL')
for name,components in primary['policy_components'].items():
    z=np.array(components).sum(axis=0);er=ex['primary']['metrics'][name]
    close(z[0]/z[1],er['row_coverage']);close(z[2]/z[4],er['demand_coverage']);close(z[3]/z[2],er['wape'])
for name,addition in [('composed_excess_lambda_1','U'),('mixed_utility_lambda_0_25','V')]:
    k=budget['sets']['K'];v=budget['sets'][addition];er=ex['primary']['metrics'][name]
    close(k['rows']+v['rows'],er['accepted_rows']);close(k['demand_mass']+v['demand_mass'],er['accepted_demand'])
    close(k['absolute_error_mass']+v['absolute_error_mass'],er['accepted_absolute_error'],1e-8)
    close(v['signed_excess']/( -k['signed_excess']),budget['budget']['additions'][addition]['fraction_of_common_slack'])
assert len(ex['secondary'])==24
assert all(z['wape_ucb']>final['wape_cap'] and z['row_coverage_lcb']>.35 for z in ex['secondary'])
checks.append('External selection: pre-opening plan amendment hashes and chronology; primary item statistics; K/U/V budget and all 48 operating endpoints')

aux=(ROOT/'paper/main.aux').read_text();match=re.search(r'\\newlabel\{mainend\}\{\{[^}]*\}\{(\d+)\}',aux)
assert match and int(match.group(1))<=9
log=(ROOT/'paper/main.log').read_text()
assert 'undefined' not in log and 'Overfull' not in log
pdf=fitz.open(ROOT/'paper/main.pdf')
assert len(pdf)>int(match.group(1)) and not pdf.is_repaired
assert all(page.get_text().strip() for page in pdf)
checks.append('PDF: valid unrepaired PDF; text on every page; main text within nine pages; no unresolved references or overfull boxes')
report={'status':'PASS','checks':checks,'main_text_pages':int(match.group(1)), 'fits_or_policy_searches':0,'evidence_files_modified':False}
(OUT/'MANUSCRIPT_VERIFICATION.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
