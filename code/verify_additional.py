"""Replay new saved-policy metrics and paired intervals without fitting or selection."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[1];E=R/'evidence/additional';checks=0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def same(x,y):
 global checks
 assert np.allclose(x,y,atol=1e-12,rtol=0),(x,y);checks+=1
p=E/'yeast';f=json.loads((p/'FROZEN.json').read_text());access=json.loads((p/'TEST_ACCESS.json').read_text());start=json.loads((p/'RUN_STARTED.json').read_text())
assert f['created_utc']<access['created_utc'];assert access['frozen_sha256']==sha(p/'FROZEN.json');assert f['classifier_sha256']==sha(p/'CLASSIFIER.joblib');assert start['script_sha256']==sha(E/'run_yeast.py');assert start['protocol_sha256']==sha(p/'PROTOCOL.json')
z=np.load(p/'TEST_PREDICTIONS.npz');L=z['L'];W=z['W'];same(L,(z['h']*(1-z['Y'])).sum(1));same(W,z['h'].sum(1));res=json.loads((p/'RESULTS.json').read_text())
for a,b in zip(f['policies'],res['policies']):
 assert a['threshold']==b['threshold'];t=a['threshold'];s=z['e'].copy()
 if a['method']=='row_excess':s-=a['cap']*W
 if a['method']=='exposure_ratio':s=np.divide(s,W,out=np.zeros(len(s)),where=W>0)
 s[W==0]=-np.inf;mask=s<=t if t is not None else np.zeros(len(W),bool);same(mask,z[f"{a['factor']}_{a['method']}"]);v=b['test'];same([mask.sum(),mask.mean(),W[mask].sum()/W.sum(),L[mask].sum()/W[mask].sum()],[v[k] for k in ['rows','c','d','risk']])
# Same paired resample sequence as the frozen run, independently recomputed.
rng=np.random.default_rng(20260909);idx=rng.integers(0,len(L),(2000,len(L)))
for c in res['contrasts']:
 a=z[f"{c['factor']}_exposure_ratio"].astype(int);b=z[f"{c['factor']}_row_excess"].astype(int);d=a-b
 arr=np.column_stack([d[idx].mean(1),(d*W)[idx].sum(1)/W[idx].sum(1)])
 same(np.quantile(arr,[.025,.975],axis=0).T,c['ci95'])
complete=json.loads((E/'sensitivity/COMPLETE.json').read_text())
for n,h in complete['files'].items():assert sha(E/'sensitivity'/n)==h
assert json.loads((E/'sensitivity/PROTOCOL.json').read_text())['script_sha256']==sha(E/'run_sensitivity.py')
S=json.loads((E/'sensitivity/RESULTS.json').read_text());F=json.loads((R/'evidence/accuracy_selection/FROZEN_SELECTORS.json').read_text());T=F['T_training'];ev=np.load(R/'evidence/accuracy_selection/evaluation_forecasts.npz');Y=ev['y'].ravel()
for model in ['group_l1','learned_l1','chronos2_univariate_raw']:
 z=np.load(R/'evidence/accuracy_selection'/f'{model}_EVAL_SELECTORS.npz');err=np.abs(Y-ev[model].ravel())
 for a in S:
  if a['model']!=model:continue
  s=z['e']-a['cap']*z['mu'];method=a['method']
  if method=='error':s=z['e']
  elif method in ['mixed','demand']:
   den=z['mu']/T if method=='demand' else .25+.75*z['mu']/T
   s=np.divide(s,den,out=np.where(s>0,np.inf,np.where(s<0,-np.inf,0.)),where=den>0)
  mask=s<=a['threshold'] if a['threshold'] is not None else np.zeros(len(Y),bool);mass=Y[mask].sum();v=a['evaluation'];same([mask.sum(),mask.mean(),mass/Y.sum()],[v[k] for k in ['rows','c','d']])
  if mass:same(err[mask].sum()/mass,v['risk'])
  else:assert v['risk'] is None
print(json.dumps({'status':'PASS','checks':checks,'sensitivity_policies':len(S),'yeast_policies':len(res['policies']),'new_fits':0,'new_threshold_searches':0}))
