"""Independent score reconstruction and small LP checks; no model fitting."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from replay_m5 import apply_saved_policy,load_json,sha256,write_json
from run_coordinate_tests import wquant,isotonic_fit,isotonic_apply
root=Path(__file__).resolve().parents[1];out=root/'evidence/coordinate_tests/analysis';d=load_json(out/'RESULTS.json');checks=[]
def check(name,a,b,tol=2e-10):
 err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert err<=tol,(name,err);checks.append(dict(name=name,max_abs_error=err))
data=np.load(sys.argv[1],allow_pickle=True);cats=data['cat_id'];v=np.load(root/'evidence/cell_audit/m5/point_validation_predictions.npz');t=np.load(root/'evidence/cell_audit/m5/later_predictions.npz');y=data['truth'][:,t['days']-1].astype(float)
for name in ['original','risk_input_only','both_inputs']:
 pol=load_json(out/('ORIGINAL_POLICIES.json' if name=='original' else f'{name}_POLICIES.json'))
 new=np.load(out/'NO_CAPACITY_PREDICTIONS.npz');b=t['raw'] if name!='both_inputs' else new['later_base'];q=t['risk'] if name=='original' else new['later_q']
 ff={'base':b*np.array([pol['risk']['category_scales'][str(c)] for c in cats])[:,None]}
 for k,p in pol.items():ff[k]=isotonic_apply(b,cats,p) if k=='isotonic32' else apply_saved_policy(b,b if k=='forecast' else q,cats,p)
 for k,f in ff.items():
  delta=y-f;val=[np.abs(delta).sum()/y.sum(),np.abs(delta).mean(),np.sqrt(np.square(delta).mean()),-delta.sum()/y.sum()]
  check(f'{name}/{k}',val,[d[name]['metrics'][k][s] for s in ['wape','mae','rmse','bias']])
 check(name+'/risk_minus_forecast',(np.abs(y-ff['risk'])-np.abs(y-ff['forecast'])).sum()/y.sum(),d[name]['risk_minus_forecast']['estimate'])
 del ff
receipts=load_json(root/'evidence/m5/bundle/PREDICTION_RECEIPTS.json');rh={(r['model'],r['origin']):r['sha256'] for r in receipts};paths=list((root/'evidence/coordinate_tests/foundation_predictions').glob('*/*.npz'))
assert len(paths)==102
for p in paths: assert sha256(p)==rh[p.parent.name,int(p.stem.split('_')[-1])]
checks.append(dict(name='102 original foundation forecast hashes',max_abs_error=0))
for m in d['foundation']:
 byday={}
 for p in (root/'evidence/coordinate_tests/foundation_predictions'/m).glob('*.npz'):
  z=np.load(p)
  for j,day in enumerate(z['days']):byday[int(day)]=z['forecast'][:,j]
 b=np.column_stack([byday[int(day)] for day in t['days']]);pp=load_json(out/f'{m}_POLICIES.json')
 for k in ['base','risk','forecast']:
  f=b*np.array([pp['risk']['category_scales'][str(c)] for c in cats])[:,None] if k=='base' else apply_saved_policy(b,t['risk'] if k=='risk' else b,cats,pp[k])
  check(m+'/'+k,np.abs(y-f).sum()/y.sum(),d['foundation'][m]['metrics'][k]['wape'])
# Tiny independent convex programs validate weighted quantiles and binned L1 isotonic fitting.
rng=np.random.default_rng(123)
for i in range(12):
 n=24;b=rng.uniform(.05,3,n);yy=rng.poisson(2,n).astype(float)
 for tau in [.5,.8,.9]:
  # y - b*s = u-v; minimize tau*u + (1-tau)*v.
  eq=np.column_stack([b,np.eye(n),-np.eye(n)])
  lp=linprog(np.r_[0,np.full(n,tau),np.full(n,1-tau)],A_eq=eq,b_eq=yy,bounds=[(0,None)]*(1+2*n),method='highs');assert lp.success
  rr=yy-wquant(yy,b,tau)*b;check(f'quantile_lp/{i}/{tau}',np.maximum(tau*rr,(tau-1)*rr).sum(),lp.fun,1e-8)
 xx=np.linspace(0,3,n)[None,:];Y=yy[None,:];cc=np.array(['g']);pol=isotonic_fit(Y,xx,cc,nb=5);f=isotonic_apply(xx,cc,pol)
 edges=pol['g']['edges'];jj=np.searchsorted(edges,xx.ravel(),side='right');k=len(edges)+1;eq=np.zeros((n,k+2*n));eq[np.arange(n),jj]=1;eq[:,k:k+n]=np.eye(n);eq[:,k+n:]=-np.eye(n);mono=np.zeros((k-1,k+2*n));mono[np.arange(k-1),np.arange(k-1)]=1;mono[np.arange(k-1),np.arange(1,k)]=-1
 lp=linprog(np.r_[np.zeros(k),np.ones(2*n)],A_ub=mono,b_ub=np.zeros(k-1),A_eq=eq,b_eq=yy,bounds=[(0,None)]*(k+2*n),method='highs');assert lp.success
 check(f'isotonic_lp/{i}',np.abs(Y-f).sum(),lp.fun,1e-8)
write_json(root/'verification/COORDINATE_TESTS.json',dict(status='PASS',checks=checks,scope='fixed score reconstructions, original forecast hashes and independent small linear programs; not a generalization guarantee'))
print('PASS',len(checks))
