"""Read-only replay of frozen non-retail masks, budgets and paired intervals."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[1];E=R/'evidence/nonretail';checks=0;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):
 global checks
 assert np.allclose(a,b,rtol=0,atol=1e-10),(a,b);checks+=1
for name in ['bike','delicious']:
 P=E/name;f=json.loads((P/'FROZEN.json').read_text());access=json.loads((P/'TEST_ACCESS.json').read_text());start=json.loads((P/'START.json').read_text());res=json.loads((P/'RESULTS.json').read_text());assert start['script_sha256']==sha(E/'run.py');assert start['protocol_sha256']==sha(E/'PROTOCOL.json');assert access['freeze_sha256']==sha(P/'FROZEN.json');assert f['model_sha256']==sha(P/'MODELS.joblib');assert start['utc']<f['utc']<access['utc']<res['completed_utc']
 variants=sorted({x['variant'] for x in res['policies']});rng=np.random.default_rng(20260910);samples=rng.integers(0,res['n_units'],(4000,res['n_units']))
 for v in variants:
  z=np.load(P/f'TEST_{v}.npz');L,W,e,w,units=[z[k] for k in ['L','W','e','w','units']];N=np.bincount(units);D=np.bincount(units,weights=W)
  for p in res['policies']:
   if p['variant']!=v:continue
   s=e-p['cap']*w;method=p['method']
   if method=='error':s=e.copy()
   if method=='weight_descending':s=-w
   if method in ['mixed','demand']:
    den=.25+.75*w/p['T'] if method=='mixed' else w;s=np.divide(s,den,out=np.where(s>0,np.inf,np.where(s<0,-np.inf,0.)),where=den>0)
   if name=='delicious':s[w==0]=-np.inf
   t=p['threshold'];a=np.ones(len(W),bool) if t=='all' else s<=t if t is not None else np.zeros(len(W),bool);equal(a,z[f"{p['factor']}_{method}"]);q=p['test'];equal([a.sum(),a.mean(),W[a].sum()/W.sum(),L[a].sum(),W[a].sum()],[q[k] for k in ['rows','c','d','loss','weight']]);assert q['risk'] is None if W[a].sum()==0 else abs(L[a].sum()/W[a].sum()-q['risk'])<1e-12
  for p in res['contrasts']:
   if p['variant']!=v:continue
   delta=z[f"{p['factor']}_demand"].astype(int)-z[f"{p['factor']}_row"].astype(int);dn=np.bincount(units,weights=delta);dd=np.bincount(units,weights=delta*W);arr=np.column_stack([dn[samples].sum(1)/N[samples].sum(1),dd[samples].sum(1)/D[samples].sum(1)])
   equal(np.quantile(arr,[.025,.975],axis=0).T,p['ci95']);equal(np.quantile(arr,[.00625,.99375],axis=0).T,p['ci_family98_75'])
b=json.loads((E/'BIKE_BUDGET.json').read_text());z=np.load(E/'bike/TEST_default.npz');r=b['cap'];a=z['0.85_row'];d=z['0.85_demand'];L=z['L'];W=z['W']
for n,m in [('common',a&d),('row_only',a&~d),('demand_only',d&~a),('union',a|d)]:
 q=b['budget'][n];equal([m.sum(),L[m].sum(),W[m].sum(),(L[m]-r*W[m]).sum()],[q[k] for k in ['rows','loss','weight','excess']])
assert b['budget']['union']['risk']>r
assert L[a].sum()/W[a].sum()<r and L[d].sum()/W[d].sum()<r
print(json.dumps({'status':'PASS','checks':checks,'new_fits':0,'threshold_searches':0,'bike_union_infeasible':True}))
