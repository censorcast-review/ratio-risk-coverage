"""Independent mass/accounting verification of the paired development audit."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def main(out):
 src=R/'evidence/matched_censoring';p=json.loads((src/'PROTOCOL.json').read_text());done=json.loads((src/'COMPLETE.json').read_text());checks=0
 for f,h in done['files'].items():assert sha(src/f)==h,f;checks+=1
 assert p['source_sha256']==sha(R/'code/run_uncensored_audit.py');checks+=1
 z=np.load(src/'TRAINING_ROWS.npz');assert z['point_days'].max()<z['head_days'].min();checks+=1
 yy=None
 for arm in ['censored','complete']:
  frozen=json.loads((src/arm/'FROZEN.json').read_text());res=json.loads((src/arm/'RESULTS.json').read_text());ev=np.load(src/arm/'EVALUATION.npz');cal=np.load(src/arm/'CALIBRATION.npz')
  if yy is not None:assert np.array_equal(yy,ev['y']);checks+=1
  yy=ev['y'];assert len(yy)==18290*114;checks+=1
  for name,h in frozen['models'].items():assert sha(src/arm/name)==h;checks+=1
  for plan,row in zip(frozen['plans'],res['menus'],strict=True):
   assert plan['selected']==row['selected'];checks+=1
   r=plan['cap'];T=frozen['T'];er=ev['e'+str(plan['e_trees'])];mu=ev['w'+str(plan['w_trees'])];L=abs(ev['y']-ev['f']);W=ev['y']
   stats={}
   for method,rule in plan['policies'].items():
    if method=='error':sc=er
    elif method=='weight_descending':sc=-mu
    elif method=='row':sc=er-r*mu
    else:
     d=mu/T if method=='demand' else .25+.75*mu/T
     sc=np.divide(er-r*mu,d,out=np.where(er-r*mu>0,np.inf,np.where(er-r*mu<0,-np.inf,0.)),where=d>0)
    t=rule['threshold'];a=np.ones(len(W),bool) if t=='all' else np.zeros(len(W),bool) if t is None else sc<=t
    v=row['test'][method];vals={'rows':int(a.sum()),'c':float(a.mean()),'d':float(W[a].sum()/W.sum()),'weight':float(W[a].sum()),'loss':float(L[a].sum())}
    units=np.repeat(ev['series_item_index'],len(ev['days']));stats[method]=np.column_stack([np.bincount(units,weights=a),np.bincount(units,weights=a*W)])
    for k,x in vals.items():assert np.isclose(x,v[k],rtol=1e-10,atol=1e-7),(arm,method,k);checks+=1
   c,d=plan['selected']['c'],plan['selected']['d']
   if c is not None and d is not None:
    for key,o in [('dc','c'),('dd','d')]:assert np.isclose(row['contrast'][key],row['test'][d][o]-row['test'][c][o]);checks+=1
   if plan['kind']=='relative' and plan['value']==.95 and c is not None and d is not None:
    ni=len(ev['items']);counts=np.random.default_rng(p['seed']).multinomial(ni,np.full(ni,1/ni),size=2000);total=np.column_stack([np.bincount(units),np.bincount(units,weights=W)]);boot=(counts@(stats[d]-stats[c]))/(counts@total);assert np.allclose(np.quantile(boot,[.025,.975],axis=0).T,row['contrast']['ci95']);checks+=1
  for k,v in res['head_diagnostics'].items():
   target=abs(ev['y']-ev['f']) if k[0]=='e' else ev['y'];assert np.isclose(v['mse'],np.mean((ev[k]-target)**2));checks+=1
 report={'status':'PASS','checks':checks,'fit_calls':done['fit_calls'],'new_external_openings':done['new_external_openings'],'scope':'hashes, paired targets, chronology, independent score masks and mass reconstruction; not population risk certification'}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=R/'reproduction_outputs/matched_censoring_verification.json');x=a.parse_args();main(x.output)
