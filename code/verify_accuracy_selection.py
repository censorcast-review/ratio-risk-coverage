"""Recompute predictor scores, masks, budgets and paired intervals from saved arrays.
Does not fit models, tune policies, or open held-out data. Outputs are separate.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from run_accuracy_selection import score,metric,choose,SEED,R,FLOOR

def main(p,out):
 checks=[]
 def ck(name,ok):
  checks.append({'name':name,'pass':bool(ok)})
  if not ok:raise AssertionError(name)
 def near(name,a,b,tol=1e-9):ck(name,abs(a-b)<tol)
 fr=json.loads((p/'FROZEN_SELECTORS.json').read_text());re=json.loads((p/'RESULTS.json').read_text());done=json.loads((p/'COMPLETE.json').read_text())
 for n,h in done['files'].items():ck('hash:'+n,hashlib.sha256((p/n).read_bytes()).hexdigest()==h)
 z=np.load(p/'evaluation_forecasts.npz');y=z['y'].ravel();it=np.load(p/'ITEM_STATISTICS.npz');total=it['total'];nd=len(z['days']);n=it['items'].shape[0]
 ck('evaluation chronology',z['days'][0]>1673);ck('completed after frozen',done['completed_utc']>fr['created_utc']);ck('four fits only',done['fit_calls']==4);ck('no new opening',done['new_holdout_openings']==0)
 for k,entry in re['predictors'].items():
  f=z[k].ravel();s=np.load(p/f'{k}_EVAL_SELECTORS.npz');near(k+':full metric',np.abs(y-f).sum()/y.sum(),entry['full']['wape'])
  for lam,stats in entry['policies'].items():
   v=score(s['e'],s['mu'],fr['T_training'],float(lam));t=fr['policies'][k][lam];a=v<=t if t is not None else np.zeros(len(y),bool);ck(k+lam+':mask',np.array_equal(a,s[lam]));m=metric(y,f,a)
   for field in ['rows','row_coverage','demand_mass','error_mass','demand_coverage','wape','excess']:
    if m[field] is None:ck(k+lam+field,stats[field] is None)
    else:near(k+lam+':'+field,m[field],stats[field])
   v=it[k+'__'+lam].sum(axis=0);near(k+lam+' item rows',v[0],m['rows']);near(k+lam+' item mass',v[1],m['demand_mass']);near(k+lam+' item errors',v[2],m['error_mass'],1e-7)
  for lam in ['0.25','0.0']:
   a=s['1.0'];b=s[lam];rec=re['budgets'][k+':'+lam]
   for name,mask in [('common',a&b),('row_only',a&~b),('other_only',~a&b),('neither',~a&~b)]:
    m=metric(y,f,mask)
    for field in ['rows','demand_mass','error_mass','excess']:near(k+lam+name+field,m[field],rec[name][field])
   near(k+lam+' row budget identity',rec['common']['excess']+rec['row_only']['excess'],entry['policies']['1.0']['excess'],1e-7)
   near(k+lam+' other budget identity',rec['common']['excess']+rec['other_only']['excess'],entry['policies'][lam]['excess'],1e-7)
 rng=np.random.default_rng(SEED);bs={k:[] for k in re['contrasts']}
 for _ in range(40):
  w=rng.multinomial(n,np.full(n,1/n),size=100);den=w@total
  for k in bs:
   model,lam=k.split(':');diff=w@(it[model+'__'+lam]-it[model+'__1.0']);bs[k].append(diff[:,:2]/den)
 for k,v in bs.items():ck('bootstrap '+k,np.max(abs(np.quantile(np.concatenate(v),[.025,.975],axis=0).T-re['contrasts'][k]['ci95']))<1e-12)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'status':'PASS','checks':checks,'count':len(checks),'inputs_unchanged':True},indent=2)+'\n');print('PASS',len(checks))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('evidence',type=Path);a.add_argument('--output',type=Path,default=Path('reproduction_outputs/accuracy_verification.json'));x=a.parse_args();main(x.evidence,x.output)
