"""Matched history intervention and factorial head-capacity audit.
Retrospective development only. Protocol/choices precede new evaluation.
"""
from pathlib import Path
import sys,json,hashlib,argparse,datetime,gc
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'forecast_archive/code'))
from replay_m5 import DesignPanel,FeatureBuilder,HierarchyFeatures
from policy_audit import METHODS,scores,mask,metrics,largest_threshold,choose,sufficient_statistics,bootstrap_counts
SEED=20260910;BOOTSTRAP_SEED=20260910;CAP=.64012983268

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def fit(X,y,trees,obj):
 m=lgb.LGBMRegressor(objective=obj,alpha=.5,n_estimators=trees,num_leaves=31,learning_rate=.05,min_child_samples=50,n_jobs=4,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
 m.fit(X,y);return m

def main(data,out):
 out.mkdir(parents=True,exist_ok=False)
 proto=dict(utc=now(),runtime={'lightgbm':lgb.__version__,'numpy':np.__version__},scope='retrospective paired development audit; no external opening; all outcomes previously evaluated',seed=SEED,bootstrap_seed=BOOTSTRAP_SEED,master_protocol_sha256=sha(ROOT/'evidence/revision_v12_seeds/PROTOCOL.json'),arms=['censored','complete'],point_days=[1314,1554],head_days=[1555,1610],calibration=[[1611,1645],[1646,1673]],evaluation=[1800,1913],sample_rows=600000,point_trees=300,head_trees=[220,660],head_objective='squared error for both moments',point_objective='median',absolute_caps=[.60,.62,CAP,.66,.68,.70],relative_caps=[.85,.90,.95,1.,1.05],relative_reference='max full-coverage calibration A/B risk per arm',primary='relative factor .95',floor=.35,features='28 common + six item aggregates; no capacity, fill, censor or hit features; prices taken at origin week',contrast='exposure-choice minus case-choice; 2x2 error/mean head sizes at primary relative cap',inference='2000 paired item-bootstrap; 95% descriptive, conditional on fits; no joint risk guarantee',source_sha256=sha(__file__),inputs={p.name:sha(p) for p in sorted((data/'data').iterdir())})
 dump(out/'PROTOCOL.json',proto)
 panel=DesignPanel.load(data/'data/design_outcomes_v0_5.npz',data/'data/calendar.csv',data/'data/sell_prices.csv');original=panel.observed;originalc=panel.censored
 n=panel.n_series;items,inv=np.unique(panel.metadata['item_id'],return_inverse=True)
 rng=np.random.default_rng(SEED)
 sample={}
 for stage,(lo,hi) in [('point',(1314,1554)),('head',(1555,1610))]:
  nd=hi-lo+1;ix=np.sort(rng.choice(n*nd,600000,replace=False));sample[stage]=(ix//nd,lo+ix%nd)
 np.savez_compressed(out/'TRAINING_ROWS.npz',**{s+'_'+v:a for s,t in sample.items() for v,a in zip(['series','days'],t)})
 allres={}; frozen={}; models={}
 for arm in proto['arms']:
  dest=out/arm;dest.mkdir();panel.observed=original if arm=='censored' else panel.truth;panel.censored=originalc if arm=='censored' else np.zeros_like(originalc)
  builder=FeatureBuilder(panel);hier=HierarchyFeatures(panel)
  def features(ss,dd):
   xx=builder.make_features(ss,dd,include_censor=False)
   origin=dd-FeatureBuilder.horizon_for_day(dd);week=panel.day_to_week_index[origin-1].astype(int)
   price=panel.price_matrix[ss,week];xx[:,13]=price
   xx[:,14]=price/np.maximum(panel.price_matrix[ss,np.maximum(week-1,0)],1e-3)-1
   xx[:,15]=price/np.maximum(panel.price_matrix[ss,np.maximum(week-4,0)],1e-3)-1
   return np.column_stack([xx,hier.make(ss,dd)[:,:6]]).astype(np.float32)
  ss,dd=sample['point'];X=features(ss,dd);target=panel.observed[ss,dd-1];m=fit(X,target,300,'quantile');m.booster_.save_model(str(dest/'POINT.txt'));print('POINT FIT',arm,flush=True);del X
  ss,dd=sample['head'];X=features(ss,dd);y=panel.truth[ss,dd-1].astype(float);f=np.maximum(m.predict(X),0);T=float(y.mean());heads={}
  for typ,tar in [('e',abs(y-f)),('w',y)]:
   hx=np.column_stack([X,f]) if typ=='e' else X
   for nt in [220,660]:
    key=typ+str(nt);heads[key]=fit(hx,tar,nt,'regression');heads[key].booster_.save_model(str(dest/(key+'.txt')));print('HEAD FIT',arm,key,flush=True)
  del X,hx,y,f;gc.collect()
  def predict(days):
   ns=len(days);ans={k:np.empty((n,ns),np.float32) for k in ['f','e220','e660','w220','w660']}
   for start in range(0,ns,7):
    ds=days[start:start+7];ss=np.repeat(np.arange(n),len(ds));dd=np.tile(ds,n);xx=features(ss,dd);ff=np.maximum(m.predict(xx),0);ans['f'][:,start:start+len(ds)]=ff.reshape(n,-1);ex=np.column_stack([xx,ff])
    for key,h in heads.items():ans[key][:,start:start+len(ds)]=np.maximum(h.predict(ex if key[0]=='e' else xx),0).reshape(n,-1)
   return {k:v.ravel().astype(float) for k,v in ans.items()}
  caldays=np.arange(1611,1674);cal=predict(caldays);cal['y']=panel.truth[:,caldays-1].ravel().astype(float);blocks=np.tile((caldays>=1646).astype(int),n);L=abs(cal['y']-cal['f']);W=cal['y'];full=max(L[blocks==i].sum()/W[blocks==i].sum() for i in [0,1]);cal['blocks']=blocks
  np.savez_compressed(dest/'CALIBRATION.npz',**cal)
  plans=[]
  for kind,values in [('absolute',proto['absolute_caps']),('relative',proto['relative_caps'])]:
   for val in values:
    r=val if kind=='absolute' else val*full
    pairs=[(220,220),(660,220),(220,660),(660,660)] if kind=='relative' and val==.95 else [(220,220)]
    for et,wt in pairs:
     menu={}
     for method in METHODS:
      t,mm=largest_threshold(scores(cal['e'+str(et)],cal['w'+str(wt)],r,method,T),L,W,blocks,r,.35);menu[method]={'threshold':t,'calibration':mm}
     plans.append(dict(kind=kind,value=val,cap=r,e_trees=et,w_trees=wt,policies=menu,selected={o:choose(menu,o) for o in ['c','d']}))
  frozen[arm]=dict(utc=now(),T=T,full_calibration_reference=float(full),plans=plans,models={p.name:sha(p) for p in dest.glob('*.txt')})
  dump(dest/'FROZEN.json',frozen[arm]);print('FROZEN',arm,flush=True)
  # All newly designed choices are fixed before this arm's new evaluation.
  days=np.arange(1800,1914);ev=predict(days);ev['y']=panel.truth[:,days-1].ravel().astype(float);np.savez_compressed(dest/'EVALUATION.npz',**ev,days=days,items=items,series_item_index=inv)
  L=abs(ev['y']-ev['f']);W=ev['y'];units=np.repeat(inv,len(days));total=np.column_stack([np.bincount(units),np.bincount(units,weights=W)]);counts=bootstrap_counts(len(items),BOOTSTRAP_SEED,2000);den=counts@total;rows=[]
  for plan in plans:
   menu={};stats={}
   for method,p in plan['policies'].items():
    aa=mask(scores(ev['e'+str(plan['e_trees'])],ev['w'+str(plan['w_trees'])],plan['cap'],method,T),p['threshold']);menu[method]=metrics(L,W,aa);stats[method]=sufficient_statistics(L,W,aa,units,len(items))
   c,d=plan['selected']['c'],plan['selected']['d'];con=None
   if c is not None and d is not None:
    b=counts@(stats[d]-stats[c]);vals=b[:,:2]/den;con=dict(dc=menu[d]['c']-menu[c]['c'],dd=menu[d]['d']-menu[c]['d'],ci95=np.quantile(vals,[.025,.975],axis=0).T.tolist())
   rows.append({**plan,'test':menu,'contrast':con})
  diag={}
  for key in ['e220','e660','w220','w660']:
   target=L if key[0]=='e' else W;diag[key]=dict(mse=float(np.mean((ev[key]-target)**2)),mae=float(np.mean(abs(ev[key]-target))),relative_mass_bias=float(ev[key].sum()/target.sum()-1))
  allres[arm]=dict(full=metrics(L,W,np.ones(len(W),bool)),head_diagnostics=diag,menus=rows)
  dump(dest/'RESULTS.json',allres[arm]);print('RESULT',arm,json.dumps(allres[arm]['full']),flush=True)
  for row in rows:
   if row['kind']=='relative' and row['value']==.95:print('PRIMARY',arm,row['e_trees'],row['w_trees'],row['selected'],row['contrast'],flush=True)
  del ev,cal,builder,hier,heads,m,L,W,counts;gc.collect()
 dump(out/'RESULTS.json',allres);dump(out/'COMPLETE.json',dict(utc=now(),fit_calls=10,new_external_openings=0,source_sha256=sha(__file__),files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--seed',type=int,required=True);x=a.parse_args()
 frozen=json.loads((ROOT/'evidence/revision_v12_seeds/PROTOCOL.json').read_text())
 assert x.seed in frozen['additional_seeds']
 for rel,h in frozen['code_sha256'].items(): assert sha(ROOT/rel)==h,rel
 for name,h in frozen['input_sha256'].items(): assert sha(x.data/'data'/name)==h,name
 SEED=x.seed
 main(x.data,x.output)
