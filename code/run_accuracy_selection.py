"""Retrospective predictor-specific selectors; no held-out opening.

Use --forecast-package for the unchanged v7 evidence package and --data for its
extracted M5 development inputs. Outputs must be new. All fits use development
truth labels deliberately; this is a mechanism audit, not natural-tail recovery.
"""
from pathlib import Path
import argparse,sys,json,hashlib,gc
from datetime import datetime,timezone
import numpy as np,pandas as pd,lightgbm as lgb,xgboost as xgb

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def metric(y,f,a=None):
 if a is None:a=np.ones(len(y),bool)
 n=int(a.sum());mass=float(y[a].sum());err=float(np.abs(y[a]-f[a]).sum());total=float(y.sum())
 return dict(rows=n,row_coverage=n/len(y),demand_mass=mass,error_mass=err,demand_coverage=mass/total,wape=err/mass if mass>0 else None,mean_forecast=float(f[a].mean()) if n else None,excess=err-R*mass)
def score(e,mu,T,lam):
 den=lam+(1-lam)*mu/T;v=e-R*mu
 return np.divide(v,den,out=np.where(v>0,np.inf,np.where(v<0,-np.inf,0.)).astype(float),where=den>0)
def choose(s,y,f,blocks):
 # Every unique threshold, including complete score ties. Both chronological
 # calibration blocks must meet the cap and row floor.
 finite=np.isfinite(s);ts=np.unique(s[finite]);ok=np.ones(len(ts),bool);front=[]
 for key in [0,1]:
  m=blocks==key;o=np.argsort(s[m],kind='stable');ss=s[m][o];yy=y[m][o];ee=np.abs(yy-f[m][o]);nn=np.searchsorted(ss,ts,side='right');dm=np.r_[0.,np.cumsum(yy)][nn];em=np.r_[0.,np.cumsum(ee)][nn];good=(nn>=FLOOR*m.sum())&(dm>0)&(em<=R*dm+1e-9);ok &= good
  take=np.unique(np.linspace(0,max(0,len(ts)-1),101,dtype=int));front.append(dict(block=key,threshold=ts[take].tolist(),row_coverage=(nn[take]/m.sum()).tolist(),demand_coverage=(dm[take]/yy.sum()).tolist(),excess=(em[take]-R*dm[take]).tolist()))
 return (float(ts[np.flatnonzero(ok)[-1]]) if ok.any() else None),front
R=.85*.7530939208;FLOOR=.35;SEED=20260908;LAMBDAS=[1.,.25,0.]

def main(args):
 out=args.output;out.mkdir(parents=True,exist_ok=False);P=args.forecast_package.resolve();sys.path.insert(0,str(P/'code'))
 from replay_m5 import DesignPanel,FeatureBuilder,HierarchyFeatures
 inputs={'base':P/'evidence/risk_calibration/primary_base/model.ubj','risk':P/'evidence/revision/m5_observable/seed_20260906/hit.txt','stack':P/'evidence/forward_controls/m5/m5_FINAL_STACK.txt','policy':P/'evidence/forward_controls/m5/m5_FROZEN_POLICIES.json','data':args.data/'data/design_outcomes_v0_5.npz'}
 proto={'created_utc':now(),'script_sha256':sha(__file__),'status':'retrospective; all evaluation dates previously consumed','new_holdout_openings':0,'predictors':['group_l1','learned_l1','chronos2_univariate_raw'],'training_targets':[1555,1610],'calibration_a':[1611,1645],'calibration_b':[1646,1673],'evaluation':[1800,1913],'cap':R,'row_floor':FLOOR,'lambdas':LAMBDAS,'seed':SEED,'head_spec':{'objective':'squared_error','trees':220,'leaves':31,'learning_rate':.05,'training_rows':600000,'n_jobs':4},'inputs':{k:sha(v) for k,v in inputs.items()},'target_information':'retained pre-censoring Y for error/demand heads and threshold calibration; forecasts fit observed sales, postprocessors calibrated with retained Y','selection':'largest common feasible threshold across two calibration blocks; all unique scores; no model/hyperparameter selection','bootstrap':'paired item clusters, 4000 draws, percentile descriptive intervals conditional on fits','planned_contrasts':['lambda .25 minus 1','lambda 0 minus 1'],'notes':'Pure-demand lambda=0 has no positive denominator floor; infeasible scores/policies are reported, not replaced.'}
 dump(out/'PROTOCOL.json',proto)
 panel=DesignPanel.load(inputs['data'],args.data/'data/calendar.csv',args.data/'data/sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8);builder=FeatureBuilder(panel);hier=HierarchyFeatures(panel)
 b=xgb.Booster();b.load_model(inputs['base']);b.set_param({'nthread':4});q=lgb.Booster(model_file=str(inputs['risk']));st=lgb.Booster(model_file=str(inputs['stack']));pol=json.loads(inputs['policy'].read_text());sc=np.array([pol['policies']['risk']['category_scales'][str(c)] for c in panel.metadata['cat_id']]);models=proto['predictors']
 def make(days,save):
  n=panel.n_series;nf=len(days);X=np.empty((n,nf,41),np.float32);F={k:np.empty((n,nf),float) for k in models};
  for start in range(0,nf,7):
   ds=days[start:start+7];ss=np.repeat(np.arange(n),len(ds));dd=np.tile(ds,n);common=builder.make_features(ss,dd,include_censor=True);xx=np.column_stack([common,hier.make(ss,dd)]).astype(np.float32);raw=np.maximum(b.predict(xgb.DMatrix(xx))-1,0);qq=np.clip(q.predict(common,num_threads=4),0,1).astype(np.float32);base=raw*np.repeat(sc,len(ds));X[:,start:start+len(ds)]=np.column_stack([xx,qq]).reshape(n,len(ds),41);F['group_l1'][:,start:start+len(ds)]=base.reshape(n,-1);F['learned_l1'][:,start:start+len(ds)]=np.maximum(base+st.predict(np.column_stack([xx,base,qq]).astype(np.float32),num_threads=4),0).reshape(n,-1)
   # Days for the later slice begin at a complete origin; last origin has 2 days.
   origin=int(ds[0]-(1+(ds[0]-1)%7));z=np.load(P/f'evidence/coordinate_tests/foundation_predictions/chronos2_univariate/origin_{origin}.npz');idx=np.searchsorted(z['days'],ds);assert np.array_equal(z['days'][idx],ds);F['chronos2_univariate_raw'][:,start:start+len(ds)]=z['forecast'][:,idx]
  y=panel.truth[:,days-1].astype(float);np.savez_compressed(out/f'{save}_forecasts.npz',days=days,y=y,**F);return X.reshape(-1,41),{k:v.ravel() for k,v in F.items()},y.ravel()
 days=np.arange(1555,1674);X,F,y=make(days,'development');dd=np.tile(days,panel.n_series);fit=dd<=1610;cal=~fit;blocks=(dd[cal]>=1646).astype(int);idx=np.flatnonzero(fit);idx=np.sort(np.random.default_rng(SEED).choice(idx,600000,replace=False));np.savez_compressed(out/'TRAINING_ROWS.npz',indices=idx)
 kwargs=dict(objective='regression',n_estimators=220,num_leaves=31,learning_rate=.05,min_child_samples=50,n_jobs=4,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
 demand=lgb.LGBMRegressor(**kwargs);demand.fit(X[idx],y[idx]);demand.booster_.save_model(str(out/'DEMAND_HEAD.txt'));mu=np.maximum(demand.predict(X[cal]),0);T=float(y[idx].mean());freeze={'created_utc':now(),'cap':R,'T_training':T,'policies':{},'calibration_metrics':{},'model_sha256':{'demand':sha(out/'DEMAND_HEAD.txt')}};heads={};curves={}
 for key in models:
  ef=np.abs(y[idx]-F[key][idx]);h=lgb.LGBMRegressor(**kwargs);h.fit(np.column_stack([X[idx],F[key][idx]]).astype(np.float32),ef);h.booster_.save_model(str(out/f'{key}_ERROR_HEAD.txt'));heads[key]=h;freeze['model_sha256'][key]=sha(out/f'{key}_ERROR_HEAD.txt');e=np.maximum(h.predict(np.column_stack([X[cal],F[key][cal]]).astype(np.float32)),0);freeze['policies'][key]={};freeze['calibration_metrics'][key]={};curves[key]={}
  for lam in LAMBDAS:
   s=score(e,mu,T,lam);t,curve=choose(s,y[cal],F[key][cal],blocks);freeze['policies'][key][str(lam)]=t;curves[key][str(lam)]=curve;freeze['calibration_metrics'][key][str(lam)]={str(j):metric(y[cal][blocks==j],F[key][cal][blocks==j],s[blocks==j]<=t) if t is not None else metric(y[cal][blocks==j],F[key][cal][blocks==j],np.zeros(sum(blocks==j),bool)) for j in [0,1]};print('CALIBRATED',key,lam,t,flush=True)
 freeze['created_utc']=now();dump(out/'FROZEN_SELECTORS.json',freeze);dump(out/'CALIBRATION_FRONTIERS.json',curves);del X,F,y;gc.collect()
 X,F,y=make(np.arange(1800,1914),'evaluation');mu=np.maximum(demand.predict(X),0);ss=np.repeat(np.arange(panel.n_series),114);item_names,ic=np.unique(panel.metadata['item_id'][ss],return_inverse=True);cats=panel.metadata['cat_id'][ss];result={'cap':R,'status':'retrospective','predictors':{},'contrasts':{},'budgets':{},'groups':{}};per={};masks={}
 for key in models:
  f=F[key];e=np.maximum(heads[key].predict(np.column_stack([X,f]).astype(np.float32)),0);result['predictors'][key]={'full':metric(y,f),'policies':{}};masks[key]={};result['groups'][key]={}
  per[key]={}
  for lam in LAMBDAS:
   t=freeze['policies'][key][str(lam)];s=score(e,mu,T,lam);a=s<=t if t is not None else np.zeros(len(y),bool);masks[key][str(lam)]=a;result['predictors'][key]['policies'][str(lam)]=metric(y,f,a);per[key][str(lam)]=np.column_stack([np.bincount(ic,weights=a),np.bincount(ic,weights=y*a),np.bincount(ic,weights=np.abs(y-f)*a)]);result['groups'][key][str(lam)]={str(c):metric(y[cats==c],f[cats==c],a[cats==c]) for c in np.unique(cats)}
  for lam in [.25,0.]:
   a=masks[key]['1.0'];bmask=masks[key][str(lam)];result['budgets'][f'{key}:{lam}']={n:metric(y,f,m) for n,m in [('common',a&bmask),('row_only',a&~bmask),('other_only',~a&bmask),('neither',~a&~bmask)]}
  np.savez_compressed(out/f'{key}_EVAL_SELECTORS.npz',e=e,mu=mu,**{k:a for k,a in masks[key].items()});print('EVALUATED',key,result['predictors'][key],flush=True)
 total=np.column_stack([np.bincount(ic),np.bincount(ic,weights=y)]);np.savez_compressed(out/'ITEM_STATISTICS.npz',items=item_names,total=total,**{f'{k}__{l}':v for k,d in per.items() for l,v in d.items()})
 rng=np.random.default_rng(SEED);boot={f'{k}:{l}':[] for k in models for l in [.25,0.]}
 for _ in range(40):
  w=rng.multinomial(len(total),np.full(len(total),1/len(total)),size=100);den=w@total
  for k in models:
   a=w@per[k]['1.0']
   for l in [.25,0.]:
    bmask=w@per[k][str(l)];boot[f'{k}:{l}'].append(np.column_stack([(bmask[:,0]-a[:,0])/den[:,0],(bmask[:,1]-a[:,1])/den[:,1]]))
 for k,v in boot.items():
  key,l=k.split(':');aa=result['predictors'][key]['policies']['1.0'];bb=result['predictors'][key]['policies'][l];result['contrasts'][k]={'row_delta':bb['row_coverage']-aa['row_coverage'],'demand_delta':bb['demand_coverage']-aa['demand_coverage'],'ci95':np.quantile(np.concatenate(v),[.025,.975],axis=0).T.tolist()}
 dump(out/'RESULTS.json',result);dump(out/'COMPLETE.json',{'completed_utc':now(),'fit_calls':4,'new_holdout_openings':0,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
 print('COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--forecast-package',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
