"""Fixed retrospective controls: calibrator information and chronological selection."""
from pathlib import Path
import sys,json,hashlib,gc,argparse
from datetime import datetime,timezone
import numpy as np,pandas as pd,lightgbm as lgb,xgboost as xgb
from sklearn.metrics import roc_auc_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/freshretail'))
from run_frozen_rc_eval import fit_policy,apply_policy,apply_category_scale,weighted_scale,metrics
from censorcast.data import infer_split_dates,make_supervised,normalize_frame,split_supervised
from censorcast.models import SmoothedTargetEncoder
from replay_m5 import DesignPanel,FeatureBuilder,HierarchyFeatures
SEED=20260910

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def wape(y,f):return float(np.abs(y-f).sum()/y.sum())
def fit_controls(d):
 y=d.y.to_numpy(float);b=d.raw.to_numpy(float);q=d.q.to_numpy(np.float32);c=d.group.astype(str).to_numpy();a=d.available.to_numpy(bool)
 risk=fit_policy(y,b,q,c,a);forecast=fit_policy(y,b,b,c,a)
 base=apply_category_scale(b,c,risk);top=a&(q>=risk['edges'][-1]);scale=max(1.,weighted_scale(y[top],base[top]));merged=json.loads(json.dumps(risk));support=[]
 for cat in risk['category_scales']:
  for j in range(8):
   mask=a&(c==cat)&(np.searchsorted(risk['edges'],q,side='right')==j)&(b>1e-10)
   n=int(d.loc[mask,'series'].nunique());fallback=n<30
   if fallback:merged['scales'][f'{cat}|{j}']=risk['category_scales'][cat]
   support.append(dict(group=cat,bin=j+1,positive_rows=int(mask.sum()),series=n,fallback=fallback))
 return dict(risk=risk,forecast=forecast,merged=merged,top_scale=scale,top_threshold=risk['edges'][-1],support=support)
def apply_controls(d,p):
 b=d.raw.to_numpy(float);q=d.q.to_numpy(np.float32);c=d.group.astype(str).to_numpy();base=apply_category_scale(b,c,p['risk']);f={'base':base,'rc':apply_policy(b,q,c,p['risk']),'forecast':apply_policy(b,b,c,p['forecast']),'merged':apply_policy(b,q,c,p['merged']),'top':base*np.where(q>=p['top_threshold'],p['top_scale'],1.)};return f
def stacking(train,valid,xf,xv,out,tag):
 pp=fit_controls(train);b=apply_controls(train,pp)['base'];vbase=apply_controls(valid,pp)['base'];m=train.available.to_numpy(bool);v=valid.available.to_numpy(bool)
 ids=np.flatnonzero(m);rng=np.random.default_rng(SEED)
 if len(ids)>200000:ids=np.sort(rng.choice(ids,200000,replace=False))
 xx=np.column_stack([xf,b,train.q.to_numpy(float)]).astype(np.float32);vx=np.column_stack([xv,vbase,valid.q.to_numpy(float)]).astype(np.float32)
 y=train.y.to_numpy(float);yv=valid.y.to_numpy(float);grid=[];best={'kind':'parent','wape':wape(yv[v],vbase[v])}
 for depth in [2,4]:
  for trees in [50,150,300]:
   model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=trees,max_depth=depth,num_leaves=2**depth,learning_rate=.03,min_child_samples=50,verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True)
   model.fit(xx[ids],(y-b)[ids]);pred=np.maximum(vbase+model.predict(vx),0);value=wape(yv[v],pred[v]);grid.append(dict(depth=depth,trees=trees,wape=value));print(tag,depth,trees,value,flush=True)
   model.booster_.save_model(str(out/f'{tag}_candidate_{depth}_{trees}.txt'))
   if value<best['wape']:best=dict(kind='residual_gbm',depth=depth,trees=trees,wape=value)
 dump(out/f'{tag}_SELECTION.json',dict(created_utc=now(),grid=grid,selected=best,parent_wape=wape(yv[v],vbase[v]),fit_max_target=str(train.day.max()),selection_min_target=str(valid.day.min()),selection_max_target=str(valid.day.max()),fit_rows=len(ids)))
 assert train.day.max()<valid.day.min()
 return best

def refit_stack(d,x,p,best,out,tag):
 if best['kind']=='parent':return None
 b=apply_controls(d,p)['base'];ids=np.flatnonzero(d.available.to_numpy(bool));rng=np.random.default_rng(SEED)
 if len(ids)>400000:ids=np.sort(rng.choice(ids,400000,replace=False))
 xx=np.column_stack([x,b,d.q.to_numpy(float)]).astype(np.float32)
 model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=best['trees'],max_depth=best['depth'],num_leaves=2**best['depth'],learning_rate=.03,min_child_samples=50,verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True)
 model.fit(xx[ids],(d.y.to_numpy(float)-b)[ids]);model.booster_.save_model(str(out/f'{tag}_FINAL_STACK.txt'));return model

def interval(d,a,b,mask,key):
 t=pd.DataFrame({'cluster':d[key].astype(str),'mass':np.where(mask,d.y,0),'delta':np.where(mask,np.abs(d.y-a)-np.abs(d.y-b),0)}).groupby('cluster',sort=True)[['mass','delta']].sum();v=t.to_numpy();rng=np.random.default_rng(SEED);vals=[]
 for _ in range(40):
  counts=rng.multinomial(len(v),np.full(len(v),1/len(v)),size=100);vals.extend((counts@v[:,1])/np.maximum(counts@v[:,0],1e-12))
 return dict(estimate=float(v[:,1].sum()/v[:,0].sum()),ci95=np.quantile(vals,[.025,.975]).tolist(),clusters=len(v),draws=4000,seed=SEED)
def report(d,pred,units):
 res={}
 for pop,mask in [('eligible',d.available.to_numpy(bool)),('all',np.ones(len(d),bool))]:
  rr={'metrics':{k:metrics(d.y.to_numpy(float),f,mask) for k,f in pred.items()},'contrasts':{}}
  for a,b in [('rc','forecast'),('rc','top'),('rc','stack'),('merged','base'),('merged','rc')]:
   rr['contrasts'][f'{a}_minus_{b}']={u:interval(d,pred[a],pred[b],mask,u) for u in units}
  for k,f in pred.items():
   dd=pd.DataFrame({'series':d.series,'delta':np.where(mask,np.abs(d.y-f)-np.abs(d.y-pred['base']),0)}).groupby('series').delta.sum();rr['metrics'][k]['series_worse']=int((dd>1e-10).sum())
  res[pop]=rr
 return res

def fresh(out):
 ev=ROOT/'evidence/freshretail/frozen_eval';t=pd.read_pickle(ROOT/'inputs/freshretail/freshretail_2000_train_only.pkl.gz');t.attrs={};t=normalize_frame(t);t['_source_split']='train';dates=infer_split_dates(t)
 rows,num,cat=make_supervised(t,horizons=range(1,8),max_train_rows=600000,seed=20260907,development_holdout_days=35);sp=split_supervised(rows,dates);fit=sp['fit'];enc=SmoothedTargetEncoder(categorical=cat,numeric=num,smoothing=30);enc.fit(fit,fit.sale_amount.to_numpy(float),uncensored=fit.is_censored.eq(0).to_numpy())
 base=xgb.Booster();base.load_model(ev/'base_l1.ubj');base.set_param({'nthread':4});risk=xgb.Booster();risk.load_model(ev/'stockout_risk.ubj');risk.set_param({'nthread':4})
 def frame(f):
  x=enc.transform(f);dm=xgb.DMatrix(x);bb=np.maximum(base.predict(dm)-1,0);q=np.clip(risk.predict(dm),0,1);d=pd.DataFrame({'series':f.series_id.astype(str).to_numpy(),'day':f.dt.astype(str).to_numpy(),'group':f.management_group_id.astype(str).to_numpy(),'y':f.sale_amount.to_numpy(float),'raw':bb,'q':q,'available':f.is_censored.eq(0).to_numpy()});return d,x
 ds=[];xs=[]
 for k in ['selection','risk_train','calibration_a','calibration_b','shadow']:
  d,x=frame(sp[k].reset_index(drop=True));d['block']=k;ds.append(d);xs.append(x);print('FRN features',k,len(d),flush=True)
 d=pd.concat(ds,ignore_index=True);x=np.concatenate(xs);m=d.block.isin(['selection','risk_train']).to_numpy();best=stacking(d[m].reset_index(drop=True),d[~m].reset_index(drop=True),x[m],x[~m],out,'fresh');p=fit_controls(d)
 original=json.loads((ev/'FINAL_POLICY.json').read_text());assert max(abs(p['risk']['scales'][k]-v) for k,v in original['scales'].items())<1e-10
 model=refit_stack(d,x,p,best,out,'fresh');dump(out/'fresh_FROZEN_POLICIES.json',dict(created_utc=now(),policies=p,selected_stack=best,feature_names=[*num,*cat,'category_scaled_base','q'],input_scope='train-only before reading already-consumed eval'))
 d.to_csv(out/'fresh_TRAIN_CALIBRATION.csv.gz',index=False);np.savez_compressed(out/'fresh_TRAIN_FEATURES.npz',x=x)
 del rows,sp,fit,x,xs,ds;gc.collect()
 # Already-consumed official file; never called a new confirmation.
 e=normalize_frame(pd.read_parquet(ROOT/'inputs/freshretail/official_eval.parquet'));e['_source_split']='eval';e=e[e.series_id.astype(str).isin(set(t.series_id.astype(str)))];c=pd.concat([t,e],ignore_index=True);r,_,_=make_supervised(c,horizons=range(1,8),max_train_rows=7,seed=20260907,development_holdout_days=35);test=split_supervised(r,infer_split_dates(c))['test'].reset_index(drop=True);td,tx=frame(test)
 td['store']=test.store_id.astype(str).to_numpy();td['product']=test.product_id.astype(str).to_numpy();pred=apply_controls(td,p);pred['stack']=pred['base'] if model is None else np.maximum(pred['base']+model.predict(np.column_stack([tx,pred['base'],td.q]).astype(np.float32)),0)
 saved=pd.read_csv(ev/'EVAL_PREDICTIONS.csv.gz');assert np.array_equal(td.series.to_numpy(),saved.series_id.astype(str).to_numpy());assert np.max(abs(pred['rc']-saved.risk_conditioned.to_numpy()))<2e-12
 for k,f in pred.items():td[k]=f
 td.to_csv(out/'fresh_EVAL_PREDICTIONS.csv.gz',index=False);np.savez_compressed(out/'fresh_EVAL_FEATURES.npz',x=tx)
 result=report(td,pred,['series','store','product','day']);result['risk_metrics']={'auc':float(roc_auc_score(~td.available,td.q)),'brier':float(brier_score_loss(~td.available,td.q)),'event_rate':float((~td.available).mean()),'constant_brier':float((~td.available).mean()*td.available.mean()),'eligible_auc':'undefined: all eligible outcomes have H=0'};result['selected_stack']=best;result['top_scale']=p['top_scale'];result['fallback_cells']=sum(v['fallback'] for v in p['support']);dump(out/'fresh_RESULTS.json',result);print('FRN COMPLETE',result['eligible']['metrics'],flush=True)

def m5(out,inputs):
 panel=DesignPanel.load(inputs/'data/design_outcomes_v0_5.npz',inputs/'data/calendar.csv',inputs/'data/sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8);builder=FeatureBuilder(panel);hier=HierarchyFeatures(panel);cats=panel.metadata['cat_id'];items=panel.metadata['item_id']
 def frame(saved):
  z=np.load(saved);days=z['days'];ss=np.repeat(np.arange(panel.n_series),len(days));dd=np.tile(days,panel.n_series);d=pd.DataFrame({'series':items[ss],'day':dd,'group':cats[ss],'y':panel.truth[ss,dd-1].astype(float),'raw':z['raw'].ravel(),'q':z['risk'].ravel(),'available':True});x=np.column_stack([builder.make_features(ss,dd,include_censor=True),hier.make(ss,dd)]).astype(np.float32);return d,x
 d,x=frame(ROOT/'evidence/cell_audit/m5/point_validation_predictions.npz');m=d.day.to_numpy()<=1373;best=stacking(d[m].reset_index(drop=True),d[~m].reset_index(drop=True),x[m],x[~m],out,'m5');p=fit_controls(d);model=refit_stack(d,x,p,best,out,'m5');dump(out/'m5_FROZEN_POLICIES.json',dict(created_utc=now(),policies=p,selected_stack=best));del x;gc.collect()
 td,tx=frame(ROOT/'evidence/cell_audit/m5/later_predictions.npz');pred=apply_controls(td,p);pred['stack']=pred['base'] if model is None else np.maximum(pred['base']+model.predict(np.column_stack([tx,pred['base'],td.q]).astype(np.float32)),0)
 # Original clipping/tolerance reproduces the archived primary policy.
 assert abs(wape(td.y.to_numpy(),pred['rc'])-.6685093103012937)<1e-10
 for k,f in pred.items():td[k]=f
 # Save item sufficient statistics rather than redundant 2M-row frames.
 rows=[]
 for key,g in td.groupby('series',sort=True):
  a={'series':key,'mass':float(g.y.sum()),'rows':len(g)}
  for k in pred:a[k+'_error']=float(np.abs(g.y-g[k]).sum())
  rows.append(a)
 pd.DataFrame(rows).to_csv(out/'m5_ITEM_STATISTICS.csv',index=False)
 result={'metrics':{k:metrics(td.y.to_numpy(),f,np.ones(len(td),bool)) for k,f in pred.items()},'selected_stack':best,'rc_minus_stack':interval(td,pred['rc'],pred['stack'],np.ones(len(td),bool),'series'),'rc_minus_top':interval(td,pred['rc'],pred['top'],np.ones(len(td),bool),'series'),'top_scale':p['top_scale']};dump(out/'m5_RESULTS.json',result);print('M5 COMPLETE',result,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['fresh','m5']);a.add_argument('--inputs',type=Path);args=a.parse_args();out=ROOT/'evidence/forward_controls'/args.mode;out.mkdir(parents=True,exist_ok=False)
 dump(out/'PROTOCOL.json',{'created_utc':now(),'script_sha256':sha(__file__),'scope':'fixed retrospective controls; no unused data claim','mode':args.mode,'seed':SEED,'bins':8,'shrink':.75,'top':'global single uplift >=1 only above original pooled 7/8 risk quantile','sparse_rule':'fallback to group if fewer than 30 distinct calibration series with positive raw forecasts','stacking':'L1 residual GBM; frozen base/risk+origin-valid history; train earlier calibration block, select later calibration block, refit combined','grid':{'depth':[2,4],'trees':[50,150,300],'learning_rate':.03,'min_child_samples':50},'max_fit_rows':200000,'max_refit_rows':400000,'bootstrap':{'seed':SEED,'draws':4000},'new_holdout_openings':0})
 fresh(out) if args.mode=='fresh' else m5(out,args.inputs)
 dump(out/'COMPLETE.json',{'finished_utc':now(),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
