"""Primary-seed occurrence/capacity-hit dual-risk calibration experiment."""
from pathlib import Path
import argparse,sys
import numpy as np
import lightgbm as lgb
import xgboost as xgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_aft_pilot import now,sha,dump,weighted_scale,scores
from run_hierarchy_pilot import HierarchyFeatures,predict
from evaluate_risk_calibration import predict_hit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'accuracy_revision_20260907'/'code'/'m5'))
from legacy_features.features import DesignPanel,FeatureBuilder

PAIRS=[(1,1),(2,1),(4,1),(8,1),(1,2),(1,4),(1,8),(2,2),(2,4),(4,2),(4,4)]
SHRINK=[0.,.25,.5,.75,1.]
PARAMS=dict(n_estimators=300,num_leaves=63,learning_rate=.05,min_child_samples=100,reg_lambda=2.,n_jobs=8,verbosity=-1,deterministic=True,force_col_wise=True)
def edges(x,n): return np.unique(np.quantile(x.ravel(),np.linspace(0,1,n+1)[1:-1])).tolist()
def fit_policy(y,b,occ,hit,cats,no,nh,sh,eo=None,eh=None):
 eo=edges(occ,no) if eo is None else eo;eh=edges(hit,nh) if eh is None else eh;go=np.searchsorted(eo,occ,side='right');gh=np.searchsorted(eh,hit,side='right');sc={};cs={}
 for c in np.unique(cats):
  mask=cats==c;cs[str(c)]=weighted_scale(y[mask],b[mask])
  for i in range(len(eo)+1):
   for j in range(len(eh)+1):
    z=mask[:,None]&(go==i)&(gh==j);gs=weighted_scale(y[z],b[z]) if np.any(b[z]>0) else cs[str(c)];sc[f'{c}|{i}|{j}']=float(np.exp((1-sh)*np.log(max(cs[str(c)],1e-8))+sh*np.log(max(gs,1e-8))))
 return {'occ_bins':no,'hit_bins':nh,'shrink':sh,'occ_edges':eo,'hit_edges':eh,'scales':sc,'category_scales':cs}
def apply(b,occ,hit,cats,p):
 go=np.searchsorted(p['occ_edges'],occ,side='right');gh=np.searchsorted(p['hit_edges'],hit,side='right');mult=np.empty_like(b,float)
 for k,c in enumerate(cats):
  for i in range(len(p['occ_edges'])+1):
   for j in range(len(p['hit_edges'])+1):
    z=(go[k]==i)&(gh[k]==j);mult[k,z]=p['scales'][f'{c}|{i}|{j}']
 return b*mult
def run(data,cache,base_path,hit_path,out):
 out.mkdir(parents=True,exist_ok=False);dump(out/'PROTOCOL.json',{'status':'FROZEN_BEFORE_OCCURRENCE_FIT','created_utc':now(),'seed':20260906,'pairs':PAIRS,'shrinkage':SHRINK,
 'fit_days':[365,1313],'fit_rows':1200000,'calibration_fit_days':[1314,1373],'family_selection_days':[1374,1433],
 'information':'occurrence label S>0 and hit label S>=C are observable; Y only calibrates/selects/scores','blocks':'previously consumed retrospective','script_sha256':sha(__file__)})
 p=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv');p.censored=(p.observed>=p.capacity).astype(np.uint8);b=FeatureBuilder(p);h=HierarchyFeatures(p);cats=p.metadata['cat_id']
 s,d=b.sample_pairs(365,1313,1200000,20260906);x=b.make_features(s,d,include_censor=True);target=(p.observed[s,d-1]>0).astype(int);occ=lgb.LGBMClassifier(objective='binary',random_state=20260906,**PARAMS).fit(x,target).booster_;occ.save_model(str(out/'occurrence.txt'));del x,target
 base=xgb.Booster();base.load_model(base_path);hm=lgb.Booster(model_file=str(hit_path));vd=np.arange(1314,1434,dtype=np.int32);vd=vd[vd-b.horizon_for_day(vd)>=1313];y=p.truth[:,vd-1].astype(float);f=predict(b,h,base,vd);o=predict_hit(b,occ,vd);r=predict_hit(b,hm,vd);cal=vd<=1373;sel=vd>=1374;recs=[]
 for no,nh in PAIRS:
  for sh in SHRINK:
   pol=fit_policy(y[:,cal],f[:,cal],o[:,cal],r[:,cal],cats,no,nh,sh);recs.append({'occ_bins':no,'hit_bins':nh,'shrink':sh,'selection':scores(y[:,sel],apply(f[:,sel],o[:,sel],r[:,sel],cats,pol))})
 choice=min(recs,key=lambda z:z['selection']['wape']);final=fit_policy(y,f,o,r,cats,choice['occ_bins'],choice['hit_bins'],choice['shrink']);cs={str(c):weighted_scale(y[cats==c],f[cats==c]) for c in np.unique(cats)}
 dump(out/'POLICY_FREEZE.json',{'created_utc':now(),'choice':choice,'all_selection':recs,'final_policy':final,'base_category_scales':cs,'occurrence_sha256':sha(out/'occurrence.txt'),'blocks_scored':0})
 rows=[]
 for block in ['calibration_a','calibration_b','shadow']:
  td=np.load(cache/f'{block}_aligned.npz')['target_days'];yy=p.truth[:,td-1].astype(float);ff=predict(b,h,base,td);oo=predict_hit(b,occ,td);rr=predict_hit(b,hm,td);cm=np.asarray([cs[str(c)] for c in cats])[:,None];p0=cm*ff;p1=apply(ff,oo,rr,cats,final)
  items,inv=np.unique(p.metadata['item_id'],return_inverse=True);mass=np.bincount(inv,weights=yy.sum(axis=1));e0=np.bincount(inv,weights=np.abs(yy-p0).sum(axis=1));e1=np.bincount(inv,weights=np.abs(yy-p1).sum(axis=1));rng=np.random.default_rng(20260906);w=rng.multinomial(len(items),np.full(len(items),1/len(items)),size=4000);vals=(w@(e1-e0))/(w@mass)
  row={'block':block,'base':scores(yy,p0),'dual_risk':scores(yy,p1),'paired_item_delta':{'estimate':float((e1-e0).sum()/mass.sum()),'ci95':np.quantile(vals,[.025,.975]).tolist()}};rows.append(row);print(row,flush=True);np.savez_compressed(out/f'{block}_item_stats.npz',items=items,mass=mass,base_error=e0,dual_error=e1)
 dump(out/'RESULTS.json',{'status':'COMPLETED','created_utc':now(),'choice':choice,'rows':rows})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--base',type=Path,required=True);a.add_argument('--hit',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.data,v.cache,v.base,v.hit,v.output)
