"""Primary-seed selection and retrospective evaluation for hierarchy base + risk calibration."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_aft_pilot import now,sha,dump,weighted_scale,scores
from run_hierarchy_pilot import HierarchyFeatures,predict
from evaluate_risk_calibration import predict_hit,fit_policy,apply_policy,BINS,SHRINK
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'accuracy_revision_20260907'/'code'/'m5'))
from legacy_features.features import DesignPanel,FeatureBuilder
def run(data,cache,model_path,hit_path,out):
 out.mkdir(parents=True,exist_ok=False);dump(out/'PROTOCOL.json',{'status':'FROZEN_BEFORE_BLOCK_SCORING','created_utc':now(),
 'calibration_fit_days':[1314,1373],'family_selection_days':[1374,1433],'candidate_bins':BINS,'candidate_shrinkage':SHRINK,
 'refit':'chosen family on all validation days','blocks':'previously consumed retrospective','model_sha256':sha(model_path),'hit_sha256':sha(hit_path),'script_sha256':sha(__file__)})
 p=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv');p.censored=(p.observed>=p.capacity).astype(np.uint8);b=FeatureBuilder(p);h=HierarchyFeatures(p);cats=p.metadata['cat_id']
 m=xgb.Booster();m.load_model(model_path);hm=lgb.Booster(model_file=str(hit_path));vd=np.arange(1314,1434,dtype=np.int32);vd=vd[vd-b.horizon_for_day(vd)>=1313]
 y=p.truth[:,vd-1].astype(float);f=predict(b,h,m,vd);r=predict_hit(b,hm,vd);fit=vd<=1373;sel=vd>=1374;records=[]
 for nb in BINS:
  for sh in SHRINK:
   pol=fit_policy(y[:,fit],f[:,fit],r[:,fit],cats,nb,sh);records.append({'nbins':nb,'shrink':sh,'selection':scores(y[:,sel],apply_policy(f[:,sel],r[:,sel],cats,pol))})
 choice=min(records,key=lambda z:z['selection']['wape']);final=fit_policy(y,f,r,cats,choice['nbins'],choice['shrink']);cs={str(c):weighted_scale(y[cats==c],f[cats==c]) for c in np.unique(cats)}
 dump(out/'POLICY_FREEZE.json',{'created_utc':now(),'choice':choice,'all_selection':records,'risk_policy':final,'base_category_scales':cs,'blocks_scored':0})
 rows=[]
 for block in ['calibration_a','calibration_b','shadow']:
  td=np.load(cache/f'{block}_aligned.npz')['target_days'];yy=p.truth[:,td-1].astype(float);ff=predict(b,h,m,td);rr=predict_hit(b,hm,td);cm=np.asarray([cs[str(c)] for c in cats])[:,None];p0=cm*ff;p1=apply_policy(ff,rr,cats,final)
  items,inv=np.unique(p.metadata['item_id'],return_inverse=True);mass=np.bincount(inv,weights=yy.sum(axis=1));e0=np.bincount(inv,weights=np.abs(yy-p0).sum(axis=1));e1=np.bincount(inv,weights=np.abs(yy-p1).sum(axis=1));rng=np.random.default_rng(20260906);w=rng.multinomial(len(items),np.full(len(items),1/len(items)),size=4000);vals=(w@(e1-e0))/(w@mass)
  row={'block':block,'base':scores(yy,p0),'risk':scores(yy,p1),'paired_item_delta':{'estimate':float((e1-e0).sum()/mass.sum()),'ci95':np.quantile(vals,[.025,.975]).tolist()}};rows.append(row);print(row,flush=True);np.savez_compressed(out/f'{block}_item_stats.npz',items=items,mass=mass,base_error=e0,risk_error=e1)
 dump(out/'RESULTS.json',{'status':'COMPLETED','created_utc':now(),'choice':choice,'rows':rows})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--model',type=Path,required=True);a.add_argument('--hit',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.data,v.cache,v.model,v.hit,v.output)
