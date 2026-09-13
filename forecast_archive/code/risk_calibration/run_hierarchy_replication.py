"""Fixed two-seed replication of hierarchy base plus risk calibration."""
from pathlib import Path
import argparse,sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_aft_pilot import now,sha,dump,weighted_scale,scores
from run_hierarchy_pilot import HierarchyFeatures,make_x,predict
from evaluate_risk_calibration import predict_hit,fit_policy,apply_policy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'accuracy_revision_20260907'/'code'/'m5'))
from legacy_features.features import DesignPanel,FeatureBuilder
SEEDS=[20260907,20260908]
def run(data,cache,hit_root,out):
 out.mkdir(parents=True,exist_ok=False);dump(out/'PROTOCOL.json',{'status':'FROZEN_BEFORE_FITS','created_utc':now(),'seeds':SEEDS,
 'base':{'objective':'reg:absoluteerror','depth':8,'rounds':750,'eta':.03,'hierarchy_features':7},'calibrator':{'bins':8,'shrink':.75},
 'provenance':'configuration and calibration family fixed by seed 20260906','blocks':'previously consumed retrospective','script_sha256':sha(__file__)})
 p=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv');p.censored=(p.observed>=p.capacity).astype(np.uint8);b=FeatureBuilder(p);h=HierarchyFeatures(p);cats=p.metadata['cat_id'];vd=np.arange(1314,1434,dtype=np.int32);vd=vd[vd-b.horizon_for_day(vd)>=1313];yv=p.truth[:,vd-1].astype(float);rows=[]
 for seed in SEEDS:
  sd=out/f'seed_{seed}';sd.mkdir();s,d=b.sample_pairs(365,1313,1200000,seed);x=make_x(b,h,s,d);dm=xgb.DMatrix(x,label=p.observed[s,d-1].astype(float)+1);del x
  dump(sd/'FIT_START.json',{'created_utc':now(),'protocol_sha256':sha(out/'PROTOCOL.json'),'blocks_scored':0})
  m=xgb.train({'objective':'reg:absoluteerror','eta':.03,'max_depth':8,'min_child_weight':100,'subsample':.9,'colsample_bytree':.9,'lambda':2.,'tree_method':'hist','max_bin':255,'nthread':8,'seed':seed},dm,750);m.save_model(sd/'model.ubj');del dm
  hm=lgb.Booster(model_file=str(hit_root/f'seed_{seed}'/'hit.txt'));fv=predict(b,h,m,vd);rv=predict_hit(b,hm,vd);cs={str(c):weighted_scale(yv[cats==c],fv[cats==c]) for c in np.unique(cats)};pol=fit_policy(yv,fv,rv,cats,8,.75)
  dump(sd/'POLICY_FREEZE.json',{'created_utc':now(),'model_sha256':sha(sd/'model.ubj'),'base_category_scales':cs,'risk_policy':pol,'blocks_scored':0})
  for block in ['calibration_a','calibration_b','shadow']:
   td=np.load(cache/f'{block}_aligned.npz')['target_days'];y=p.truth[:,td-1].astype(float);f=predict(b,h,m,td);r=predict_hit(b,hm,td);cm=np.asarray([cs[str(c)] for c in cats])[:,None];p0=cm*f;p1=apply_policy(f,r,cats,pol)
   items,inv=np.unique(p.metadata['item_id'],return_inverse=True);mass=np.bincount(inv,weights=y.sum(axis=1));e0=np.bincount(inv,weights=np.abs(y-p0).sum(axis=1));e1=np.bincount(inv,weights=np.abs(y-p1).sum(axis=1));np.savez_compressed(sd/f'{block}_item_stats.npz',items=items,mass=mass,base_error=e0,risk_error=e1)
   row={'seed':seed,'block':block,'base':scores(y,p0),'risk':scores(y,p1)};rows.append(row);print(row,flush=True)
 dump(out/'RESULTS.json',{'status':'COMPLETED','created_utc':now(),'rows':rows})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--hit-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.data,v.cache,v.hit_root,v.output)
