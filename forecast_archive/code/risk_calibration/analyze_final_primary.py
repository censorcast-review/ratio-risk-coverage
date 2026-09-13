"""Descriptive decomposition of the frozen primary hierarchy-risk policy."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import xgboost as xgb
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_aft_pilot import now,sha,dump,scores
from run_hierarchy_pilot import HierarchyFeatures,predict
from evaluate_risk_calibration import predict_hit,apply_policy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'accuracy_revision_20260907'/'code'/'m5'))
from legacy_features.features import DesignPanel,FeatureBuilder
def run(data,cache,base_path,hit_path,policy_path,out):
 out.mkdir(parents=True,exist_ok=False);dump(out/'PROTOCOL.json',{'status':'POST_HOC_DESCRIPTIVE','created_utc':now(),'base_sha256':sha(base_path),'hit_sha256':sha(hit_path),'policy_sha256':sha(policy_path),'script_sha256':sha(__file__)})
 p=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv');p.censored=(p.observed>=p.capacity).astype(np.uint8);b=FeatureBuilder(p);h=HierarchyFeatures(p);cats=p.metadata['cat_id'];m=xgb.Booster();m.load_model(base_path);hm=lgb.Booster(model_file=str(hit_path));fr=json.load(open(policy_path));pol=fr['risk_policy'];cs=fr['base_category_scales'];rows=[]
 for block in ['calibration_a','calibration_b','shadow']:
  td=np.load(cache/f'{block}_aligned.npz')['target_days'];y=p.truth[:,td-1].astype(float);f=predict(b,h,m,td);r=predict_hit(b,hm,td);cm=np.asarray([cs[str(c)] for c in cats])[:,None];p0=cm*f;p1=apply_policy(f,r,cats,pol);strict=y>p.capacity[:,td-1]
  cat={str(c):{'base':scores(y[cats==c],p0[cats==c]),'risk':scores(y[cats==c],p1[cats==c])} for c in np.unique(cats)}
  horizons={str(k):{'base':scores(y[:,b.horizon_for_day(td)==k],p0[:,b.horizon_for_day(td)==k]),'risk':scores(y[:,b.horizon_for_day(td)==k],p1[:,b.horizon_for_day(td)==k])} for k in range(1,8)}
  strata={'strict':{'base':scores(y[strict],p0[strict]),'risk':scores(y[strict],p1[strict])},'other':{'base':scores(y[~strict],p0[~strict]),'risk':scores(y[~strict],p1[~strict])}}
  item=np.unique(p.metadata['item_id']);_,inv=np.unique(p.metadata['item_id'],return_inverse=True);e0=np.bincount(inv,weights=np.abs(y-p0).sum(axis=1));e1=np.bincount(inv,weights=np.abs(y-p1).sum(axis=1));mass=np.bincount(inv,weights=y.sum(axis=1))
  items={'count':len(item),'improved':int((e1<e0).sum()),'tied':int((e1==e0).sum()),'demand_mass_on_improved_items':float(mass[e1<e0].sum()/mass.sum())}
  rows.append({'block':block,'categories':cat,'horizons':horizons,'strata':strata,'items':items})
 dump(out/'RESULTS.json',{'status':'COMPLETED','created_utc':now(),'rows':rows});print(json.dumps(rows[-1],indent=2),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--base',type=Path,required=True);a.add_argument('--hit',type=Path,required=True);a.add_argument('--policy',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.data,v.cache,v.base,v.hit,v.policy,v.output)
