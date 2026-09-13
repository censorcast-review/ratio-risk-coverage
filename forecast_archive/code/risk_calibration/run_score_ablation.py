"""Post-hoc matched-score attribution for the primary risk calibrator."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_aft_pilot import now,sha,dump,scores,predict_days
from evaluate_risk_calibration import predict_hit,fit_policy,apply_policy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'accuracy_revision_20260907'/'code'/'m5'))
from legacy_features.features import DesignPanel,FeatureBuilder

def feature_scores(builder,days,batch_days=7):
 n=builder.panel.n_series;out={k:np.empty((n,len(days)),np.float32) for k in ['censor_rate_28','origin_fill_ratio','capacity_pressure']}
 for start in range(0,len(days),batch_days):
  ds=days[start:start+batch_days];s=np.repeat(np.arange(n,dtype=np.int32),len(ds));d=np.tile(ds,n)
  x=builder.make_features(s,d,include_censor=True)
  out['censor_rate_28'][:,start:start+len(ds)]=x[:,29].reshape(n,-1)
  out['origin_fill_ratio'][:,start:start+len(ds)]=x[:,31].reshape(n,-1)
  out['capacity_pressure'][:,start:start+len(ds)]=-x[:,32].reshape(n,-1)
 return out

def run(data,cache,model_path,hit_path,output):
 output.mkdir(parents=True,exist_ok=False)
 dump(output/'PROTOCOL.json',{'status':'POST_HOC_ATTRIBUTION','created_utc':now(),'bins':8,'shrink':.75,
  'scores':['learned_hit_risk','base_forecast','censor_rate_28','origin_fill_ratio','capacity_pressure'],
  'selection':'none; identical fixed calibrator applied to every score','evaluation_blocks':'previously consumed',
  'model_sha256':sha(model_path),'hit_sha256':sha(hit_path),'script_sha256':sha(__file__)})
 panel=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8)
 builder=FeatureBuilder(panel);cats=panel.metadata['cat_id'];base=xgb.Booster();base.load_model(model_path);hit=lgb.Booster(model_file=str(hit_path))
 vd=np.arange(1314,1434,dtype=np.int32);vd=vd[vd-builder.horizon_for_day(vd)>=1313];yv=panel.truth[:,vd-1].astype(float);bv=predict_days(builder,base,vd)
 sv=feature_scores(builder,vd);sv['learned_hit_risk']=predict_hit(builder,hit,vd);sv['base_forecast']=bv
 policies={k:fit_policy(yv,bv,v,cats,8,.75) for k,v in sv.items()};rows=[]
 for block in ['calibration_a','calibration_b','shadow']:
  td=np.load(cache/f'{block}_aligned.npz')['target_days'];y=panel.truth[:,td-1].astype(float);b=predict_days(builder,base,td)
  ss=feature_scores(builder,td);ss['learned_hit_risk']=predict_hit(builder,hit,td);ss['base_forecast']=b
  row={'block':block,'scores':{k:scores(y,apply_policy(b,ss[k],cats,policies[k])) for k in policies}};rows.append(row);print(row,flush=True)
 dump(output/'RESULTS.json',{'status':'COMPLETED','created_utc':now(),'policies':policies,'rows':rows})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--hit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.data,a.cache,a.model,a.hit,a.output)
