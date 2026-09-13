"""Reconstruct calibration support from train-only data and frozen heads.

No official evaluation file is opened. Counts concern calibration rows;
repeated horizons are not independent samples.
"""
from pathlib import Path
import sys,json,gc
import numpy as np
import pandas as pd
import xgboost as xgb
from replay_m5 import sha256,write_json
from reproduce_m5_missing_controls import now
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'code/freshretail'))
from censorcast.data import infer_split_dates,make_supervised,normalize_frame,split_supervised
from censorcast.models import SmoothedTargetEncoder
from run_frozen_rc_eval import fit_policy
out=root/'evidence/coordinate_tests/fresh_calibration_support';out.mkdir(exist_ok=False)
ev=root/'evidence/freshretail/frozen_eval';pol=json.loads((ev/'FINAL_POLICY.json').read_text())
write_json(out/'PROTOCOL.json',dict(created_utc=now(),mode='train-only support audit',new_forecast_head_fits=0,encoder_reconstructions=1,eval_accesses=0,policy_changes=0,script_sha256=sha256(__file__),train_sha256=sha256(root/'inputs/freshretail/freshretail_2000_train_only.pkl.gz')))
t=pd.read_pickle(root/'inputs/freshretail/freshretail_2000_train_only.pkl.gz');t.attrs={};t=normalize_frame(t);t['_source_split']='train';dates=infer_split_dates(t)
rows,num,cat=make_supervised(t,horizons=range(1,8),max_train_rows=600000,seed=20260907,development_holdout_days=35)
splits=split_supervised(rows,dates);fit=splits['fit'];enc=SmoothedTargetEncoder(categorical=cat,numeric=num,smoothing=30);enc.fit(fit,fit.sale_amount.to_numpy(float),uncensored=fit.is_censored.eq(0).to_numpy())
base=xgb.Booster();base.load_model(ev/'base_l1.ubj');base.set_param({'nthread':4});risk=xgb.Booster();risk.load_model(ev/'stockout_risk.ubj');risk.set_param({'nthread':4})
frames=[]
for name in ['selection','risk_train','calibration_a','calibration_b','shadow']:
 f=splits[name].reset_index(drop=True);dm=xgb.DMatrix(enc.transform(f));bb=np.maximum(base.predict(dm)-1,0);q=np.clip(risk.predict(dm),0,1)
 frames.append(pd.DataFrame(dict(group=f.management_group_id.astype(str),series=f.series_id.astype(str),day=f.dt.astype(str),y=f.sale_amount.to_numpy(float),available=f.is_censored.eq(0).to_numpy(),raw=bb,q=q)));print(name,len(f),flush=True)
df=pd.concat(frames,ignore_index=True);reconstructed=fit_policy(df.y.to_numpy(),df.raw.to_numpy(),df.q.to_numpy(),df.group.to_numpy(),df.available.to_numpy())
err=max(abs(reconstructed['scales'][k]-v) for k,v in pol['scales'].items());assert err<1e-10,err
df=df.loc[df.available].copy();df['bin']=np.searchsorted(pol['edges'],df.q,side='right')+1
cells=[]
for c in sorted(pol['category_scales']):
 for j in range(1,9):
  f=df.loc[df.group.eq(c)&df.bin.eq(j)];b=f.raw.to_numpy(float)
  cells.append(dict(group=c,bin=j,rows=len(f),unique_series=f.series.nunique(),unique_dates=f.day.nunique(),unique_series_target_dates=f[['series','day']].drop_duplicates().shape[0],positive_rows=int((b>0).sum()),weight_effective_n=float(b.sum()**2/(np.square(b).sum()+1e-30)),target_mass=float(f.y.sum()),scale=pol['scales'][f'{c}|{j-1}']))
pd.DataFrame(cells).to_csv(out/'CALIBRATION_SUPPORT.csv',index=False)
write_json(out/'RESULTS.json',dict(status='PASS',calibration_rows=len(df),max_policy_scale_difference=err,model_fits=0,eval_file_accesses=0,cells=cells,interpretation='row-weight effective size is a concentration diagnostic, not an independence-adjusted sample size'))
print('COMPLETE',err,flush=True)
