"""M5-fixed risk-calibration family transferred within UCI validation only."""
from pathlib import Path
import argparse,importlib.util,sys
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_aft_pilot import now,sha,dump,weighted_scale,scores
from run_dual_risk import fit_policy,apply
def load_module(path):
 spec=importlib.util.spec_from_file_location('uci_frozen_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def run(run_dir,runner,out):
 out.mkdir(parents=True,exist_ok=False);dump(out/'PROTOCOL.json',{'status':'FROZEN_BEFORE_EVALUATION','created_utc':now(),
 'family_source':'M5 primary seed: 8 hit-risk bins, shrinkage 0.75','fit':'first half of UCI validation','evaluation':'second half of UCI validation only',
 'uci_test_accesses':0,'runner_sha256':sha(runner),'l1_sha256':sha(run_dir/'models'/'l1.txt'),'hit_sha256':sha(run_dir/'models'/'hit.txt'),'script_sha256':sha(__file__)})
 mod=load_module(runner);z=np.load(run_dir/'development.npz');obs=z['observed'];cap=z['capacity'];truth=z['truth'];vd=z['valid_days'];dates=pd.date_range('2009-12-01',periods=obs.shape[1]);x=mod.features(obs,cap,dates,vd)
 lm=lgb.Booster(model_file=str(run_dir/'models'/'l1.txt'));hm=lgb.Booster(model_file=str(run_dir/'models'/'hit.txt'));b=np.maximum(lm.predict(x,num_threads=8),0).reshape(len(obs),-1);r=np.clip(hm.predict(x,num_threads=8),0,1).reshape(len(obs),-1);y=truth[:,vd].astype(float);cats=np.array(['ALL']*len(obs));cut=len(vd)//2
 scale=weighted_scale(y[:,:cut],b[:,:cut]);pol=fit_policy(y[:,:cut],b[:,:cut],np.zeros_like(r[:,:cut]),r[:,:cut],cats,1,8,.75);p0=scale*b[:,cut:];p1=apply(b[:,cut:],np.zeros_like(r[:,cut:]),r[:,cut:],cats,pol);yy=y[:,cut:]
 mass=yy.sum(axis=1);e0=np.abs(yy-p0).sum(axis=1);e1=np.abs(yy-p1).sum(axis=1);rng=np.random.default_rng(20260906);w=rng.multinomial(len(obs),np.full(len(obs),1/len(obs)),size=4000);vals=(w@(e1-e0))/(w@mass)
 result={'status':'COMPLETED','created_utc':now(),'fit_days':vd[:cut].tolist(),'evaluation_days':vd[cut:].tolist(),'baseline_scale':scale,
 'baseline':scores(yy,p0),'risk':scores(yy,p1),'delta_wape':float((e1-e0).sum()/mass.sum()),'paired_product_ci95':np.quantile(vals,[.025,.975]).tolist(),'policy':pol}
 dump(out/'RESULTS.json',result);print(result,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--runner',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.run_dir,v.runner,v.output)
