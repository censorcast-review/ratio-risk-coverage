"""Independent model inference and exact calibration-threshold replay."""
from pathlib import Path
import argparse,sys,json
import numpy as np,lightgbm as lgb
from run_accuracy_selection import score,choose,metric

a=argparse.ArgumentParser();a.add_argument('--forecast-package',type=Path,required=True);a.add_argument('--data',type=Path,required=True);a.add_argument('--evidence',type=Path,required=True);a.add_argument('--output',type=Path,required=True);p=a.parse_args();sys.path.insert(0,str(p.forecast_package/'code'))
from replay_m5 import DesignPanel,FeatureBuilder,HierarchyFeatures
panel=DesignPanel.load(p.data/'data/design_outcomes_v0_5.npz',p.data/'data/calendar.csv',p.data/'data/sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8);b=FeatureBuilder(panel);h=HierarchyFeatures(panel);risk=lgb.Booster(model_file=str(p.forecast_package/'evidence/revision/m5_observable/seed_20260906/hit.txt'));muhead=lgb.Booster(model_file=str(p.evidence/'DEMAND_HEAD.txt'));fr=json.loads((p.evidence/'FROZEN_SELECTORS.json').read_text());checks=[]
def ck(n,yes):
 checks.append({'name':n,'pass':bool(yes)})
 if not yes:raise AssertionError(n)
# Check that all newer forecasts/head outputs are aligned on a deterministic
# 50k-row later sample, independently rebuilding history features and inference.
z=np.load(p.evidence/'evaluation_forecasts.npz');days=z['days'];rng=np.random.default_rng(49021);ids=np.sort(rng.choice(z['y'].size,50000,replace=False));ss=ids//len(days);dd=days[ids%len(days)];common=b.make_features(ss,dd,include_censor=True);q=np.clip(risk.predict(common,num_threads=4),0,1).astype(np.float32);x=np.column_stack([common,h.make(ss,dd),q]).astype(np.float32);mu=np.maximum(muhead.predict(x,num_threads=4),0)
for k in fr['policies']:
 model=lgb.Booster(model_file=str(p.evidence/(k+'_ERROR_HEAD.txt')));saved=np.load(p.evidence/(k+'_EVAL_SELECTORS.npz'));e=np.maximum(model.predict(np.column_stack([x,z[k].ravel()[ids]]).astype(np.float32),num_threads=4),0);ck(k+' mu replay',np.max(abs(mu-saved['mu'][ids]))<1e-12);ck(k+' error replay',np.max(abs(e-saved['e'][ids]))<1e-12)
# Full calibration inference; reproduce all distinct-score threshold searches.
z=np.load(p.evidence/'development_forecasts.npz');days=z['days'];maskdays=days>1610;ds=days[maskdays];ss=np.repeat(np.arange(panel.n_series),len(ds));dd=np.tile(ds,panel.n_series);common=b.make_features(ss,dd,include_censor=True);q=np.clip(risk.predict(common,num_threads=4),0,1).astype(np.float32);x=np.column_stack([common,h.make(ss,dd),q]).astype(np.float32);mu=np.maximum(muhead.predict(x,num_threads=4),0);y=z['y'][:,maskdays].ravel();blocks=(dd>=1646).astype(int)
for k in fr['policies']:
 model=lgb.Booster(model_file=str(p.evidence/(k+'_ERROR_HEAD.txt')));f=z[k][:,maskdays].ravel();e=np.maximum(model.predict(np.column_stack([x,f]).astype(np.float32),num_threads=4),0)
 for l,t in fr['policies'][k].items():
  s=score(e,mu,fr['T_training'],float(l));tt,_=choose(s,y,f,blocks);ck(k+l+' exact threshold',tt==t)
  for j in [0,1]:
   m=blocks==j;rec=metric(y[m],f[m],s[m]<=t);saved=fr['calibration_metrics'][k][l][str(j)];ck(k+l+str(j)+' calibration',abs(rec['wape']-saved['wape'])<1e-12 and rec['rows']==saved['rows'])
p.output.parent.mkdir(parents=True,exist_ok=True);p.output.write_text(json.dumps({'status':'PASS','checks':checks,'count':len(checks),'new_fits':0,'later_model_replay_rows':50000,'calibration_replay_rows':len(y)},indent=2)+'\n');print('PASS',len(checks))
