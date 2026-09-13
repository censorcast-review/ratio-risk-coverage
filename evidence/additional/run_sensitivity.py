from pathlib import Path
import sys,json,hashlib,time
import numpy as np,lightgbm as lgb
R=Path('review_revision/CENSORCAST').resolve();E=R/'evidence/accuracy_selection';out=Path('additional_experiments/sensitivity');out.mkdir(exist_ok=False)
def dump(name,obj):(out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
caps=[.60,.62,.85*.7530939208,.66,.68,.70];floors=[0.,.35,.60,.80];methods=['error','row','mixed','demand'];models=['group_l1','learned_l1','chronos2_univariate_raw']
dump('PROTOCOL.json',{'time':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'caps':caps,'floors':floors,'methods':methods,'models':models,'scope':'retrospective development only; original saved predictors and heads; no fits or holdout access','baseline':'Conditional absolute-error regression ranking, following Franc et al. (2023) regression-based uncertainty construction; thresholds recalibrated to WAPE cap, not original selective-risk guarantee','selection':'largest common feasible whole-tie threshold on calibration A/B; no threshold selection using evaluation outcomes'})
sys.path.insert(0,str(R/'forecast_archive/code'))
from replay_m5 import DesignPanel,FeatureBuilder,HierarchyFeatures
D=Path('integration_inputs');panel=DesignPanel.load(D/'data/design_outcomes_v0_5.npz',D/'data/calendar.csv',D/'data/sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8)
b=FeatureBuilder(panel);h=HierarchyFeatures(panel);qmodel=lgb.Booster(model_file=str(R/'forecast_archive/evidence/revision/m5_observable/seed_20260906/hit.txt'));muhead=lgb.Booster(model_file=str(E/'DEMAND_HEAD.txt'));fr=json.loads((E/'FROZEN_SELECTORS.json').read_text());T=fr['T_training']
z=np.load(E/'development_forecasts.npz');days=z['days'];m=days>1610;ds=days[m];ss=np.repeat(np.arange(panel.n_series),len(ds));dd=np.tile(ds,panel.n_series);common=b.make_features(ss,dd,include_censor=True);q=np.clip(qmodel.predict(common,num_threads=4),0,1).astype(np.float32);x=np.column_stack([common,h.make(ss,dd),q]).astype(np.float32);mu=np.maximum(muhead.predict(x,num_threads=4),0);y=z['y'][:,m].ravel();blocks=(dd>=1646).astype(int);ev=np.load(E/'evaluation_forecasts.npz');yv=ev['y'].ravel();records=[]
def scores(e,w,r,method):
 if method=='error':return e
 if method=='row':return e-r*w
 den=w/T if method=='demand' else .25+.75*w/T
 v=e-r*w
 return np.divide(v,den,out=np.where(v>0,np.inf,np.where(v<0,-np.inf,0.)),where=den>0)
def metrics(y,f,a):
 n=int(a.sum());d=float(y[a].sum());err=float(np.abs(y[a]-f[a]).sum())
 return {'rows':n,'c':n/len(y),'d':d/float(y.sum()),'risk':err/d if d else None}
for model in models:
 f=z[model][:,m].ravel();error=lgb.Booster(model_file=str(E/(model+'_ERROR_HEAD.txt')));e=np.maximum(error.predict(np.column_stack([x,f]).astype(np.float32),num_threads=4),0);es=np.load(E/(model+'_EVAL_SELECTORS.npz'));fv=ev[model].ravel()
 np.savez_compressed(out/(model+'_CALIBRATION.npz'),e=e,mu=mu,y=y,f=f,blocks=blocks)
 for cap in caps:
  for method in methods:
   score=scores(e,mu,cap,method);ts=np.unique(score[np.isfinite(score)]);ok=np.ones(len(ts),bool);counts=[]
   for j in [0,1]:
    mm=blocks==j;o=np.argsort(score[mm],kind='stable');sscore=score[mm][o];yy=y[mm][o];ff=f[mm][o];n=np.searchsorted(sscore,ts,side='right');mass=np.r_[0.,np.cumsum(yy)][n];err=np.r_[0.,np.cumsum(np.abs(yy-ff))][n];ok&=(mass>0)&(err<=cap*mass+1e-9);counts.append(n/len(yy))
   se=scores(es['e'],es['mu'],cap,method)
   for floor in floors:
    good=ok&(counts[0]>=floor)&(counts[1]>=floor);ix=np.flatnonzero(good);t=float(ts[ix[-1]]) if len(ix) else None;a=se<=t if t is not None else np.zeros(len(se),bool)
    cal=[metrics(y[blocks==j],f[blocks==j],score[blocks==j]<=t) if t is not None else metrics(y[blocks==j],f[blocks==j],np.zeros(sum(blocks==j),bool)) for j in [0,1]]
    records.append({'model':model,'cap':cap,'floor':floor,'method':method,'threshold':t,'calibration':cal,'evaluation':metrics(yv,fv,a)})
  print(model,cap,flush=True)
 dump('RESULTS.json',records)
dump('COMPLETE.json',{'fit_calls':0,'heldout_access':0,'records':len(records),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}})
