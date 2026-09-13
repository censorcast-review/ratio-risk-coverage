"""Expanded adapter grid, chosen on selection only; no risk fits are changed."""
from pathlib import Path
import json
import numpy as np
from run_selective_study import load,dump,origin
from run_point_baselines import metrics
p=Path(__file__).resolve().parents[1]
s=load(p/'inputs/strong_point/selection_base_v0_5.npz');h=load(p/'inputs/strong_point/shadow_v0_5.npz')
keep=origin(s['target_days'])>=1313
ys=s['truth'][:,keep].astype(float);ps=s['baseline'][:,keep]
cs=np.maximum(s['censored_poisson_em'][:,keep]-ps,0);ph=s['censor_probability'][:,keep]
yh=h['truth'].astype(float);p0=h['baseline'];ch=np.maximum(h['censored_poisson_em']-p0,0);prob=h['censor_probability']
rows=[]
for alpha in [0,.25,.5,.75,1,1.5,2,3,4,6,8]:
 for gamma in [.25,.5,1,2,3,4,6]:
  pred=ps+alpha*ph**gamma*cs
  rows.append(dict(alpha=alpha,gamma=gamma,selection_wape=metrics(ys,pred)['wape']))
best=min(rows,key=lambda z:(z['selection_wape'],z['alpha'],z['gamma']))
new=p0+best['alpha']*prob**best['gamma']*ch
result={'status':'SELECTION_ONLY_EXPANDED_GRID_COMPLETE','point_seed':20260906,'grid':rows,'selected':best,'old_alpha2_gamma2_shadow':metrics(yh,h['proposal']),'expanded_shadow':metrics(yh,new),'training_calls':0,'risk_comparison_forecast_unchanged':True,'selection_only':True,'guardian_opened':False}
result['selection_boundary']=best['alpha'] in [0,8] or best['gamma'] in [.25,6]
dump(p/'results/review_revision/EXPANDED_ADAPTER_GRID.json',result)
print({k:v for k,v in result.items() if k!='grid'})
