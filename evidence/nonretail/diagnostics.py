from pathlib import Path
import json,numpy as np
from scipy.stats import kendalltau
R=Path('experiment_revision/CENSORCAST');E=R/'evidence/additional';O=Path('nonretail_experiments');res=[];rank=[]
S=json.loads((E/'sensitivity/RESULTS.json').read_text());old=R/'evidence/accuracy_selection';ev=np.load(old/'evaluation_forecasts.npz');y=ev['y'].ravel();r=.85*.7530939208
for model in ['group_l1','learned_l1','chronos2_univariate_raw']:
 z=np.load(E/'sensitivity'/f'{model}_CALIBRATION.npz');score=-z['mu'];ts=np.unique(score);masks=[];ok=np.ones(len(ts),bool);L=np.abs(z['y']-z['f']);W=z['y'];blocks=z['blocks']
 for j in [0,1]:
  ix=np.flatnonzero(blocks==j);o=ix[np.argsort(score[ix],kind='stable')];n=np.searchsorted(score[o],ts,side='right');err=np.r_[0.,np.cumsum(L[o])][n];mass=np.r_[0.,np.cumsum(W[o])][n];ok&=(mass>0)&(err<=r*mass+1e-9)&(n/len(ix)>=.35)
 ids=np.flatnonzero(ok);t=float(ts[ids[-1]]) if len(ids) else None;v=np.load(old/f'{model}_EVAL_SELECTORS.npz');mu=v['mu'];e=v['e'];a=-mu<=t if t is not None else np.zeros(len(y),bool);mass=y[a].sum();res.append(dict(model=model,cap=r,floor=.35,threshold=t,rows=int(a.sum()),c=float(a.mean()),d=float(mass/y.sum()),risk=float(np.abs(y[a]-ev[model].ravel()[a]).sum()/mass) if mass else None))
 inds=np.random.default_rng(20260910).choice(len(mu),min(100000,len(mu)),replace=False);p=mu[inds]>0;ee=e[inds][p];mm=mu[inds][p];rank.append(dict(dataset=model,n=len(ee),kendall=float(kendalltau(ee-r*mm,ee/mm).statistic),positive_cost_kendall=float(kendalltau((ee-r*mm)[ee>r*mm],(ee/mm)[ee>r*mm]).statistic)))
yz=np.load(E/'yeast/TEST_PREDICTIONS.npz');p=yz['W']>0;e=yz['e'][p];w=yz['W'][p];yr=.27558845861807135;rank.append(dict(dataset='Yeast',n=int(p.sum()),kendall=float(kendalltau(e-yr*w,e/w).statistic),positive_cost_kendall=float(kendalltau((e-yr*w)[e>yr*w],(e/w)[e>yr*w]).statistic)))
(O/'M5_DESCENDING.json').write_text(json.dumps(res,indent=2)+'\n');(O/'RANK_DIAGNOSTICS.json').write_text(json.dumps(rank,indent=2)+'\n')
z=np.load(O/'bike/TEST_default.npz');a=z['0.85_row'];b=z['0.85_demand'];cap=json.loads((O/'bike/RESULTS.json').read_text())['policies'][5]['cap'];L=z['L'];W=z['W'];budget={}
for name,m in [('common',a&b),('row_only',a&~b),('demand_only',b&~a),('union',a|b)]:
 budget[name]=dict(rows=int(m.sum()),loss=float(L[m].sum()),weight=float(W[m].sum()),risk=float(L[m].sum()/W[m].sum()),excess=float((L[m]-cap*W[m]).sum()))
(O/'BIKE_BUDGET.json').write_text(json.dumps({'cap':cap,'budget':budget},indent=2)+'\n');print(json.dumps({'descending':res,'ranks':rank,'budget':budget},indent=2))
