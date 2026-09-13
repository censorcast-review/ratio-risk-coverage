"""Independent saved-policy/model replay and sufficient-statistic arithmetic."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd,lightgbm as lgb
from replay_m5 import DesignPanel,FeatureBuilder,HierarchyFeatures
R=Path(__file__).resolve().parents[1];out=R/'evidence/forward_controls';checks=[]
def ck(name,cond):assert cond,name;checks.append(name)
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for tag in ['fresh','m5']:
 p=out/tag;complete=json.loads((p/'COMPLETE.json').read_text());proto=json.loads((p/'PROTOCOL.json').read_text());ck(tag+' protocol script hash',digest(R/'code/run_forward_controls.py')==proto['script_sha256'])
 for n,h in complete['files'].items():ck(tag+'/'+n,digest(p/n)==h)
 sel=json.loads((p/f'{tag}_SELECTION.json').read_text());ck(tag+' temporal separation',sel['fit_max_target']<sel['selection_min_target']);ck(tag+' selected only validation',sel['selected']['wape']==min(sel['parent_wape'],*[v['wape'] for v in sel['grid']]))
 ck(tag+' freeze after selection',json.loads((p/f'{tag}_FROZEN_POLICIES.json').read_text())['created_utc']>=sel['created_utc'])
p=out/'fresh';d=pd.read_csv(p/'fresh_EVAL_PREDICTIONS.csv.gz');d['raw']=d.raw.astype(np.float32).astype(float);res=json.loads((p/'fresh_RESULTS.json').read_text());pol=json.loads((p/'fresh_FROZEN_POLICIES.json').read_text())['policies'];y=d.y.to_numpy();base=d.raw.to_numpy()*np.array([pol['risk']['category_scales'][str(c)] for c in d.group]);ck('fresh parent replay',np.max(abs(base-d.base))<1e-12)
for key,coord in [('rc',d.q.to_numpy(np.float32)),('forecast',d.raw.to_numpy()),('merged',d.q.to_numpy(np.float32))]:
 pp=pol['risk' if key=='rc' else key];j=np.searchsorted(pp['edges'],coord,side='right');f=d.raw.to_numpy()*np.array([pp['scales'][f'{c}|{b}'] for c,b in zip(d.group,j)]);ck('fresh '+key+' policy replay',np.max(abs(f-d[key]))<1e-12)
f=base*np.where(d.q.to_numpy(np.float32)>=pol['top_threshold'],pol['top_scale'],1.);ck('fresh top replay',np.max(abs(f-d.top))<1e-12)
x=np.load(p/'fresh_EVAL_FEATURES.npz')['x'];model=lgb.Booster(model_file=str(p/'fresh_FINAL_STACK.txt'));f=np.maximum(base+model.predict(np.column_stack([x,base,d.q.to_numpy(np.float32)]).astype(np.float32),num_threads=4),0);ck('fresh final model replay',np.max(abs(f-d['stack']))<1e-10)
for pop,m in [('eligible',d.available.to_numpy(bool)),('all',np.ones(len(d),bool))]:
 for k in ['base','rc','forecast','top','merged','stack']:
  v=float(np.abs(y[m]-d[k].to_numpy()[m]).sum()/y[m].sum());ck('fresh '+pop+' '+k,abs(v-res[pop]['metrics'][k]['wape'])<1e-12)
 for contrast,units in res[pop]['contrasts'].items():
  a,b=contrast.split('_minus_')
  for unit,saved in units.items():
   t=pd.DataFrame({'u':d[unit].astype(str),'y':np.where(m,y,0),'delta':np.where(m,np.abs(y-d[a])-np.abs(y-d[b]),0)}).groupby('u',sort=True).sum();mass=t.y.to_numpy();delta=t.delta.to_numpy();rng=np.random.default_rng(saved['seed']);v=[]
   for _ in range(40):
    w=rng.multinomial(len(t),np.full(len(t),1/len(t)),size=100);v.extend(w@delta/np.maximum(w@mass,1e-12))
   ck('fresh '+pop+' '+contrast+' '+unit,np.max(abs(np.quantile(v,[.025,.975])-saved['ci95']))<1e-12)
p=out/'m5';stats=pd.read_csv(p/'m5_ITEM_STATISTICS.csv');res=json.loads((p/'m5_RESULTS.json').read_text())
for k,v in res['metrics'].items():ck('m5 item '+k,abs(stats[k+'_error'].sum()/stats.mass.sum()-v['wape'])<1e-12)
for key,a,b in [('rc_minus_stack','rc','stack'),('rc_minus_top','rc','top')]:
 saved=res[key];mass=stats.mass.to_numpy();delta=(stats[a+'_error']-stats[b+'_error']).to_numpy();rng=np.random.default_rng(saved['seed']);v=[]
 for _ in range(40):
  w=rng.multinomial(len(stats),np.full(len(stats),1/len(stats)),size=100);v.extend(w@delta/np.maximum(w@mass,1e-12))
 ck('m5 interval '+key,np.max(abs(np.quantile(v,[.025,.975])-saved['ci95']))<1e-12)
# Full M5 stack inference, built from original origin-valid features.
inputs=Path(sys.argv[1]);panel=DesignPanel.load(inputs/'data/design_outcomes_v0_5.npz',inputs/'data/calendar.csv',inputs/'data/sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8);builder=FeatureBuilder(panel);hier=HierarchyFeatures(panel);z=np.load(R/'evidence/cell_audit/m5/later_predictions.npz');days=z['days'];cats=panel.metadata['cat_id'];pol=json.loads((p/'m5_FROZEN_POLICIES.json').read_text())['policies']['risk'];model=lgb.Booster(model_file=str(p/'m5_FINAL_STACK.txt'));err=0.;mass=0.
for start in range(0,len(days),7):
 ds=days[start:start+7];ss=np.repeat(np.arange(panel.n_series),len(ds));dd=np.tile(ds,panel.n_series);b=z['raw'][:,start:start+len(ds)].ravel()*np.array([pol['category_scales'][str(c)] for c in cats[ss]]);q=z['risk'][:,start:start+len(ds)].ravel();x=np.column_stack([builder.make_features(ss,dd,include_censor=True),hier.make(ss,dd),b,q]).astype(np.float32);f=np.maximum(b+model.predict(x,num_threads=4),0);yy=panel.truth[ss,dd-1];err+=np.abs(yy-f).sum();mass+=yy.sum()
ck('m5 full final stack replay',abs(err/mass-res['metrics']['stack']['wape'])<1e-12)
(R/'verification/FORWARD_CONTROLS.json').write_text(json.dumps({'status':'PASS','checks':checks,'count':len(checks)},indent=2)+'\n');print('PASS',len(checks))
