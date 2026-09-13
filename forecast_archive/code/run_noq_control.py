"""Adaptive retrospective follow-up: omit learned q from the residual calibrator."""
from run_forward_controls import *
a=argparse.ArgumentParser();a.add_argument('mode',choices=['fresh','m5']);a.add_argument('--inputs',type=Path);args=a.parse_args();tag=args.mode;out=ROOT/'evidence/forward_controls'/('noq_'+tag);out.mkdir(exist_ok=False);prev=ROOT/'evidence/forward_controls'/tag
dump(out/'PROTOCOL.json',{'created_utc':now(),'script_sha256':sha(__file__),'scope':'Follow-up after observing forward-stacking superiority; consumed outcomes, not confirmation','change':'omit learned q only; retain original history and availability features; same six-setting chronological selection','grid':{'depth':[2,4],'trees':[50,150,300]},'seed':SEED,'fit_cap':200000,'refit_cap':400000})
if tag=='fresh':
 d=pd.read_csv(prev/'fresh_TRAIN_CALIBRATION.csv.gz');d['raw']=d.raw.astype(np.float32).astype(float);d['q']=d.q.astype(np.float32);d['group']=d.group.astype(str);x=np.load(prev/'fresh_TRAIN_FEATURES.npz')['x'];td=pd.read_csv(prev/'fresh_EVAL_PREDICTIONS.csv.gz');td['raw']=td.raw.astype(np.float32).astype(float);td['q']=td.q.astype(np.float32);td['group']=td.group.astype(str);tx=np.load(prev/'fresh_EVAL_FEATURES.npz')['x'];mask=d.block.isin(['selection','risk_train']).to_numpy()
else:
 panel=DesignPanel.load(args.inputs/'data/design_outcomes_v0_5.npz',args.inputs/'data/calendar.csv',args.inputs/'data/sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8);builder=FeatureBuilder(panel);hier=HierarchyFeatures(panel)
 def load(path):
  z=np.load(path);days=z['days'];ss=np.repeat(np.arange(panel.n_series),len(days));dd=np.tile(days,panel.n_series);df=pd.DataFrame({'series':panel.metadata['item_id'][ss],'day':dd,'group':panel.metadata['cat_id'][ss],'y':panel.truth[ss,dd-1].astype(float),'raw':z['raw'].ravel(),'q':z['risk'].ravel(),'available':True});features=np.column_stack([builder.make_features(ss,dd,include_censor=True),hier.make(ss,dd)]).astype(np.float32);return df,features
 d,x=load(ROOT/'evidence/cell_audit/m5/point_validation_predictions.npz');mask=d.day.to_numpy()<=1373
# Independent tuning: parent baseline is fitted only on the earlier calibration rows.
p=fit_controls(d[mask]);bp=apply_controls(d,p)['base'];ok=d.available.to_numpy(bool);ids=np.flatnonzero(mask&ok);rng=np.random.default_rng(SEED)
if len(ids)>200000:ids=np.sort(rng.choice(ids,200000,replace=False))
xx=np.column_stack([x,bp]).astype(np.float32);sel=(~mask)&ok;yv=d.y.to_numpy(float);best={'kind':'parent','wape':wape(yv[sel],bp[sel])};grid=[]
for depth in [2,4]:
 for trees in [50,150,300]:
  model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=trees,max_depth=depth,num_leaves=2**depth,learning_rate=.03,min_child_samples=50,verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True);model.fit(xx[ids],(yv-bp)[ids]);pred=np.maximum(bp[sel]+model.predict(xx[sel]),0);v=wape(yv[sel],pred);grid.append({'depth':depth,'trees':trees,'wape':v});model.booster_.save_model(str(out/f'candidate_{depth}_{trees}.txt'))
  if v<best['wape']:best={'kind':'gbm','depth':depth,'trees':trees,'wape':v}
  print(tag,'noq',depth,trees,v,flush=True)
dump(out/'SELECTION.json',{'created_utc':now(),'selected':best,'grid':grid,'fit_max_target':str(d[mask].day.max()),'selection_min_target':str(d[~mask].day.min())})
p=fit_controls(d);bp=apply_controls(d,p)['base'];xx=np.column_stack([x,bp]).astype(np.float32);ids=np.flatnonzero(ok);rng=np.random.default_rng(SEED)
if len(ids)>400000:ids=np.sort(rng.choice(ids,400000,replace=False))
model=None
if best['kind']!='parent':
 model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=best['trees'],max_depth=best['depth'],num_leaves=2**best['depth'],learning_rate=.03,min_child_samples=50,verbosity=-1,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True);model.fit(xx[ids],(yv-bp)[ids]);model.booster_.save_model(str(out/'FINAL_MODEL.txt'))
dump(out/'FREEZE.json',{'created_utc':now(),'parent_scales':p['risk']['category_scales'],'selected':best});del x,xx;gc.collect()
if tag=='m5':td,tx=load(ROOT/'evidence/cell_audit/m5/later_predictions.npz')
b=apply_controls(td,p)['base'];f=b if model is None else np.maximum(b+model.predict(np.column_stack([tx,b]).astype(np.float32)),0);yp=td.y.to_numpy(float);result={}
for pop,m in [('eligible',td.available.to_numpy(bool)),('all',np.ones(len(td),bool))]:
 result[pop]=metrics(yp,f,m)
 if tag=='fresh':result[pop]['noq_minus_q']={u:interval(td,f,td['stack'].to_numpy(),m,u) for u in ['series','store','product','day']}
rows=pd.DataFrame({'series':td.series,'mass':yp,'noq_error':np.abs(yp-f)}).groupby('series',sort=True).sum();rows.to_csv(out/'ITEM_STATISTICS.csv');
if tag=='m5':
 q=pd.read_csv(prev/'m5_ITEM_STATISTICS.csv').set_index('series');delta=rows.noq_error-q.stack_error;mass=rows.mass;rng=np.random.default_rng(SEED);v=[]
 for _ in range(40):
  w=rng.multinomial(len(rows),np.full(len(rows),1/len(rows)),size=100);v.extend(w@delta/(w@mass))
 result['noq_minus_q']={'estimate':float(delta.sum()/mass.sum()),'ci95':np.quantile(v,[.025,.975]).tolist()}
else:
 pd.DataFrame({'series':td.series,'day':td.day,'noq':f}).to_csv(out/'EVAL_PREDICTIONS.csv.gz',index=False)
dump(out/'RESULTS.json',result);dump(out/'COMPLETE.json',{'finished_utc':now(),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}});print(tag,'NOQ COMPLETE',result,flush=True)
