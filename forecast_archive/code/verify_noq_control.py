from verify_forward_controls import *
# The imported verifier first rechecks the full-q controls (104 checks).
for tag in ['fresh','m5']:
 p=out/('noq_'+tag);z=json.loads((p/'COMPLETE.json').read_text());proto=json.loads((p/'PROTOCOL.json').read_text());ck('noq '+tag+' script',digest(R/'code/run_noq_control.py')==proto['script_sha256'])
 for n,h in z['files'].items():ck('noq '+tag+'/'+n,digest(p/n)==h)
 s=json.loads((p/'SELECTION.json').read_text());ck('noq '+tag+' chronology',s['fit_max_target']<s['selection_min_target']);ck('noq '+tag+' selection',s['selected']['wape']==min(t['wape'] for t in s['grid']))
p=out/'noq_fresh';d=pd.read_csv(out/'fresh/fresh_EVAL_PREDICTIONS.csv.gz');d['raw']=d.raw.astype(np.float32).astype(float);f=pd.read_csv(p/'EVAL_PREDICTIONS.csv.gz').noq.to_numpy();xx=np.load(out/'fresh/fresh_EVAL_FEATURES.npz')['x'];model=lgb.Booster(model_file=str(p/'FINAL_MODEL.txt'));regen=np.maximum(d.base.to_numpy()+model.predict(np.column_stack([xx,d.base]).astype(np.float32),num_threads=4),0);ck('noq fresh model replay',np.max(abs(regen-f))<1e-10)
r=json.loads((p/'RESULTS.json').read_text());y=d.y.to_numpy()
for pop,mask in [('eligible',d.available.to_numpy(bool)),('all',np.ones(len(d),bool))]:
 ck('noq fresh '+pop+' metric',abs(np.abs(y[mask]-f[mask]).sum()/y[mask].sum()-r[pop]['wape'])<1e-12)
 for unit,saved in r[pop]['noq_minus_q'].items():
  t=pd.DataFrame({'u':d[unit].astype(str),'mass':np.where(mask,y,0),'err':np.where(mask,np.abs(y-f)-np.abs(y-d['stack']),0)}).groupby('u',sort=True).sum();rng=np.random.default_rng(saved['seed']);vals=[]
  for _ in range(40):
   w=rng.multinomial(len(t),np.full(len(t),1/len(t)),size=100);vals.extend(w@t.err/(w@t.mass))
  ck('noq fresh '+pop+' '+unit,np.max(abs(np.quantile(vals,[.025,.975])-saved['ci95']))<1e-12)
p=out/'noq_m5';r=json.loads((p/'RESULTS.json').read_text());s=pd.read_csv(p/'ITEM_STATISTICS.csv');ck('noq m5 stats',abs(s.noq_error.sum()/s.mass.sum()-r['all']['wape'])<1e-12)
# Original panel, builder, hierarchy and days remain from full-q verification.
zz=np.load(R/'evidence/cell_audit/m5/later_predictions.npz');pp=json.loads((p/'FREEZE.json').read_text())['parent_scales'];model=lgb.Booster(model_file=str(p/'FINAL_MODEL.txt'));err=0.;mass=0.
for start in range(0,len(days),7):
 ds=days[start:start+7];ss=np.repeat(np.arange(panel.n_series),len(ds));dd=np.tile(ds,panel.n_series);bb=zz['raw'][:,start:start+len(ds)].ravel()*np.array([pp[str(c)] for c in cats[ss]]);xx=np.column_stack([builder.make_features(ss,dd,include_censor=True),hier.make(ss,dd),bb]).astype(np.float32);pred=np.maximum(bb+model.predict(xx,num_threads=4),0);yy=panel.truth[ss,dd-1];err+=np.abs(yy-pred).sum();mass+=yy.sum()
ck('noq m5 full replay',abs(err/mass-r['all']['wape'])<1e-12)
(R/'verification/FORWARD_AND_NOQ.json').write_text(json.dumps({'status':'PASS','count':len(checks),'checks':checks},indent=2)+'\n');print('FULL PASS',len(checks))
