"""Two frozen, development-only Hobbies error-head interventions.

Only the already consumed design shard is accepted. No guardian or external
loader is imported. Original point/error/demand objects are read-only. The
specialists use the Hobbies subset of the original first-seed 600,000 risk rows.
"""
from pathlib import Path
import argparse, io, json, sys, time
from importlib.metadata import version
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/design'))
from censorcast_v05.features import DesignPanel,FeatureBuilder
from r2_io import write,dump,sha
from run_selective_study import origin,R as CAP
from run_review2_objectives import curve,select_threshold
SEED=20260906
BLOCKS={'calibration_a':(1555,1673),'calibration_b':(1674,1793),'shadow':(1800,1913)}
INPUT_HASHES={'design_outcomes_v0_5.npz':'5d63b6aff7854ed1811f6b5bffd2f397a09d671c6754e0f1345e2313b35e48fd','calendar.csv':'d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5','sell_prices.csv':'9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef'}
MODELS={'bundle':'inputs/forecast_bundle_v0_5.joblib','point':'results/point_baselines/observed_l1_s20260906.txt','error':'results/selective_strong/mean_error_s20260906.txt','demand':'results/review_revision/demand_s20260906.txt'}
EXTRA=[f'{name}_{w}' for w in [56,112] for name in ['observed_positive_count','observed_positive_mean','observed_positive_cv2','observed_positive_recency']]
PARAMS=dict(objective='regression',n_estimators=220,num_leaves=31,learning_rate=.05,min_child_samples=100,reg_lambda=3,n_jobs=4,random_state=SEED,deterministic=True,force_col_wise=True,verbosity=-1)

def save_npz(p,**v):
    b=io.BytesIO();np.savez_compressed(b,**v);write(p,b.getvalue())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    srcs=['code/run_hobbies_intervention.py','code/run_selective_study.py','code/run_review2_objectives.py','upstream/design/censorcast_v05/features.py','upstream/design/censorcast_v05/models.py']
    spec={'schema':'hobbies-prescription-1','status':'FROZEN_BEFORE_REPLAY_AND_FITS','scope':'POST_HOC_CONSUMED_DEVELOPMENT_ONLY','seed':SEED,'cap':CAP,'row_floor':.35,
      'methods':['shared_error_21','hobbies_error_21','hobbies_error_29'],'new_fits':2,'fixed_score':'clip(error,0)-r*clip(frozen_demand,0)',
      'parameters':PARAMS,'categorical_columns':[17,18,19,20],'training_days':[1436,1553],
      'sample':'np.random.default_rng(20260906).choice(18290*118,600000,replace=False); retain rows whose series category is HOBBIES; identical indices and labels for both new fits',
      'target':'abs(pre-mechanical-censoring truth - frozen adapted point forecast); squared-error loss',
      'point_forecast':'first-seed observed-sales L1 plus frozen alpha=2 gamma=2 censor adapter',
      'base_features':'exact original 21 origin-aligned risk features; inherited historical strict-censor flags retained',
      'extra_features':EXTRA,'extra_conventions':'observed history d_(origin-window+1) through d_origin only; positive mean and population CV2 are zero for no positive observations; recency=origin-last_positive_day, capped at window and equal to window if none',
      'threshold_selection':'largest common score threshold satisfying realized WAPE<=cap and rows>=.35 in BOTH development A and B; no threshold means empty acceptance in all blocks',
      'evaluation_blocks':{k:list(v) for k,v in BLOCKS.items()},
      'retained_diagnostics':['all-row error-head MSE','positive-truth-day error-head MSE','mean error bias','A/B transferred policy metrics','development-shadow exact-score-prefix cap feasibility'],
      'prefix_diagnostic':'each block may use its own truth only to describe the largest feasible prefix; not a deployable or independently validated threshold; all three methods retained',
      'selection_before_shadow_metrics':True,'guardian_inputs_read':False,'external_inputs_read':False,'certificate_issued':False,
      'input_hashes':INPUT_HASHES,'frozen_model_hashes':{k:sha(ROOT/v) for k,v in MODELS.items()},'source_hashes':{s:sha(ROOT/s) for s in srcs},
      'runtime':{p:version(p) for p in ['numpy','scipy','scikit-learn','lightgbm','joblib','pandas']}}
    p=out/'PROTOCOL.json'
    if p.exists():assert json.loads(p.read_text())==spec,'Frozen intervention protocol changed'
    else:dump(p,spec)
    return spec

class ExtraBuilder:
    def __init__(self,observed):
        x=np.asarray(observed,dtype=np.float64);positive=x>0
        self.n=np.pad(np.cumsum(positive,axis=1,dtype=np.int32),((0,0),(1,0)))
        self.s=np.pad(np.cumsum(x,axis=1),((0,0),(1,0)))
        self.ss=np.pad(np.cumsum(x*x,axis=1),((0,0),(1,0)))
        self.last=np.maximum.accumulate(np.where(positive,np.arange(1,x.shape[1]+1)[None,:],0),axis=1)
    def make(self,series,days):
        s=np.asarray(series);o=origin(np.asarray(days));cols=[]
        assert np.all(o>=112)
        for w in [56,112]:
            n=self.n[s,o]-self.n[s,o-w];summ=self.s[s,o]-self.s[s,o-w];ss=self.ss[s,o]-self.ss[s,o-w]
            mean=np.divide(summ,n,out=np.zeros(len(s),float),where=n>0)
            m2=np.divide(ss,n,out=np.zeros(len(s),float),where=n>0)
            cv=np.divide(np.maximum(m2-mean*mean,0),mean*mean,out=np.zeros(len(s),float),where=mean>0)
            rec=np.minimum(o-self.last[s,o-1],w)
            cols += [n,mean,cv,rec]
        return np.column_stack(cols).astype(np.float32)

def invariance_test():
    rng=np.random.default_rng(314159);obs=rng.poisson(.3,(9,230)).astype(float)
    days=np.full(9,211);o=int(origin(days)[0]);s=np.arange(9)
    ref=ExtraBuilder(obs).make(s,days);changed=obs.copy();changed[:,o:]+=1000
    assert np.array_equal(ref,ExtraBuilder(changed).make(s,days))
    changed=obs.copy();changed[0,o-1]+=1000
    assert not np.array_equal(ref[0],ExtraBuilder(changed).make(s,days)[0])
    zero=ExtraBuilder(np.zeros_like(obs)).make(s,days)
    assert np.array_equal(zero,np.tile([0,0,0,56,0,0,0,112],(9,1)))
    # Direct window calculations independently establish positive-only CV2 semantics.
    for i in s:
      for j,w in enumerate([56,112]):
        win=obs[i,o-w:o];pos=win[win>0];expected=[len(pos),pos.mean() if len(pos) else 0,pos.var()/pos.mean()**2 if len(pos) else 0,min(o-(np.flatnonzero(obs[i,:o]>0)[-1]+1),w) if np.any(obs[i,:o]>0) else w]
        assert np.allclose(ref[i,4*j:4*j+4],expected,atol=1e-6,rtol=1e-6)
    return {'future_observation_perturbation':'PASS','at_origin_negative_control':'PASS','zero_history_conventions':'PASS','direct_window_comparisons':18}

def replay(data,out,protocol):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/'READY.json').exists():
        r=json.loads((cache/'READY.json').read_text());assert r['protocol_sha256']==sha(out/'PROTOCOL.json');return cache,r
    for n,h in INPUT_HASHES.items():assert sha(data/n)==h,n
    # Loading is restricted by exact hashes and the development-only loader day range.
    panel=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv')
    assert panel.n_series==18290
    hob=panel.metadata['cat_id']=='HOBBIES';hs=np.flatnonzero(hob)
    forecast_names=['baseline','proposal','raw_poisson_histgb','censored_poisson_em','censor_probability']
    fp=cache/'replayed_forecasts.npz'
    if fp.exists():forecasts=load(fp)
    else:
        import joblib
        from threadpoolctl import threadpool_limits
        b=FeatureBuilder(panel);bundle=joblib.load(ROOT/MODELS['bundle']);point=lgb.Booster(model_file=str(ROOT/MODELS['point']))
        forecasts={k:np.empty((panel.n_series,600),np.float32) for k in forecast_names};days=np.arange(1314,1914,dtype=np.int32)
        with threadpool_limits(limits=4):
          for st in range(0,600,7):
            dd=days[st:st+7];s=np.repeat(np.arange(panel.n_series),len(dd));d=np.tile(dd,panel.n_series)
            x=b.make_features(s,d,include_censor=False);cx=b.make_features(s,d,include_censor=True)
            f=np.maximum(point.predict(x,num_threads=4),0).reshape(panel.n_series,-1).astype(np.float32)
            values=bundle.predict(x,cx);forecasts['baseline'][:,st:st+len(dd)]=f
            for k,v in values.items():forecasts[k][:,st:st+len(dd)]=v.reshape(panel.n_series,-1)
            if st%70==0:print('Historical development replay',st,'/600',flush=True)
        forecasts['proposal']=forecasts['baseline']+2*forecasts['censor_probability']**2*np.maximum(forecasts['censored_poisson_em']-forecasts['baseline'],0)
        save_npz(fp,**forecasts)
        del b,bundle,point
    replay_report={}
    for key in BLOCKS:
        retained=load(ROOT/'results/review2/cache_design'/f'{key}_aligned.npz');idx=retained['target_days']-1314
        for name in ['baseline','proposal']:
            assert np.array_equal(retained[name],forecasts[name][:,idx]),f'Original forecast mismatch {key}/{name}'
        assert np.array_equal(retained['truth'],panel.truth[:,retained['target_days']-1]),key
        replay_report[key]='BITWISE_IDENTICAL_POINT_FORECASTS_AND_TRUTH'
    # All-series observed error history is needed by the original empirical-Bayes prior.
    p=forecasts['proposal'];obs=panel.observed.astype(np.float32,copy=False);cen=panel.censored.astype(np.float32,copy=False)
    allo=obs[:,1313:1913];ce=np.pad(np.cumsum(np.abs(allo.astype(float)-p),axis=1),((0,0),(1,0)));cd=np.pad(np.cumsum(allo.astype(float),axis=1),((0,0),(1,0)))
    gp=(ce.sum(0)+30*.75)/(cd.sum(0)+30)
    oc=np.pad(np.cumsum(obs,dtype=float,axis=1),((0,0),(1,0)));oc2=np.pad(np.cumsum(obs*obs,dtype=float,axis=1),((0,0),(1,0)))
    zc=np.pad(np.cumsum(obs==0,axis=1),((0,0),(1,0)));cc=np.pad(np.cumsum(cen,axis=1),((0,0),(1,0)))
    def make(s,d):
        o=origin(d);j=d-1314;i=o-1313;prior=(ce[s,i]+30*gp[i])/(cd[s,i]+30)
        vals={k:forecasts[k][s,j] for k in forecast_names};means={w:(oc[s,o]-oc[s,o-w])/w for w in [7,28,56]}
        cols=[np.log1p(vals[k]) for k in ['baseline','proposal','raw_poisson_histgb','censored_poisson_em']]
        cols += [vals['censor_probability'],np.maximum(vals['proposal']-vals['baseline'],0),prior,d-o,means[7],means[28],means[56],np.sqrt(np.maximum((oc2[s,o]-oc2[s,o-28])/28-means[28]**2,0)),(zc[s,o]-zc[s,o-28])/28,(cc[s,o]-cc[s,o-7])/7,(cc[s,o]-cc[s,o-28])/28,obs[s,d-8],obs[s,d-15]]
        cols += [panel.static_codes[k][s] for k in ['cat_id','dept_id','store_id','state_id']]
        return np.column_stack(cols).astype(np.float32)
    extra=ExtraBuilder(panel.observed[hs]);full_to_hob=np.full(panel.n_series,-1,int);full_to_hob[hs]=np.arange(len(hs))
    days=np.arange(1436,1554);idx=np.random.default_rng(SEED).choice(panel.n_series*len(days),600000,replace=False);s=idx//len(days);d=days[idx%len(days)];keep=hob[s];original_idx=idx[keep];s=s[keep];d=d[keep]
    x=make(s,d);xx=extra.make(full_to_hob[s],d);y=panel.truth[s,d-1];f=forecasts['proposal'][s,d-1314]
    save_npz(cache/'train.npz',x=x,extra=xx,truth=y,f=f,original_flat_index=original_idx,series=s,days=d)
    del x,xx
    for key,(lo,hi) in BLOCKS.items():
        days=np.arange(lo,hi+1);s=np.repeat(hs,len(days));d=np.tile(days,len(hs));x=make(s,d);xx=extra.make(full_to_hob[s],d)
        save_npz(cache/(key+'.npz'),x=x,extra=xx,truth=panel.truth[s,d-1],f=forecasts['proposal'][s,d-1314],series=s,days=d)
    report={'status':'READY','protocol_sha256':sha(out/'PROTOCOL.json'),'original_forecast_replay':replay_report,'hobbies_series':len(hs),'training_rows':len(original_idx),'training_indices_sha256':__import__('hashlib').sha256(original_idx.tobytes()).hexdigest(),'feature_invariance':invariance_test(),'files':{p.name:sha(p) for p in cache.iterdir() if p.is_file()},'guardian_inputs_read':False,'external_inputs_read':False}
    dump(cache/'READY.json',report);return cache,report

def stats(y,f,a):
    y=np.asarray(y,float);f=np.asarray(f,float);e=np.abs(y-f);mass=float(y[a].sum());n=int(a.sum())
    return {'n':len(y),'accepted_n':n,'row_coverage':n/len(y),'demand_coverage':mass/float(y.sum()),'error_mass':float(e[a].sum()),'demand_mass':mass,'total_demand':float(y.sum()),'wape':float(e[a].sum()/mass) if mass else None}

def summarize_prefix(c):
    good=np.flatnonzero(c['feasible'])
    if not len(good):return {'row_coverage':0.,'demand_coverage':0.,'wape':None,'threshold':None}
    i=good[-1];return {k:float(c[k][i]) for k in ['row','demand','wape','threshold']}|{'row_coverage':float(c['row'][i]),'demand_coverage':float(c['demand'][i])}

def run(data,out):
    protocol=freeze(out);cache,reconstruction=replay(data,out,protocol)
    modeldir=out/'models';modeldir.mkdir(exist_ok=True);tr=load(cache/'train.npz');target=np.abs(tr['truth'].astype(float)-tr['f'])
    models={'shared_error_21':lgb.Booster(model_file=str(ROOT/MODELS['error']))};newfits=0
    for name,aug in [('hobbies_error_21',False),('hobbies_error_29',True)]:
        p=modeldir/(name+'.txt');rr=p.with_suffix('.json')
        if rr.exists():
            r=json.loads(rr.read_text());assert sha(p)==r['model_sha256'];assert r['protocol_sha256']==sha(out/'PROTOCOL.json');models[name]=lgb.Booster(model_file=str(p));continue
        x=np.column_stack([tr['x'],tr['extra']]) if aug else tr['x'];m=lgb.LGBMRegressor(**PARAMS)
        st=time.time();m.fit(x,target,categorical_feature=[17,18,19,20]);write(p,m.booster_.model_to_string().encode());models[name]=m.booster_;newfits+=1
        dump(rr,{'model_sha256':sha(p),'protocol_sha256':sha(out/'PROTOCOL.json'),'training_rows':len(target),'num_features':x.shape[1],'fit_seconds':time.time()-st,'target':'abs(truth-f)','labels':'pre-mechanical-censoring design truth'})
        print('Fitted',name,len(target),flush=True)
    demand=lgb.Booster(model_file=str(ROOT/MODELS['demand']));data_by_block={k:load(cache/(k+'.npz')) for k in BLOCKS};records=[]
    predictions={}
    for name,m in models.items():
        predictions[name]={}
        for k,v in data_by_block.items():
            x=np.column_stack([v['x'],v['extra']]) if name=='hobbies_error_29' else v['x'];e=np.maximum(m.predict(x,num_threads=4).astype(np.float32),0)
            mu=np.maximum(demand.predict(v['x'],num_threads=4).astype(np.float32),0);s=e-CAP*mu
            predictions[name][k]={'error':e,'demand':mu,'score':s}
        cs={k:curve(predictions[name][k]['score'],data_by_block[k]['truth'],data_by_block[k]['f']) for k in ['calibration_a','calibration_b']}
        t,obj=select_threshold(cs['calibration_a'],cs['calibration_b'],'row')
        records.append({'method':name,'selected_threshold':t,'minimum_calibration_row_coverage':obj,'feature_count':29 if name=='hobbies_error_29' else 21,'model_sha256':sha(ROOT/MODELS['error']) if name=='shared_error_21' else sha(modeldir/(name+'.txt'))})
    dump(out/'SELECTED_POLICIES.json',{'scope':'A_B_SELECTION_COMPLETED_BEFORE_SHADOW_METRICS','protocol_sha256':sha(out/'PROTOCOL.json'),'records':records})
    # Outcomes below report all prespecified methods; none is chosen on shadow.
    for rec in records:
        name=rec['method'];rec['blocks']={}
        for k,v in data_by_block.items():
            pp=predictions[name][k];error=np.abs(v['truth'].astype(float)-v['f']);de=pp['error'].astype(float)-error;positive=v['truth']>0
            t=rec['selected_threshold'];a=np.zeros(len(error),bool) if t is None else pp['score']<=t
            cc=curve(pp['score'],v['truth'],v['f']);metric=stats(v['truth'],v['f'],a)
            rec['blocks'][k]={'policy':metric,'head':{'error_mse':float(np.mean(de*de)),'positive_day_error_mse':float(np.mean(de[positive]**2)),'error_bias':float(de.mean()),'positive_day_n':int(positive.sum()),'n':len(error),'squared_error_sum':float(np.sum(de*de)),'positive_day_squared_error_sum':float(np.sum(de[positive]**2))},'samplewise_prefix_diagnostic':summarize_prefix(cc)}
            save_npz(cache/f'predictions_{name}_{k}.npz',**pp)
        print(name,rec['selected_threshold'],rec['blocks']['shadow'],flush=True)
    result={'status':'COMPLETE','scope':protocol['scope'],'protocol_sha256':sha(out/'PROTOCOL.json'),'total_new_fits':2,'fits_this_invocation':newfits,'reconstruction':reconstruction,'records':records,'frozen_models_unchanged':all(sha(ROOT/MODELS[k])==h for k,h in protocol['frozen_model_hashes'].items()),'guardian_inputs_read':False,'external_inputs_read':False,'certificate_issued':False}
    dump(out/'RESULTS.json',result)
    print('HOBBIES_INTERVENTION_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--out',type=Path,default=ROOT/'results/review7_hobbies');p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.freeze_only:freeze(a.out);print('FROZEN',sha(a.out/'PROTOCOL.json'))
    else:
        if a.data is None:p.error('--data required')
        run(a.data,a.out)
