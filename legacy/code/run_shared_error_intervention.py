"""One matched shared error-head intervention on consumed development data.

No guardian/external loader or cache is used. The original 600,000 rows, labels,
21 base features, tree capacity and frozen point/demand objects are held fixed;
only eight origin-safe intermittency features are added to one new error fit.
"""
from pathlib import Path
import argparse, copy, hashlib, json, time
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import lightgbm as lgb
from r2_io import write, dump, sha
from run_hobbies_intervention import (ROOT, MODELS, SEED, CAP, BLOCKS, EXTRA,
    PARAMS, ExtraBuilder, invariance_test, load, save_npz, origin, stats,
    summarize_prefix)
from run_review2_objectives import curve, select_threshold

PRIOR = ROOT/'results/error_interventions'
RAW_HASH='5d63b6aff7854ed1811f6b5bffd2f397a09d671c6754e0f1345e2313b35e48fd'
METHOD='shared_error_29'
FIT_PARAMS=PARAMS | {'alpha': .85}

def now(): return datetime.now(timezone.utc).isoformat()
def ahash(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    path=out/'PROTOCOL.json'
    if path.exists():
        p=json.loads(path.read_text())
        assert p['source_sha256']==sha(__file__),'Frozen executable changed'
        assert p['parameters']==FIT_PARAMS and p['extra_features']==EXTRA
        return p
    p={'schema':'shared-error-intervention-1','status':'FROZEN_BEFORE_FEATURE_RECONSTRUCTION_AND_FIT',
       'frozen_utc':now(),'scope':'POST_HOC_CONSUMED_DEVELOPMENT_ONLY','seed':SEED,
       'new_fits':1,'new_method':METHOD,'retained_comparators':['shared_error_21','hobbies_error_21','hobbies_error_29'],
       'sample':'Exact original first-seed np.random.default_rng(20260906).choice(18290*118,600000,replace=False); all categories retained, in original order',
       'training_days':[1436,1553],'training_rows':600000,'parameters':FIT_PARAMS,
       'categorical_columns':[17,18,19,20],'base_feature_count':21,'extra_features':EXTRA,
       'target':'abs(float64(pre-mechanical-censoring truth) - frozen point forecast)',
       'point_demand_cap_unchanged':True,'cap':CAP,'row_floor':.35,
       'score':'float32 clip(error,0) - r * float32 clip(frozen demand,0)',
       'evaluation_population':'Hobbies series only, all three development blocks',
       'evaluation_blocks':{k:list(v) for k,v in BLOCKS.items()},
       'threshold_selection':'Largest common score threshold satisfying WAPE<=cap and row coverage>=.35 in both development A/B; if absent accept no rows',
       'diagnostics':['all-row error-head MSE','positive-truth-day error-head MSE','mean error bias','fixed A/B-threshold metrics','outcome-informed blockwise largest feasible prefix'],
       'prefix_scope':'Descriptive and outcome-informed; neither deployed nor independently validated',
       'selection_before_later_block_metrics':True,'point_forecast_fit':False,
       'guardian_inputs_read':False,'external_inputs_read':False,'certificate_issued':False,
       'raw_development_hash':RAW_HASH,'source_sha256':sha(__file__),
       'source_dependencies':{x:sha(ROOT/x) for x in ['code/run_hobbies_intervention.py','code/run_selective_study.py','code/run_review2_objectives.py']},
       'prior_evidence_hashes':{x:sha(PRIOR/x) for x in ['PROTOCOL.json','RESULTS.json','cache/READY.json','cache/replayed_forecasts.npz','cache/train.npz','cache/calibration_a.npz','cache/calibration_b.npz','cache/shadow.npz']},
       'frozen_model_hashes':{k:sha(ROOT/v) for k,v in MODELS.items()},
       'runtime':{k:version(k) for k in ['numpy','scipy','scikit-learn','lightgbm','joblib','pandas']}}
    dump(path,p)
    print('PROTOCOL_FROZEN',p['frozen_utc'],sha(path),flush=True)
    return p

def reconstruct(data,out,p):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    ready=cache/'READY.json'
    if ready.exists():
        r=json.loads(ready.read_text());assert r['protocol_sha256']==sha(out/'PROTOCOL.json')
        assert r['training_cache_sha256']==sha(cache/'train.npz')
        return cache,r
    raw=data/'design_outcomes_v0_5.npz';assert sha(raw)==RAW_HASH
    assert sha(PRIOR/'cache/replayed_forecasts.npz')==p['prior_evidence_hashes']['cache/replayed_forecasts.npz']
    with np.load(raw,allow_pickle=False) as z:
        assert int(z['day_start'][0])==1 and int(z['day_end'][0])==1913
        observed=z['observed'].astype(np.float32);truth=z['truth'].astype(np.float32)
        censored=z['censored'].astype(np.float32)
        meta={k:z[k].astype(str) for k in ['cat_id','dept_id','store_id','state_id']}
    assert observed.shape==(18290,1913)
    forecasts=load(PRIOR/'cache/replayed_forecasts.npz')
    days=np.arange(1436,1554);idx=np.random.default_rng(SEED).choice(18290*118,600000,replace=False)
    s=idx//118;d=days[idx%118];o=origin(d);j=d-1314;i=o-1313
    # Reconstruct only the requested 600k rows, reusing the retained full-series
    # forecast history needed by the original origin-aligned pooled prior.
    allo=observed[:,1313:1913]
    ce=np.pad(np.cumsum(np.abs(allo.astype(float)-forecasts['proposal']),axis=1),((0,0),(1,0)))
    cd=np.pad(np.cumsum(allo.astype(float),axis=1),((0,0),(1,0)))
    gp=(ce.sum(0)+30*.75)/(cd.sum(0)+30)
    prior=(ce[s,i]+30*gp[i])/(cd[s,i]+30)
    del ce,cd
    oc=np.pad(np.cumsum(observed,dtype=float,axis=1),((0,0),(1,0)))
    means={w:(oc[s,o]-oc[s,o-w])/w for w in [7,28,56]}
    del oc
    oc2=np.pad(np.cumsum(observed*observed,dtype=float,axis=1),((0,0),(1,0)))
    std=np.sqrt(np.maximum((oc2[s,o]-oc2[s,o-28])/28-means[28]**2,0));del oc2
    zc=np.pad(np.cumsum(observed==0,axis=1),((0,0),(1,0)))
    zero=(zc[s,o]-zc[s,o-28])/28;del zc
    cc=np.pad(np.cumsum(censored,axis=1),((0,0),(1,0)))
    cr7=(cc[s,o]-cc[s,o-7])/7;cr28=(cc[s,o]-cc[s,o-28])/28;del cc,censored
    vals={k:forecasts[k][s,j] for k in forecasts}
    cols=[np.log1p(vals[k]) for k in ['baseline','proposal','raw_poisson_histgb','censored_poisson_em']]
    cols += [vals['censor_probability'],np.maximum(vals['proposal']-vals['baseline'],0),prior,d-o,means[7],means[28],means[56],std,zero,cr7,cr28,observed[s,d-8],observed[s,d-15]]
    codes={k:np.unique(meta[k],return_inverse=True)[1].astype(np.float32) for k in ['cat_id','dept_id','store_id','state_id']}
    cols += [codes[k][s] for k in codes]
    x=np.column_stack(cols).astype(np.float32)
    extra=ExtraBuilder(observed).make(s,d)
    y=truth[s,d-1];f=vals['proposal'];hob=meta['cat_id'][s]=='HOBBIES'
    old=load(PRIOR/'cache/train.npz')
    comparisons={}
    for k,v in {'x':x,'extra':extra,'truth':y,'f':f,'series':s,'days':d,'original_flat_index':idx}.items():
        assert np.array_equal(v[hob],old[k]),f'Prior Hobbies training mismatch: {k}'
        comparisons[k]='BITWISE_IDENTICAL_ON_107384_HOBBIES_ROWS'
    assert int(hob.sum())==107384
    save_npz(cache/'train.npz',x=x,extra=extra,truth=y,f=f,series=s,days=d,original_flat_index=idx,hobbies_mask=hob)
    r={'status':'READY','protocol_sha256':sha(out/'PROTOCOL.json'),'training_rows':len(idx),'hobbies_training_rows':int(hob.sum()),
       'training_indices_sha256':ahash(idx),'base_features_sha256':ahash(x),'extra_features_sha256':ahash(extra),
       'target_sha256':ahash(np.abs(y.astype(float)-f)),'training_cache_sha256':sha(cache/'train.npz'),
       'prior_hobbies_comparisons':comparisons,'feature_invariance':invariance_test(),
       'historical_point_forecast_fits':0,'historical_forecast_replay_performed':False,
       'guardian_inputs_read':False,'external_inputs_read':False}
    dump(ready,r);print('MATCHED_TRAINING_READY',len(idx),int(hob.sum()),flush=True)
    return cache,r

def run(data,out):
    p=freeze(out)
    for rel,h in p['source_dependencies'].items():assert sha(ROOT/rel)==h
    for rel,h in p['prior_evidence_hashes'].items():assert sha(PRIOR/rel)==h
    cache,reconstruction=reconstruct(data,out,p)
    modeldir=out/'models';modeldir.mkdir(exist_ok=True)
    mp=modeldir/(METHOD+'.txt');receipt=mp.with_suffix('.json')
    newfits=0
    if receipt.exists():
        fit=json.loads(receipt.read_text());assert fit['protocol_sha256']==sha(out/'PROTOCOL.json');assert fit['model_sha256']==sha(mp)
        model=lgb.Booster(model_file=str(mp))
    else:
        assert not mp.exists(),'Model without receipt: do not silently refit'
        tr=load(cache/'train.npz');x=np.column_stack([tr['x'],tr['extra']]);target=np.abs(tr['truth'].astype(float)-tr['f'])
        assert x.shape==(600000,29) and np.isfinite(x).all()
        m=lgb.LGBMRegressor(**FIT_PARAMS);started=now();t=time.time()
        m.fit(x,target,categorical_feature=[17,18,19,20]);write(mp,m.booster_.model_to_string().encode())
        model=m.booster_;newfits=1
        fit={'protocol_sha256':sha(out/'PROTOCOL.json'),'model_sha256':sha(mp),'training_rows':len(target),'feature_count':29,'fit_started_utc':started,'fit_completed_utc':now(),'fit_seconds':time.time()-t,'fit_count':1}
        dump(receipt,fit);del tr,x,target,m
        print('SHARED_29_FITTED',fit,flush=True)
    demand=lgb.Booster(model_file=str(ROOT/MODELS['demand']))
    baseline=lgb.Booster(model_file=str(ROOT/MODELS['error']))
    old=json.loads((PRIOR/'RESULTS.json').read_text())
    records=copy.deepcopy(old['records']);blockdata={};preds={};baseline_checks={}
    for k in BLOCKS:
        v=load(PRIOR/'cache'/(k+'.npz'));blockdata[k]=v
        x=np.column_stack([v['x'],v['extra']])
        e=np.maximum(model.predict(x,num_threads=4).astype(np.float32),0)
        mu=np.maximum(demand.predict(v['x'],num_threads=4).astype(np.float32),0)
        preds[k]={'error':e,'demand':mu,'score':e-CAP*mu}
        oldpred=load(PRIOR/'cache'/f'predictions_shared_error_21_{k}.npz')
        ebase=np.maximum(baseline.predict(v['x'],num_threads=4).astype(np.float32),0)
        assert np.array_equal(ebase,oldpred['error']) and np.array_equal(mu,oldpred['demand'])
        priorrec=next(x for x in old['records'] if x['method']=='shared_error_21')['blocks'][k]['head']
        de=ebase.astype(float)-np.abs(v['truth'].astype(float)-v['f'])
        assert np.isclose(np.mean(de*de),priorrec['error_mse'],rtol=1e-14)
        assert np.isclose(de.mean(),priorrec['error_bias'],rtol=1e-14)
        baseline_checks[k]='BITWISE_IDENTICAL_ERROR_AND_DEMAND_PREDICTIONS_AND_MATCHED_MSE_BIAS'
        save_npz(cache/f'predictions_{METHOD}_{k}.npz',**preds[k])
    cs={k:curve(preds[k]['score'],blockdata[k]['truth'],blockdata[k]['f']) for k in ['calibration_a','calibration_b']}
    threshold,obj=select_threshold(cs['calibration_a'],cs['calibration_b'],'row')
    rec={'method':METHOD,'selected_threshold':threshold,'minimum_calibration_row_coverage':obj,'feature_count':29,'training_rows':600000,'model_sha256':sha(mp),'blocks':{}}
    dump(out/'SELECTED_POLICY.json',{'scope':'A_B_SELECTION_COMPLETED_BEFORE_LATER_BLOCK_METRICS','selection_utc':now(),'protocol_sha256':sha(out/'PROTOCOL.json'),'record':rec})
    for k,v in blockdata.items():
        pp=preds[k];err=np.abs(v['truth'].astype(float)-v['f']);de=pp['error'].astype(float)-err;positive=v['truth']>0
        a=np.zeros(len(err),bool) if threshold is None else pp['score']<=threshold
        cc=curve(pp['score'],v['truth'],v['f'])
        rec['blocks'][k]={'policy':stats(v['truth'],v['f'],a),'head':{'error_mse':float(np.mean(de*de)),'positive_day_error_mse':float(np.mean(de[positive]**2)),
          'error_bias':float(de.mean()),'positive_day_n':int(positive.sum()),'n':len(err),'squared_error_sum':float(np.sum(de*de)),'positive_day_squared_error_sum':float(np.sum(de[positive]**2))},
          'samplewise_prefix_diagnostic':summarize_prefix(cc)}
    for x in records:x['training_rows']=600000 if x['method']=='shared_error_21' else 107384
    records.append(rec)
    r={'status':'COMPLETE','scope':p['scope'],'protocol_sha256':sha(out/'PROTOCOL.json'),'total_new_fits':1,'fits_this_invocation':newfits,'records':records,
       'reconstruction':reconstruction,'baseline_replay':baseline_checks,'fit_receipt_sha256':sha(receipt),'prior_results_sha256':sha(PRIOR/'RESULTS.json'),
       'frozen_models_unchanged':all(sha(ROOT/MODELS[k])==h for k,h in p['frozen_model_hashes'].items()),'guardian_inputs_read':False,'external_inputs_read':False,'certificate_issued':False,
       'interpretation':'The matched shared-head comparison isolates adding eight features at fixed training population, target and capacity. It does not test a changed point forecast, establish a point-forecast bottleneck, or identify the cause of any MSE change.'}
    assert r['frozen_models_unchanged'];dump(out/'RESULTS.json',r)
    print('SHARED_ERROR_INTERVENTION_COMPLETE',json.dumps(rec),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path);ap.add_argument('--out',type=Path,default=ROOT/'results/shared_error_control');ap.add_argument('--freeze-only',action='store_true');a=ap.parse_args()
    if a.freeze_only:freeze(a.out)
    else:
        if a.data is None:ap.error('--data is required')
        run(a.data,a.out)
