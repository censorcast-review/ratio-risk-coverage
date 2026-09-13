"""Matched observable-input correction and ensemble controls on consumed M5.

Only the design shard is accepted. Validation chooses all correction parameters;
the three evaluation blocks are retrospective. No fresh test is claimed.
"""
from pathlib import Path
import argparse, gc, json, sys, time
from datetime import datetime, timezone
import numpy as np
import lightgbm as lgb
from scipy.stats import poisson
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'m5'))
from legacy_features.features import DesignPanel,FeatureBuilder
from audit_observed_data import sha,write,metrics,rmsse_scale,optimal_scale

SEEDS=[20260906,20260907,20260908]
ALPHAS=[0.,.25,.5,.75,1.,1.5,2.]
POWERS=[.5,1.,2.]
PARAMS=dict(n_estimators=300,num_leaves=63,learning_rate=.05,
            min_child_samples=100,reg_lambda=2.,n_jobs=4,
            verbosity=-1,deterministic=True,force_col_wise=True)

def now():return datetime.now(timezone.utc).isoformat()

def predict_heads(builder,models,days):
    n=builder.panel.n_series
    preds={k:np.empty((n,len(days)),np.float32) for k in models}
    for col in range(0,len(days),14):
        ds=days[col:col+14];s=np.repeat(np.arange(n),len(ds));d=np.tile(ds,n)
        x=builder.make_features(s,d,include_censor=True)
        for k,m in models.items():
            preds[k][:,col:col+len(ds)]=np.maximum(m.predict(x,num_threads=4),0).reshape(n,-1)
    return preds

def candidate(pred,config,cats):
    b=pred['l1'].astype(float)
    if config['kind']=='category_scaled_l1':
        return b*np.array([config['scales'][str(c)] for c in cats])[:,None]
    if config['kind'] in ['l1','scaled_l1']:return config.get('scale',1.)*b
    m=pred[config['mean']].astype(float);h=pred['hit'].astype(float)
    return config['scale']*(b+config['alpha']*h**config['power']*np.maximum(m-b,0))

def select(y,pred,cats):
    b=pred['l1'].astype(float);records=[];chosen={}
    for config in [dict(kind='l1',scale=1.),dict(kind='scaled_l1',scale=optimal_scale(y,b)),
                   dict(kind='category_scaled_l1',scales={str(c):optimal_scale(y[cats==c],b[cats==c]) for c in np.unique(cats)})]:
        r=dict(config=config,wape=metrics(y,candidate(pred,config,cats))['wape']);records.append(r);chosen[config['kind']]=r
    for kind,mean,powers in [('observed_mean_gate','poisson',POWERS),
                             ('censored_mean_gate','em',POWERS),
                             ('censored_mean_ungated','em',[0.])]:
        own=[]
        for alpha in ALPHAS:
            for power in powers:
                unscaled=b+alpha*pred['hit'].astype(float)**power*np.maximum(pred[mean].astype(float)-b,0)
                scale=optimal_scale(y,unscaled)
                config=dict(kind=kind,mean=mean,alpha=alpha,power=power,scale=scale)
                r=dict(config=config,wape=metrics(y,scale*unscaled)['wape']);own.append(r);records.append(r)
        chosen[kind]=min(own,key=lambda r:r['wape'])
    return chosen,records

def information_checks(panel,builder):
    """Features cannot change when hidden truth or the strict flag changes."""
    s=np.arange(100,dtype=np.int32);d=np.full(100,1317,dtype=np.int32)
    old=builder.make_features(s,d,include_censor=True)
    truth=panel.truth;flag=panel.censored
    panel.truth=np.zeros_like(truth);panel.censored=1-flag
    same=builder.make_features(s,d,include_censor=True)
    panel.truth=truth;panel.censored=flag
    assert np.array_equal(old,same)
    # Verify observable rolling features directly at every tested origin.
    o=d-builder.horizon_for_day(d)
    for j,w in [(28,7),(29,28)]:
        expected=np.array([(panel.observed[i,t-w:t]>=panel.capacity[i,t-w:t]).mean() for i,t in zip(s,o)])
        assert np.allclose(old[:,j],expected,rtol=0,atol=1e-7)
    # Alter observations strictly after origin; lag and hit features stay fixed.
    # All other fields are fixed calendar/price covariates or static metadata.
    original=panel.observed.copy();hitcs=builder.censor_cumsum.copy()
    panel.observed[:,int(o.min()):]=12345
    same=builder.make_features(s,d,include_censor=True)
    panel.observed=original
    assert np.array_equal(old,same)
    return dict(hidden_target_perturbation=True,strict_flag_perturbation=True,
                direct_observable_hit_history=True,post_origin_sales_perturbation=True,
                feature_count=int(old.shape[1]))

def run(data,cache,out):
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    protocol=dict(created_utc=now(),status='FROZEN_BEFORE_FITS',scope='POST_HOC_CONSUMED_M5_DESIGN_ONLY',
        seeds=SEEDS,parameters=PARAMS,training_rows=1200000,training_days=[365,1313],
        selection_days=[1314,1433],purge='weekly origin >= 1313 for selection',
        eval_days='identical to saved Chronos comparison aligned arrays',
        alphas=ALPHAS,powers=POWERS,em_iterations=2,
        scale='exact nonnegative weighted-median optimum, separately for every configuration; no grid ceiling',
        mean_controls=['observed Poisson','capacity-hit Poisson EM'],
        extra_controls=['unscaled L1','exact global rescaling','exact category rescaling','ungated EM'],
        information='all heads use identical 33 observable origin features; hit = S >= C; no strict Y>C label or history',
        selection_targets='pre-censoring recorded-sales Y only in point validation',
        primary_seed=SEEDS[0],first_seed_predictions_saved=True,
        uncertainty='paired item bootstrap conditional on fit, 2000 draws; descriptive',
        guardian_accesses=0,external_accesses=0,uci_test_accesses=0,
        input_sha256={n:sha(data/n) for n in ['design_outcomes_v0_5.npz','calendar.csv','sell_prices.csv']},
        code_sha256=sha(__file__),utility_sha256=sha(Path(__file__).with_name('audit_observed_data.py')),
        feature_sha256=sha(Path(__file__).resolve().parents[1]/'m5/legacy_features/features.py'))
    write(out/'PROTOCOL.json',protocol)
    panel=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv')
    # Replace privileged labels BEFORE constructing any rolling statistic.
    panel.censored=(panel.observed>=panel.capacity).astype(np.uint8)
    builder=FeatureBuilder(panel)
    write(out/'INFORMATION_CHECKS.json',information_checks(panel,builder))
    print('Observable information checks PASS',flush=True)
    vd=np.arange(1314,1434);vd=vd[vd-builder.horizon_for_day(vd)>=1313]
    yv=panel.truth[:,vd-1].astype(float);cats=panel.metadata['cat_id']
    blocks={k:np.load(cache/f'{k}_aligned.npz')['target_days'] for k in ['calibration_a','calibration_b','shadow']}
    q=rmsse_scale(panel.truth[:,:1313]);item,inv=np.unique(panel.metadata['item_id'],return_inverse=True)
    results=[]
    for seed in SEEDS:
        sd=out/f'seed_{seed}';sd.mkdir();models={}
        s,d=builder.sample_pairs(365,1313,1200000,seed)
        x=builder.make_features(s,d,include_censor=True)
        observed=panel.observed[s,d-1];capacity=panel.capacity[s,d-1];hit=(observed>=capacity)
        write(sd/'FIT_START.json',dict(created_utc=now(),protocol_sha256=sha(out/'PROTOCOL.json'),
            training_rows=len(s),strict_flag_labels_used=0,hidden_target_labels_used=0))
        for name,objective,target in [('l1','regression_l1',observed),('poisson','poisson',observed),('hit','binary',hit.astype(int))]:
            print('Fitting',seed,name,flush=True);ts=time.time()
            cls=lgb.LGBMClassifier if name=='hit' else lgb.LGBMRegressor
            m=cls(objective=objective,random_state=seed,**PARAMS).fit(x,target)
            models[name]=m.booster_;m.booster_.save_model(str(sd/f'{name}.txt'))
            print('Fit completed',seed,name,round(time.time()-ts,1),flush=True)
        em=models['poisson'];fallback=[]
        for iteration in range(2):
            mu=np.maximum(em.predict(x,num_threads=4),1e-6);den=poisson.sf(capacity-1,mu)
            target=np.divide(mu*poisson.sf(capacity-2,mu),den,out=capacity.astype(float).copy(),where=den>1e-250)
            target=np.where(hit,np.maximum(target,capacity),observed)
            fallback.append(int((hit&(den<=1e-250)).sum()))
            assert np.isfinite(target).all()
            print('Fitting',seed,'EM',iteration+1,flush=True)
            em=lgb.LGBMRegressor(objective='poisson',random_state=seed,**PARAMS).fit(x,target).booster_
        models['em']=em;em.save_model(str(sd/'em.txt'));del x;gc.collect()
        pv=predict_heads(builder,models,vd);chosen,records=select(yv,pv,cats)
        write(sd/'POLICY_FREEZE.json',dict(created_utc=now(),choices=chosen,all_validation=records,
             model_sha256={k:sha(sd/f'{k}.txt') for k in models},em_underflow_fallback_counts=fallback,
             evaluation_blocks_scored=0,protocol_sha256=sha(out/'PROTOCOL.json')))
        np.savez_compressed(sd/'validation_heads.npz',days=vd,**pv)
        print('Selected',seed,{k:(v['config'],v['wape']) for k,v in chosen.items()},flush=True)
        del pv
        for block,days in blocks.items():
            pred=predict_heads(builder,models,days);y=panel.truth[:,days-1].astype(float)
            if seed==SEEDS[0]:np.savez_compressed(sd/f'{block}_heads.npz',days=days,**pred)
            pp={k:candidate(pred,v['config'],cats) for k,v in chosen.items()}
            row=dict(seed=seed,block=block,metrics={k:metrics(y,p,q) for k,p in pp.items()},
                categories={str(c):{k:metrics(y[cats==c],p[cats==c],q[cats==c]) for k,p in pp.items()} for c in np.unique(cats)})
            # A comparison conditional on censoring is descriptive, never a gate.
            strict=y>panel.capacity[:,days-1]
            row['strata']={name:{k:metrics(y[mask],p[mask]) for k,p in pp.items()}
                           for name,mask in [('strict_censored',strict),('not_strict_censored',~strict)]}
            mass=np.bincount(inv,weights=y.sum(axis=1));err={k:np.bincount(inv,weights=np.abs(y-p).sum(axis=1)) for k,p in pp.items()}
            np.savez_compressed(sd/f'{block}_item_stats.npz',items=item,mass=mass,**err)
            if seed==SEEDS[0]:
                rng=np.random.default_rng(seed);boot=rng.multinomial(len(item),np.full(len(item),1/len(item)),size=2000)
                ci={}
                for k in ['scaled_l1','category_scaled_l1','observed_mean_gate','censored_mean_ungated']:
                    diff=err['censored_mean_gate']-err[k];vals=(boot@diff)/(boot@mass)
                    ci[k]=dict(delta_wape=float(diff.sum()/mass.sum()),ci95=np.quantile(vals,[.025,.975]).tolist())
                row['paired_item_intervals']=ci
            results.append(row);write(sd/f'{block}_results.json',row)
            print('EVALUATED',seed,block,{k:round(v['wape'],6) for k,v in row['metrics'].items()},flush=True)
            del pp,pred;gc.collect()
        del models;gc.collect()
    write(out/'RESULTS.json',dict(status='COMPLETED',created_utc=now(),seconds=time.time()-start,
        protocol_sha256=sha(out/'PROTOCOL.json'),rows=results,information='observable inputs only',
        statistical_scope='retrospective development comparison; no fresh held-out claim',
        versions=dict(numpy=np.__version__,lightgbm=lgb.__version__)))
    print('COMPLETED',time.time()-start,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.data,v.cache,v.output)
