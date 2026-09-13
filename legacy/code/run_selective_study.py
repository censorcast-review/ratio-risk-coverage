"""Origin-aligned, contract-targeted selection; all analyses are development-only.

Risk labels are pre-artificial-censoring recorded sales. They are a benchmark audit
resource, not identified latent demand on naturally stocked-out transactions.
"""
from __future__ import annotations
import argparse, hashlib, json, time, os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

R=.85*.7530939208313988
FLOOR=.35
BLOCKS=['risk_train','calibration_a','calibration_b','shadow']
CUTOFF={'risk_train':1433,'calibration_a':1553,'calibration_b':1673,'shadow':1793}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files if k!='risk_features'}

def save_npz(path,**arrays):
    temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def origin(day):return day-(1+(day-1)%7)

def metric(y,p,b,a=None):
    if a is None:a=np.ones(y.shape,bool)
    a=np.array(a,dtype=bool,copy=True)&np.isfinite(y)&np.isfinite(p)&np.isfinite(b)
    e=np.abs(np.asarray(y,dtype=np.float64)-p)
    eb=np.abs(np.asarray(y,dtype=np.float64)-b)
    dy=float(y[a].sum()); nb=float(eb[a].sum());n=int(a.sum())
    return dict(n=int(y.size),accepted_n=n,coverage=n/y.size,
                wape=float(e[a].sum()/dy) if dy else None,
                baseline_same_subset_wape=float(eb[a].sum()/dy) if dy else None,
                paired_error_ratio=float(e[a].sum()/nb) if nb else None,
                demand_coverage=float(dy/y.sum()) if y.sum() else None)

def construct(root,out):
    """Build only forecast-origin features and purge fitted-object time boundaries."""
    with np.load(root/'design_outcomes_v0_5.npz',allow_pickle=False) as z:
        meta={k:z[k].astype(str) for k in ['id','item_id','cat_id','dept_id','store_id','state_id']}
        obs=z['observed'].astype(np.float32);cen=z['censored'].astype(np.float32)
    data={k:load(root/(k+'_v0_5.npz')) for k in BLOCKS}
    sel=load(root/'selection_base_v0_5.npz')
    if 'proposal' not in sel:
        sel['proposal']=sel['raw_poisson_histgb']+2*sel['censor_probability']*np.maximum(sel['censored_poisson_em']-sel['raw_poisson_histgb'],0)
    allp=np.concatenate([sel['proposal']]+[data[k]['proposal'] for k in BLOCKS],axis=1)
    allo=np.concatenate([sel['observed']]+[data[k]['observed'] for k in BLOCKS],axis=1)
    ce=np.pad(np.cumsum(np.abs(allo.astype(np.float64)-allp),axis=1),((0,0),(1,0)))
    cd=np.pad(np.cumsum(allo.astype(np.float64),axis=1),((0,0),(1,0)))
    # History begins at d_1314; cumulative indices count observations <= origin.
    ge=ce.sum(axis=0);gd=cd.sum(axis=0);gp=(ge+30*.75)/(gd+30)
    oc=np.pad(np.cumsum(obs,dtype=np.float64,axis=1),((0,0),(1,0)))
    oc2=np.pad(np.cumsum(obs*obs,dtype=np.float64,axis=1),((0,0),(1,0)))
    zc=np.pad(np.cumsum(obs==0,axis=1),((0,0),(1,0)))
    cc=np.pad(np.cumsum(cen,axis=1),((0,0),(1,0)))
    codes={k:np.unique(meta[k],return_inverse=True)[1].astype(np.float32) for k in ['cat_id','dept_id','store_id','state_id']}
    audit=[]
    names=['log_baseline','log_proposal','log_raw','log_em','censor_probability','positive_correction',
           'prior_observed_wape_origin','horizon','observed_mean7','observed_mean28','observed_mean56',
           'observed_std28','observed_zero_fraction28','past_censor_rate7','past_censor_rate28',
           'observed_seasonal7','observed_seasonal14','cat_id','dept_id','store_id','state_id']
    np.savez_compressed(out/'metadata.npz',**meta)
    for key,v in data.items():
        d=v['target_days'];o=origin(d);indices=np.maximum(o-1313,0)
        prior=(ce[:,indices]+30*gp[indices])/(cd[:,indices]+30)
        late=d-1-o
        old=v['prior_observed_wape']
        audit.append(dict(block=key,total_rows=int(old.size),rows_with_after_origin_history=int(np.sum(late>0)*len(old)),
                          max_after_origin_days=int(late.max()),mean_absolute_prior_change=float(np.abs(old-prior).mean()),
                          p95_absolute_prior_change=float(np.quantile(np.abs(old-prior),.95)),
                          dropped_days=d[o<CUTOFF[key]].tolist()))
        cols=[]
        for f in ['baseline','proposal','raw_poisson_histgb','censored_poisson_em']:cols.append(np.log1p(v[f]))
        cols += [v['censor_probability'],np.maximum(v['proposal']-v['baseline'],0),prior,np.broadcast_to((d-o)[None,:],prior.shape)]
        means={w:(oc[:,o]-oc[:,o-w])/w for w in (7,28,56)}
        cols += [means[7],means[28],means[56],np.sqrt(np.maximum((oc2[:,o]-oc2[:,o-28])/28-means[28]**2,0)),
                 (zc[:,o]-zc[:,o-28])/28,(cc[:,o]-cc[:,o-7])/7,(cc[:,o]-cc[:,o-28])/28,
                 obs[:,d-8],obs[:,d-15]]
        cols += [np.broadcast_to(codes[k][:,None],prior.shape) for k in codes]
        keep=o>=CUTOFF[key]
        x=np.stack([np.asarray(c[:,keep],np.float32) for c in cols],axis=-1)
        assert x.shape[-1]==len(names) and np.isfinite(x).all()
        np.save(out/(key+'_features.npy'),x)
        arr={k:(a[:,keep] if a.ndim==2 else a[keep]) for k,a in v.items()}
        arr['prior_origin']=prior[:,keep].astype(np.float32)
        np.savez_compressed(out/(key+'_aligned.npz'),**arr)
        print('Features prepared',key,x.shape,flush=True)
    dump(out/'ORIGIN_AUDIT.json',dict(status='ORIGIN_AND_SPLIT_BOUNDARIES_REPAIRED',feature_names=names,blocks=audit,
         old_risk_predictions='diagnostic negative control only; not used by repaired learners',
         raw_dataset='already-consumed design shard only',fresh_guardian_opened=False,external_opened=False))

def frontier(score,y,p,b,kind):
    s=np.asarray(score,dtype=np.float64).ravel();y=y.ravel().astype(np.float64)
    ep=np.abs(y-p.ravel());eb=np.abs(y-b.ravel());idx=np.argsort(s,kind='stable')
    s=s[idx];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    count=ends+1;err=np.cumsum(ep[idx])[ends];mass=np.cumsum(y[idx])[ends];base=np.cumsum(eb[idx])[ends]
    denom=mass if kind=='absolute' else base
    cap=R if kind=='absolute' else .95
    ratio=np.divide(err,denom,out=np.full(len(err),np.inf),where=denom>0)
    return dict(threshold=s[ends],n=count,coverage=count/len(y),ratio=ratio,point_feasible=(ratio<=cap)&(count/len(y)>=FLOOR))

def choose(fa,fb):
    thresholds=np.union1d(fa['threshold'],fb['threshold'])
    ia=np.searchsorted(fa['threshold'],thresholds,side='right')-1
    ib=np.searchsorted(fb['threshold'],thresholds,side='right')-1
    valid=(ia>=0)&(ib>=0)
    ia=np.maximum(ia,0);ib=np.maximum(ib,0)
    good=valid&fa['point_feasible'][ia]&fb['point_feasible'][ib]
    if not good.any():return None
    objective=np.minimum(fa['coverage'][ia],fb['coverage'][ib]);objective[~good]=-1
    return float(thresholds[np.argmax(objective)])

def summarize(a,v,meta,group=None):
    rows=np.ones(len(a),bool) if group is None else meta['cat_id']==group
    return metric(v['truth'][rows],v['proposal'][rows],v['baseline'][rows],a[rows])

def bootstrap(v,accept,meta,reps=3000,seed=20260906,cluster='item'):
    y=v['truth'].astype(np.float64);ep=np.abs(y-v['proposal']);eb=np.abs(y-v['baseline'])
    # Copies are essential: category filtering must never mutate a shared policy mask.
    a=np.array(accept,dtype=bool,copy=True)
    if cluster=='item':
        labels,codes=np.unique(meta['item_id'],return_inverse=True)
        components=np.column_stack([np.bincount(codes,weights=z,minlength=len(labels)) for z in
                     [a.sum(axis=1),np.full(len(a),a.shape[1]),(ep*a).sum(axis=1),(y*a).sum(axis=1),(eb*a).sum(axis=1)]])
    elif cluster=='week':
        labels,codes=np.unique(origin(v['target_days']),return_inverse=True)
        components=np.column_stack([np.bincount(codes,weights=z,minlength=len(labels)) for z in
                     [a.sum(axis=0),np.full(a.shape[1],len(a)),(ep*a).sum(axis=0),(y*a).sum(axis=0),(eb*a).sum(axis=0)]])
    n=len(labels);rng=np.random.default_rng(seed);stats=[]
    for st in range(0,reps,200):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(200,reps-st));s=weights@components
        stats.append(np.column_stack([s[:,0]/s[:,1],s[:,2]/np.maximum(s[:,3],1e-15),s[:,2]/np.maximum(s[:,4],1e-15)]))
    z=np.concatenate(stats);result={'clusters':n,'reps':reps,'type':cluster,'scope':'conditional_descriptive_not_confirmatory'}
    for j,name in enumerate(['coverage','wape','paired_error_ratio']):
        result[name+'_ci95']=np.quantile(z[:,j],[.025,.975]).tolist()
        result[name+'_lcb95']=float(np.quantile(z[:,j],.05))
        result[name+'_ucb95']=float(np.quantile(z[:,j],.95))
        result[name+'_lcb_bonf8']=float(np.quantile(z[:,j],.05/8))
        result[name+'_ucb_bonf8']=float(np.quantile(z[:,j],1-.05/8))
    return result

def run(args):
    t0=time.time();out=args.output;out.mkdir(parents=True,exist_ok=True);cache=out/'cache';cache.mkdir(exist_ok=True)
    if not (cache/'ORIGIN_AUDIT.json').exists():construct(args.input,cache)
    meta=load(cache/'metadata.npz');data={k:load(cache/(k+'_aligned.npz')) for k in BLOCKS}
    xs={k:np.load(cache/(k+'_features.npy'),mmap_mode='r') for k in BLOCKS}
    train=data['risk_train'];err=np.abs(train['truth'].astype(np.float64)-train['proposal']).ravel()
    baseerr=np.abs(train['truth'].astype(np.float64)-train['baseline']).ravel()
    targets={'mean_error':err,'quantile_error':err,'contract_excess':err-R*train['truth'].ravel(),
             'paired_excess':err-.95*baseerr}
    # Primary contrasts use identical features, training rows and tree capacity.
    records=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed);idx=rng.choice(err.size,600000,replace=False)
        x=xs['risk_train'].reshape(-1,xs['risk_train'].shape[-1])[idx]
        for name,target in targets.items():
            model_path=out/f'{name}_s{seed}.txt';score_path=out/f'{name}_s{seed}_scores.npz'
            if score_path.exists() and zipfile.is_zipfile(score_path):continue
            if score_path.exists():score_path.replace(score_path.with_suffix('.corrupt'))
            objective='quantile' if name=='quantile_error' else 'regression'
            m=lgb.LGBMRegressor(objective=objective,alpha=.85,n_estimators=220,num_leaves=31,
                   learning_rate=.05,min_child_samples=100,reg_lambda=3,n_jobs=4,random_state=seed,
                   deterministic=True,force_col_wise=True,verbosity=-1)
            if model_path.exists():
                print('Restoring fitted risk model',name,seed,flush=True)
                booster=lgb.Booster(model_file=str(model_path))
            else:
                print('Fitting risk',name,seed,flush=True);m.fit(x,target[idx],categorical_feature=[17,18,19,20])
                m.booster_.save_model(str(model_path));booster=m.booster_
            pred={}
            for k in BLOCKS[1:]:
                xx=xs[k].reshape(-1,x.shape[-1]);parts=[]
                for st in range(0,len(xx),200000):parts.append(booster.predict(xx[st:st+200000],num_threads=4))
                pred[k]=np.concatenate(parts).reshape(xs[k].shape[:2]).astype(np.float32)
            save_npz(score_path,**pred)
    for seed in args.seeds:
        learned={name:load(out/f'{name}_s{seed}_scores.npz') for name in targets}
        families={name:{k:learned[name][k] for k in BLOCKS[1:]} for name in targets}
        families['relative_error']={k:np.maximum(learned['mean_error'][k],0)/np.maximum(data[k]['proposal'],.25) for k in BLOCKS[1:]}
        if seed==args.seeds[0]:
            families['historical_origin']={k:data[k]['prior_origin'] for k in BLOCKS[1:]}
            families['high_predicted_mass']={k:-data[k]['proposal'] for k in BLOCKS[1:]}
            if not args.skip_legacy:
                families['legacy_leaky_hybrid']={k:.25*data[k]['instantaneous_absolute_error']+.75*np.maximum(data[k]['proposal'],1)*data[k]['prior_observed_wape']-.90*.7530939208313988*np.maximum(data[k]['proposal'],1) for k in BLOCKS[1:]}
        for name,scores in families.items():
            checkpoint=out/f'evaluation_{name}_s{seed}.json'
            if checkpoint.exists():
                saved_reports=json.loads(checkpoint.read_text())
                # A completed JSON can survive an interrupted mask-archive write.
                # Recover that artifact from the fixed score and recorded threshold.
                for report in saved_reports:
                    mp=out/f"mask_{name}_{report['threshold_mode']}_s{seed}.npz"
                    if mp.exists() and zipfile.is_zipfile(mp):continue
                    recovered={k:np.zeros(data[k]['truth'].shape,bool) for k in BLOCKS[1:]}
                    for group,threshold in report['thresholds'].items():
                        if threshold is None:continue
                        rows=np.ones(len(meta['id']),bool) if group=='ALL' else meta['cat_id']==group
                        for k in recovered:recovered[k][rows]=scores[k][rows]<=threshold
                    assert summarize(recovered['shadow'],data['shadow'],meta)['accepted_n']==report['metrics']['shadow']['accepted_n']
                    save_npz(mp,**recovered)
                    print('Restored recorded acceptance mask',mp.name,flush=True)
                records+=saved_reports;continue
            local=[];kind='paired' if name=='paired_excess' else 'absolute'
            for mode in ['pooled','category']:
                groups=['ALL'] if mode=='pooled' else sorted(np.unique(meta['cat_id']).tolist())
                mask={k:np.zeros(data[k]['truth'].shape,bool) for k in BLOCKS[1:]};thresholds={};has_all=True
                for group in groups:
                    rows=np.ones(len(meta['id']),bool) if group=='ALL' else meta['cat_id']==group
                    ff={k:frontier(scores[k][rows],data[k]['truth'][rows],data[k]['proposal'][rows],data[k]['baseline'][rows],kind) for k in ['calibration_a','calibration_b']}
                    threshold=choose(ff['calibration_a'],ff['calibration_b']);thresholds[group]=threshold
                    if threshold is None:has_all=False;continue
                    for k in mask:mask[k][rows]=scores[k][rows]<=threshold
                report=dict(score=name,seed=seed,threshold_mode=mode,contract_kind=kind,thresholds=thresholds,
                    every_category_has_calibration_candidate=has_all,calibration_point_only=True)
                report['metrics']={k:summarize(mask[k],data[k],meta) for k in mask}
                report['shadow_categories']={g:summarize(mask['shadow'],data['shadow'],meta,g) for g in sorted(np.unique(meta['cat_id']))}
                if mask['shadow'].any():
                    report['shadow_bootstrap']=bootstrap(data['shadow'],mask['shadow'],meta,reps=args.bootstrap_reps)
                    report['shadow_week_bootstrap']=bootstrap(data['shadow'],mask['shadow'],meta,reps=args.bootstrap_reps,cluster='week')
                sh=data['shadow'];errors=np.abs(sh['truth']-sh['proposal']).ravel();score=scores['shadow'].ravel()
                high=errors>=np.quantile(errors,.8)
                report['score_diagnostics']={'auc_top20_absolute_error':float(roc_auc_score(high,score)),
                    'spearman_contract_excess':float(spearmanr(score,errors-R*sh['truth'].ravel()).statistic)}
                local.append(report)
                save_npz(out/f'mask_{name}_{mode}_s{seed}.npz',**mask)
                print(name,seed,mode,report['metrics']['shadow'],flush=True)
            dump(checkpoint,local);records+=local
    dump(out/'SELECTIVE_RESULTS.json',dict(status='DEVELOPMENT_REPAIRED_SELECTORS_COMPLETE',absolute_wape_cap=R,coverage_floor=FLOOR,
          records=records,origin_audit=json.loads((cache/'ORIGIN_AUDIT.json').read_text()),seconds=time.time()-t0,
          risk_label_scope='pre-artificial-censoring recorded M5 sales in development audit data',
          fresh_guardian_opened=False,external_opened=False,certificate_issued=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--bootstrap-reps',type=int,default=3000)
    ap.add_argument('--seeds',type=int,nargs='+',default=[20260906,20260907,20260908])
    ap.add_argument('--skip-legacy',action='store_true')
    ap.add_argument('--absolute-cap',type=float,default=R)
    args=ap.parse_args();R=args.absolute_cap;run(args)
