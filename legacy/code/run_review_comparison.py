"""Review revision: matched conditional-error / conditional-demand decomposition.

All thresholds are chosen on development calibration A and B. The consumed
shadow is descriptive. No guardian or external input is accepted by this CLI.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from run_selective_study import (R,FLOOR,BLOCKS,construct,load,dump,save_npz,frontier,choose,metric,bootstrap,origin)
SEEDS=[20260906,20260907,20260908]

def predict(booster,x,iterations=None):
    flat=x.reshape(-1,x.shape[-1]);parts=[]
    for st in range(0,len(flat),200000):
        parts.append(booster.predict(flat[st:st+200000],num_threads=4,num_iteration=iterations))
    return np.concatenate(parts).reshape(x.shape[:2]).astype(np.float32)

def metrics_by_group(v,mask,meta):
    result={}
    for g in ['ALL']+sorted(set(meta['cat_id'].tolist())):
        rows=np.ones(len(mask),bool) if g=='ALL' else meta['cat_id']==g
        result[g]=metric(v['truth'][rows],v['proposal'][rows],v['baseline'][rows],mask[rows])
    return result

def policy(scores,data,meta,mode='pooled',cap=R):
    import run_selective_study as core
    old=core.R;core.R=cap
    masks={k:np.zeros(data[k]['truth'].shape,bool) for k in scores};ts={}
    for g in ['ALL'] if mode=='pooled' else sorted(set(meta['cat_id'])):
        rows=np.ones(len(meta['cat_id']),bool) if g=='ALL' else meta['cat_id']==g
        ff={k:frontier(scores[k][rows],data[k]['truth'][rows],data[k]['proposal'][rows],data[k]['baseline'][rows],'absolute') for k in ['calibration_a','calibration_b']}
        t=choose(ff['calibration_a'],ff['calibration_b']);ts[g]=t
        if t is not None:
            for k in scores:masks[k][rows]=scores[k][rows]<=t
    core.R=old
    return ts,masks

def paired_delta(v,aa,bb,meta,reps=5000):
    _,code=np.unique(meta['item_id'],return_inverse=True);n=code.max()+1
    y=v['truth'].astype(float);err=np.abs(y-v['proposal']);comp=[]
    for mask in [aa,bb]:
        comp.extend([np.bincount(code,weights=q,minlength=n) for q in [mask.sum(1),np.full(len(mask),mask.shape[1]),(y*mask).sum(1),(err*mask).sum(1),y.sum(1)]])
    mat=np.column_stack(comp);rng=np.random.default_rng(20260906);vals=[]
    for start in range(0,reps,200):
        w=rng.multinomial(n,np.full(n,1/n),size=min(200,reps-start));q=w@mat
        vals.append(np.column_stack([q[:,0]/q[:,1]-q[:,5]/q[:,6],q[:,2]/q[:,4]-q[:,7]/q[:,9],q[:,3]/np.maximum(q[:,2],1e-12)-q[:,8]/np.maximum(q[:,7],1e-12)]))
    arr=np.concatenate(vals)
    return {name:np.quantile(arr[:,j],[.025,.975]).tolist() for j,name in enumerate(['row_coverage_delta_ci95','demand_coverage_delta_ci95','wape_delta_ci95'])}|{'bootstrap_reps':reps,'item_clusters':int(n),'scope':'descriptive_consumed_design_shadow'}

def volume_diagnostic(v,masks,meta):
    p=v['proposal'];y=v['truth'].astype(float);e=np.abs(y-p);cat=meta['cat_id']
    # Fixed interpretable unit bins; no selected threshold is derived here.
    edges=[0,.25,1,2,5,10,float('inf')];out=[]
    for g in ['ALL']+sorted(set(cat)):
      rows=np.ones(len(cat),bool) if g=='ALL' else cat==g
      yy=y[rows];pp=p[rows];ee=e[rows]
      for lo,hi in zip(edges[:-1],edges[1:]):
        membership=(pp>=lo)&(pp<hi)
        for name,a in masks.items():
          take=membership&a[rows];n=int(take.sum());mass=float(yy[take].sum());zero=take&(yy==0)
          out.append(dict(group=g,forecast_bin=[lo,None if np.isinf(hi) else hi],score=name,bin_rows=int(membership.sum()),accepted_n=n,accepted_demand=mass,wape=float(ee[take].sum()/mass) if mass else None,zero_demand_fraction=float(zero.sum()/n) if n else None,absolute_error_on_zero=float(ee[zero].sum()),positive_demand_rows=int((take&(yy>0)).sum())))
    return out

def main(a):
    assert hashlib.sha256((a.input/'design_outcomes_v0_5.npz').read_bytes()).hexdigest()=='5d63b6aff7854ed1811f6b5bffd2f397a09d671c6754e0f1345e2313b35e48fd', 'Only the already-consumed design panel is allowed'
    t0=time.time();a.output.mkdir(parents=True,exist_ok=True);cache=a.output/'cache';cache.mkdir(exist_ok=True)
    if not (cache/'shadow_features.npy').exists():construct(a.input,cache)
    meta=load(cache/'metadata.npz');data={k:load(cache/(k+'_aligned.npz')) for k in BLOCKS}
    xs={k:np.load(cache/(k+'_features.npy'),mmap_mode='r') for k in BLOCKS};records=[];contrasts=[]
    for seed in SEEDS:
        head_path=a.output/f'demand_s{seed}.txt'
        if not head_path.exists():
            y=data['risk_train']['truth'].ravel();idx=np.random.default_rng(seed).choice(y.size,600000,replace=False)
            x=xs['risk_train'].reshape(-1,21)[idx]
            model=lgb.LGBMRegressor(objective='regression',n_estimators=220,num_leaves=31,learning_rate=.05,min_child_samples=100,reg_lambda=3,n_jobs=4,random_state=seed,deterministic=True,force_col_wise=True,verbosity=-1)
            print('Training conditional demand head',seed,flush=True);model.fit(x,y[idx],categorical_feature=[17,18,19,20]);model.booster_.save_model(str(head_path))
        models={'error':a.fitted/f'mean_error_s{seed}.txt','direct':a.fitted/f'contract_excess_s{seed}.txt','demand':head_path}
        preds={}
        for name,path in models.items():
            dest=a.output/f'{name}_s{seed}_predictions.npz'
            if dest.exists():preds[name]=load(dest);continue
            b=lgb.Booster(model_file=str(path));v={}
            for k in BLOCKS[1:]:
                v[k]=predict(b,xs[k])
                if name in ['error','demand']:v[k+'_110']=predict(b,xs[k],110)
            save_npz(dest,**v);preds[name]=v;print('Scored',name,seed,flush=True)
        e={k:np.maximum(preds['error'][k],0) for k in BLOCKS[1:]};mu={k:np.maximum(preds['demand'][k],0) for k in BLOCKS[1:]}
        family={n:{} for n in ['mean_error','relative_error_f','relative_error_mu','excess_f','composite_excess','composite_excess_budget220','direct_excess']}
        for k in BLOCKS[1:]:
            family['mean_error'][k]=e[k]
            family['relative_error_f'][k]=e[k]/np.maximum(data[k]['proposal'],.25)
            family['relative_error_mu'][k]=e[k]/np.maximum(mu[k],.25)
            family['excess_f'][k]=e[k]-R*data[k]['proposal']
            family['composite_excess'][k]=e[k]-R*mu[k]
            family['composite_excess_budget220'][k]=np.maximum(preds['error'][k+'_110'],0)-R*np.maximum(preds['demand'][k+'_110'],0)
            family['direct_excess'][k]=preds['direct'][k]
        allmasks={}
        for name,scores in family.items():
            for mode in ['pooled','category']:
                dest=a.output/f'report_{name}_{mode}_s{seed}.json'
                ts,masks=policy(scores,data,meta,mode)
                rr=dict(seed=seed,score=name,threshold_mode=mode,thresholds=ts,metrics={k:metrics_by_group(data[k],masks[k],meta) for k in scores},trees_total=440 if name in ['relative_error_mu','composite_excess'] else 220)
                save_npz(a.output/f'mask_{name}_{mode}_s{seed}.npz',**masks)
                if mode=='pooled':
                    allmasks[name]=masks['shadow'];err=np.abs(data['shadow']['truth']-data['shadow']['proposal']).ravel();s=scores['shadow'].ravel()
                    rr['absolute_error_auc']=float(roc_auc_score(err>=np.quantile(err,.8),s))
                dump(dest,rr);records.append(rr)
                print(seed,name,mode,rr['metrics']['shadow']['ALL'],flush=True)
        sh=data['shadow']
        for other in ['relative_error_f','composite_excess','composite_excess_budget220']:
            contrasts.append(dict(seed=seed,contrast='direct_excess minus '+other,**paired_delta(sh,allmasks['direct_excess'],allmasks[other],meta)))
        if seed==SEEDS[0]:
            dump(a.output/'ACCEPTED_VOLUME_DIAGNOSTIC.json',volume_diagnostic(sh,allmasks,meta))
            group_summary=[]
            for g in sorted(set(meta['cat_id'])):
                rows=meta['cat_id']==g;y=sh['truth'][rows].astype(float);p=sh['proposal'][rows];ee=np.abs(y-p)
                group_summary.append(dict(group=g,n=int(y.size),zero_demand_fraction=float((y==0).mean()),mean_units=float(y.mean()),zero_day_error_share=float(ee[y==0].sum()/ee.sum()),nonzero_day_wape=float(ee[y>0].sum()/y.sum()),full_wape=float(ee.sum()/y.sum()),past28_zero_mean=float(xs['shadow'][rows,:,12].mean()),row_censor_rate=float(sh['censored'][rows].mean())))
            dump(a.output/'INTERMITTENCY_DIAGNOSTIC.json',group_summary)
            sensitivity=[]
            for cap in [.55,.60,R,.65,.70,.75]:
                ss={k:e[k]-cap*mu[k] for k in BLOCKS[1:]};ts,mm=policy(ss,data,meta,cap=cap)
                sensitivity.append(dict(cap=cap,thresholds=ts,metrics=metrics_by_group(sh,mm['shadow'],meta),refit_count=0))
            dump(a.output/'CAP_SENSITIVITY_TWO_HEAD.json',sensitivity)
    dump(a.output/'REVIEW_COMPARISON.json',dict(status='DESIGN_COMPARISON_COMPLETED',cap=R,floor=FLOOR,records=records,contrasts=contrasts,seconds=time.time()-t0,additional_fit_calls=3,fresh_guardian_opened=False,external_opened=False,scope='consumed_design_only'))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for x in ['input','output','fitted']:ap.add_argument('--'+x,type=Path,required=True)
    main(ap.parse_args())
