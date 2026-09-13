"""Matched demand-head loss ablation; six design-only fits, consumed holdout."""
from pathlib import Path
import argparse, json,datetime,time
import numpy as np,lightgbm as lgb
from r2_io import install_numpy_writers,sha,dump
from run_selective_study import load,construct,R
from run_review_comparison import predict,SEEDS,metrics_by_group
from run_review2_objectives import curve,select_threshold,apply,KEYS

def main(a):
    install_numpy_writers();a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(status='POST_HOC_LOSS_ABLATION',losses=['poisson','tweedie'],tweedie_variance_power=1.5,seeds=SEEDS,training_rows=600000,trees=220,leaves=31,learning_rate=.05,min_child_samples=100,reg_lambda=3,fitting_scope='already_consumed_design_risk_train',guardian_scope='posthoc_consumed_cache_only',new_guardian_accesses=0,external_opened=False)
    dump(a.output/'LOSS_PROTOCOL.json',protocol)
    dc=a.design/'cache_design';dd={k:load(dc/(k+'_aligned.npz')) for k in KEYS};dm=load(dc/'metadata.npz')
    train=load(dc/'risk_train_aligned.npz')['truth'].ravel();xt=np.load(dc/'risk_train_features.npy',mmap_mode='r').reshape(-1,21)
    dx={k:np.load(dc/(k+'_features.npy'),mmap_mode='r') for k in KEYS}
    gc=a.output/'guardian_features';gc.mkdir(exist_ok=True)
    if not (gc/'ORIGIN_AUDIT.json').exists():construct(a.guardian_predictions,gc)
    gd={k:load(a.guardian/'cache'/(k+'_aligned.npz')) for k in KEYS};gm=load(a.guardian/'cache/metadata.npz')
    for k in KEYS:
        rebuilt=load(gc/(k+'_aligned.npz'))
        for key,value in gd[k].items():assert np.array_equal(value,rebuilt[key]),(k,key)
    gx={k:np.load(gc/(k+'_features.npy'),mmap_mode='r') for k in KEYS}
    records=[];fits=0
    for seed in SEEDS:
      idx=np.random.default_rng(seed).choice(len(train),600000,replace=False)
      for objective in protocol['losses']:
        path=a.output/f'demand_{objective}_s{seed}.txt'
        if not path.exists():
            kw=dict(objective=objective,n_estimators=220,num_leaves=31,learning_rate=.05,min_child_samples=100,reg_lambda=3,n_jobs=4,random_state=seed,deterministic=True,force_col_wise=True,verbosity=-1)
            if objective=='tweedie':kw['tweedie_variance_power']=1.5
            model=lgb.LGBMRegressor(**kw);model.fit(xt[idx],train[idx],categorical_feature=[17,18,19,20]);model.booster_.save_model(str(path));fits+=1
            print('Fit completed',seed,objective,flush=True)
        b=lgb.Booster(model_file=str(path));dest=a.output/f'predictions_{objective}_s{seed}.npz'
        if dest.exists():pred=load(dest)
        else:
            pred={}
            for cohort,xx in [('design',dx),('guardian',gx)]:
                for k,x in xx.items():pred[cohort+'_'+k]=predict(b,x)
            np.savez_compressed(dest,**pred)
        # Select only on development calibration A/B before application.
        ep=load(a.design/f'design_predictions_s{seed}.npz')
        families={name:{} for name in ['composite_excess','relative_error_mu']}
        for k in KEYS:
            e=np.maximum(ep['error_'+k],0);m=np.maximum(pred['design_'+k],0)
            families['composite_excess'][k]=e-R*m;families['relative_error_mu'][k]=e/np.maximum(m,.25)
        choices={}
        for name,ss in families.items():
            ca={k:curve(ss[k],dd[k]['truth'],dd[k]['proposal']) for k in KEYS[:2]}
            t,obj=select_threshold(ca['calibration_a'],ca['calibration_b'],'demand');choices[name]=t
        dump(a.output/f'POLICIES_{objective}_s{seed}.json',dict(thresholds=choices,scope='development_selected_posthoc'))
        gp=load(a.guardian/f'predictions_s{seed}.npz')
        for name,ss in families.items():
            gs={}
            for k in KEYS:
                e=np.maximum(gp['error_'+k],0);m=np.maximum(pred['guardian_'+k],0)
                gs[k]=e-R*m if name=='composite_excess' else e/np.maximum(m,.25)
            records.append(dict(seed=seed,demand_loss=objective,score=name,threshold=choices[name],design=apply(dd,dm,ss,choices[name]),guardian_posthoc=apply(gd,gm,gs,choices[name]),demand_shadow_mse={cohort:float(np.mean((np.maximum(pred[cohort+'_'+'shadow'],0).astype(float)-data['shadow']['truth'])**2)) for cohort,data in [('design',dd),('guardian',gd)]}))
        print('Loss evaluation complete',seed,objective,flush=True)
    dump(a.output/'LOSS_RESULTS.json',dict(status='MATCHED_LOSS_CONTROLS_COMPLETED',protocol=protocol,fit_calls_this_invocation=fits,total_fitted_models=6,records=records))
    print('SIX MATCHED LOSS CONTROLS COMPLETED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['design','guardian','guardian-predictions','output']:ap.add_argument('--'+k,type=Path,required=True)
    main(ap.parse_args())
