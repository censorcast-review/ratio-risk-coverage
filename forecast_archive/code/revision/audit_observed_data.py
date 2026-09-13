"""Post-hoc diagnostics of saved predictions; new scales use validation only."""
from pathlib import Path
import argparse, hashlib, importlib.util, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def rmsse_scale(y):
    """First difference MSE after each series' first positive training target."""
    y=np.asarray(y,dtype=float);q=np.full(len(y),np.nan)
    for i,row in enumerate(y):
        nz=np.flatnonzero(row>0)
        if len(nz) and nz[0]<len(row)-1:q[i]=np.mean(np.diff(row[nz[0]:])**2)
    return q

def metrics(y,f,q=None):
    y=np.asarray(y,dtype=float);f=np.asarray(f,dtype=float);e=f-y
    out=dict(rows=int(y.size),target_mass=float(y.sum()),wape=float(np.abs(e).sum()/y.sum()),
             mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),
             signed_bias=float(e.sum()/y.sum()))
    if q is not None:
        ok=np.isfinite(q)&(q>0);r=np.sqrt(np.mean(e**2,axis=1)[ok]/q[ok])
        out.update(mean_series_rmsse=float(r.mean()),median_series_rmsse=float(np.median(r)),
                   rmsse_eligible_series=int(ok.sum()),rmsse_excluded_series=int((~ok).sum()))
    return out

def severity(y,s,c):
    y=np.asarray(y,dtype=float);s=np.asarray(s,dtype=float);c=np.asarray(c,dtype=float)
    assert np.array_equal(s,np.minimum(y,c))
    hit=s==c;strict=y>c;eq=y==c;mass=y.sum()
    return dict(rows=int(y.size),target_mass=float(mass),observed_mass=float(s.sum()),
                capacity_hit_rows=int(hit.sum()),capacity_hit_fraction=float(hit.mean()),
                strictly_censored_rows=int(strict.sum()),strictly_censored_fraction=float(strict.mean()),
                equality_only_fraction=float(eq.mean()),hidden_mass=float((y-s).sum()),
                hidden_mass_fraction=float((y-s).sum()/mass),
                target_mass_on_censored_rows_fraction=float(y[strict].sum()/mass),
                zero_target_fraction=float((y==0).mean()))

def optimal_scale(y,b):
    y=np.asarray(y,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel();ok=b>0
    if not ok.any():return 1.
    ratio=y[ok]/b[ok];w=b[ok];order=np.argsort(ratio,kind='stable')
    return float(ratio[order][np.searchsorted(np.cumsum(w[order]),.5*w.sum())])

def concentration(y,b,p):
    y=np.asarray(y,dtype=float);eb=np.abs(y-b);ep=np.abs(y-p);rows=[]
    for unit in ['row','product']:
        mass=y.ravel() if unit=='row' else y.sum(axis=1)
        be=eb.ravel() if unit=='row' else eb.sum(axis=1)
        pe=ep.ravel() if unit=='row' else ep.sum(axis=1)
        order=np.argsort(-mass,kind='stable')
        for frac in [.001,.01,.05,.1]:
            n=max(1,int(np.ceil(frac*len(mass))));idx=order[:n];rest=order[n:]
            rows.append(dict(unit=unit,top_fraction=frac,count=n,
                 target_share=float(mass[idx].sum()/mass.sum()),
                 baseline_error_share=float(be[idx].sum()/be.sum()),
                 proposal_error_share=float(pe[idx].sum()/pe.sum()),
                 share_of_total_error_reduction=float((be[idx]-pe[idx]).sum()/(be-pe).sum()),
                 remaining_baseline_wape=float(be[rest].sum()/mass[rest].sum()),
                 remaining_proposal_wape=float(pe[rest].sum()/mass[rest].sum())))
    return rows

def run(root,data,out):
    out.mkdir(parents=True,exist_ok=True)
    u=root/'evidence/uci/run_001'
    write(out/'AUDIT_PROTOCOL.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='POST_HOC_DESCRIPTIVE_AUDIT_AND_VALIDATION_ONLY_SCALE_DIAGNOSTIC',
        code_sha256=sha(__file__),uci_final_freeze_sha256=sha(u/'FINAL_FREEZE.json'),
        scales=[2.,2.5,3.,4.],continuous_scale='exact nonnegative weighted-median solution',
        new_test_policies=0,refits=0,
        rmsse='macro series RMSSE using pre-fit-cutoff recorded targets; not official hierarchical WRMSSE',
        concentration='outcome-ranked descriptive strata; never changes the primary target population'))
    spec=importlib.util.spec_from_file_location('uci_frozen',root/'code/run_study.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    z=np.load(u/'development.npz');vd=z['valid_days'];td=z['train_days']
    obs=z['observed'];cap=z['capacity'];dates=pd.date_range('2009-12-01',periods=obs.shape[1])
    xv=mod.features(obs,cap,dates,vd)
    model=lgb.Booster(model_file=str(u/'models/l1.txt'))
    b=np.maximum(model.predict(xv,num_threads=4),0).reshape(len(obs),-1);yv=z['truth'][:,vd]
    sopt=optimal_scale(yv,b);scales=[2.,2.5,3.,4.,sopt]
    vs=[dict(scale=s,**metrics(yv,s*b)) for s in scales]
    dev=json.loads((u/'DEVELOPMENT_RESULTS.json').read_text())
    assert abs(vs[0]['wape']-dev['baseline']['wape'])<1e-12
    np.savez_compressed(out/'UCI_VALIDATION_PREDICTIONS.npz',days=vd,truth=yv,unscaled_l1=b)
    context=np.load(u/'prediction_context.npz');test=np.load(u/'sealed_targets.npz');pr=np.load(u/'FROZEN_PREDICTIONS.npz')
    assert np.array_equal(test['days'],pr['days'])
    days=test['days'];yt=test['truth'].astype(float);bt=pr['baseline'];pt=pr['proposal']
    fit_end=int(td[-1]);q=rmsse_scale(z['truth'][:,:fit_end+1])
    uci=dict(validation_scale_diagnostic=vs,continuous_optimum_scale=sopt,
        frozen_correction_validation_wape=dev['proposal']['wape'],
        validation_scaled_base_minus_frozen_correction=vs[-1]['wape']-dev['proposal']['wape'],
        severity=dict(train=severity(z['truth'][:,td],obs[:,td],cap[:,td]),
            validation=severity(yv,obs[:,vd],cap[:,vd]),
            test=severity(yt,context['observed'][:,days],context['capacity'][:,days])),
        frozen_test_metrics=dict(baseline=metrics(yt,bt,q),proposal=metrics(yt,pt,q)),
        concentration=concentration(yt,bt,pt),test_new_predictions_computed=0,
        inputs={n:sha(u/n) for n in ['FINAL_FREEZE.json','FROZEN_PREDICTIONS.npz','sealed_targets.npz','development.npz']})
    write(out/'UCI_AUDIT.json',uci)
    print('UCI audit',json.dumps({k:uci[k] for k in ['validation_scale_diagnostic','severity','frozen_test_metrics']},indent=2),flush=True)
    m=np.load(data/'data/design_outcomes_v0_5.npz');my=m['truth'];ms=m['observed'];mc=m['capacity'];meta=m['cat_id']
    q=rmsse_scale(my[:,:1313]);m5={}
    for key in ['calibration_a','calibration_b','shadow']:
        ca=np.load(data/'cache'/f'{key}_aligned.npz');d=ca['target_days'];yy=my[:,d-1]
        assert np.array_equal(yy,ca['truth'])
        entry=dict(days=[int(d.min()),int(d.max())],severity=severity(yy,ms[:,d-1],mc[:,d-1]),
            metrics={n:metrics(yy,ca[n],q) for n in ['baseline','proposal']},
            categories={str(cat):severity(yy[meta==cat],ms[meta==cat][:,d-1],mc[meta==cat][:,d-1]) for cat in np.unique(meta)})
        for flag in [False,True]:
            ix=(yy>mc[:,d-1])==flag
            entry['censored_rows' if flag else 'non_censored_rows']={n:metrics(yy[ix],ca[n][ix]) for n in ['baseline','proposal']}
        m5[key]=entry
    write(out/'M5_AUDIT.json',m5)
    print('M5 audit',json.dumps({k:v['severity'] for k,v in m5.items()},indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args();run(v.root,v.data,v.output)
