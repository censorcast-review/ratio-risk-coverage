"""Seven predeclared post-hoc scores on consumed caches; no fits or external loader."""
from pathlib import Path
import argparse,json,time
import numpy as np
from r2_io import dump,sha,install_numpy_writers
from run_selective_study import load,R,FLOOR
from run_review2_objectives import curve,select_threshold,apply,KEYS
SEED=20260906
LAMBDAS=[0.,.25,.5,.75,1.]

def finite_ratio(e,mu):
    out=np.divide(e,mu,out=np.full(e.shape,np.finfo(np.float64).max),where=mu>0)
    out[(mu==0)&(e==0)]=0
    return out

def make_scores(pred,data,mean_ref):
    out={f'mixed_{lam:g}':{} for lam in LAMBDAS}
    out.update({n:{} for n in ['forecast_ratio_no_floor','demand_ratio_no_floor']})
    for k,v in data.items():
        e=np.maximum(pred['error_'+k],0).astype(float);mu=np.maximum(pred['demand_'+k],0).astype(float)
        out['forecast_ratio_no_floor'][k]=finite_ratio(e,v['proposal'].astype(float))
        out['demand_ratio_no_floor'][k]=finite_ratio(e,mu)
        for lam in LAMBDAS:
            if lam==0:
                s=np.full(e.shape,np.finfo(np.float64).max);valid=(mu>0)|(e==0)
                s[valid]=mean_ref*(finite_ratio(e[valid],mu[valid])-R)
            else:s=(e-R*mu)/(lam+(1-lam)*mu/mean_ref)
            out[f'mixed_{lam:g}'][k]=np.minimum(s,np.finfo(np.float64).max)
    return out

def main(a):
    install_numpy_writers();a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(status='POST_HOC_CONSUMED_CACHE',seed=SEED,lambdas=LAMBDAS,denominator_floor=0,zero_rule='0/0 is0; positive/0 sorts last',threshold_rule='largest common feasible threshold in development A/B; point WAPE<=r and rows>=c0',r=R,c0=FLOOR,training_calls=0,external_opened=False,code_sha256=sha(__file__),claim='Transferred operating curve, not an empirical Pareto guarantee or an independent test')
    dump(a.output/'TRADEOFF_PROTOCOL.json',protocol)
    dd={k:load(a.design/'cache_design'/f'{k}_aligned.npz') for k in KEYS};dm=load(a.design/'cache_design/metadata.npz');dp=load(a.design/f'design_predictions_s{SEED}.npz')
    mean_ref=sum(float(dd[k]['truth'].sum(dtype=np.float64)) for k in KEYS[:2])/sum(dd[k]['truth'].size for k in KEYS[:2]);ss=make_scores(dp,dd,mean_ref)
    records=[]
    for name,s in ss.items():
        cc={k:curve(s[k],dd[k]['truth'],dd[k]['proposal']) for k in KEYS[:2]}
        t,_=select_threshold(cc['calibration_a'],cc['calibration_b'],'row')
        records.append(dict(score=name,threshold=t,design=apply(dd,dm,s,t)))
        print('Selected on development',name,t,flush=True)
    dump(a.output/'TRADEOFF_DEVELOPMENT_SELECTED.json',dict(protocol=protocol,reference_mean=mean_ref,records=records))
    # Only after every development choice has been written load consumed guardian caches.
    guardian_sha=sha(a.guardian/'GUARDIAN_COMPARISON.json')
    gd={k:load(a.guardian/'cache'/f'{k}_aligned.npz') for k in KEYS};gm=load(a.guardian/'cache/metadata.npz');gp=load(a.guardian/f'predictions_s{SEED}.npz');gs=make_scores(gp,gd,mean_ref)
    for rec in records:rec['guardian_posthoc']=apply(gd,gm,gs[rec['score']],rec['threshold'])
    assert guardian_sha==sha(a.guardian/'GUARDIAN_COMPARISON.json')
    inp={str(p.relative_to(a.design)):sha(p) for p in [a.design/f'design_predictions_s{SEED}.npz',a.design/'cache_design/metadata.npz']}
    result=dict(protocol=protocol,reference_mean=mean_ref,records=records,input_hashes=inp,guardian_result_unchanged=True)
    dump(a.output/'TRADEOFF_RESULTS.json',result)
    for rec in records:print(rec['score'],rec['guardian_posthoc']['shadow']['ALL'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['design','guardian','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
