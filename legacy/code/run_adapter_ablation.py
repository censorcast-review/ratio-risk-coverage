"""Zero-fit ablations of post-processing against a metric-matched forecaster."""
from pathlib import Path
import argparse,json
import numpy as np
from run_point_baselines import metrics,dump,atomic_npz

def weighted_median(v,w):
    idx=np.argsort(v);vv=v[idx];ww=w[idx];return float(vv[np.searchsorted(np.cumsum(ww),ww.sum()/2)])

def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    with np.load(a.input/'design_outcomes_v0_5.npz') as z:cat=z['cat_id'];items=z['item_id']
    with np.load(a.input/'selection_base_v0_5.npz') as z:selection={k:z[k] for k in z.files}
    with np.load(a.input/'shadow_v0_5.npz') as z:shadow={k:z[k] for k in z.files}
    report=[]
    for seed in (20260906,20260907,20260908):
        with np.load(a.points/f'observed_l1_s{seed}.npz') as z:pred={k:z[k] for k in z.files}
        ix=np.searchsorted(selection['target_days'],pred['selection_days']);y=selection['truth'][:,ix];ps=pred['selection']
        variants={'plain':(ps,pred['shadow'],{})}
        # Generic global and category scaling are essential controls for censor-specific correction.
        fac=weighted_median((y/np.maximum(ps,1e-8)).ravel(),ps.ravel())
        fac=float(np.clip(fac,.5,2))
        variants['global_scale']=(ps*fac,pred['shadow']*fac,{'factor':fac})
        s=ps.copy();h=pred['shadow'].copy();factors={}
        for c in np.unique(cat):
            m=cat==c;f=float(np.clip(weighted_median((y[m]/np.maximum(ps[m],1e-8)).ravel(),ps[m].ravel()),.5,2));factors[str(c)]=f
            s[m]*=f;h[m]*=f
        variants['category_scale']=(s,h,{'factors':factors})
        grid_ungated=[]
        for weight in [0,.125,.25,.5,.75,1,1.5,2]:
            corr=weight*np.maximum(selection['censored_poisson_em'][:,ix]-ps,0)
            grid_ungated.append((metrics(y,ps+corr)['wape'],weight))
        _,weight=min(grid_ungated)
        variants['ungated_em_correction']=(ps+weight*np.maximum(selection['censored_poisson_em'][:,ix]-ps,0),
            pred['shadow']+weight*np.maximum(shadow['censored_poisson_em']-pred['shadow'],0),{'weight':weight})
        grid=[]
        for strength in [0,.25,.5,.75,1,1.5,2]:
            for power in [.5,1,2]:
                cor=strength*selection['censor_probability'][:,ix]**power*np.maximum(selection['censored_poisson_em'][:,ix]-ps,0)
                grid.append((metrics(y,ps+cor)['wape'],strength,power))
        _,strength,power=min(grid)
        ss=ps+strength*selection['censor_probability'][:,ix]**power*np.maximum(selection['censored_poisson_em'][:,ix]-ps,0)
        hh=pred['shadow']+strength*shadow['censor_probability']**power*np.maximum(shadow['censored_poisson_em']-pred['shadow'],0)
        variants['censor_adapter']=(ss,hh,dict(strength=strength,power=power,selection_candidates=grid))
        for name,(s,h,info) in variants.items():
            row=dict(seed=seed,method=name,choice_on_selection=info,selection=metrics(y,s),shadow=metrics(shadow['truth'],h))
            report.append(row)
            atomic_npz(a.output/f'{name}_s{seed}.npz',dict(selection=s,shadow=h,selection_days=pred['selection_days'],shadow_days=pred['shadow_days']))
            print(seed,name,row['shadow'],flush=True)
    dump(a.output/'ADAPTER_RESULTS.json',dict(status='METRIC_MATCHED_ADAPTER_ABLATION_COMPLETE',records=report,
         scope='post_hoc_development; all choices use selection only',fresh_guardian_opened=False,external_opened=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--points',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    main(ap.parse_args())
