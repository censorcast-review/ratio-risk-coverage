"""Fixed chronological held-out-half floor diagnostic from replayed predictions.

Fit only scales and pooled edges on days <=1373; score days >=1374.  The
configuration (8 bins, .75 shrinkage, absolute multiplier floor .5) is fixed.
The input policy for Later remains the original full-validation frozen policy.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from replay_m5 import apply_saved_policy, load_json, sha256, write_json
from reproduce_m5_missing_controls import weighted_scale
from analyze_m5_cells_floor import metric_mass


def fit_fixed(y,b,q,cats):
    edges=np.unique(np.quantile(q,np.arange(1,8)/8)).tolist()
    bins=np.searchsorted(edges,q,side='right')
    cs={};sc={}
    for cat in np.unique(cats):
        catmask=cats==cat
        parent=weighted_scale(y[catmask],b[catmask]);cs[str(cat)]=parent
        for j in range(8):
            mask=catmask[:,None] & (bins==j)
            median=weighted_scale(y[mask],b[mask]) if np.any(b[mask]>0) else parent
            sc[f'{cat}|{j}']=float(np.exp(.25*np.log(max(parent,1e-8))+.75*np.log(max(median,1e-8))))
    return {'nbins':8,'shrink':.75,'edges':edges,'category_scales':cs,'scales':sc}


def render(result,later,paper):
    lines=[r'\begin{table}[htbp]',
           r'\caption{Fixed absolute-multiplier floor diagnostic, primary seed. Validation-half scales and pooled bin edges are fitted only on days through 1373 and evaluated on days 1374--1433. Later uses the original frozen full-validation policy. The floor $\max(\widetilde s_{c,j},0.5)$ multiplies the raw hierarchy forecast. This single floor is a post-hoc diagnostic, not a newly selected method.}',
           r'\label{tab:m5-floor}',r'\centering\small',
           r'\begin{tabular}{llrrr}\toprule',
           r'Population & Category & Base WAPE & RC WAPE & RC floor WAPE \\ \midrule']
    for display,metrics in [('Validation half',result['heldout_metrics']),('Later',later['populations']['later']['metrics'])]:
        for cat in ['pooled','FOODS','HOBBIES','HOUSEHOLD']:
            m=metrics[cat]
            lines.append(f"{display} & {cat.title()} & {m['base']['wape']:.5f} & {m['rc']['wape']:.5f} & {m['floor_0_5']['wape']:.5f} "+r'\\')
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (paper/'risk_floor_table.tex').write_text('\n'.join(lines)+'\n')


def main(args):
    root=args.package.resolve();directory=args.audit.resolve()
    if args.tables_only:
        render(load_json(directory/'SPLIT_RESULTS.json'),load_json(directory/'RESULTS.json'),root/'paper');return
    assert not (directory/'SPLIT_START.json').exists(), 'Use a new audit directory for a fresh run'
    amendment=directory/'AMENDMENT_SPLIT.json'
    assert amendment.exists(),'Record amendment before split scoring'
    start={'created_utc':datetime.now(timezone.utc).isoformat(),'amendment_sha256':sha256(amendment),
           'script_sha256':sha256(__file__),'weighted_scale_source_sha256':sha256(Path(__file__).with_name('reproduce_m5_missing_controls.py')),
           'prediction_sha256':sha256(directory/'point_validation_predictions.npz'),
           'input_sha256':sha256(args.inputs/'data/design_outcomes_v0_5.npz'),
           'status':'FIXED_CONFIG_BEFORE_CALIBRATION_FIT_AND_SCORING','model_fit_calls':0,
           'fixed_calibration_fit_calls':1,'policy_selection_calls':0,'new_holdout_accesses':0}
    write_json(directory/'SPLIT_START.json',start)
    z=np.load(directory/'point_validation_predictions.npz');days=z['days'];b=z['raw'];q=z['risk']
    panel=np.load(args.inputs/'data/design_outcomes_v0_5.npz',allow_pickle=True)
    # The archive has separate metadata arrays, matching DesignPanel's loader.
    cats=panel['cat_id'];y=panel['truth'][:,days-1].astype(float)
    fit=days<=1373;heldout=days>=1374
    policy=fit_fixed(y[:,fit],b[:,fit],q[:,fit],cats)
    write_json(directory/'SPLIT_POLICY.json',policy)
    baseline=np.asarray([policy['category_scales'][str(c)] for c in cats])[:,None]*b[:,heldout]
    floored=dict(policy);floored['scales']={k:max(v,.5) for k,v in policy['scales'].items()}
    forecasts={'base':baseline,'rc':apply_saved_policy(b[:,heldout],q[:,heldout],cats,policy),
               'floor_0_5':apply_saved_policy(b[:,heldout],q[:,heldout],cats,floored)}
    yy=y[:,heldout];metrics={}
    for cat in ['pooled',*list(np.unique(cats))]:
        mask=np.ones_like(yy,dtype=bool) if cat=='pooled' else np.broadcast_to((cats==cat)[:,None],yy.shape)
        metrics[cat]={name:metric_mass(yy[mask],p[mask]) for name,p in forecasts.items()}
    frozen=load_json(root/'evidence/risk_calibration/primary_policy/POLICY_FREEZE.json')
    original=frozen['choice']['selection']
    checks=[{'name':f'original_split_RC_{m}','difference':metrics['pooled']['rc'][m]-original[m],
             'pass':abs(metrics['pooled']['rc'][m]-original[m])<1e-9}
            for m in ['wape','mae','rmse','bias']]
    for cat in ['pooled',*list(np.unique(cats))]:
        for method in forecasts:
            m=metrics[cat][method]
            checks.append({'name':f'{cat}:{method}:error_mass_identity','pass':abs(m['absolute_error_mass']/m['demand_mass']-m['wape'])<1e-12})
    result={'finished_utc':datetime.now(timezone.utc).isoformat(),
            'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL',
            'fit_days':days[fit].tolist(),'heldout_days':days[heldout].tolist(),
            'heldout_metrics':metrics,'checks':checks,
            'start_sha256':sha256(directory/'SPLIT_START.json'),'policy_sha256':sha256(directory/'SPLIT_POLICY.json'),
            'scope':'post-hoc held-out time-half diagnostic, not independent confirmation or selection'}
    write_json(directory/'SPLIT_RESULTS.json',result)
    render(result,load_json(directory/'RESULTS.json'),root/'paper')
    print({'status':result['status'],'metrics':metrics,'checks':checks},flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--tables-only',action='store_true')
    main(p.parse_args())
