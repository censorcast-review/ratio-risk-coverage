"""Audit M5 cell attribution and a fixed post-hoc absolute multiplier floor.

No fit or policy search is performed.  The floor is max(saved_cell_scale, .5)
and multiplies the raw hierarchy forecast, not the category-scaled comparator.
The historical evidence is read only; all new outputs go to a fresh directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from replay_m5 import (DesignPanel, FeatureBuilder, HierarchyFeatures,
                       apply_saved_policy, load_json, predict_saved,
                       scores, sha256, write_json)


def utc():
    return datetime.now(timezone.utc).isoformat()


def metric_mass(y, f):
    result = scores(y, f)
    result.update(rows=int(y.size), demand_mass=float(y.sum()),
                  absolute_error_mass=float(np.abs(y-f).sum()),
                  squared_error_mass=float(((y-f)**2).sum()),
                  prediction_mass=float(f.sum()))
    return result


def render_tables(result, paper):
    """Generate table values from the same audited sufficient statistics."""
    rows = [r for r in result['cells'] if r['population'] == 'later']
    lines = [r'\begin{table}[htbp]',
             r'\caption{All 24 primary-seed cells on Later. Bin edges are pooled across categories on point validation. Shares use the full Later population. Mult. is the absolute multiplier on the raw hierarchy forecast; Base includes the frozen category scale. Strict denotes benchmark demand above capacity.}',
             r'\label{tab:cells}', r'\centering\scriptsize',
             r'\begin{tabular}{lrrrrrrr}\toprule',
             r'Category & Bin & Mult. & Rows (\%) & Demand (\%) & Strict (\%) & Base WAPE & RC WAPE \\ \midrule']
    for i, row in enumerate(rows):
        if i and row['category'] != rows[i-1]['category']:
            lines.append(r'\addlinespace')
        mult = r'$1.0\!\times\!10^{-6}$' if row['multiplier'] < 1e-5 else f"{row['multiplier']:.4f}"
        lines.append(f"{row['category'].title()} & {row['bin']} & {mult} & "
                     f"{100*row['row_share']:.2f} & {100*row['demand_share']:.2f} & "
                     f"{100*row['strict_row_share']:.2f} & {row['base_wape']:.5f} & "
                     f"{row['calibrated_wape']:.5f} " + r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}']
    (paper / 'risk_cells_table.tex').write_text('\n'.join(lines)+'\n')
    lines = [r'\begin{table}[htbp]',
             r'\caption{Fixed multiplier-floor diagnostic, primary seed. RC floor uses $\max(\widetilde s_{c,j},0.5)b$ without refitting. Validation is the final calibration sample, so its loss is in-sample; Later was previously consumed. No floor is selected from these results.}',
             r'\label{tab:m5-floor}', r'\centering\small',
             r'\begin{tabular}{llrrr}\toprule',
             r'Population & Category & Base WAPE & RC WAPE & RC floor WAPE \\ \midrule']
    for population, display in [('point_validation', 'Validation'), ('later', 'Later')]:
        for cat in ['pooled', 'FOODS', 'HOBBIES', 'HOUSEHOLD']:
            m=result['populations'][population]['metrics'][cat]
            lines.append(f"{display} & {cat.title()} & {m['base']['wape']:.5f} & "
                         f"{m['rc']['wape']:.5f} & {m['floor_0_5']['wape']:.5f} " + r'\\')
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}']
    (paper / 'risk_floor_table.tex').write_text('\n'.join(lines)+'\n')


def main(args):
    root=args.package.resolve(); inputs=args.inputs.resolve(); out=args.output.resolve()
    if args.tables_only:
        render_tables(load_json(out/'RESULTS.json'), root/'paper')
        return
    out.mkdir(parents=True, exist_ok=False)
    source = {
        'hierarchy_base': root/'evidence/risk_calibration/primary_base/model.ubj',
        'hit_model': root/'evidence/revision/m5_observable/seed_20260906/hit.txt',
        'policy': root/'evidence/risk_calibration/primary_policy/POLICY_FREEZE.json',
        'validation': root/'evidence/risk_calibration/primary_base/validation.npz',
        'old_cells': root/'evidence/risk_calibration/attribution_controls/RESULTS.json',
        'original_results': root/'evidence/risk_calibration/primary_policy/RESULTS.json',
        'category_results': root/'evidence/risk_calibration/primary_analysis/RESULTS.json',
        'design': inputs/'data/design_outcomes_v0_5.npz',
        'calendar': inputs/'data/calendar.csv', 'prices': inputs/'data/sell_prices.csv',
        'later_index': inputs/'cache/shadow_aligned.npz',
    }
    input_hashes={key: sha256(path) for key,path in source.items()}
    protocol={'created_utc': utc(), 'status': 'FIXED_BEFORE_NEW_SCORING',
              'analysis_type': 'post-hoc descriptive audit on previously consumed M5',
              'seed': 20260906, 'fit_calls': 0, 'policy_search_calls': 0,
              'new_holdout_accesses': 0, 'floor': 0.5,
              'floor_definition': 'absolute multiplier on raw hierarchy forecast: max(saved shrunk scale, 0.5) * raw forecast',
              'comparator_definition': 'base_category_scale[category] * raw hierarchy forecast',
              'validation_scope': 'entire point-validation set used for final policy refit; in-sample calibration loss',
              'quantile_scope': 'pooled over every category and row in point validation; frozen edges applied unchanged',
              'no_selection': 'exactly one floor (0.5), no grid, no method replacement using Later',
              'input_sha256': input_hashes, 'script_sha256': sha256(__file__),
              'replay_helper_sha256': sha256(Path(__file__).with_name('replay_m5.py'))}
    write_json(out/'PROTOCOL.json',protocol)
    start=time.monotonic()
    panel=DesignPanel.load(source['design'],source['calendar'],source['prices'])
    panel.censored=(panel.observed>=panel.capacity).astype(np.uint8)
    builder=FeatureBuilder(panel); hierarchy=HierarchyFeatures(panel)
    cats=panel.metadata['cat_id']
    model=xgb.Booster(params={'nthread':args.threads}); model.load_model(source['hierarchy_base'])
    model.set_param({'nthread':args.threads})
    hit=lgb.Booster(model_file=str(source['hit_model']))
    frozen=load_json(source['policy']); policy=frozen['risk_policy']
    old=load_json(source['old_cells']); reference=load_json(source['original_results'])
    all_checks=[]
    def check(label,actual,expected,tol=1e-9):
        actual=np.asarray(actual);expected=np.asarray(expected)
        diff=float(np.max(np.abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
        all_checks.append({'name':label,'pass':actual.shape==expected.shape and diff<=tol,'max_abs_diff':diff,'tolerance':tol})
    result={'populations':{},'cells':[], 'input_sha256':input_hashes,
            'root_cause': {'type':'comparator definition mismatch, not different model bytes',
                          'source':'code/risk_calibration/run_attribution_and_integration_controls.py: cell_rows and its raw-base call sites',
                          'old_base_column':'absolute error of raw hierarchy forecast',
                          'correct_base_column':'absolute error of category-scaled hierarchy forecast',
                          'rc_column':'saved absolute cell multiplier times raw hierarchy forecast; unchanged'},
            'base_category_scales':frozen['base_category_scales'],
            'risk_edges':policy['edges']}
    for pop,days in [('point_validation',np.load(source['validation'])['days']),
                     ('later',np.load(source['later_index'])['target_days'])]:
        print(f'Predicting {pop}, {len(days)} days',flush=True)
        raw,q,_=predict_saved(builder,hierarchy,model,hit,days,args.threads)
        y=panel.truth[:,days-1].astype(float);cap=panel.capacity[:,days-1]
        bins=np.searchsorted(policy['edges'],q,side='right')
        category_multiplier=np.asarray([frozen['base_category_scales'][str(c)] for c in cats])[:,None]
        mult=np.empty_like(raw,dtype=float)
        for cat in np.unique(cats):
            catmask=cats==cat
            mult[catmask]=np.asarray([policy['scales'][f'{cat}|{j}'] for j in range(8)])[bins[catmask]]
        forecasts={'raw':raw,'base':category_multiplier*raw,'rc':mult*raw,
                   'floor_0_5':np.maximum(mult,0.5)*raw}
        check(f'{pop}:rc_policy_reconstruction',forecasts['rc'],apply_saved_policy(raw,q,cats,policy),0)
        if pop=='point_validation':
            check('validation_saved_forecast',raw,np.load(source['validation'])['prediction'],0)
            check('pooled_quantile_edges',np.unique(np.quantile(q,np.arange(1,8)/8)),policy['edges'],0)
        metrics={}
        for cat in ['pooled',*list(np.unique(cats))]:
            mask=np.ones_like(y,dtype=bool) if cat=='pooled' else np.broadcast_to((cats==cat)[:,None],y.shape)
            metrics[cat]={name:metric_mass(y[mask],p[mask]) for name,p in forecasts.items()}
        if pop=='later':
            ref=next(r for r in reference['rows'] if r['block']=='shadow')
            for name,oldname in [('base','base'),('rc','risk')]:
                for metric in ['wape','mae','rmse','bias']:
                    check(f'later:{name}:{metric}',metrics['pooled'][name][metric],ref[oldname][metric])
        for cat in np.unique(cats):
            for j in range(8):
                mask=(cats==cat)[:,None] & (bins==j)
                n=int(mask.sum());mass=float(y[mask].sum())
                row={'population':pop,'category':str(cat),'bin':j+1,
                     'multiplier':policy['scales'][f'{cat}|{j}'],
                     'relative_to_base_multiplier':policy['scales'][f'{cat}|{j}']/frozen['base_category_scales'][str(cat)],
                     'rows':n,'row_share':n/y.size,'demand_mass':mass,'demand_share':mass/float(y.sum()),
                     'mean_risk':float(q[mask].mean()),'strict_row_share':float((y[mask]>cap[mask]).mean())}
                for name,p in forecasts.items():
                    mm=metric_mass(y[mask],p[mask])
                    for key,value in mm.items(): row[f'{name}_{key}']=value
                row['calibrated_wape']=row['rc_wape']
                row['delta_absolute_error_mass']=row['rc_absolute_error_mass']-row['base_absolute_error_mass']
                row['pooled_wape_delta_contribution']=row['delta_absolute_error_mass']/float(y.sum())
                previous=next(r for r in old['cells'] if r['population']==pop and r['category']==cat and r['bin']==j+1)
                check(f'{pop}:{cat}:{j+1}:old_base_is_raw',row['raw_wape'],previous['base_wape'])
                # Original cell routine multiplies float32 by a scalar; tiny rounding differs from float64 policy application.
                check(f'{pop}:{cat}:{j+1}:rc_matches_old',row['rc_wape'],previous['calibrated_wape'],2e-7)
                check(f'{pop}:{cat}:{j+1}:row_count',n,previous['rows'],0)
                check(f'{pop}:{cat}:{j+1}:demand_mass',mass,previous['demand_mass'],0)
                result['cells'].append(row)
        population_cells=[r for r in result['cells'] if r['population']==pop]
        for cat in ['pooled',*list(np.unique(cats))]:
            subset=[r for r in population_cells if cat=='pooled' or r['category']==cat]
            for method in forecasts:
                error=sum(r[f'{method}_absolute_error_mass'] for r in subset)
                mass=sum(r['demand_mass'] for r in subset)
                check(f'{pop}:{cat}:{method}:cell_sum_error',error,metrics[cat][method]['absolute_error_mass'],1e-6)
                check(f'{pop}:{cat}:{method}:cell_weighted_wape',error/mass,metrics[cat][method]['wape'])
        low=mult<1e-5
        low_metrics={name:metric_mass(y[low],p[low]) for name,p in forecasts.items()}
        result['populations'][pop]={'days':[int(days.min()),int(days.max())], 'metrics':metrics,
            'low_multiplier_cells':{'threshold':1e-5,'rows':int(low.sum()),'row_share':float(low.mean()),
                                    'demand_mass':float(y[low].sum()),'demand_share':float(y[low].sum()/y.sum()),
                                    'metrics':low_metrics,'raw_positive_forecast_rows':int((raw[low]>0).sum()),
                                    'pooled_wape_delta_contribution':float((np.abs(y[low]-forecasts['rc'][low]).sum()-np.abs(y[low]-forecasts['base'][low]).sum())/y.sum())}}
        np.savez_compressed(out/f'{pop}_predictions.npz',days=days,raw=raw,risk=q)
        print({pop:metrics['pooled']},flush=True)
    receipt={'finished_utc':utc(),'elapsed_seconds':time.monotonic()-start,'checks':all_checks,
             'check_count':len(all_checks),'status':'PASS' if all(c['pass'] for c in all_checks) else 'FAIL',
             'protocol_sha256':sha256(out/'PROTOCOL.json'),
             'old_inputs_unchanged':all(sha256(path)==input_hashes[key] for key,path in source.items())}
    write_json(out/'RESULTS.json',result);write_json(out/'RECEIPT.json',receipt)
    with (out/'cells.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['cells'][0]));writer.writeheader();writer.writerows(result['cells'])
    render_tables(result,root/'paper')
    print({'status':receipt['status'],'checks':len(all_checks),'failed':[c for c in all_checks if not c['pass']]},flush=True)
    if receipt['status']!='PASS' or not receipt['old_inputs_unchanged']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--tables-only',action='store_true')
    main(p.parse_args())
