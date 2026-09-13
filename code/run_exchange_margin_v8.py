"""Retrospective fixed-policy exchange intervals and frozen calibration margins.

Run --init before --run. The protocol is bound to code/input hashes. No fitting,
threshold changes, policy reselection, or external-data access occurs here.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import numpy as np
from policy_audit import METHODS, scores, mask, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/revision_v8_exchange'
SOURCE = ROOT/'evidence/matched_censoring'
ARMS = ('censored', 'complete')
HEADS = ((220, 220), (660, 220), (220, 660), (660, 660))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inputs():
    paths = [Path(__file__), ROOT/'code/policy_audit.py', SOURCE/'PROTOCOL.json']
    for arm in ARMS:
        paths += [SOURCE/arm/f for f in ('FROZEN.json', 'RESULTS.json',
                                         'CALIBRATION.npz', 'EVALUATION.npz')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def init():
    OUT.mkdir(exist_ok=False)
    dump(OUT/'PROTOCOL.json', dict(created_utc=now(), analysis='revision_v8_exchange',
        scope='Retrospective development sensitivity on already evaluated data; no fresh confirmation.',
        inputs=inputs(), arms=list(ARMS), heads=[list(p) for p in HEADS],
        exchange=dict(relative_cap_factor=.95, definition='delta_D / (-delta_C)',
            contrast='original frozen exposure policy minus original frozen case policy',
            bootstrap=dict(draws=4000, seed=20260915, unit='M5 item, jointly across stores and days',
                pairing='Identical item multiplicities for all arms, cells, and branch comparisons.',
                conditional_on='fixed original fitted models, heads, caps, thresholds, and menu choices',
                individual_percentiles=[.025, .975],
                four_arm_contrast_bonferroni_percentiles=[.00625, .99375],
                invalid_rule='If any draw has nonpositive case-loss denominator or total exposure, report no ratio interval; never silently omit such draws.'),
            arm_comparison='complete minus censored exchange ratio at each identical head pair',
            range='point-estimate minimum and maximum over all four head pairs per arm'),
        calibration=dict(relative_factors=[.85, .90, .95], heads=[220, 220], floor=.35,
            sources='original frozen thresholds; reconstruct all five candidate masks and calibration-window metrics',
            margin='minimum calibration exposure coverage of Ratio minus Descending only when both feasible',
            other_margin='Ratio minus strongest feasible other candidate; report winning candidate and eligible count'),
        branch=dict(arm='complete', heads=[220,220], relative_factor=.95,
            comparisons=['original frozen Excess versus Ratio', 'original frozen Excess versus Descending'],
            interpretation='Deterministic alternative fixed-policy comparisons; not selection probabilities or optimality.'),
        safeguards=['No model fits', 'No new thresholds', 'No menu choices changed',
            'No external outcomes opened', 'No claim that a fixed-policy interval includes calibration selection uncertainty',
            'No causal conclusion about the error head from candidate margins']))
    print('PROTOCOL WRITTEN', sha(OUT/'PROTOCOL.json'), flush=True)


def plan(frozen, factor, heads):
    rows = [p for p in frozen['plans'] if p['kind']=='relative' and
            p['value']==factor and (p['e_trees'],p['w_trees'])==heads]
    assert len(rows)==1
    return rows[0]


def independent_quantile(values, qs):
    ordered = sorted(float(v) for v in values)
    result=[]
    for q in qs:
        x=q*(len(ordered)-1); i=int(np.floor(x)); j=int(np.ceil(x))
        result.append(ordered[i]*(j-x)+ordered[j]*(x-i) if j!=i else ordered[i])
    return result


def interval(values, qs=(.025,.975)):
    direct=np.quantile(values, qs)
    assert np.allclose(direct, independent_quantile(values, qs), atol=1e-12, rtol=1e-12)
    return direct.tolist()


def cluster_total(W, units, n):
    return np.column_stack([np.bincount(units, minlength=n),
                            np.bincount(units, weights=W, minlength=n)])


def cluster_policy(W, a, units, n, series_items, days):
    result=cluster_total(W*a, units, n)
    result[:,0]=np.bincount(units, weights=a, minlength=n)
    # Independent aggregation first by series, then item, catches wrong row order.
    per_series=np.column_stack([a.reshape(-1,days).sum(axis=1),
                               (W*a).reshape(-1,days).sum(axis=1)])
    independent=np.zeros((n,2)); np.add.at(independent,series_items,per_series)
    assert np.array_equal(result, independent)
    assert np.array_equal(result.sum(axis=0), [a.sum(),W[a].sum()])
    return result


def exchange_summary(diff, totals, counts):
    point=diff.sum(axis=0)/totals.sum(axis=0)
    denominator=counts@totals
    draw_delta=(counts@diff)/denominator
    case_loss=-draw_delta[:,0]
    bad=(case_loss<=0)|(~np.isfinite(case_loss))|(denominator[:,1]<=0)
    point_ratio=float(point[1]/(-point[0])) if point[0]<0 else None
    rr=np.full(len(counts),np.nan)
    rr[~bad]=draw_delta[~bad,1]/case_loss[~bad]
    summary=dict(delta_case_pp=float(100*point[0]), delta_exposure_pp=float(100*point[1]),
        exchange_ratio=point_ratio,
        ratio_ci95=None if bad.any() else interval(rr),
        case_loss_pp_ci95=interval(100*case_loss),
        exposure_gain_pp_ci95=interval(100*draw_delta[:,1]),
        denominator=dict(bad_draws=int(bad.sum()), nonpositive_case_loss_draws=int((case_loss<=0).sum()),
            case_loss_min_pp=float(100*case_loss.min()), case_loss_max_pp=float(100*case_loss.max()),
            total_exposure_min=float(denominator[:,1].min())),
        interval_scope='Fixed fits, caps, thresholds, and policy choices; evaluation item resampling only.')
    return summary, rr, draw_delta


def run():
    proto=json.loads((OUT/'PROTOCOL.json').read_text())
    assert proto['inputs']==inputs(), 'Protocol-bound code or input changed'
    if (OUT/'RESULTS.json').exists():
        raise FileExistsError('Refusing to overwrite completed analysis')
    print('PROTOCOL VERIFIED', flush=True)
    frozen={a:json.loads((SOURCE/a/'FROZEN.json').read_text()) for a in ARMS}
    old={a:json.loads((SOURCE/a/'RESULTS.json').read_text()) for a in ARMS}
    result=dict(protocol_sha256=sha(OUT/'PROTOCOL.json'), scope=proto['scope'],
                exchange_cells=[], paired_arm_contrasts=[], calibration=[], branches=[])
    arrays={}; common=None; counts=None
    for arm in ARMS:
        with np.load(SOURCE/arm/'EVALUATION.npz') as z:
            ev={k:z[k] for k in z.files}
        W=ev['y']; L=abs(W-ev['f']); si=ev['series_item_index']; nd=len(ev['days'])
        units=np.repeat(si, nd); n=len(ev['items']); total=cluster_total(W,units,n)
        if common is None:
            common={k:ev[k] for k in ('y','items','series_item_index','days')}
            counts=np.random.default_rng(20260915).multinomial(n,np.full(n,1/n),size=4000)
            arrays.update(counts=counts, total=total, item_names=ev['items'])
        else:
            assert all(np.array_equal(ev[k],v) for k,v in common.items())
            assert np.array_equal(total,arrays['total'])
        for heads in HEADS:
            p=plan(frozen[arm],.95,heads); e=ev['e'+str(heads[0])]; w=ev['w'+str(heads[1])]
            selected=p['selected']; stats={}
            for method in set(selected.values())|({'demand','weight_descending'} if arm=='complete' and heads==(220,220) else set()):
                a=mask(scores(e,w,p['cap'],method,frozen[arm]['T']),p['policies'][method]['threshold'])
                stats[method]=cluster_policy(W,a,units,n,si,nd)
            diff=stats[selected['d']]-stats[selected['c']]
            summary, rr, deltas=exchange_summary(diff,total,counts)
            summary.update(arm=arm, heads=list(heads), cap=p['cap'], selected=selected)
            expected=next(x for x in old[arm]['menus'] if x['kind']=='relative' and x['value']==.95 and (x['e_trees'],x['w_trees'])==heads)
            assert np.allclose([summary['delta_case_pp'],summary['delta_exposure_pp']],
                               100*np.array([expected['contrast']['dc'],expected['contrast']['dd']]),atol=1e-10)
            result['exchange_cells'].append(summary)
            key=f'{arm}_{heads[0]}_{heads[1]}'
            arrays[key+'_ratio']=rr; arrays[key+'_diff']=diff; arrays[key+'_draw_delta']=deltas
            if arm=='complete' and heads==(220,220):
                for method in ('demand','weight_descending'):
                    branch,br,bd=exchange_summary(stats[method]-stats[selected['c']],total,counts)
                    branch.update(case_method=selected['c'], exposure_method=method, arm=arm, heads=list(heads),cap=p['cap'])
                    result['branches'].append(branch)
                    arrays['branch_'+method+'_ratio']=br
            print('EXCHANGE',key,summary['exchange_ratio'],summary['ratio_ci95'],flush=True)
        del ev, L, W
    for heads in HEADS:
        key=f'{heads[0]}_{heads[1]}'
        a=arrays['complete_'+key+'_ratio']; b=arrays['censored_'+key+'_ratio']; v=a-b
        points=[next(x['exchange_ratio'] for x in result['exchange_cells'] if x['arm']==arm and x['heads']==list(heads)) for arm in ARMS]
        valid=np.isfinite(v).all()
        result['paired_arm_contrasts'].append(dict(heads=list(heads), complete_minus_censored=points[1]-points[0],
            ci95=interval(v) if valid else None,
            four_contrast_bonferroni_interval=interval(v,(.00625,.99375)) if valid else None,
            bootstrap_fraction_positive=float((v>0).mean()) if valid else None,
            caveat='Approximate descriptive bootstrap intervals conditional on original fits and choices; multiplicity adjustment does not establish population coverage.'))
    result['arm_ranges']={a:[min(x['exchange_ratio'] for x in result['exchange_cells'] if x['arm']==a),
                            max(x['exchange_ratio'] for x in result['exchange_cells'] if x['arm']==a)] for a in ARMS}
    for arm in ARMS:
        with np.load(SOURCE/arm/'CALIBRATION.npz') as z:
            cal={k:z[k] for k in z.files}
        L=abs(cal['y']-cal['f']); W=cal['y']; blocks=cal['blocks']
        for factor in (.85,.90,.95):
            p=plan(frozen[arm],factor,(220,220)); candidates={}
            for method in METHODS:
                policy=p['policies'][method]
                a=mask(scores(cal['e220'],cal['w220'],p['cap'],method,frozen[arm]['T']),policy['threshold'])
                got=[metrics(L[blocks==b],W[blocks==b],a[blocks==b]) for b in np.unique(blocks)]
                for g, expected in zip(got,policy['calibration']):
                    for k,v in expected.items():
                        assert g[k] is None if v is None else np.isclose(g[k],v,atol=1e-9,rtol=1e-12),(arm,factor,method,k)
                feasible=policy['threshold'] is not None and all(g['weight']>0 and g['c']>=.35 and g['loss']<=p['cap']*g['weight']+1e-9 for g in got)
                candidates[method]=dict(threshold=policy['threshold'], feasible=feasible,
                    min_case_pp=min(g['c'] for g in got)*100, min_exposure_pp=min(g['d'] for g in got)*100,
                    calibration=got)
            eligible=[m for m in METHODS if candidates[m]['feasible']]
            winner=max(eligible,key=lambda m:candidates[m]['min_exposure_pp']) if eligible else None
            assert winner==p['selected']['d']
            other=[m for m in eligible if m!='demand']
            runner=max(other,key=lambda m:candidates[m]['min_exposure_pp']) if other else None
            ratio=candidates['demand']; desc=candidates['weight_descending']
            row=dict(arm=arm, heads=[220,220],relative_factor=factor, cap=p['cap'],candidates=candidates,
                eligible_count=len(eligible), exposure_winner=winner,
                ratio_minus_descending_pp=ratio['min_exposure_pp']-desc['min_exposure_pp'] if ratio['feasible'] and desc['feasible'] else None,
                descending_feasible=desc['feasible'], ratio_feasible=ratio['feasible'],
                strongest_other_feasible=runner,
                ratio_minus_strongest_other_pp=ratio['min_exposure_pp']-candidates[runner]['min_exposure_pp'] if ratio['feasible'] and runner is not None else None)
            result['calibration'].append(row)
            print('MARGIN',arm,factor,row['ratio_minus_descending_pp'],row['strongest_other_feasible'],row['ratio_minus_strongest_other_pp'],flush=True)
    np.savez_compressed(OUT/'BOOTSTRAP.npz',**arrays)
    dump(OUT/'RESULTS.json',result)
    lines=['# Retrospective exchange and calibration-margin audit (v8)','',
      'Original models, thresholds, and menu choices are fixed. Intervals describe item resampling of previously evaluated outcomes; they do not include training or calibration selection uncertainty. No distribution under policy reselection is asserted.','',
      '| Arm | Heads | Exchange ratio | Conditional 95% interval | Minimum bootstrap case loss (pp) |',
      '|---|---|---:|---|---:|']
    for x in result['exchange_cells']:
        ci=x['ratio_ci95']; lines.append(f"| {x['arm']} | {'/'.join(map(str,x['heads']))} | {x['exchange_ratio']:.6f} | {ci[0]:.6f}, {ci[1]:.6f} | {x['denominator']['case_loss_min_pp']:.4f} |")
    lines+=['','| Arm | Relative cap | Eligible | Exposure winner | Ratio minus Descending (pp) | Ratio minus best other (pp) |','|---|---:|---:|---|---:|---:|']
    for x in result['calibration']:
        m=x['ratio_minus_descending_pp']; q=x['ratio_minus_strongest_other_pp']
        lines.append(f"| {x['arm']} | {x['relative_factor']:.2f} | {x['eligible_count']} | {x['exposure_winner']} | {'infeasible comparator' if m is None else f'{m:.6f}'} | {'NA' if q is None else f'{q:.6f}'} |")
    lines+=['','A missing Ratio–Descending margin means at least one family had no feasible original threshold under the risk constraint and 35% floor. It is not an exposure-score margin. Margin results do not isolate the causal value of error estimation.','',
        'Files: PROTOCOL.json fixes conditions and input/code hashes; RESULTS.json contains all candidate window metrics, all eight exchange cells, four paired arm contrasts, denominator diagnostics, and both reference-cell branches; BOOTSTRAP.npz contains common multiplicities, sufficient statistics, and draws.','',
        'Reproduce in a clean output location with `python code/run_exchange_margin_v8.py --init` followed by `python code/run_exchange_margin_v8.py --run`. The script refuses overwriting completed evidence.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'CHECKS.json',dict(completed_utc=now(),same_item_pairs_all_arms_and_cells=True,
        direct_mask_sums_equal_independent_series_then_item_sums=True,
        all_eight_point_contrasts_match_original_results=True,
        all_30_calibration_candidates_match_original_frozen_metrics=True,
        interval_endpoints_equal_independent_sorted_linear_interpolation=True,
        quantile_axis='one-dimensional replicate axis per contrast',
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--init',action='store_true'); parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.init==args.run: parser.error('Choose exactly one of --init or --run')
    init() if args.init else run()
