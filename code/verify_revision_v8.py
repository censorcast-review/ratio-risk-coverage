"""Independent, read-only checks of v8 evidence; no production audit imports.

Writes a report only under reproduction_outputs unless --output is provided.
Uses original arrays, an independently expressed selection rule, and item sums
computed via series/day matrices to check the saved evidence.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('error', 'row', 'mixed', 'demand', 'weight_descending')
CHECKS = {}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def close(a, b, label='', atol=1e-9):
    np.testing.assert_allclose(a, b, atol=atol, rtol=2e-11, err_msg=label)


def ranking(e, w, r, method, scale):
    if method == 'error':
        return e
    if method == 'weight_descending':
        return -w
    excess = e-r*w
    if method == 'row':
        return excess
    den = w/scale if method == 'demand' else .25+.75*w/scale
    result = np.zeros_like(excess)
    pos = den > 0
    result[pos] = excess[pos]/den[pos]
    result[(~pos) & (excess > 0)] = np.inf
    result[(~pos) & (excess < 0)] = -np.inf
    return result


def acceptance(s, t):
    if t is None:
        return np.zeros(s.shape, bool)
    if t == 'all':
        return np.ones(s.shape, bool)
    return s <= t


def measure(a, y, loss):
    n = int(np.count_nonzero(a))
    w, l = float(y[a].sum()), float(loss[a].sum())
    return dict(rows=n, c=n/len(y), d=w/y.sum(), weight=w, loss=l,
                risk=l/w if w else None)


def compare_metrics(got, stored):
    for k in ('rows', 'c', 'd', 'weight', 'loss', 'risk'):
        if got[k] is None:
            assert stored[k] is None
        else:
            close(got[k], stored[k], k, atol=1e-7 if k in ('weight','loss') else 1e-10)


def series_item_sum(values, item_index, days, n):
    # Independent from production bincount: sum temporal rows, then add series.
    totals = np.asarray(values).reshape(len(item_index), days).sum(axis=1)
    result = np.zeros(n)
    np.add.at(result, item_index, totals)
    return result


def item_stats(a, y, loss, ii, days, n):
    return np.column_stack([series_item_sum(v, ii, days, n)
                            for v in (a, a*y, a*loss)])


def select(utilities, eligible, epsilon):
    ids = [i for i in range(len(METHODS)) if eligible[i]]
    if not ids:
        return -1
    best = max(utilities[i][1] for i in ids)
    band = [i for i in ids if best-utilities[i][1] <= epsilon+1e-12]
    return sorted(band, key=lambda i: (-utilities[i][0], -utilities[i][1], i))[0]


def provenance():
    hashed = {}
    def check_hashes(mapping):
        for p, digest in mapping.items():
            if p not in hashed:
                hashed[p] = sha(ROOT/p)
            assert hashed[p] == digest, p
    tp = ROOT/'evidence/revision_v8_tolerance'
    fp = ROOT/'evidence/revision_v8_floor'
    ep = ROOT/'evidence/revision_v8_exchange'
    t = read(tp/'PROTOCOL.json'); check_hashes(t['input_sha256'])
    assert sha(ROOT/'code/run_tolerance_v8.py') == t['code_sha256']
    f = read(fp/'START.json'); check_hashes(f['inputs_sha256']); check_hashes(f['code_sha256'])
    e = read(ep/'PROTOCOL.json'); check_hashes(e['inputs'])
    for folder, protocol_time, freeze_name, open_name in (
        (tp,t['utc'],'FROZEN_CHOICES.json','EVALUATION_OPEN.json'),
        (fp,read(fp/'PROTOCOL.json')['utc'],'FROZEN.json','EVALUATION_STARTED.json')):
        freeze = read(folder/freeze_name); opened = read(folder/open_name)
        times = [protocol_time,read(folder/'START.json')['utc'],freeze['utc'],opened['utc'],read(folder/'COMPLETE.json')['utc']]
        assert times == sorted(times)
        recorded = opened.get('frozen_choices_sha256',opened.get('freeze_sha256'))
        assert recorded == sha(folder/freeze_name)
        assert freeze['protocol_sha256'] == sha(folder/'PROTOCOL.json')
        completion = read(folder/'COMPLETE.json')
        for name, digest in completion.get('files',completion.get('files_sha256')).items():
            assert sha(folder/name) == digest, name
    assert read(ep/'RESULTS.json')['protocol_sha256'] == sha(ep/'PROTOCOL.json')
    for folder in (tp,fp,ep):
        for path in folder.glob('*.npz'):
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None
                names = z.namelist(); assert len(names)==len(set(names))
            with np.load(path) as z:
                for name in z.files:
                    a=z[name]
                    assert a.dtype.kind != 'O'
                    if a.dtype.kind in 'fc':
                        assert np.isfinite(a).all(), (path.name,name)
    CHECKS['provenance'] = dict(status='PASS',unique_input_code_hashes=len(hashed),
        calibration_choices_frozen_before_consumed_evaluation=True,
        final_npz_zip_crc_unique_members_and_finite_numeric_arrays=True,
        limit='Timestamps and hashes document execution order; already consumed data remain retrospective.')
    print('provenance PASS', flush=True)


def calibration_checks():
    tol=read(ROOT/'evidence/revision_v8_tolerance/RESULTS.json')
    exc=read(ROOT/'evidence/revision_v8_exchange/RESULTS.json')
    count=0
    for arm in ('censored','complete'):
        folder=ROOT/f'evidence/matched_censoring/{arm}'
        frozen=read(folder/'FROZEN.json')
        with np.load(folder/'CALIBRATION.npz') as z:
            data={k:z[k] for k in z.files}
        y=data['y']; loss=np.abs(y-data['f']); blocks=data['blocks']
        for setting in [s for s in tol['settings'] if s['arm']==arm]:
            util=[]; eligible=[]
            for m in METHODS:
                candidate=setting['candidates'][m]
                s=ranking(data[f"e{setting['e_trees']}"],data[f"w{setting['w_trees']}"],setting['cap'],m,setting['T'])
                a=acceptance(s,candidate['threshold']); metrics=[]
                for b in sorted(set(blocks)):
                    ix=blocks==b; met=measure(a[ix],y[ix],loss[ix]); metrics.append(met)
                    compare_metrics(met,candidate['calibration'][int(b)])
                u=[min(q[k] for q in metrics) for k in ('c','d')]
                close(u,[candidate['min_case_coverage'],candidate['min_exposure_coverage']])
                feasible=candidate['threshold'] is not None and all(q['weight']>0 and q['c']>=.35 and q['loss']<=setting['cap']*q['weight']+1e-9 for q in metrics)
                assert feasible==candidate['eligible']; eligible.append(feasible);util.append(u);count+=1
            for band in setting['bands']:
                got=select(util,eligible,band['epsilon'])
                assert METHODS[got]==band['selected']['d']
                concession=max(util[i][1] for i in range(5) if eligible[i])-util[got][1]
                close(concession,band['calibration_exposure_concession'])
                assert concession<=band['epsilon']+1e-12
        for row in [q for q in exc['calibration'] if q['arm']==arm]:
            util={};eligible={}
            for m in METHODS:
                rec=row['candidates'][m]
                a=acceptance(ranking(data['e220'],data['w220'],row['cap'],m,frozen['T']),rec['threshold'])
                mm=[]
                for b in sorted(set(blocks)):
                    ix=blocks==b;met=measure(a[ix],y[ix],loss[ix]);mm.append(met)
                    compare_metrics(met,rec['calibration'][int(b)])
                util[m]=min(q['d'] for q in mm)*100
                eligible[m]=rec['threshold'] is not None and all(q['weight']>0 and q['c']>=.35 and q['loss']<=row['cap']*q['weight']+1e-9 for q in mm)
                assert eligible[m]==rec['feasible'];count+=1
            margin=util['demand']-util['weight_descending'] if eligible['demand'] and eligible['weight_descending'] else None
            assert (margin is None)==(row['ratio_minus_descending_pp'] is None)
            if margin is not None:close(margin,row['ratio_minus_descending_pp'])
            other=[m for m in METHODS if m!='demand' and eligible[m]]
            winner=max(other,key=util.get) if other else None
            assert winner==row['strongest_other_feasible']
            if eligible['demand'] and winner is not None:
                close(util['demand']-util[winner],row['ratio_minus_strongest_other_pp'])
    CHECKS['calibration']=dict(status='PASS',raw_candidate_window_metrics=count*2,
        tolerance_point_choices=40,strict_cap_rows=6,
        direct_strict_complete_ratio_descending_comparison='not defined: Descending infeasible',
        complete_ratio_minus_best_other_pp=[q['ratio_minus_strongest_other_pp'] for q in exc['calibration'] if q['arm']=='complete'])
    print('calibration PASS',flush=True)


def stability_checks():
    result=read(ROOT/'evidence/revision_v8_tolerance/RESULTS.json')
    with np.load(ROOT/'evidence/revision_v8_tolerance/STABILITY_CHOICES.npz') as z:
        saved={k:z[k] for k in z.files}
    findings={}
    for setting in result['stability']:
        name=setting['setting'];folder=ROOT/f'evidence/revision_v7_stability/{name}'
        point=read(folder/'POINT.json')
        with np.load(folder/'FULL_DRAWS.npz') as z:
            stats=z['statistics'];total=z['totals'];exists=z['exists']
        utility=np.stack([(stats[...,0]/total[:,None,:,0]).min(axis=2),
                          (stats[...,1]/total[:,None,:,1]).min(axis=2)],axis=-1)
        epsrisk=1e-12 if name=='bike' else 1e-9
        eligible=exists & np.all((stats[...,1]>0)&(stats[...,0]>=.35*total[:,None,:,0])&
                  (stats[...,2]<=point['design_cap']*stats[...,1]+epsrisk),axis=2)
        for idx,band in enumerate(setting['bands']):
            choices=np.array([select(u,ok,band['epsilon']) for u,ok in zip(utility,eligible)])
            assert np.array_equal(choices,saved[name+'_band_choices'][idx])
            for j,m in list(enumerate(METHODS))+[(-1,'no_choice')]:
                assert int((choices==j).sum())==band['frequencies'][m]['count']
        findings[name]={m:x['count'] for m,x in setting['bands'][3]['frequencies'].items()}
    CHECKS['stability']=dict(status='PASS',independent_reused_draw_selections=2000,
        primary_tolerance_counts=findings,
        limit='Conditional on reused threshold designs and fitted heads; positive epsilon changes utility preference.')
    print('stability PASS',flush=True)


def floor_checks():
    folder=ROOT/'evidence/revision_v8_floor';frozen=read(folder/'FROZEN.json');result=read(folder/'RESULTS.json')
    with np.load(folder/'CALIBRATION_ITEM_STATISTICS.npz') as z:calstats={k:z[k] for k in z.files}
    with np.load(folder/'EVALUATION_ITEM_STATISTICS.npz') as z:evstats={k:z[k] for k in z.files}
    with np.load(ROOT/'evidence/matched_censoring/complete/CALIBRATION.npz') as z:data={k:z[k] for k in z.files}
    n=len(calstats['items']); ii=calstats['series_item_index'];y=data['y'];loss=abs(y-data['f']);A=data['blocks']==0;B=~A
    counts=np.random.default_rng(20260912).multinomial(n,np.full(n,1/n),4000)
    threshold_checks=0;aggchecks=0
    for et,wt in ((220,220),(660,220),(220,660),(660,660)):
        relevant=[p for p in frozen['plans'] if (p['e_trees'],p['w_trees'])==(et,wt)]
        for method in METHODS:
            s=ranking(data[f'e{et}'],data[f'w{wt}'],frozen['cap'],method,frozen['T'])
            sa=s[A];order=np.argsort(sa,kind='stable');sorted_s=sa[order]
            ends=np.r_[np.flatnonzero(sorted_s[1:]!=sorted_s[:-1]),len(sa)-1]
            endpoints=sorted_s[ends];cum_y=np.cumsum(y[A][order])[ends];cum_l=np.cumsum(loss[A][order])[ends]
            for p in relevant:
                rec=p['policies'][method];floor=p['floor'];stem=f"f{round(floor*100):02d}__{p['cell']}__{method}"
                good=(endpoints<np.inf)&(cum_y>0)&(ends+1>=floor*len(sa))&(cum_l<=frozen['design_cap']*cum_y+1e-9)
                t=float(endpoints[np.flatnonzero(good)[-1]]) if good.any() else None
                if loss[A].sum()/y[A].sum()<=frozen['design_cap']:t='all'
                assert t==rec['threshold'],(p['cell'],floor,method,t,rec['threshold']);threshold_checks+=1
                for ix,label,nd in ((A,'A',35),(B,'B',28)):
                    a=acceptance(s[ix],t);compare_metrics(measure(a,y[ix],loss[ix]),rec[label])
                    stat=item_stats(a,y[ix],loss[ix],ii,nd,n)
                    close(stat,calstats[stem+'__'+label],stem,atol=1e-7);aggchecks+=1
                bstat=calstats[stem+'__B'];signed=counts@(bstat[:,2]-frozen['cap']*bstat[:,1])/n
                for family in (20,60):
                    bound=float(np.quantile(signed,1-.05/family));scr=rec['screens'][str(family)]
                    close(bound,scr['signed_excess_upper'],atol=1e-8)
                    assert scr['pass']==(t is not None and bstat[:,1].sum()>0 and bound<=0)
        for p in relevant:
            for family in (20,60):
                eligible=[m for m in METHODS if p['policies'][m]['screens'][str(family)]['pass']]
                for objective in ('c','d'):
                    choice=max(eligible,key=lambda m:p['policies'][m]['A'][objective]) if eligible else None
                    assert choice==p['selected_by_family'][str(family)][objective]
    del data,y,loss
    with np.load(ROOT/'evidence/matched_censoring/complete/EVALUATION.npz') as z:data={k:z[k] for k in z.files}
    y=data['y'];loss=abs(y-data['f']);total=evstats['total'];den=counts@total
    for p in frozen['plans']:
        later=next(q for q in result['plans'] if q['cell']==p['cell'] and q['floor']==p['floor'])
        for method in METHODS:
            stem=f"f{round(p['floor']*100):02d}__{p['cell']}__{method}"
            s=ranking(data[f"e{p['e_trees']}"],data[f"w{p['w_trees']}"],frozen['cap'],method,frozen['T'])
            a=acceptance(s,p['policies'][method]['threshold']);got=measure(a,y,loss)
            compare_metrics(got,later['candidates'][method]);st=item_stats(a,y,loss,ii,len(data['days']),n)
            close(st,evstats[stem],stem,atol=1e-7);aggchecks+=1
            bootstrap=counts@st
            if got['weight']:
                risks=bootstrap[:,2]/bootstrap[:,1]
                for family in (20,60):
                    close(np.quantile(risks,1-.05/family),later['candidates'][method][f'risk_upper_family{family}'])
            close(np.quantile(bootstrap[:,:2]/den,[.025,.975],axis=0).T,later['candidates'][method]['coverage_ci95'])
        c,d=[later['selected'][o] for o in ('c','d')]
        stem=f"f{round(p['floor']*100):02d}__{p['cell']}__"
        delta=evstats[stem+d]-evstats[stem+c];draw=(counts@delta)[:,:2]/den
        close(np.quantile(draw,[.025,.975],axis=0).T,later['by_family']['60']['contrast']['ci95'])
    CHECKS['floor']=dict(status='PASS',raw_thresholds=threshold_checks,
        raw_item_statistics=aggchecks,all_family20_and_family60_screens_and_choices=True,
        all_candidate_risk_upper_and_coverage_interval_axes=True,
        higher_floor_choices={f:[p['selected']['d'] for p in result['plans'] if p['floor']==f] for f in (.35,.4,.45)},
        selected_later_risk_u60_max=max(p['candidates'][m]['risk_upper_family60'] for p in result['plans'] for m in p['selected'].values()),
        ratio_A_floor45_min=min(p['policies']['demand']['A']['c'] for p in frozen['plans'] if p['floor']==.45),
        limit='Family60 tail has about 3.3 expected bootstrap exceedances among 4000 draws; approximate conditional sensitivity.')
    print('floor PASS',flush=True)


def evaluation_exchange():
    folder=ROOT/'evidence/revision_v8_exchange';result=read(folder/'RESULTS.json')
    tol=read(ROOT/'evidence/revision_v8_tolerance/RESULTS.json')
    with np.load(folder/'BOOTSTRAP.npz') as z:stored={k:z[k] for k in z.files}
    n=len(stored['item_names']);counts=np.random.default_rng(20260915).multinomial(n,np.full(n,1/n),4000)
    assert np.array_equal(counts,stored['counts']);ratios={};checked=0
    for arm in ('censored','complete'):
        source=ROOT/f'evidence/matched_censoring/{arm}';frozen=read(source/'FROZEN.json')
        with np.load(source/'EVALUATION.npz') as z:data={k:z[k] for k in z.files}
        y=data['y'];loss=abs(y-data['f']);ii=data['series_item_index'];days=len(data['days'])
        assert np.array_equal(data['items'],stored['item_names'])
        total=np.column_stack([series_item_sum(np.ones(len(y)),ii,days,n),series_item_sum(y,ii,days,n)])
        close(total,stored['total']);den=counts@total
        for setting in [s for s in tol['settings'] if s['arm']==arm]:
            et,wt=setting['e_trees'],setting['w_trees'];sufficient={}
            for m in METHODS:
                s=ranking(data[f'e{et}'],data[f'w{wt}'],setting['cap'],m,frozen['T'])
                a=acceptance(s,setting['candidates'][m]['threshold']);got=measure(a,y,loss)
                compare_metrics(got,setting['evaluation_candidates'][m]);checked+=1
                sufficient[m]=item_stats(a,y,loss,ii,days,n)[:,:2]
            for band in setting['bands']:
                method=band['selected']['d'];old=setting['original_selected']['d'];case=setting['original_selected']['c']
                for i,k in enumerate(('c','d')):
                    close((sufficient[method]-sufficient[case]).sum(axis=0)[i]/total.sum(axis=0)[i],band['evaluation']['contrast_vs_case'][k])
                    close((sufficient[method]-sufficient[old]).sum(axis=0)[i]/total.sum(axis=0)[i],band['evaluation']['change_vs_original_exposure'][k])
            row=next(q for q in result['exchange_cells'] if q['arm']==arm and q['heads']==[et,wt])
            c,d=row['selected']['c'],row['selected']['d'];diff=sufficient[d]-sufficient[c]
            key=f'{arm}_{et}_{wt}';close(diff,stored[key+'_diff'])
            draw=(counts@diff)/den;assert np.all(draw[:,0]<0)
            ratio=-draw[:,1]/draw[:,0];ratios[key]=ratio
            close(draw,stored[key+'_draw_delta']);close(ratio,stored[key+'_ratio'])
            close(np.quantile(ratio,[.025,.975]),row['ratio_ci95'])
            point=diff.sum(axis=0)/total.sum(axis=0)
            close(-point[1]/point[0],row['exchange_ratio'])
            if arm=='complete' and (et,wt)==(220,220):
                for branch in result['branches']:
                    m=branch['exposure_method'];dd=(counts@(sufficient[m]-sufficient[c]))/den
                    br=-dd[:,1]/dd[:,0];close(br,stored['branch_'+m+'_ratio'])
                    close(np.quantile(br,[.025,.975]),branch['ratio_ci95'])
    for row in result['paired_arm_contrasts']:
        et,wt=row['heads'];delta=ratios[f'complete_{et}_{wt}']-ratios[f'censored_{et}_{wt}']
        close(np.quantile(delta,[.00625,.99375]),row['four_contrast_bonferroni_interval'])
    CHECKS['evaluation_exchange']=dict(status='PASS',raw_tolerance_candidate_metrics=checked,
        raw_paired_exchange_statistics=8,conditional_intervals_reconstructed=10,
        same_seed_item_weights_across_arms_and_cells=True,
        all_case_loss_bootstrap_denominators_positive=True,
        arm_ranges=result['arm_ranges'],four_paired_adjusted_difference_intervals=[q['four_contrast_bonferroni_interval'] for q in result['paired_arm_contrasts']],
        limit='Intervals hold fits, candidate thresholds, and menu choices fixed; no calibration selection uncertainty included.')
    print('evaluation/exchange PASS',flush=True)


def main(output):
    provenance();calibration_checks();stability_checks();floor_checks();evaluation_exchange()
    report=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reviewer='Independent v8 QA; no imports from production analysis code',
        verifier_sha256=sha(__file__),checks=CHECKS,
        material_issues=[],
        interpretation_limits=['Tolerance is a declared preference change, not a statistical equivalence band.',
         'Later exposure concession can exceed calibration epsilon.',
         'Bike Ratio selection frequency decreases at the primary positive tolerance.',
         'Floor .45 does not eliminate proximity to a feasibility boundary.',
         'Strict-cap Descending infeasibility prevents a direct two-eligible-candidate margin.',
         'All analyses remain retrospective and conditional; no population risk guarantee.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/revision_v8_independent/INDEPENDENT_REVIEW.json')
    main(parser.parse_args().output)
