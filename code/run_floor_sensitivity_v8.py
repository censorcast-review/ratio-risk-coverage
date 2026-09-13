"""Retrospective row-floor sensitivity for the complete-sales A/B screen.

Three floors x four fitted head settings x five rankings are all retained.
No fitting, cap adaptation, or fresh evaluation data is used. Choices under
both the inherited 20-candidate and joint 60-candidate corrections are frozen
before reopening the already-consumed later evaluation arrays.
"""
from pathlib import Path
import argparse
import numpy as np
from policy_audit import METHODS, scores, mask, metrics, largest_threshold, choose, sufficient_statistics, bootstrap_counts
from run_complete_screen_v7 import (ROOT, PAIRS, NAMES, now, sha, read, write, key,
    finite_quantile, independently_scored, independently_masked, independent_stats,
    verify_threshold, linear_quantile, independently_choose)

FLOORS=(.35,.40,.45)
FAMILIES=(20,60)

def fkey(floor): return f'f{int(round(100*floor)):02d}'

def assert_metrics(actual, expected):
    assert actual['rows']==expected['rows']
    for k in ('c','d','loss','weight','risk'):
        if expected[k] is None: assert actual[k] is None
        else: np.testing.assert_allclose(actual[k],expected[k],rtol=1e-11,atol=1e-7)

def interval(v):
    result=finite_quantile(v,[.025,.975]).T.tolist()
    assert np.shape(result)==(2,2)
    for j in range(2):
        for k,q in enumerate((.025,.975)):
            assert abs(result[j][k]-linear_quantile(v[:,j],q))<1e-12
    return result

def main(out):
    source=ROOT/'evidence/matched_censoring/complete'
    prior=ROOT/'evidence/revision_v7_complete_screen'
    base=read(source/'FROZEN.json')
    oldplans={p['cell']:p for p in read(prior/'FROZEN.json')['plans']}
    r=.6326834234117924; design=.95*r; T=base['T']
    assert read(prior/'PROTOCOL.json')['report_cap']==r
    out.mkdir(parents=True,exist_ok=True)
    if (out/'PROTOCOL.json').exists(): raise FileExistsError('Preserve an existing run before restarting')
    protocol={'utc':now(), 'scope':'Retrospective sensitivity analysis on consumed development data; conditional on one existing fit.',
      'floors':list(FLOORS), 'head_pairs':[list(p) for p in PAIRS], 'methods':list(METHODS),
      'candidate_family_size':60, 'inherited_within_floor_family_size':20,
      'primary_reporting_family':60, 'report_cap':r, 'score_cap':r, 'design_cap':design,
      'design_factor':.95, 'T':T,
      'windows':{'A':[1611,1645], 'B':[1646,1673], 'already_consumed_later':[1800,1913]},
      'threshold_design':'For every floor/head/ranking condition, redesign on A only: largest complete-score-tie threshold with positive mass, risk <= .95r and case coverage >= floor. No B retuning.',
      'screen':'Positive accepted B exposure and item-bootstrap signed-excess upper <= 0; report both alpha=.05/20 and alpha=.05/60.',
      'selection':'Separately maximize A case or A exposure coverage among B-screen passers, with fixed METHODS order for exact ties. Retain no-choice.',
      'bootstrap':{'unit':'item with all stores and days clustered together', 'draws':4000, 'seed':20260912,
        'screen_one_sided_alpha':[.05/20,.05/60], 'later_pointwise_risk_ci':[.025,.975],
        'later_risk_upper_quantiles':[1-.05/20,1-.05/60], 'coverage_contrast_ci':[.025,.975],
        'quantile_method':'linear; replicate axis zero',
        'counts':'Same item counts shared across every condition, both windows, and later evaluation.'},
      'comparators':'Compare each floor against floor .35 using the same family adjustment. Replicate v7 floor .35/family20 thresholds, screens, choices, and evaluation point metrics.',
      'freeze':'Write all thresholds, both screen decisions, and both sets of utility choices before first evaluation-array access.',
      'limitations':['Approximate conditional bootstrap statements, not population risk guarantees.',
        'A and B share item clusters and fitted heads; temporal splitting does not imply independent item samples.',
        'B screening and A design margin jointly form the procedure; no causal effect is attributed to B alone.',
        'Three floors and four shared-head settings are sensitivity conditions, not independent replications.',
        'Thresholds and selected choices are held fixed in later bootstrap intervals; fit and selection uncertainty is excluded.',
        'All later data were consumed previously; this is not prospective validation.',
        'Family60 upper quantile is based on only about 3.3 expected tail draws among 4000 and is Monte Carlo sensitive.'],
      'fit_calls':0,'new_external_openings':0}
    write(out/'PROTOCOL.json',protocol)
    inputs=[source/'FROZEN.json',source/'CALIBRATION.npz',source/'EVALUATION.npz',
      ROOT/'legacy/results/objectives/cache_design/metadata.npz',prior/'PROTOCOL.json',prior/'FROZEN.json',prior/'RESULTS.json']
    codes=[Path(__file__),ROOT/'code/policy_audit.py',ROOT/'code/run_complete_screen_v7.py']
    write(out/'START.json',{'utc':now(),'protocol_sha256':sha(out/'PROTOCOL.json'),
      'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
      'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in codes},
      'evaluation_status':'Only raw evaluation file hashed; no evaluation array read before freeze.'})
    meta=np.load(ROOT/'legacy/results/objectives/cache_design/metadata.npz')
    items,inv=np.unique(meta['item_id'],return_inverse=True); ni=len(items)
    counts=bootstrap_counts(ni,20260912,4000)
    cal=np.load(source/'CALIBRATION.npz'); L=abs(cal['y']-cal['f']); W=cal['y']; blocks=cal['blocks']
    A=blocks==0; B=blocks==1; ua=np.repeat(inv,35); ub=np.repeat(inv,28)
    assert A.sum()==len(inv)*35 and B.sum()==len(inv)*28
    assert np.array_equal(blocks,np.tile(np.r_[np.zeros(35,int),np.ones(28,int)],len(inv)))
    stats={'items':items,'series_item_index':inv,
      'A_total':np.column_stack([np.bincount(ua),np.bincount(ua,weights=W[A])]),
      'B_total':np.column_stack([np.bincount(ub),np.bincount(ub,weights=W[B])])}
    checks={'thresholds_independently_enumerated':0,'masks_independently_reconstructed':0,
      'item_aggregates_independently_reconstructed':0,'screen_bounds_independently_interpolated':0,
      'choices_independently_reconstructed':0,'v7_candidates_replicated':0,
      'v7_evaluation_candidates_replicated':0,'evaluation_aggregates_independently_reconstructed':0}
    plans=[]
    for et,wt in PAIRS:
      cell=key(et,wt); e=cal[f'e{et}']; w=cal[f'w{wt}']
      scorecache={m:scores(e,w,r,m,T) for m in METHODS}
      for m,s in scorecache.items():
        np.testing.assert_array_equal(s,independently_scored(e,w,r,m,T))
      for floor in FLOORS:
        menu={}; eligible={str(k):{} for k in FAMILIES}
        for m in METHODS:
          s=scorecache[m]
          t,mm=largest_threshold(s[A],L[A],W[A],np.zeros(A.sum(),int),design,floor)
          verify_threshold(s[A],L[A],W[A],design,floor,t); checks['thresholds_independently_enumerated']+=1
          aa=mask(s[A],t); ab=mask(s[B],t)
          assert np.array_equal(aa,independently_masked(s[A],t))
          assert np.array_equal(ab,independently_masked(s[B],t)); checks['masks_independently_reconstructed']+=2
          sta=sufficient_statistics(L[A],W[A],aa,ua,ni); stb=sufficient_statistics(L[B],W[B],ab,ub,ni)
          va=independent_stats(L[A],W[A],aa,ua,ni); vb=independent_stats(L[B],W[B],ab,ub,ni)
          np.testing.assert_allclose(sta,va,rtol=1e-12,atol=1e-7); np.testing.assert_allclose(stb,vb,rtol=1e-12,atol=1e-7)
          checks['item_aggregates_independently_reconstructed']+=2
          stem=f'{fkey(floor)}__{cell}__{m}'; stats[stem+'__A']=sta; stats[stem+'__B']=stb
          ex=stb[:,2]-r*stb[:,1]; boot=counts@ex/ni
          rec={'threshold':t,'calibration':mm,'A':metrics(L[A],W[A],aa),'B':metrics(L[B],W[B],ab),
            'B_mean_signed_excess_per_item':float(ex.mean()),'screens':{}}
          for fam in FAMILIES:
            q=1-.05/fam; upper=float(finite_quantile(boot,q))
            independent_upper=linear_quantile(counts@(vb[:,2]-r*vb[:,1])/ni,q)
            assert abs(upper-independent_upper)<1e-8; checks['screen_bounds_independently_interpolated']+=1
            good=t is not None and stb[:,1].sum()>0 and upper<=0
            eligible[str(fam)][m]=bool(good)
            rec['screens'][str(fam)]={'alpha':.05/fam,'signed_excess_upper':upper,'pass':bool(good)}
          if stb[:,1].sum()>0:
            bb=counts@stb; rr=bb[:,2]/bb[:,1]
            rec['B_risk_ci95']=finite_quantile(rr,[.025,.975]).tolist()
            for fam in FAMILIES: rec[f'B_risk_upper_family{fam}']=float(finite_quantile(rr,1-.05/fam))
          else:
            rec['B_risk_ci95']=None
            for fam in FAMILIES: rec[f'B_risk_upper_family{fam}']=None
          original=oldplans[cell]['policies'][m]
          rec['floor35_threshold']=original['threshold']
          rec['floor35_A_case_coverage']=original['A']['c']
          rec['threshold_change']='unchanged' if t==original['threshold'] else ('became_infeasible' if t is None else 'different_feasible_threshold')
          if floor==.35:
            assert t==original['threshold']; assert_metrics(rec['A'],original['A']); assert_metrics(rec['B'],original['B'])
            assert rec['screens']['20']['pass']==original['B_screen_pass']
            assert abs(rec['screens']['20']['signed_excess_upper']-original['B_adjusted_excess_upper'])<1e-8
            checks['v7_candidates_replicated']+=1
          menu[m]=rec
        choices={}
        for fam in FAMILIES:
          elig=eligible[str(fam)]; choice={o:choose(menu,o,elig) for o in ('c','d')}
          for o in ('c','d'):
            assert choice[o]==independently_choose(menu,o,elig); checks['choices_independently_reconstructed']+=1
          if floor==.35 and fam==20: assert choice==oldplans[cell]['selected']
          choices[str(fam)]=choice
        plan={'floor':floor,'cell':cell,'e_trees':et,'w_trees':wt,'policies':menu,'selected_by_family':choices,
          'selected':choices['60'],'nonempty_candidates':sum(z['threshold'] is not None for z in menu.values()),
          'passing_candidates_by_family':{str(f):sum(eligible[str(f)].values()) for f in FAMILIES}}
        plans.append(plan)
        print('CALIBRATION',cell,floor,choices,'nonempty',plan['nonempty_candidates'],flush=True)
    assert len(plans)==12 and sum(len(p['policies']) for p in plans)==60
    np.savez_compressed(out/'CALIBRATION_ITEM_STATISTICS.npz',**stats)
    frozen={'utc':now(),'cap':r,'design_cap':design,'T':T,'plans':plans,
      'protocol_sha256':sha(out/'PROTOCOL.json'),'start_sha256':sha(out/'START.json'),
      'calibration_stats_sha256':sha(out/'CALIBRATION_ITEM_STATISTICS.npz')}
    write(out/'CALIBRATION_RESULTS.json',frozen); write(out/'FROZEN.json',frozen)
    write(out/'EVALUATION_STARTED.json',{'utc':now(),'freeze_sha256':sha(out/'FROZEN.json'),
      'evaluation_sha256':sha(source/'EVALUATION.npz'),
      'note':'First evaluation-array read in this run. Evaluation outcomes were already consumed by earlier revisions.'})
    ev=np.load(source/'EVALUATION.npz')
    assert np.array_equal(ev['items'],items) and np.array_equal(ev['series_item_index'],inv)
    assert np.array_equal(ev['days'],np.arange(1800,1914))
    oldresults={p['cell']:p for p in read(prior/'RESULTS.json')['plans']}
    u=np.repeat(inv,len(ev['days'])); L=abs(ev['y']-ev['f']); W=ev['y']
    total=np.column_stack([np.bincount(u),np.bincount(u,weights=W)]); den=counts@total
    saved={'items':items,'total':total}; results=[]; perplan={}; boottotal={}
    for plan in plans:
      cell=plan['cell']; floor=plan['floor']; mets={}; boots={}
      e=ev[f"e{plan['e_trees']}"]; w=ev[f"w{plan['w_trees']}"]
      for m in METHODS:
        t=plan['policies'][m]['threshold']; s=scores(e,w,r,m,T); a=mask(s,t)
        ia=independently_masked(independently_scored(e,w,r,m,T),t); assert np.array_equal(a,ia)
        st=sufficient_statistics(L,W,a,u,ni); ist=independent_stats(L,W,ia,u,ni)
        np.testing.assert_allclose(st,ist,rtol=1e-12,atol=1e-7); checks['evaluation_aggregates_independently_reconstructed']+=1
        met=metrics(L,W,a); bb=counts@st; cv=bb[:,:2]/den
        met['coverage_ci95']=interval(cv)
        if met['weight']>0:
          rr=bb[:,2]/bb[:,1]; met['risk_ci95']=finite_quantile(rr,[.025,.975]).tolist()
          for fam in FAMILIES:
            q=1-.05/fam; bound=float(finite_quantile(rr,q)); assert abs(bound-linear_quantile(rr,q))<1e-12
            met[f'risk_upper_family{fam}']=bound
        else:
          met['risk_ci95']=None
          for fam in FAMILIES: met[f'risk_upper_family{fam}']=None
        if floor==.35:
          assert_metrics(met,oldresults[cell]['candidates'][m]); checks['v7_evaluation_candidates_replicated']+=1
          if met['weight']>0:
            np.testing.assert_allclose(met['risk_ci95'],oldresults[cell]['candidates'][m]['risk_ci95'],rtol=0,atol=1e-12)
        mets[m]=met; boots[m]=bb; saved[f'{fkey(floor)}__{cell}__{m}']=st
      rec={'floor':floor,'cell':cell,'e_trees':plan['e_trees'],'w_trees':plan['w_trees'],
        'selected_by_family':plan['selected_by_family'],'selected':plan['selected'],'candidates':mets,'by_family':{}}
      results.append(rec); perplan[(cell,floor)]=rec; boottotal[(cell,floor)]=boots
      print('EVALUATION_AGGREGATES',cell,floor,flush=True)
    summary=[]
    for rec in results:
      cell=rec['cell']; floor=rec['floor']; boots=boottotal[(cell,floor)]; mets=rec['candidates']
      baseline=perplan[(cell,.35)]; baseboots=boottotal[(cell,.35)]
      for fam in FAMILIES:
        selections=rec['selected_by_family'][str(fam)]; basechoices=baseline['selected_by_family'][str(fam)]
        selected={}; contrast=None
        for o in ('c','d'):
          m=selections[o]; bm=basechoices[o]
          if m is None:
            selected[o]={'method':None,'no_choice':True,'baseline_floor35_method':bm}; continue
          selected[o]={'method':m,'no_choice':False,'metrics':mets[m], 'baseline_floor35_method':bm}
          if bm is not None:
            point=[mets[m]['c']-baseline['candidates'][bm]['c'],mets[m]['d']-baseline['candidates'][bm]['d']]
            change=(boots[m]-baseboots[bm])[:,:2]/den
            selected[o].update({'coverage_change_vs_floor35':point,'coverage_change_ci95':interval(change),
              'coverage_cost_floor35_minus_current':[-x for x in point]})
          summary.append({'floor':floor,'cell':cell,'family':fam,'utility':o,'method':m,
            'case_percent':100*mets[m]['c'],'exposure_percent':100*mets[m]['d'],
            'risk':mets[m]['risk'],'risk_ci95':mets[m]['risk_ci95'],
            'risk_upper_family60':mets[m]['risk_upper_family60'],
            'coverage_change_vs_floor35_pp':[100*x for x in selected[o].get('coverage_change_vs_floor35',[])]})
        c,d=selections['c'],selections['d']
        if c is not None and d is not None:
          dc=mets[d]['c']-mets[c]['c']; dd=mets[d]['d']-mets[c]['d']
          contrast={'dc':dc,'dd':dd,'ci95':interval((boots[d]-boots[c])[:,:2]/den),
            'exchange_ratio':dd/(-dc) if dc<0 else None}
        rec['by_family'][str(fam)]={'selected_policies':selected,'contrast':contrast}
      print('RESULT',cell,floor,rec['selected'],rec['by_family']['60']['contrast'],flush=True)
    result={'utc':now(),'scope':protocol['scope'],'cap':r,'design_cap':design,'floors':list(FLOORS),
      'primary_screen_family':60,'plans':results,'summary':summary,
      'all_nonempty_candidates_pass_by_family':{str(f):all(p['nonempty_candidates']==p['passing_candidates_by_family'][str(f)] for p in plans) for f in FAMILIES},
      'candidate_condition_count':60,'selection_record_count_including_no_choice':48,
      'population_risk_guarantee':False,'fit_calls':0,'new_external_openings':0}
    np.savez_compressed(out/'EVALUATION_ITEM_STATISTICS.npz',**saved); write(out/'RESULTS.json',result)
    checks.update({'status':'PASS','utc':now(), 'candidate_conditions':60,'selection_records_including_no_choice':48,
      'checks':['Protocol/code/input hashes saved before calibration calculations.',
        'All 60 condition thresholds independently checked across complete score ties.',
        'All A/B and evaluation item statistics checked using reduceat independent of bincount.',
        'Both corrected screens and A-only choices retained, including no-choice.',
        'Floor .35 family20 thresholds, B decisions, choices, evaluation point metrics and risk intervals reproduce v7.',
        'All conditions frozen before reopening consumed later evaluation arrays.',
        'Floor contrasts use the identical family correction on both sides.',
        'Quantiles checked by independent scalar linear interpolation; coverage interval arrays have shape (2,2).']})
    write(out/'VERIFICATION.json',checks)
    lines=['# Complete-sales floor sensitivity (v8)','', 'Retrospective analysis; no population risk guarantee. Primary adjustment covers all 60 floor/head/ranking conditions.','',
      f"Report cap {r:.12f}; A design cap {design:.12f}; item bootstrap 4,000 draws, seed 20260912.",'',
      '| Floor | Heads e/w | Selected case / exposure | Δcase pp | Δexposure pp | Case risk U60 | Exposure risk U60 |',
      '|---|---|---|---:|---:|---:|---:|']
    for rec in results:
      z=rec['by_family']['60']; c=z['selected_policies']['c']; d=z['selected_policies']['d']; contrast=z['contrast']
      lines.append(f"| {rec['floor']:.2f} | {rec['e_trees']}/{rec['w_trees']} | {c['method']} / {d['method']} | {100*contrast['dc']:.4f} | {100*contrast['dd']:.4f} | {c['metrics']['risk_upper_family60']:.6f} | {d['metrics']['risk_upper_family60']:.6f} |" if contrast is not None else f"| {rec['floor']:.2f} | {rec['e_trees']}/{rec['w_trees']} | {c['method']} / {d['method']} | no choice | | | |")
    lines.extend(['',f"All nonempty candidates pass B by adjustment family: {result['all_nonempty_candidates_pass_by_family']}",
      '', 'At 4,000 draws the 60-way upper quantile has about 3.3 expected tail draws; tail Monte Carlo resolution is limited.',
      'A design margin and B screening are evaluated as one combined procedure; successful screening does not establish a separate effect of B.',
      'Bootstrap intervals hold fitted heads, thresholds, and selections fixed and condition on already-consumed data.'])
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    write(out/'COMPLETE.json',{'utc':now(),'status':'PASS','files_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=ROOT/'evidence/revision_v8_floor')
    main(parser.parse_args().output)
