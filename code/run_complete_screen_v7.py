"""Retrospective complete-sales A-design / B-screen audit (v7).

No fitting or new external data. Twenty candidates are fixed before screening,
then all choices are frozen before reopening the already-consumed later data.
All bootstrap statements are approximate and conditional on the existing fit.
"""
from pathlib import Path
import argparse, datetime, hashlib, json
import numpy as np
from policy_audit import (METHODS, scores, mask, metrics, largest_threshold,
                         choose, sufficient_statistics, bootstrap_counts)

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ((220,220),(660,220),(220,660),(660,660))
NAMES = {'error':'Error','row':'Excess','mixed':'Mixed','demand':'Ratio',
         'weight_descending':'Exposure descending'}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(e,w): return f'e{e}_w{w}'
def finite_quantile(x,q):
    if not np.isfinite(x).all(): raise ValueError('Undefined bootstrap draw')
    return np.quantile(x,q,axis=0)

def independently_scored(e,w,r,m,T):
    if m=='error': return np.array(e)
    if m=='weight_descending': return 0.-w
    z=e-r*w
    if m=='row': return z
    den=w/T if m=='demand' else .25+.75*w/T
    out=np.zeros_like(z)
    ok=den>0
    out[ok]=z[ok]/den[ok]
    out[~ok & (z>0)]=np.inf
    out[~ok & (z<0)]=-np.inf
    return out

def independently_masked(s,t):
    if t is None: return np.zeros(s.shape,dtype=bool)
    if t=='all': return np.ones(s.shape,dtype=bool)
    return np.logical_not(np.greater(s,t))

def independent_stats(L,W,a,u,n):
    """Use sorted item groups and reduceat, rather than bincount."""
    order=np.argsort(u,kind='stable'); groups=u[order]
    starts=np.r_[0,1+np.flatnonzero(np.diff(groups))]
    out=np.zeros((n,3))
    for j,v in enumerate((a.astype(float),W*a,L*a)):
        out[groups[starts],j]=np.add.reduceat(v[order],starts)
    return out

def verify_threshold(s,L,W,cap,floor,t):
    """Independent complete-tie enumeration in one A window."""
    if W.sum()>0 and L.sum()/W.sum()<=cap:
        assert t=='all'; return
    order=np.argsort(s,kind='stable'); ss=s[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    cw=np.cumsum(W[order]); cl=np.cumsum(L[order]); nn=ends+1
    feasible=(cw[ends]>0)&(nn>=floor*len(W))&(cl[ends]<=cap*cw[ends]+1e-9)&(ss[ends]<np.inf)
    valid=ends[feasible]
    if len(valid)==0: assert t is None
    else: assert t==float(ss[valid[-1]])

def linear_quantile(v,q):
    """Independent linear order-statistic interpolation."""
    a=np.sort(v); loc=(len(a)-1)*q; lo=int(np.floor(loc)); hi=int(np.ceil(loc))
    return float(a[lo]+(a[hi]-a[lo])*(loc-lo))

def independently_choose(menu,obj,eligible):
    ranked=[(min(z[obj] for z in menu[m]['calibration']),-i,m)
            for i,m in enumerate(METHODS) if menu[m]['threshold'] is not None and eligible[m]]
    return sorted(ranked,reverse=True)[0][2] if ranked else None

def main(out):
    source=ROOT/'evidence/matched_censoring/complete'
    baseline=read(source/'FROZEN.json')
    oldplans={(p['e_trees'],p['w_trees']):p for p in baseline['plans']
              if p['kind']=='relative' and p['value']==.95}
    assert set(oldplans)==set(PAIRS)
    r=oldplans[(220,220)]['cap']; T=baseline['T']; design=.95*r
    assert all(p['cap']==r for p in oldplans.values())
    out.mkdir(parents=True,exist_ok=True)
    if (out/'PROTOCOL.json').exists(): raise FileExistsError('Existing completed or partial run must be preserved before rerun')
    protocol={
      'utc':now(),'scope':'retrospective audit of already-consumed development windows; conditional on one existing fit',
      'arm':'complete recorded sales','head_pairs':[list(p) for p in PAIRS],
      'methods':list(METHODS),'candidate_family_size':20,'report_cap':r,
      'report_cap_source':'matched_censoring/complete/FROZEN.json, relative factor 0.95',
      'score_cap':r,'design_cap':design,'design_factor':.95,'row_floor_A':.35,'T':T,
      'windows':{'A':[1611,1645],'B':[1646,1673],'later_already_consumed':[1800,1913]},
      'threshold_design':'A only; largest eligible complete-score-tie threshold at design cap; positive accepted exposure; no B threshold retuning',
      'screen':'B pooled signed-excess item-bootstrap upper quantile <= 0 and positive B accepted exposure',
      'bootstrap':{'unit':'item (stores and days clustered together)','draws':4000,'seed':20260912,
                   'screen_one_sided_alpha':.05/20,'screen_family_size':20,
                   'later_pointwise_risk_ci':[.025,.975],
                   'later_selected_eight_upper_quantile':1-.05/8,
                   'later_twenty_candidate_upper_quantile':1-.05/20,
                   'coverage_and_contrast_ci':[.025,.975],'quantile_method':'linear'},
      'selection':'among B-screen passers, maximize A case/exposure coverage separately, declared METHODS order for exact ties; no-choice if none pass',
      'baseline':'each cell\'s existing unscreened relative-0.95 primary case/exposure choices, frozen without refit',
      'limitations':['approximate bootstrap, not population risk guarantee','B shares items with A and prior fits; temporal split is not independent item sampling',
                     'four shared-head cells are sensitivity conditions, not independent replications',
                     'one fit seed 20260910; bootstrap does not include fit or policy-selection uncertainty',
                     'later outcomes have previously been consumed; no prospective claim','no cap tuning after any outcomes'],
      'fit_calls':0,'new_external_openings':0}
    write(out/'PROTOCOL.json',protocol)
    inputs=[source/'FROZEN.json',source/'CALIBRATION.npz',source/'EVALUATION.npz',
            ROOT/'evidence/matched_censoring/PROTOCOL.json',ROOT/'legacy/results/objectives/cache_design/metadata.npz']
    codes=[Path(__file__),ROOT/'code/policy_audit.py',ROOT/'code/run_policy_choice.py',ROOT/'code/run_uncensored_audit.py']
    write(out/'START.json',{'utc':now(),'protocol_sha256':sha(out/'PROTOCOL.json'),
        'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in codes},
        'evaluation_status':'only raw file hashed; no evaluation array read until FROZEN and EVALUATION_STARTED'})
    meta=np.load(ROOT/'legacy/results/objectives/cache_design/metadata.npz')
    items,inv=np.unique(meta['item_id'],return_inverse=True); ni=len(items)
    counts=bootstrap_counts(ni,20260912,4000)
    cal=np.load(source/'CALIBRATION.npz'); L=abs(cal['y']-cal['f']); W=cal['y']; blocks=cal['blocks']
    A=blocks==0; B=blocks==1
    assert A.sum()==len(inv)*35 and B.sum()==len(inv)*28
    ua=np.repeat(inv,35); ub=np.repeat(inv,28)
    assert np.array_equal(blocks,np.tile(np.r_[np.zeros(35,int),np.ones(28,int)],len(inv)))
    stats_save={'items':items,'series_item_index':inv,
        'A_total':np.column_stack([np.bincount(ua),np.bincount(ua,weights=W[A])]),
        'B_total':np.column_stack([np.bincount(ub),np.bincount(ub,weights=W[B])])}
    frozen={'utc':None,'cap':r,'design_cap':design,'T':T,'plans':[],
            'protocol_sha256':sha(out/'PROTOCOL.json'),'start_sha256':sha(out/'START.json')}
    checks={'independent_score_mask':0,'independent_largest_A_threshold':0,'independent_item_aggregation':0,
            'independent_B_quantile_and_pass':0,'independent_selection':0,'old_baseline_calibration':0,
            'later_independent_aggregation':0,'later_independent_quantiles':0,'checks':[]}
    for et,wt in PAIRS:
        cell=key(et,wt); e=cal[f'e{et}']; w=cal[f'w{wt}']; menu={}; elig={}
        for m in METHODS:
            s=scores(e,w,r,m,T); independent=independently_scored(e,w,r,m,T)
            np.testing.assert_allclose(s,independent,rtol=0,atol=0)
            t,mm=largest_threshold(s[A],L[A],W[A],np.zeros(A.sum(),int),design,.35)
            verify_threshold(independent[A],L[A],W[A],design,.35,t); checks['independent_largest_A_threshold']+=1
            aa=mask(s[A],t); ab=mask(s[B],t)
            assert np.array_equal(aa,independently_masked(independent[A],t))
            assert np.array_equal(ab,independently_masked(independent[B],t)); checks['independent_score_mask']+=1
            sta=sufficient_statistics(L[A],W[A],aa,ua,ni); stb=sufficient_statistics(L[B],W[B],ab,ub,ni)
            vsta=independent_stats(L[A],W[A],aa,ua,ni); vstb=independent_stats(L[B],W[B],ab,ub,ni)
            np.testing.assert_allclose(sta,vsta,rtol=1e-12,atol=1e-7); np.testing.assert_allclose(stb,vstb,rtol=1e-12,atol=1e-7)
            checks['independent_item_aggregation']+=2
            stats_save[cell+'__'+m+'__A']=sta; stats_save[cell+'__'+m+'__B']=stb
            ex=stb[:,2]-r*stb[:,1]; boot=counts@ex/ni
            upper=float(finite_quantile(boot,1-.05/20)); valid=t is not None and stb[:,1].sum()>0 and upper<=0
            vboot=counts@(vstb[:,2]-r*vstb[:,1])/ni; vupper=linear_quantile(vboot,1-.05/20)
            assert abs(upper-vupper)<1e-8
            assert valid==(t is not None and np.sum(W[B][independently_masked(independent[B],t)])>0 and vupper<=0)
            checks['independent_B_quantile_and_pass']+=1
            elig[m]=bool(valid)
            menu[m]={'threshold':t,'calibration':mm,'A':metrics(L[A],W[A],aa),'B':metrics(L[B],W[B],ab),
                     'B_mean_signed_excess_per_item':float(ex.mean()),'B_adjusted_excess_upper':upper,
                     'B_screen_pass':bool(valid),'screen_alpha':.05/20}
            if stb[:,1].sum()>0:
                bb=counts@stb; rr=np.divide(bb[:,2],bb[:,1],out=np.full(len(bb),np.nan),where=bb[:,1]>0)
                menu[m]['B_risk_ci95']=finite_quantile(rr,[.025,.975]).tolist()
                menu[m]['B_risk_upper_family20']=float(finite_quantile(rr,1-.05/20))
            else: menu[m].update(B_risk_ci95=None,B_risk_upper_family20=None)
            oldt=oldplans[(et,wt)]['policies'][m]['threshold']; olda=mask(s,oldt)
            for ib in (0,1):
                actual=metrics(L[blocks==ib],W[blocks==ib],olda[blocks==ib]); expected=oldplans[(et,wt)]['policies'][m]['calibration'][ib]
                assert actual['rows']==expected['rows']
                for k in ['c','d','loss','weight']:
                    assert np.isclose(actual[k],expected[k],rtol=1e-11,atol=1e-7)
            checks['old_baseline_calibration']+=1
        selected={o:choose(menu,o,elig) for o in ('c','d')}
        for o in ('c','d'): assert selected[o]==independently_choose(menu,o,elig); checks['independent_selection']+=1
        frozen['plans'].append({'cell':cell,'e_trees':et,'w_trees':wt,'policies':menu,'selected':selected,
            'baseline_primary':oldplans[(et,wt)]})
        print('B_SCREEN',cell,selected,{m:menu[m]['B_screen_pass'] for m in METHODS},flush=True)
    np.savez_compressed(out/'CALIBRATION_ITEM_STATISTICS.npz',**stats_save)
    write(out/'CALIBRATION_RESULTS.json',{'utc':now(),'plans':frozen['plans']})
    frozen['utc']=now(); frozen['calibration_stats_sha256']=sha(out/'CALIBRATION_ITEM_STATISTICS.npz')
    write(out/'FROZEN.json',frozen)
    write(out/'EVALUATION_STARTED.json',{'utc':now(),'freeze_sha256':sha(out/'FROZEN.json'),
        'evaluation_sha256':sha(source/'EVALUATION.npz'),'new_external_openings':0,
        'note':'First evaluation array access in this run; later outcomes have already been used in previous work.'})
    ev=np.load(source/'EVALUATION.npz')
    assert np.array_equal(ev['items'],items) and np.array_equal(ev['series_item_index'],inv)
    assert np.array_equal(ev['days'],np.arange(1800,1914))
    u=np.repeat(inv,len(ev['days'])); L=abs(ev['y']-ev['f']); W=ev['y']
    total=np.column_stack([np.bincount(u),np.bincount(u,weights=W)]); den=counts@total
    saved={'items':items,'total':total}; result={'scope':protocol['scope'],'cap':r,'design_cap':design,'plans':[],
            'selection_policy_count':8,'candidate_count':20,'pointwise_and_adjusted_intervals_are_descriptive':True}
    summary=[]
    for plan in frozen['plans']:
        cell=plan['cell']; e=ev[f"e{plan['e_trees']}"]; w=ev[f"w{plan['w_trees']}"]
        mets={}; sts={}; bootstats={}; olds={}; oldmetrics={}
        for m in METHODS:
            s=scores(e,w,r,m,T); a=mask(s,plan['policies'][m]['threshold']); st=sufficient_statistics(L,W,a,u,ni)
            independent=independently_scored(e,w,r,m,T); am=independently_masked(independent,plan['policies'][m]['threshold'])
            assert np.array_equal(a,am)
            vst=independent_stats(L,W,am,u,ni); np.testing.assert_allclose(st,vst,rtol=1e-12,atol=1e-7)
            checks['later_independent_aggregation']+=1
            met=metrics(L,W,a); bb=counts@st; cv=bb[:,:2]/den
            met['coverage_ci95']=finite_quantile(cv,[.025,.975]).T.tolist()
            assert np.shape(met['coverage_ci95'])==(2,2)
            for j in (0,1):
                for k,q in enumerate((.025,.975)):
                    assert abs(met['coverage_ci95'][j][k]-linear_quantile(cv[:,j],q))<1e-12
            if met['weight']>0:
                rr=bb[:,2]/bb[:,1]; met['risk_ci95']=finite_quantile(rr,[.025,.975]).tolist()
                met['risk_upper_family20']=float(finite_quantile(rr,1-.05/20))
                met['risk_upper_selected8']=float(finite_quantile(rr,1-.05/8))
                assert abs(met['risk_ci95'][1]-linear_quantile(rr,.975))<1e-12
                assert abs(met['risk_upper_family20']-linear_quantile(rr,1-.05/20))<1e-12
                checks['later_independent_quantiles']+=1
            else: met.update(risk_ci95=None,risk_upper_family20=None,risk_upper_selected8=None)
            mets[m]=met; sts[m]=st; bootstats[m]=bb; saved[cell+'__'+m]=st
            oa=mask(s,plan['baseline_primary']['policies'][m]['threshold'])
            olds[m]=sufficient_statistics(L,W,oa,u,ni); oldmetrics[m]=metrics(L,W,oa)
            saved[cell+'__baseline__'+m]=olds[m]
        rec={'cell':cell,'e_trees':plan['e_trees'],'w_trees':plan['w_trees'],'selected':plan['selected'],
             'candidates':mets,'baseline_candidates':oldmetrics,'contrast':None,'selected_policies':{}}
        for o in ('c','d'):
            m=plan['selected'][o]; bm=plan['baseline_primary']['selected'][o]
            if m is None:
                rec['selected_policies'][o]={'method':None,'no_choice':True,'baseline_method':bm};continue
            change=(counts@(sts[m]-olds[bm]))[:,:2]/den
            point=[mets[m]['c']-oldmetrics[bm]['c'],mets[m]['d']-oldmetrics[bm]['d']]
            cc={'method':m,'no_choice':False,'metrics':mets[m],'baseline_method':bm,'baseline_metrics':oldmetrics[bm],
                'coverage_change_vs_unscreened_primary':point,
                'coverage_change_ci95':finite_quantile(change,[.025,.975]).T.tolist(),
                'coverage_cost_unscreened_minus_screened':[-x for x in point]}
            rec['selected_policies'][o]=cc
            summary.append({'cell':cell,'utility':o,'method':m,'case_percent':100*mets[m]['c'],
                'exposure_percent':100*mets[m]['d'],'risk':mets[m]['risk'],'risk_ci95':mets[m]['risk_ci95'],
                'risk_upper_selected8':mets[m]['risk_upper_selected8'],
                'risk_upper_family20':mets[m]['risk_upper_family20'],
                'case_cost_pp':-100*point[0],'exposure_cost_pp':-100*point[1]})
        c,d=plan['selected']['c'],plan['selected']['d']
        if c is not None and d is not None:
            vals=(bootstats[d]-bootstats[c])[:,:2]/den
            rec['contrast']={'dc':mets[d]['c']-mets[c]['c'],'dd':mets[d]['d']-mets[c]['d'],
                'ci95':finite_quantile(vals,[.025,.975]).T.tolist()}
            assert np.shape(rec['contrast']['ci95'])==(2,2)
            for j in (0,1):
                for k,q in enumerate((.025,.975)):
                    assert abs(rec['contrast']['ci95'][j][k]-linear_quantile(vals[:,j],q))<1e-12
        result['plans'].append(rec)
        print('LATER',cell,rec['selected'],rec['contrast'],flush=True)
    result['summary']=summary; result['completed_utc']=now()
    np.savez_compressed(out/'EVALUATION_ITEM_STATISTICS.npz',**saved)
    write(out/'RESULTS.json',result)
    checks['status']='PASS';checks['utc']=now(); checks['checks'].extend([
      'every declared head cell and all five methods reported; no omission of failed screens',
      'score and threshold masks independently reconstructed',
      'complete A tie endpoints independently checked for maximal feasibility',
      'item statistics independently reconstructed with sorted reduceat',
      'B pass indicators recomputed from independent statistics and independent quantile interpolation',
      'selection independently reconstructed from eligible A utilities with fixed tie order',
      'all 20 original frozen baseline calibration metrics reproduced',
      'evaluation items, ordering, and days agree with pre-freeze metadata',
      'all eight selected-policy and four contrast records retained, including possible no-choice'])
    checks['files_sha256']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    write(out/'VERIFICATION.json',checks)
    write(out/'COMPLETE.json',{'utc':now(),'fit_calls':0,'new_external_openings':0,
        'files_sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({'cap':r,'summary':summary,'verification':'PASS'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'evidence/revision_v7_complete_screen')
    main(p.parse_args().output)
