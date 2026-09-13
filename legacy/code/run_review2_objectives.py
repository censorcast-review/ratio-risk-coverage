"""Post-hoc objectives and head diagnostics on consumed design/guardian caches.

No fit; no raw guardian loader; no external input. Thresholds and method choices
are computed on development calibration A/B, then applied to cached guardian
predictions. This sequencing does not turn an after-review analysis into a new
independent test. The original frozen comparison remains immutable.
"""
from pathlib import Path
import argparse,json,datetime,time
import numpy as np
from r2_io import dump,sha,install_numpy_writers
from run_selective_study import load,metric,R,FLOOR
from run_review_comparison import SEEDS,metrics_by_group,paired_delta
KEYS=['calibration_a','calibration_b','shadow']
PRIMARY=['mean_error','relative_error_f','relative_error_mu','excess_f','composite_excess','composite_excess_budget220','direct_excess']

def curve(score,y,p,cap=R,floor=FLOOR):
    s=np.asarray(score).ravel();y=np.asarray(y,dtype=float).ravel();p=np.asarray(p).ravel()
    assert np.isfinite(s).all() and np.isfinite(y).all() and (y>=0).all()
    order=np.argsort(s,kind='stable');s=s[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    mass=np.cumsum(y[order])[ends];error=np.cumsum(np.abs(y-p)[order])[ends]
    wape=np.divide(error,mass,out=np.full(len(mass),np.inf),where=mass>0)
    row=(ends+1)/len(y);dem=mass/y.sum() if y.sum()>0 else np.zeros(len(mass))
    return dict(threshold=s[ends],row=row,demand=dem,wape=wape,feasible=(wape<=cap)&(row>=floor))

def select_threshold(a,b,objective):
    grid=np.union1d(a['threshold'],b['threshold'])
    ia=np.searchsorted(a['threshold'],grid,side='right')-1
    ib=np.searchsorted(b['threshold'],grid,side='right')-1
    valid=(ia>=0)&(ib>=0);ia=np.maximum(ia,0);ib=np.maximum(ib,0)
    good=valid&a['feasible'][ia]&b['feasible'][ib]
    if not good.any():return None,0.
    value=np.minimum(a[objective][ia],b[objective][ib]);best=value[good].max()
    # Largest threshold among maximizers: same convention for both objectives.
    chosen=np.flatnonzero(good&(value==best))[-1]
    return float(grid[chosen]),float(best)

def scores(pred,data,floor=.25):
    names=PRIMARY+['error110_demand220','error220_demand110']
    out={k:{} for k in names}
    for k in KEYS:
        e=np.maximum(pred['error_'+k],0);mu=np.maximum(pred['demand_'+k],0)
        e1=np.maximum(pred['error110_'+k],0);m1=np.maximum(pred['demand110_'+k],0)
        p=data[k]['proposal']
        v=[e,e/np.maximum(p,floor),e/np.maximum(mu,floor),e-R*p,e-R*mu,e1-R*m1,pred['direct_'+k],e1-R*mu,e-R*m1]
        for name,x in zip(names,v):out[name][k]=x
    return out

def apply(data,meta,ss,t):
    return {k:metrics_by_group(v,np.zeros(v['truth'].shape,bool) if t is None else ss[k]<=t,meta) for k,v in data.items()}

def compressed_curve(c):
    take=np.unique(np.r_[np.linspace(0,len(c['row'])-1,min(700,len(c['row']))).astype(int),np.flatnonzero(c['feasible'])[-1:]])
    return {k:[float(v[i]) if np.isfinite(v[i]) else None for i in take] for k,v in c.items() if k!='feasible'}|{'exact_threshold_count':len(c['row'])}

def head_diagnostics(pred,data,meta,cohort,seed):
    out=[]
    for block,v in data.items():
      for g in ['ALL','HOBBIES','HOUSEHOLD','FOODS']:
        rows=np.ones(len(meta['item_id']),bool) if g=='ALL' else meta['cat_id']==g
        y=v['truth'][rows].astype(float);true_e=np.abs(y-v['proposal'][rows])
        for ne,nm in [(110,110),(110,220),(220,110),(220,220)]:
            pe=np.maximum(pred[('error110' if ne==110 else 'error')+'_'+block][rows],0).astype(float)
            pm=np.maximum(pred[('demand110' if nm==110 else 'demand')+'_'+block][rows],0).astype(float)
            de=pe-true_e;dm=pm-y;res=de-R*dm
            ee=float(np.mean(de*de));mm=float(np.mean(dm*dm));cross=float(np.mean(de*dm))
            combined=float(np.mean(res*res))
            assert np.isclose(combined,ee+R*R*mm-2*R*cross,rtol=1e-10)
            out.append(dict(cohort=cohort,seed=seed,block=block,group=g,error_trees=ne,demand_trees=nm,error_mse=ee,demand_mse=mm,error_bias=float(de.mean()),demand_bias=float(dm.mean()),cross_error_moment=cross,excess_mse=combined,excess_residual_variance=float(res.var()),excess_bias=float(res.mean())))
    return out

def main(a):
    t0=time.time();install_numpy_writers();a.output.mkdir(parents=True,exist_ok=True)
    original=a.guardian/'GUARDIAN_COMPARISON.json';before=sha(original)
    assert before=='aaaf3e1f72a576f229f4a31ea42cb46470206cb1dbd33678beb24b9609c0077f'
    protocol={'status':'POST_HOC_CONSUMED_CACHE_DIAGNOSTIC','objectives':['row','demand'],'criterion':'maximize minimum calibration A/B coverage of the requested type, subject to WAPE<=cap and row coverage>=floor in both blocks','cap':R,'row_floor':FLOOR,'threshold_tie_break':'largest maximizer','family_tie_break':'lexicographic family name','primary_families':PRIMARY,'denominator_floors':[.01,.05,.1,.25,.5,1.0],'seed':SEEDS,'training_calls':0,'new_guardian_accesses':0,'external_opened':False,'certificate_issued':False,'original_guardian_sha256':before,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    dump(a.output/'PROTOCOL.json',protocol)
    dc=a.design/'cache_design';dm=load(dc/'metadata.npz');dd={k:load(dc/(k+'_aligned.npz')) for k in KEYS}
    records=[];families=[];frontiers=[];heads=[];floors=[];design_predictions={}
    # All threshold/family selection is completed before this program loads
    # the guardian outcomes below. Prior review feedback is already observed.
    for seed in SEEDS:
        pred=load(a.design/f'design_predictions_s{seed}.npz');design_predictions[seed]=pred
        ss=scores(pred,dd);family_candidates={o:[] for o in ['row','demand']}
        for name,s in ss.items():
            ca={k:curve(s[k],dd[k]['truth'],dd[k]['proposal']) for k in KEYS[:2]}
            tr,vr=select_threshold(ca['calibration_a'],ca['calibration_b'],'row')
            td,vd=select_threshold(ca['calibration_a'],ca['calibration_b'],'demand')
            assert tr==td,(seed,name,tr,td)
            rec=dict(seed=seed,score=name,row_threshold=tr,demand_threshold=td,row_objective=vr,demand_objective=vd,thresholds_identical=True,design=apply(dd,dm,s,tr))
            if name in PRIMARY:
                old=json.loads((a.design.parent/'review_revision'/f'report_{name}_pooled_s{seed}.json').read_text())['thresholds']['ALL']
                rec['original_row_threshold']=old
                rec['changed_shadow_rows_from_tie_break']=int(np.sum((s['shadow']<=tr)!=(s['shadow']<=old))) if tr is not None and old is not None else 0
                if tr is not None:
                    for o,v in [('row',vr),('demand',vd)]:family_candidates[o].append((v,name,tr))
            records.append(rec)
            if seed==SEEDS[0] and name in PRIMARY:
                for k in KEYS:
                    c=ca[k] if k in ca else curve(s[k],dd[k]['truth'],dd[k]['proposal'])
                    frontiers.append(dict(cohort='design',score=name,block=k,**compressed_curve(c)))
        for obj,opts in family_candidates.items():
            opts.sort(key=lambda z:(-z[0],z[1]));v,n,t=opts[0]
            families.append(dict(seed=seed,objective=obj,selected_score=n,threshold=t,minimum_calibration_objective=v))
        heads+=head_diagnostics(pred,dd,dm,'design',seed)
        if seed==SEEDS[0]:
            for f in protocol['denominator_floors']:
                ff=scores(pred,dd,f)
                for name in ['relative_error_f','relative_error_mu']:
                    ca={k:curve(ff[name][k],dd[k]['truth'],dd[k]['proposal']) for k in KEYS[:2]}
                    t,val=select_threshold(ca['calibration_a'],ca['calibration_b'],'demand')
                    floors.append(dict(floor=f,score=name,threshold=t,minimum_calibration_demand_coverage=val,design=apply(dd,dm,ff[name],t)))
        print('Development objectives selected',seed,flush=True)
    dump(a.output/'DEVELOPMENT_SELECTED_POLICIES.json',dict(records=records,family_choices=families,floor_sensitivity=floors,scope='selection_on_consumed_design_after_prior_guardian_results'))
    gc=a.guardian/'cache';gm=load(gc/'metadata.npz');gd={k:load(gc/(k+'_aligned.npz')) for k in KEYS}
    assert len(np.unique(gm['item_id']))==610 and not(set(gm['item_id'])&set(dm['item_id']))
    for seed in SEEDS:
        pred=load(a.guardian/f'predictions_s{seed}.npz');ss=scores(pred,gd)
        for rec in [x for x in records if x['seed']==seed]:rec['guardian_posthoc']=apply(gd,gm,ss[rec['score']],rec['demand_threshold'])
        heads+=head_diagnostics(pred,gd,gm,'guardian_posthoc',seed)
        for rec in [x for x in families if x['seed']==seed]:
            n=rec['selected_score'];t=rec['threshold']
            rec['design']=apply(dd,dm,scores(design_predictions[seed],dd)[n],t)
            rec['guardian_posthoc']=apply(gd,gm,ss[n],t)
        if seed==SEEDS[0]:
            for rec in floors:rec['guardian_posthoc']=apply(gd,gm,scores(pred,gd,rec['floor'])[rec['score']],rec['threshold'])
            for name in PRIMARY:
                for k in KEYS:
                    c=curve(ss[name][k],gd[k]['truth'],gd[k]['proposal'])
                    frontiers.append(dict(cohort='guardian_posthoc',score=name,block=k,**compressed_curve(c)))
            pair=[next(x for x in families if x['seed']==seed and x['objective']==obj) for obj in ['demand','row']]
            contrast=paired_delta(gd['shadow'],ss[pair[0]['selected_score']]['shadow']<=pair[0]['threshold'],ss[pair[1]['selected_score']]['shadow']<=pair[1]['threshold'],gm,reps=5000)
            contrast['scope']='descriptive_posthoc_same_consumed_guardian_not_new_confirmatory_test'
            dump(a.output/'OBJECTIVE_CONTRAST.json',contrast)
        print('Applied to cached guardian',seed,flush=True)
    oracle=[]
    for cohort,meta,data,pred in [('design',dm,dd,design_predictions[SEEDS[0]]),('guardian_posthoc',gm,gd,load(a.guardian/f'predictions_s{SEEDS[0]}.npz'))]:
      v=data['shadow'];y=v['truth'].astype(float);p=v['proposal'];e=np.abs(y-p)
      pe=np.maximum(pred['error_shadow'],0);mu=np.maximum(pred['demand_shadow'],0)
      for ranking in ['excess','ratio']:
        for replacement,ee,mm in [('none',pe,mu),('realized_error_only',e,mu),('realized_demand_only',pe,y),('both_realized',e,y)]:
          # Ratio oracle: exact zero convention, not the deployable .25 floor.
          if ranking=='excess':s=ee-R*mm
          elif replacement in ['realized_demand_only','both_realized']:
              s=np.divide(ee,mm,out=np.full(ee.shape,np.inf),where=mm>0);s[(mm==0)&(ee==0)]=0
              s=np.minimum(s,np.finfo(np.float64).max)
          else:s=ee/np.maximum(mm,.25)
          for group in ['ALL','HOBBIES','HOUSEHOLD','FOODS']:
            rows=np.ones(len(meta['id']),bool) if group=='ALL' else meta['cat_id']==group
            c=curve(s[rows],y[rows],p[rows]);good=np.flatnonzero(c['feasible']);i=int(good[-1]) if len(good) else None
            oracle.append(dict(cohort=cohort,ranking=ranking,replacement=replacement,group=group,row_coverage=float(c['row'][i]) if i is not None else 0.,demand_coverage=float(c['demand'][i]) if i is not None else 0.,wape=float(c['wape'][i]) if i is not None else None,scope='truth_only_samplewise_substitution_not_conditional_expectation_oracle'))
    assert sha(original)==before
    result=dict(status='REVIEW2_ZERO_FIT_DIAGNOSTICS_COMPLETE',protocol=protocol,records=records,family_choices=families,floor_sensitivity=floors,head_diagnostics=heads,partial_oracles=oracle,seconds=time.time()-t0,original_guardian_result_unchanged=True)
    dump(a.output/'REVIEW2_RESULTS.json',result);dump(a.output/'FRONTIERS.json',frontiers)
    print(json.dumps({'status':result['status'],'row_demand_thresholds_identical':len(records),'family_choices':[{k:v for k,v in r.items() if k not in ['design','guardian_posthoc']} for r in families]},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['design','guardian','output']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
