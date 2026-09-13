"""Independently recompute only the registered external statistics from saved predictions.

Never imports the frozen evaluator, reads sealed/raw outcomes, fits a model,
changes thresholds, or evaluates another policy. The same prespecified
bootstrap draws are reproduced for numerical verification, not new inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
sys.path.insert(0,'/tmp/censorcast_external_legacy')
import numpy as np
from r2_io import dump, sha
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'code/review5_external/EXTERNAL_PLAN.json'
OUT=ROOT/'results/external_confirmation'
CACHE=OUT/'prediction_cache'
REPORT=OUT/'EXTERNAL_COMPARISON.json'
checks=[]; diffs=[]
def check(name, passed, evidence=None):
    checks.append({'check':name,'pass':bool(passed),'evidence':evidence})
def same(name, actual, expected, tol=5e-11):
    if actual is None or expected is None:
        ok=actual is expected
    elif isinstance(actual,(bool,np.bool_)) or isinstance(expected,(bool,np.bool_)):
        ok=bool(actual)==bool(expected)
    elif isinstance(actual,(int,float,np.number)) and isinstance(expected,(int,float,np.number)):
        ok=np.isfinite(actual) and np.isfinite(expected) and abs(float(actual)-float(expected))<=tol*max(1.,abs(float(actual)),abs(float(expected)))
    else:ok=actual==expected
    if not ok:diffs.append({'field':name,'recomputed':actual,'reported':expected})
    return bool(ok)
def compare_metrics(name,a,b):
    for key,value in a.items():same(name+'.'+key,value,b.get(key))

def read_cache(name):
    with np.load(CACHE/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
plan=json.loads(PLAN.read_text());report=json.loads(REPORT.read_text())
receipt=json.loads((CACHE/'PREDICTION_RECEIPT.json').read_text())
opened=json.loads((OUT/'EXTERNAL_OPENED.json').read_text())
check('freeze_hash_matches_result_prediction_opening',all(x.get('freeze_sha256')==sha(PLAN) for x in [report,receipt,opened]),sha(PLAN))
immutable_fail=[rel for rel,h in plan['immutable_files'].items() if sha(ROOT/rel)!=h]
check('all_frozen_dependencies_byte_exact',not immutable_fail,{'file_count':len(plan['immutable_files']),'mismatches':immutable_fail})
check('one_approved_external_use_no_certificate',report.get('external_use_count')==opened.get('external_use_count')==1 and report.get('certificate_issued') is False and opened.get('certificate_issued') is False)
check('exactly_two_registered_policies_and_no_fit_search',report.get('policy_count')==2 and report.get('fit_calls')==0 and report.get('threshold_selection_calls')==0)
check('bootstrap_matches_registered_settings',report['bootstrap_reps']==plan['bootstrap']['reps']==10000 and report['bootstrap_seed']==plan['bootstrap']['seed']==20260906)
data=read_cache('external_predictions.npz');heads=read_cache('external_head_scores.npz');meta=read_cache('external_metadata.npz')
y=np.asarray(data['truth'],dtype=np.float64);prediction=np.asarray(data['proposal'],dtype=np.float64)
days=np.asarray(data['target_days'],dtype=int);items=np.asarray(meta['item_id']).astype(str);cats=np.asarray(meta['cat_id']).astype(str)
labels,counts=np.unique(items,return_counts=True)
item_hash=hashlib.sha256(('\n'.join(labels.tolist())+'\n').encode()).hexdigest()
check('saved_cache_target_and_partition_shape',y.shape==prediction.shape==(6100,28) and np.array_equal(days,np.arange(1914,1942)) and len(labels)==610 and np.all(counts==10) and item_hash==plan['item_set_sha256'],{'array_shape':list(y.shape),'items':len(labels),'item_set_sha256':item_hash})
check('saved_targets_and_predictions_finite_nonnegative',np.isfinite(y).all() and np.isfinite(prediction).all() and (y>=0).all() and (prediction>=0).all())
primary_cols=(days>=plan['primary_target_days'][0])&(days<=plan['primary_target_days'][1])
origins=days-(1+(days-1)%7)
check('primary_count_and_later_origins',int(primary_cols.sum())*len(items)==plan['expected_primary_rows']==140300 and (origins[primary_cols]>1913).all(),{'primary_rows':int(primary_cols.sum())*len(items),'primary_origins':np.unique(origins[primary_cols]).tolist(),'excluded_first_five_origin':np.unique(origins[~primary_cols]).tolist()})

# Reconstruct only the two already-frozen acceptance masks from saved heads.
cap=float(plan['wape_cap']);floor=float(plan['row_floor']);T=float(plan['reference_mean'])
error=np.maximum(np.asarray(heads['error'],dtype=np.float32),0).astype(np.float64)
demand=np.maximum(np.asarray(heads['demand'],dtype=np.float32),0).astype(np.float64)
masks={}
for policy in plan['policies']:
    lam=policy['lambda'];score=(error-cap*demand)/(lam+(1-lam)*demand/T)
    masks[policy['name']]=np.isfinite(score)&(score<=policy['threshold'])
names=[p['name'] for p in plan['policies']]
check('policy_order_and_utility_roles',names==['composed_excess_lambda_1','mixed_utility_lambda_0_25'] and [p['lambda'] for p in plan['policies']]==[1.,.25])

# Unlike the evaluator's series-level bincount construction, directly sum each
# item's flattened stores/days into sufficient statistics.
def stats_and_metric(row_selection,col_selection,acceptance):
    yy=y[row_selection][:,col_selection];pp=prediction[row_selection][:,col_selection];aa=acceptance[row_selection][:,col_selection]
    ids=items[row_selection];unique=np.unique(ids)
    n=yy.size;accepted=int(np.count_nonzero(aa));mass=float(np.sum(yy[aa],dtype=np.float64));total=float(np.sum(yy,dtype=np.float64));err=float(np.sum(np.abs(yy[aa]-pp[aa]),dtype=np.float64))
    metric={'rows':n,'accepted_rows':accepted,'row_coverage':accepted/n if n else None,'demand_coverage':mass/total if total>0 else None,'wape':err/mass if mass>0 else None,'accepted_demand':mass,'total_demand':total,'accepted_absolute_error':err,'point_operating_check_pass':bool(n and mass>0 and err/mass<=cap and accepted/n>=floor)}
    matrix=np.empty((len(unique),5),dtype=np.float64)
    for j,item in enumerate(unique):
        ss=ids==item;yj=yy[ss].ravel();pj=pp[ss].ravel();aj=aa[ss].ravel()
        matrix[j]=[np.count_nonzero(aj),yj.size,np.sum(yj[aj],dtype=np.float64),np.sum(np.abs(yj[aj]-pj[aj]),dtype=np.float64),np.sum(yj,dtype=np.float64)]
    return matrix,metric

def resample(matrices):
    n=len(matrices[0]);rng=np.random.default_rng(plan['bootstrap']['seed']);reps=plan['bootstrap']['reps']
    out=[np.empty((reps,3),dtype=np.float64) for _ in matrices]
    for i in range(0,reps,200):
        count=min(200,reps-i);weights=rng.multinomial(n,np.repeat(1/n,n),size=count)
        for target,matrix in zip(out,matrices):
            totals=weights@matrix
            with np.errstate(divide='ignore',invalid='ignore'):
                target[i:i+count,0]=totals[:,0]/totals[:,1]
                target[i:i+count,1]=totals[:,2]/totals[:,4]
                target[i:i+count,2]=np.where(totals[:,2]>0,totals[:,3]/totals[:,2],np.inf)
    return out
all_rows=np.ones(len(items),dtype=bool)
primary_parts=[];primary_metrics={}
for name in names:
    matrix,metric=stats_and_metric(all_rows,primary_cols,masks[name]);primary_parts.append(matrix);primary_metrics[name]=metric
    compare_metrics('primary.metrics.'+name,metric,report['primary']['metrics'][name])
boot=resample(primary_parts);deltas=boot[1]-boot[0]
valid=np.isfinite(deltas[:,:2]).all(axis=1)
lo=float(np.quantile(deltas[:,1],.025)) if valid.all() else None
hi=float(np.quantile(deltas[:,0],.975)) if valid.all() else None
primary_recomputed={'row_coverage_difference':primary_metrics[names[1]]['row_coverage']-primary_metrics[names[0]]['row_coverage'],'demand_coverage_difference':primary_metrics[names[1]]['demand_coverage']-primary_metrics[names[0]]['demand_coverage'],'row_difference_upper_one_sided_97_5':hi,'demand_difference_lower_one_sided_97_5':lo,'directional_confirmation_pass':bool(valid.all() and lo>0 and hi<0),'familywise_alpha':.05,'one_sided_endpoint_alpha':.025,'invalid_bootstrap_demand_draws':int(np.sum(~valid))}
for key,value in primary_recomputed.items():same('primary.'+key,value,report['primary'][key])

registered={(period,group,name) for period in plan['secondary']['periods'] for group in plan['secondary']['strata'] for name in names}
record_keys=[(v['period'],v['group'],v['policy']) for v in report['secondary']]
check('secondary_complete_nonduplicated_24_records_48_endpoints',len(record_keys)==len(set(record_keys))==24 and set(record_keys)==registered and report['secondary_endpoint_count']==48,{'records':len(record_keys),'bounds_per_record':2,'endpoint_count':len(record_keys)*2})
reported_secondary={key:value for key,value in zip(record_keys,report['secondary'])}
alpha=plan['secondary']['per_endpoint_alpha']
check('separate_secondary_bonferroni_family',alpha==.05/48 and report['secondary_endpoint_alpha']==alpha)
secondary_recomputed=[]
for period,(start,end) in plan['secondary']['periods'].items():
    cols=(days>=start)&(days<=end)
    for group in plan['secondary']['strata']:
        rows=all_rows if group=='ALL' else cats==group
        if not rows.any():
            raise ValueError('Registered category absent: independent audit requires explicit absent-stratum branch.')
        parts=[];metrics=[]
        for name in names:
            matrix,metric=stats_and_metric(rows,cols,masks[name]);parts.append(matrix);metrics.append(metric)
        draws=resample(parts)
        for name,metric,z in zip(names,metrics,draws):
            cl=float(np.quantile(z[:,0],alpha));wu=float(np.quantile(z[:,2],1-alpha,method='higher'))
            got={'period':period,'group':group,'policy':name,'item_clusters':int(len(np.unique(items[rows]))),'metrics':metric,'row_coverage_lcb':cl,'wape_ucb':wu if np.isfinite(wu) else None,'undefined_wape_draws':int(np.count_nonzero(~np.isfinite(z[:,2]))),'adjusted_operating_check_pass':bool(cl>=floor and wu<=cap)}
            ref=reported_secondary[(period,group,name)]
            compare_metrics('secondary.'+'.'.join([period,group,name])+'.metrics',metric,ref['metrics'])
            for key,value in got.items():
                if key!='metrics':same('secondary.'+'.'.join([period,group,name])+'.'+key,value,ref.get(key))
            secondary_recomputed.append(got)
same('joint_secondary_operating_check_pass',all(x['adjusted_operating_check_pass'] for x in secondary_recomputed),report['joint_secondary_operating_check_pass'])
full={}
for name in names:
    _,metric=stats_and_metric(all_rows,np.ones(len(days),dtype=bool),masks[name]);full[name]=metric
    compare_metrics('descriptive_full_28_sensitivity.metrics.'+name,metric,report['descriptive_full_28_sensitivity']['metrics'][name])
check('registered_metrics_and_quantiles_reproduce',not diffs,{'mismatch_count':len(diffs),'relative_or_absolute_tolerance':5e-11,'primary_endpoint_count':2,'secondary_endpoint_count':48,'point_metric_records':28,'primary_invalid_draws':int(np.sum(~valid))})

paths=[PLAN,REPORT,CACHE/'external_predictions.npz',CACHE/'external_head_scores.npz',CACHE/'external_metadata.npz',CACHE/'PREDICTION_RECEIPT.json',OUT/'EXTERNAL_OPENED.json',Path(__file__)]
result={'status':'PASS' if all(v['pass'] for v in checks) and not diffs else 'FAIL','completed_utc':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(PLAN),'auditor':'Independent implementation of registered aggregate and bootstrap calculations; frozen evaluator was read but not imported.','actual_scope':{'raw_external_inputs_read':False,'new_model_predictions':False,'model_fits':0,'new_threshold_selection':0,'policy_count':2,'new_policy_evaluations':False,'saved_prediction_cache_read':True,'same_registered_bootstrap_recomputed':True,'new_statistical_hypotheses':0,'predictor_feature_construction_independently_reexecuted':False},'checks':checks,'mismatches':diffs,'primary_recomputed':primary_recomputed,'primary_metrics_recomputed':primary_metrics,'secondary_recomputed':secondary_recomputed,'full_28_metrics_recomputed':full,'input_and_audit_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'runtime_numpy':np.__version__,'inferential_limits':['Primary two-endpoint family and secondary 48-endpoint family each have nominal alpha=.05; no joint 50-endpoint alpha=.05 claim is justified.','Item-cluster percentile bootstrap is approximate and conditions on frozen fits and policies; common calendar/store shocks and training uncertainty are not removed.','The pair was selected after consumed-guardian exploration but frozen before external access; only its registered two-policy coverage contrast is confirmed if both directions pass.','Lambda .25 optimizes a stated mixed utility, not pure denominator coverage; this experiment does not prove its population optimality or compare every possible selector.','Targets are recorded sales under mechanical censoring; latent demand under natural stockouts is not identified by these results.','Raw-input integrity and forecast-origin construction were not independently rerun here; this audit checks their stored receipt and frozen-code hashes, plus only cached registered statistics.']}
dump(ROOT/'results/external_audits/EXTERNAL_INDEPENDENT_AUDIT.json',result)
print(json.dumps({'status':result['status'],'checks':len(checks),'mismatches':len(diffs),'primary':primary_recomputed,'joint_secondary_check_pass':all(x['adjusted_operating_check_pass'] for x in secondary_recomputed)},indent=2))
