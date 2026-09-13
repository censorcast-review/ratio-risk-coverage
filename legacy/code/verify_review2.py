"""Verify historical metrics and new decomposition without changing results."""
from pathlib import Path
import json,numpy as np
from r2_io import sha,dump
from run_selective_study import load,metric
from run_review2_objectives import scores,KEYS,PRIMARY
ROOT=Path(__file__).resolve().parents[1];R2=ROOT/'results/objectives';G=ROOT.parent/'evaluation_checkpoints'
freeze=ROOT/'provenance/FROZEN_REVIEW_GUARDIAN_COMPARISON.json'
assert sha(freeze)=='3d94aec80d993b92684a61f058ed4e3ceb2351233f07fb8eea3f9f542fe2b156'
f=json.loads(freeze.read_text())
for p,h in f['immutable_files'].items():assert sha(__import__('publication_paths').recorded_path(p))==h,p
original=G/'GUARDIAN_COMPARISON.json';assert sha(original)=='aaaf3e1f72a576f229f4a31ea42cb46470206cb1dbd33678beb24b9609c0077f'
count=0
for cohort,cache,recordpath in [('design',R2/'cache_design',ROOT/'results/conditional_heads/REVIEW_COMPARISON.json'),('guardian',G/'cache',original)]:
    data={k:load(cache/(k+'_aligned.npz')) for k in KEYS};meta=load(cache/'metadata.npz');old=json.loads(recordpath.read_text())
    for seed in f['seeds']:
        pred=load((R2/f'design_predictions_s{seed}.npz') if cohort=='design' else G/f'predictions_s{seed}.npz');ss=scores(pred,data)
        for rec in [x for x in old['records'] if x['seed']==seed]:
            for k in KEYS:
                mask=np.zeros(data[k]['truth'].shape,bool)
                for group,t in rec['thresholds'].items():
                    if t is None:continue
                    rows=np.ones(len(meta['id']),bool) if group=='ALL' else meta['cat_id']==group
                    mask[rows]=ss[rec['score']][k][rows]<=t
                for group,expected in rec['metrics'][k].items():
                    rows=np.ones(len(meta['id']),bool) if group=='ALL' else meta['cat_id']==group
                    now=metric(data[k]['truth'][rows],data[k]['proposal'][rows],data[k]['baseline'][rows],mask[rows])
                    for m,v in expected.items():
                        if v is None:assert now[m] is None,(cohort,seed,rec['score'],k,group,m)
                        else:assert np.isclose(v,now[m],rtol=0,atol=1e-10),(cohort,seed,rec['score'],k,group,m,v,now[m])
                    count+=1
d=json.loads((R2/'objectives/REVIEW2_RESULTS.json').read_text());loss=json.loads((R2/'loss_controls/LOSS_RESULTS.json').read_text())
assert len(d['records'])==27 and all(x['row_threshold']==x['demand_threshold'] for x in d['records'])
assert len(loss['records'])==12 and len(list((R2/'loss_controls').glob('demand_*.txt')))==6
assert all(x['guardian_posthoc']['calibration_b']['ALL']['wape']>d['protocol']['cap'] for x in loss['records'])
for x in d['head_diagnostics']:
    cap=d['protocol']['cap'];assert np.isclose(x['excess_mse'],x['error_mse']+cap*cap*x['demand_mse']-2*cap*x['cross_error_moment'],rtol=1e-10)
audit=dict(status='REVIEW2_NUMERICAL_VERIFICATION_PASS',historical_cohort_block_group_metric_records_reproduced=count,frozen_files_unchanged=len(f['immutable_files']),original_guardian_result_unchanged=True,exact_nested_objective_pairs=27,head_decomposition_checks=len(d['head_diagnostics']),new_development_only_fitted_models=6,external_opened=False,new_independent_guardian_evaluations=0,certificate_issued=False,human_author_verification='NOT_ATTESTED_BY_AUTOMATED_CHECKS')
dump(R2/'VERIFICATION.json',audit);print(json.dumps(audit,indent=2))
