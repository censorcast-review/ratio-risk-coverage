"""Audit matched-feature control from published aggregates; raw replay is optional.

The default does not fit a model, load outcome/prediction caches, or alter any
archived evidence. Use --arrays only for consumed-development cache verification.
"""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json
import numpy as np
from r2_io import dump,sha
from reproduction_io import safe_output

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'results/error_interventions'
MODELS={'bundle':'inputs/forecast_bundle_v0_5.joblib','point':'results/point_baselines/observed_l1_s20260906.txt','error':'results/selective_strong/mean_error_s20260906.txt','demand':'results/conditional_heads/demand_s20260906.txt'}

def read(path):return json.loads(path.read_text())
def close(a,b):assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(a,b)
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def verify(out,output,with_arrays=False):
    output=safe_output(output);p=read(out/'PROTOCOL.json');r=read(out/'RESULTS.json')
    old=read(PRIOR/'RESULTS.json');fit=read(out/'models/shared_error_29.json')
    assert r['status']=='COMPLETE' and r['total_new_fits']==1 and fit['fit_count']==1
    assert r['protocol_sha256']==fit['protocol_sha256']==sha(out/'PROTOCOL.json')
    assert r['prior_results_sha256']==sha(PRIOR/'RESULTS.json')
    assert r['fit_receipt_sha256']==sha(out/'models/shared_error_29.json')
    assert p['training_rows']==fit['training_rows']==600000 and fit['feature_count']==29
    assert p['new_method']=='shared_error_29' and p['base_feature_count']==21 and len(p['extra_features'])==8
    assert datetime.fromisoformat(p['frozen_utc'])<datetime.fromisoformat(fit['fit_started_utc'])<=datetime.fromisoformat(fit['fit_completed_utc'])
    assert all(not r[k] for k in ['guardian_inputs_read','external_inputs_read','certificate_issued'])
    assert r['frozen_models_unchanged'] and not p['point_forecast_fit']
    close(p['cap'],.85*.7530939208313988);assert p['row_floor']==.35
    for name,h in p['frozen_model_hashes'].items():assert sha(ROOT/MODELS[name])==h
    assert sha(out/'models/shared_error_29.txt')==fit['model_sha256']
    rr=r['reconstruction'];assert rr['training_rows']==600000 and rr['hobbies_training_rows']==107384
    idx=np.random.default_rng(p['seed']).choice(18290*118,600000,replace=False)
    assert hashlib.sha256(idx.tobytes()).hexdigest()==rr['training_indices_sha256']
    assert len(rr['prior_hobbies_comparisons'])==7 and all(x=='BITWISE_IDENTICAL_ON_107384_HOBBIES_ROWS' for x in rr['prior_hobbies_comparisons'].values())
    assert not rr['historical_forecast_replay_performed'] and rr['historical_point_forecast_fits']==0
    assert len(r['baseline_replay'])==3 and all('BITWISE_IDENTICAL' in x for x in r['baseline_replay'].values())
    methods={x['method']:x for x in r['records']};assert set(methods)=={'shared_error_21','shared_error_29','hobbies_error_21','hobbies_error_29'}
    for rec in old['records']:
        got=dict(methods[rec['method']]);assert got.pop('training_rows')==(600000 if rec['method']=='shared_error_21' else 107384)
        assert got==rec,'Previously reported comparator was modified'
    assert methods['shared_error_29']['model_sha256']==fit['model_sha256']
    count=0;array_count=0
    for name,rec in methods.items():
      for block,z in rec['blocks'].items():
        m=z['policy'];h=z['head'];q=z['samplewise_prefix_diagnostic'];t=rec['selected_threshold']
        close(m['row_coverage'],m['accepted_n']/m['n']);close(m['demand_coverage'],m['demand_mass']/m['total_demand'])
        close(h['error_mse'],h['squared_error_sum']/h['n']);close(h['positive_day_error_mse'],h['positive_day_squared_error_sum']/h['positive_day_n'])
        assert h['n']==m['n'] and h['positive_day_n']<=h['n']
        if not m['demand_mass']:assert m['wape'] is None and m['accepted_n']==0
        else:close(m['wape'],m['error_mass']/m['demand_mass'])
        if t is None:assert m['accepted_n']==0
        elif block!='shadow':assert m['wape']<=p['cap']+1e-12 and m['row_coverage']>=p['row_floor']
        if q['threshold'] is None:assert q['row_coverage']==q['demand_coverage']==0 and q['wape'] is None
        else:assert q['wape']<=p['cap']+1e-12 and q['row_coverage']>=p['row_floor']
        count+=1
        if with_arrays:
            from run_review2_objectives import curve
            v=arrays(PRIOR/'cache'/(block+'.npz'))
            cp=out/'cache' if name=='shared_error_29' else PRIOR/'cache'
            zpred=arrays(cp/f'predictions_{name}_{block}.npz')
            err=np.abs(v['truth'].astype(float)-v['f']);de=zpred['error'].astype(float)-err
            close(np.sum(de*de),h['squared_error_sum']);close(np.sum(de[v['truth']>0]**2),h['positive_day_squared_error_sum'])
            cv=curve(zpred['score'],v['truth'],v['f']);good=np.flatnonzero(cv['feasible'])
            if len(good):close(q['threshold'],cv['threshold'][good[-1]]);close(q['row_coverage'],cv['row'][good[-1]])
            else:assert q['threshold'] is None
            array_count+=1
    if with_arrays:
        from run_hobbies_intervention import invariance_test
        from run_review2_objectives import curve,select_threshold
        assert invariance_test()==rr['feature_invariance']
        assert sha(out/'cache/train.npz')==rr['training_cache_sha256']
        tr=arrays(out/'cache/train.npz');oldtr=arrays(PRIOR/'cache/train.npz');mask=tr['hobbies_mask']
        assert np.array_equal(idx,tr['original_flat_index'])
        for k in ['x','extra','truth','f','series','days','original_flat_index']:assert np.array_equal(tr[k][mask],oldtr[k])
        cs=[]
        for block in ['calibration_a','calibration_b']:
            pp=arrays(out/'cache'/f'predictions_shared_error_29_{block}.npz');v=arrays(PRIOR/'cache'/(block+'.npz'))
            cs.append(curve(pp['score'],v['truth'],v['f']))
        t,obj=select_threshold(*cs,'row');assert t==methods['shared_error_29']['selected_threshold'];close(obj,methods['shared_error_29']['minimum_calibration_row_coverage'])
    audit={'status':'PASS','protocol_sha256':sha(out/'PROTOCOL.json'),'results_sha256':sha(out/'RESULTS.json'),'aggregate_records_checked':count,
       'development_array_records_checked':array_count,'original_comparator_records_preserved':3,'same_training_sample_rows':600000,'same_hobbies_subset_rows':107384,
       'new_fits_verified':1,'freeze_precedes_fit':True,'guardian_inputs_read':False,'external_inputs_read':False,'raw_or_prediction_cache_read':with_arrays,
       'scope':'Aggregate consistency plus retained fit/hash/sample audit; independent development array checks only when --arrays is requested'}
    dump(output,audit);print(json.dumps(audit,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=ROOT/'results/shared_error_control');ap.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/shared_error_control/AUDIT.json');ap.add_argument('--arrays',action='store_true');a=ap.parse_args();verify(a.results,a.output,a.arrays)
