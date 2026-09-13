"""Independent aggregate and optional array audit; never alters frozen evidence."""
from pathlib import Path
import argparse,json
import numpy as np
from r2_io import dump,sha
from run_hobbies_intervention import ROOT,MODELS,PARAMS,CAP,EXTRA,invariance_test,load
from run_review2_objectives import curve,select_threshold

def check(out,output):
    p=json.loads((out/'PROTOCOL.json').read_text());r=json.loads((out/'RESULTS.json').read_text());checks=[]
    assert r['status']=='COMPLETE' and r['total_new_fits']==2
    assert r['protocol_sha256']==sha(out/'PROTOCOL.json')
    assert not r['guardian_inputs_read'] and not r['external_inputs_read'] and not r['certificate_issued']
    assert p['parameters']==PARAMS and p['extra_features']==EXTRA
    for rel,d in p['source_hashes'].items():assert sha(__import__('publication_paths').recorded_path(rel))==d,rel
    for name,d in p['frozen_model_hashes'].items():assert sha(ROOT/MODELS[name])==d,name
    checks.append('Frozen source, model, scope and protocol hashes')
    assert len(r['records'])==3 and set(x['method'] for x in r['records'])==set(p['methods'])
    assert all(v=='BITWISE_IDENTICAL_POINT_FORECASTS_AND_TRUTH' for v in r['reconstruction']['original_forecast_replay'].values())
    checks.append('Three original development block point replays recorded bitwise-identical')
    checks.append(invariance_test())
    # Independent exact enumeration checks common-threshold logic under ties.
    toy=[(np.array([0,0,1,2]),np.array([1,1,1,10]),np.array([1,1,1,0])),(np.array([0,1,1,3]),np.array([2,1,1,10]),np.array([2,1,1,0]))]
    cc=[curve(s,y,f) for s,y,f in toy];t,_=select_threshold(*cc,'row');valid=[]
    for z in np.unique(np.concatenate([x[0] for x in toy])):
        good=True
        for s,y,f in toy:
            a=s<=z;mass=y[a].sum();good &= a.mean()>=.35 and mass>0 and np.abs(y[a]-f[a]).sum()/mass<=CAP
        if good:valid.append(float(z))
    assert t==max(valid)
    checks.append('Common-threshold selection agrees with independent enumeration and tie handling')
    cache=out/'cache';array_checks=0;aggregate_checks=0
    for rec in r['records']:
        name=rec['method'];path=ROOT/MODELS['error'] if name=='shared_error_21' else out/'models'/(name+'.txt')
        assert sha(path)==rec['model_sha256']
        if name!='shared_error_21':
            info=json.loads(path.with_suffix('.json').read_text());assert info['training_rows']==r['reconstruction']['training_rows'];assert info['num_features']==rec['feature_count']
        t=rec['selected_threshold']
        for k,z in rec['blocks'].items():
            m=z['policy'];h=z['head'];q=z['samplewise_prefix_diagnostic']
            assert np.isclose(m['row_coverage'],m['accepted_n']/m['n'],rtol=0,atol=1e-15)
            assert np.isclose(m['demand_coverage'],m['demand_mass']/m['total_demand'],rtol=0,atol=1e-15)
            assert np.isclose(h['error_mse'],h['squared_error_sum']/h['n'],rtol=1e-14)
            assert np.isclose(h['positive_day_error_mse'],h['positive_day_squared_error_sum']/h['positive_day_n'],rtol=1e-14)
            if m['demand_mass']>0:assert np.isclose(m['wape'],m['error_mass']/m['demand_mass'],rtol=1e-14)
            else:assert m['wape'] is None and m['accepted_n']==0
            if t is None:assert m['accepted_n']==0
            elif k!='shadow':assert m['wape']<=CAP+1e-12 and m['row_coverage']>=.35
            if q['wape'] is not None:assert q['wape']<=CAP+1e-12 and q['row_coverage']>=.35
            aggregate_checks+=1
            pp=cache/f'predictions_{name}_{k}.npz';vp=cache/(k+'.npz')
            if pp.exists() and vp.exists():
                pred=load(pp);v=load(vp);y=v['truth'].astype(float);err=np.abs(y-v['f']);a=np.zeros(len(y),bool) if t is None else pred['score']<=t
                assert int(a.sum())==m['accepted_n'];assert np.isclose(np.sum((pred['error'].astype(float)-err)**2),h['squared_error_sum'],rtol=1e-13)
                cv=curve(pred['score'],y,v['f']);good=np.flatnonzero(cv['feasible'])
                if len(good):assert q['threshold']==float(cv['threshold'][good[-1]]) and q['row_coverage']==float(cv['row'][good[-1]])
                else:assert q['threshold'] is None
                array_checks+=1
        if (cache/'train.npz').exists():
            tr=load(cache/'train.npz');assert len(tr['truth'])==r['reconstruction']['training_rows'];assert np.unique(tr['original_flat_index']).size==len(tr['truth'])
            assert np.array_equal(tr['days'],1436+tr['original_flat_index']%118)
    # Verify unchanged shared-head outputs against already published DEVELOPMENT-only aggregates.
    published=json.loads((ROOT/'results/objectives/objectives/REVIEW2_RESULTS.json').read_text())['head_diagnostics']
    baseline=next(z for z in r['records'] if z['method']=='shared_error_21')
    replay=0
    for z in published:
        if z['cohort']=='design' and z['seed']==20260906 and z['group']=='HOBBIES' and z['error_trees']==220 and z['demand_trees']==220:
            h=baseline['blocks'][z['block']]['head'];assert np.isclose(h['error_mse'],z['error_mse'],rtol=1e-12,atol=1e-12),(h,z)
            assert np.isclose(h['error_bias'],z['error_bias'],rtol=1e-12,atol=1e-12)
            replay+=1
    assert replay==3
    checks.append('Original shared Hobbies error MSE/bias reproduce all 3 published development blocks')
    result={'status':'PASS','results_sha256':sha(out/'RESULTS.json'),'protocol_sha256':sha(out/'PROTOCOL.json'),'aggregate_records_checked':aggregate_checks,'optional_array_records_checked':array_checks,'baseline_head_replays':replay,'checks':checks,'guardian_inputs_read':False,'external_inputs_read':False}
    dump(output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=ROOT/'results/error_interventions');ap.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/hobbies/AUDIT.json');a=ap.parse_args();check(a.results,__import__('reproduction_io').safe_output(a.output))
