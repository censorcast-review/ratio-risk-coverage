"""Audit compact fine-grid evidence; no raw data/model load required."""
from pathlib import Path
import argparse,json,math
from reproduction_io import safe_output
from r2_io import sha,dump
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/review5_nb_grid'
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
def run(output):
    p=json.loads((OUT/'PROTOCOL.json').read_text());r=json.loads((OUT/'RESULTS.json').read_text())
    assert r['status']=='COMPLETE' and r['protocol_sha256']==sha(OUT/'PROTOCOL.json')
    assert not any(r[x] for x in ['external_opened','fresh_guardian_opened','certificate_issued'])
    assert r['identical_forecast_verified'] and r['kappa2_retained_result_replay_verified']
    assert r['selection_candidates']==[1.,1.5,2.,3.,5.]
    selected=min([z for z in r['records'] if z['shape'] in r['selection_candidates']],key=lambda x:x['observed_validation_nll'])
    assert selected['shape']==r['selected_shape_by_observed_validation']
    tested=0
    def check(v,implied=False):
        nonlocal tested
        close(v['row_coverage'],v['accepted_n']/v['n'])
        close(v['demand_coverage'],v['demand_mass']/v['total_demand'])
        if v['demand_mass']>0:close(v['wape'],v['error_mass']/v['demand_mass'])
        else:assert v['wape'] is None
        if implied:
            if v['implied_demand_mass']>0:close(v['implied_wape'],v['implied_error_mass']/v['implied_demand_mass'])
            else:assert v['implied_wape'] is None
            if v['error_mass']>0:close(v['error_mass_relative_bias'],v['implied_error_mass']/v['error_mass']-1)
            if v['demand_mass']>0:close(v['demand_mass_relative_bias'],v['implied_demand_mass']/v['demand_mass']-1)
        tested+=1
    def aggregation(gs,implied=False):
        for v in gs.values():check(v,implied)
        for key in ['n','accepted_n','demand_mass','error_mass','total_demand']+(['implied_demand_mass','implied_error_mass'] if implied else []):
            close(gs['ALL'][key],sum(v[key] for k,v in gs.items() if k!='ALL'))
    for rec in r['records']:
        assert rec['hidden_labels_used_for_fit']==0 and rec['hidden_labels_used_for_threshold']==0
        for block in ['calibration_a','calibration_b','shadow']:
            gs=rec['metrics'][block];aggregation(gs,True)
            cm=rec['matched_supervised'][block]
            for name in ['exact_count_metrics','fractional_boundary_metrics']:
                aggregation(cm[name]);close(cm[name]['ALL']['accepted_n'],gs['ALL']['accepted_n'])
            close(cm['matching']['target_count'],gs['ALL']['accepted_n'])
            assert 0<=cm['matching']['boundary_probability']<=1
            if rec['threshold'] is not None and block!='shadow':
                assert gs['ALL']['implied_wape']<=p['cap']+1e-12 and gs['ALL']['row_coverage']>=p['minimum_row_coverage']
    for gs in r['supervised_control']['metrics'].values():aggregation(gs)
    for filename,digest in p['source_hashes'].items():assert sha(ROOT/'code'/filename)==digest,filename
    out={'status':'PASS','moment_records_checked':tested,'grid_models':len(r['records']),'exact_count_matches':3*len(r['records']),
         'source_hashes_verified':True,'selected_shape_by_observed_validation':selected['shape'],'result_sha256':sha(OUT/'RESULTS.json')}
    dump(safe_output(output),out);print(json.dumps(out))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/nb_grid/AGGREGATE_AUDIT.json')
    run(parser.parse_args().output)
