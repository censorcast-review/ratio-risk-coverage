"""Aggregate retained moment audits; no fitting or outcome-file access.

Inputs are summary JSONs and saved threshold readouts only. Uniform-thinning
controls match acceptance *probability mass*, not a realized random mask and
not the ranking of a learned baseline at a new threshold.
"""
from pathlib import Path
import argparse, hashlib, json, math
from statistics import mean

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); source=args.source; out=__import__('reproduction_io').safe_output(args.output)
    out.mkdir(parents=True,exist_ok=True)
    files={
        'capacity_summary': source/'results/capacity_hit_extension/CAPACITY_HIT_VERIFIED_SUMMARY.json',
        'capacity_readouts': source/'results/capacity_hit_extension/CAPACITY_HIT_RESULTS.json',
        'strict_summary': source/'results/observable_extension/OBSERVABLE_VERIFIED_SUMMARY.json',
        'strict_readouts': source/'results/observable_extension/OBSERVABLE_RESULTS.json',
    }
    xx={k:json.loads(p.read_text()) for k,p in files.items()}
    records=[]; checks=0
    for branch,prefix in [('capacity','capacity_hit'),('strict','strict_flag')]:
        summary=xx[branch+'_summary']; raw=xx[branch+'_readouts']
        moments=summary['conditional_moment_audit']
        for v in moments:
            matching=[z for z in raw['records'] if z['method']==v['method'] and z['seed']==v['seed'] and z['score']=='excess' and z['calibration']=='model_implied_observable']
            assert len(matching)==1
            m=matching[0]['metrics'][v['block']]['ALL']
            assert m['accepted_n']==v['accepted_n']
            row=dict(v,branch=prefix,row_coverage=m['row_coverage'],demand_coverage=m['demand_coverage'],threshold=matching[0]['threshold'])
            if v['accepted_n']:
                assert math.isclose(v['implied_total_error']/v['implied_demand'],v['model_implied_wape'],rel_tol=1e-12)
                assert math.isclose(v['realized_total_error']/v['realized_demand'],v['realized_wape'],rel_tol=1e-12)
                assert math.isclose(v['realized_wape'],m['wape'],rel_tol=1e-12)
                be=v['implied_total_error']/v['realized_total_error']-1
                bd=v['implied_demand']/v['realized_demand']-1
                ratio=(1+be)/(1+bd)
                assert math.isclose(ratio,v['model_implied_wape']/v['realized_wape'],rel_tol=1e-12)
                row.update(error_mass_relative_bias=be,demand_mass_relative_bias=bd,
                           implied_minus_realized_wape=v['model_implied_wape']-v['realized_wape'],
                           implied_to_realized_wape_ratio=ratio)
            else:
                assert v['implied_demand']==v['realized_demand']==v['implied_total_error']==v['realized_total_error']==0
                assert v['model_implied_wape'] is None and v['realized_wape'] is None
            records.append(row); checks+=1
    groups=[]
    fields=['row_coverage','demand_coverage','model_implied_wape','realized_wape','error_mass_relative_bias','demand_mass_relative_bias','implied_minus_realized_wape']
    for branch,method,block in sorted(set((v['branch'],v['method'],v['block']) for v in records)):
        rr=[v for v in records if (v['branch'],v['method'],v['block'])==(branch,method,block)]
        assert len(rr)==3
        item={'branch':branch,'method':method,'block':block,'seed_count':3}
        for field in fields:
            values=[v[field] for v in rr if v.get(field) is not None]
            item[field]={'mean':mean(values),'min':min(values),'max':max(values)} if values else None
        groups.append(item)
    controls=[]
    for target in (v for v in records if v['method']=='capacity_hit_nb_2'):
        for method,branch in [('capacity_hit_poisson','capacity_hit'),('truth_poisson_reference','strict_flag')]:
            reference=next(v for v in records if (v['method'],v['branch'],v['seed'],v['block'])==(method,branch,target['seed'],target['block']))
            p=target['row_coverage']/reference['row_coverage']
            assert 0<=p<=1
            mass=p*reference['realized_demand']; error=p*reference['realized_total_error']
            assert math.isclose(error/mass,reference['realized_wape'],rel_tol=1e-12)
            controls.append({'block':target['block'],'seed':target['seed'],'reference_method':method,
                'thinning_probability':p,'row_coverage':target['row_coverage'],
                'demand_coverage':p*reference['demand_coverage'],
                'ratio_of_expected_error_to_expected_demand':error/mass,
                'expected_error_mass':error,'expected_demand_mass':mass,
                'nb2_demand_coverage':target['demand_coverage'],'nb2_realized_wape':target['realized_wape']})
    control_means=[]
    for method,block in sorted(set((v['reference_method'],v['block']) for v in controls)):
        rr=[v for v in controls if (v['reference_method'],v['block'])==(method,block)]
        control_means.append({'reference_method':method,'block':block,
           **{k:mean(v[k] for v in rr) for k in ['row_coverage','demand_coverage','ratio_of_expected_error_to_expected_demand','nb2_demand_coverage','nb2_realized_wape']}})
    result={'status':'RETAINED_SUMMARY_MOMENT_AUDIT_PASS','scope':'POST_HOC_CONSUMED_DEVELOPMENT_SUMMARIES_ONLY',
        'source_hashes':{str(p.relative_to(source)):sha(p) for p in files.values()},
        'new_fit_calls':0,'raw_outcome_files_opened':0,'guardian_files_opened':0,'external_files_opened':0,
        'audited_moment_records':checks,'cap':xx['capacity_summary']['cap'],
        'aggregation':'Arithmetic means of the three fit-specific ratios/biases; not pooled across seeds.',
        'interpretation':'NB shape 2 is conservative in aggregate on its accepted rows: positive error-mass bias exceeds positive demand-mass bias. This is not conditional calibration or a causal decomposition of fitted-model differences.',
        'records':records,'means':groups,
        'validation_selection':xx['capacity_readouts']['validation_selection'],
        'uniform_thinning':{'definition':'Independently retain each row accepted by the saved reference with probability p=c_NB2/c_reference. Risks are ratios of expected masses, not expectations of finite random-sample WAPE.',
            'scope':'Exact fractional-acceptance control on the same empirical development rows and fixed point forecast. Does not compare learned rankings at matched coverage.',
            'records':controls,'means':control_means},
        'matched_learned_ranking_status':{'status':'NOT_COMPUTABLE_FROM_RETAINED_SUMMARIES',
            'missing':['Per-row capacity-hit NB2/Poisson and truth-reference moments (mu, expected_error) for each seed and development block.',
              'Aligned unadapted observed-L1 forecasts and hidden outcomes for these same rows, plus capacity-hit feature arrays if moments are to be regenerated.'],
            'present':'Saved LightGBM model files, aggregate moment audits, saved-threshold metrics, and a separate adapted-forecast review2 cache.',
            'reason':'Aggregates at saved thresholds do not identify alternative-threshold errors or ranking comparisons. The review2 supervised selector uses a different adapted forecast, so reusing its error scores would confound the point forecast.'}}
    __import__('r2_io').dump(out/'OBSERVABLE_MOMENT_DIAGNOSTICS.json',result)
    print(json.dumps({'status':result['status'],'records':checks,'nb2':[v for v in groups if v['method']=='capacity_hit_nb_2'],'matched_controls':control_means},indent=2))

if __name__=='__main__':main()
