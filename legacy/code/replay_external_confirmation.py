"""Replay the fixed external statistical evaluation from item sufficient statistics.

No raw data, row-level arrays, model fitting, threshold selection, or new
holdout opening is used. The original frozen evaluator is imported unchanged.
"""
from pathlib import Path
import argparse, importlib.util, json
import numpy as np
from r2_io import dump, sha

ROOT=Path(__file__).resolve().parents[1]


def main(output):
    folder=ROOT/'results/external_confirmation'
    report_path=folder/'EXTERNAL_COMPARISON.json'
    stats_path=folder/'EXTERNAL_ITEM_STATISTICS.json'
    report=json.loads(report_path.read_text());stats=json.loads(stats_path.read_text())
    evaluator=ROOT/'code/external_reproduction/evaluate.py'
    if not evaluator.is_file():evaluator=ROOT/'code/review5_external/evaluate.py'
    spec=importlib.util.spec_from_file_location('frozen_external_evaluator',evaluator)
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    source_hashes=stats['source_files_sha256']
    assert sha(report_path)==source_hashes['results/external_confirmation/EXTERNAL_COMPARISON.json']
    assert sha(evaluator)==source_hashes['code/review5_external/evaluate.py']
    assert stats['freeze_sha256']==report['freeze_sha256']
    assert stats['external_use_count']==report['external_use_count']==1
    assert stats['actual_item_ids_included'] is False
    assert stats['component_columns']==['accepted_rows','total_rows','accepted_demand','accepted_absolute_error','total_demand']
    assert stats['policy_names']==list(ev.THRESHOLDS)
    assert stats['bootstrap_reps']==report['bootstrap_reps']==10000
    assert stats['bootstrap_seed']==report['bootstrap_seed']==20260906
    assert stats['primary_endpoint_alpha']==.025
    assert stats['secondary_endpoint_alpha']==report['secondary_endpoint_alpha']==.05/48
    assert ev.CAP==.85*.7530939208313988 and ev.ROW_FLOOR==.35
    assert ev.THRESHOLDS=={'composed_excess_lambda_1':.2888724531967165,'mixed_utility_lambda_0_25':.37635288579749315}
    assert ev.REFERENCE_MEAN==1.3048948713314772
    names=stats['policy_names'];checks=0;max_error=0.;point_records=0;secondary_bounds=0
    def compare(value,expected,label):
        nonlocal checks,max_error
        if isinstance(expected,bool):assert bool(value)==expected,label
        elif expected is None:assert value is None,label
        elif isinstance(expected,(int,float)):
            assert value is not None and np.isclose(value,expected,rtol=1e-12,atol=1e-10), (label,value,expected)
            max_error=max(max_error,abs(float(value)-float(expected)))
        else:assert value==expected,label
        checks+=1
    def point(matrix):
        v=matrix.sum(axis=0);rows=int(round(v[1]));accepted=int(round(v[0]));mass=float(v[2]);error=float(v[3]);total=float(v[4])
        return dict(rows=rows,accepted_rows=accepted,row_coverage=accepted/rows if rows else None,
            demand_coverage=mass/total if total>0 else None,wape=error/mass if mass>0 else None,
            accepted_demand=mass,total_demand=total,accepted_absolute_error=error,
            point_operating_check_pass=bool(rows and mass>0 and error/mass<=ev.CAP and accepted/rows>=ev.ROW_FLOOR))
    def compare_point(matrix,expected,label):
        nonlocal point_records
        actual=point(matrix)
        assert set(actual)==set(expected),label
        for k,v in expected.items():compare(actual[k],v,label+'/'+k)
        point_records+=1
        return actual
    records={(r['period'],r['group']):r for r in stats['records']}
    assert len(records)==len(stats['records'])==13
    required={(period,group) for period in ev.PERIODS for group in ev.GROUPS}|{('descriptive_full_28','ALL')}
    assert set(records)==required,set(records)
    all_secondary=[];primary_recomputed=None
    for period,group in sorted(records):
        rec=records[(period,group)];n=rec['item_clusters']
        assert rec['anonymous_item_index']==list(range(n))
        matrices=[np.asarray(rec['policy_components'][name],dtype=np.float64) for name in names]
        assert all(m.shape==(n,5) and np.isfinite(m).all() and (m>=0).all() for m in matrices)
        assert np.array_equal(matrices[0][:,[1,4]],matrices[1][:,[1,4]])
        for m in matrices:
            assert np.all(m[:,0]<=m[:,1]) and np.all(m[:,2]<=m[:,4]+1e-10)
            assert np.array_equal(m[:,:2],np.rint(m[:,:2]))
        if period=='descriptive_full_28':
            assert rec['target_days']==[1914,1941]
            for name,m in zip(names,matrices):compare_point(m,report['descriptive_full_28_sensitivity']['metrics'][name],period+'/'+name)
            continue
        assert rec['target_days']==ev.PERIODS[period]
        draws=ev.bootstrap_matrices(matrices,reps=10000,seed=20260906)
        if period=='primary_23' and group=='ALL':
            primary=report['primary'];points={}
            for name,m in zip(names,matrices):points[name]=compare_point(m,primary['metrics'][name],'primary/'+name)
            delta=draws[1]-draws[0];valid=np.isfinite(delta[:,:2]).all(axis=1)
            low=float(np.quantile(delta[valid,1],.025)) if valid.all() else float('nan')
            upper=float(np.quantile(delta[valid,0],.975)) if valid.all() else float('nan')
            primary_recomputed=dict(
                row_coverage_difference=points[names[1]]['row_coverage']-points[names[0]]['row_coverage'],
                demand_coverage_difference=points[names[1]]['demand_coverage']-points[names[0]]['demand_coverage'],
                row_difference_upper_one_sided_97_5=ev.finite(upper),
                demand_difference_lower_one_sided_97_5=ev.finite(low),
                directional_confirmation_pass=bool(valid.all() and low>0 and upper<0),
                invalid_bootstrap_demand_draws=int((~valid).sum()))
            for k,v in primary_recomputed.items():compare(v,primary[k],'primary/'+k)
        for name,m,z in zip(names,matrices,draws):
            matches=[x for x in report['secondary'] if (x['period'],x['group'],x['policy'])==(period,group,name)]
            assert len(matches)==1
            old=matches[0];compare_point(m,old['metrics'],period+'/'+group+'/'+name)
            cl=float(np.quantile(z[:,0],.05/48));wu=float(np.quantile(z[:,2],1-.05/48,method='higher'))
            current=dict(row_coverage_lcb=ev.finite(cl),wape_ucb=ev.finite(wu),
                undefined_wape_draws=int((~np.isfinite(z[:,2])).sum()),
                adjusted_operating_check_pass=bool(cl>=ev.ROW_FLOOR and wu<=ev.CAP))
            compare(n,old['item_clusters'],'item_clusters')
            for k,v in current.items():compare(v,old[k],period+'/'+group+'/'+name+'/'+k)
            secondary_bounds+=2;all_secondary.append(current)
    assert primary_recomputed is not None and secondary_bounds==report['secondary_endpoint_count']==48
    compare(all(v['adjusted_operating_check_pass'] for v in all_secondary),report['joint_secondary_operating_check_pass'],'joint_secondary')
    result=dict(status='FROZEN_EXTERNAL_ITEM_STATISTIC_REPLAY_PASS',
        item_statistic_records=len(records),point_metric_records=point_records,
        primary_directional_bounds=2,secondary_bounds=secondary_bounds,numerical_and_decision_checks=checks,
        maximum_absolute_replay_difference=max_error,float_comparison_rtol=1e-12,float_comparison_atol=1e-10,
        primary=primary_recomputed,
        row_secondary_bounds_pass=sum(x['row_coverage_lcb'] is not None and x['row_coverage_lcb']>=ev.ROW_FLOOR for x in all_secondary),
        wape_secondary_bounds_pass=sum(x['wape_ucb'] is not None and x['wape_ucb']<=ev.CAP for x in all_secondary),
        raw_or_row_level_arrays_read=False,new_fit_calls=0,new_threshold_selections=0,new_holdout_openings=0,
        original_frozen_evaluator_sha256=sha(evaluator),external_result_sha256=sha(report_path),
        sufficient_statistic_sha256=sha(stats_path),numpy_version=np.__version__,
        scope='Exact frozen statistical formulas and bootstrap seed replayed from saved item sums. Forecast generation and threshold-mask construction are not rerun by this audit.')
    dump(output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reproduction_outputs/EXTERNAL_REPLAY_AUDIT.json')
    args=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main(__import__('reproduction_io').safe_output(args.output))
