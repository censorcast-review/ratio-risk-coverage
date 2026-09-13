"""Check group-budget results from compact sufficient statistics; no caches."""
from pathlib import Path
import json
import numpy as np
from run_review5_group_constraints import R, FLOOR, KEYS, rows_lp

ROOT=Path(__file__).resolve().parents[1]


def main():
    result=json.loads((ROOT/'results/review5/GROUP_RESULTS.json').read_text())
    checked=0
    for record in result['records']:
        stat=record['development_bin_statistics']
        n=record['cells']
        for key in KEYS:
            v={k:np.asarray(x) if isinstance(x,list) else x for k,x in stat[key].items()}
            assert abs(v['count'].sum()-1)<1e-12
            assert np.allclose(v['error']-R*v['demand'],v['excess'],atol=1e-14)
            for label,probability in [('strict',record['strict_policy_probabilities']),('cap_only',record['cap_only_lp']['probabilities'])]:
                p=np.asarray(probability)
                assert np.min(p)>=-1e-10 and np.max(p)<=1+1e-10
                expected=record['development_'+label][key]
                assert abs(v['count']@p-expected['row_coverage'])<1e-10
                assert abs(v['excess']@p-expected['excess_per_total_row'])<1e-10
                mass=float(v['demand']@p)
                if mass>0: assert abs((v['error']@p)/mass-expected['wape'])<1e-10
                else: assert expected['wape'] is None
                checked+=1
        cert=record['infeasibility_diagnostic']; matrix=[]; rhs=[]
        for key in KEYS[:2]:
            v=stat[key]
            matrix.extend([np.r_[v['excess'],-1.],np.r_[-np.asarray(v['count']),0.]])
            rhs.extend([0.,-FLOOR])
        matrix=np.asarray(matrix);rhs=np.asarray(rhs)
        y=np.asarray(cert['dual_inequality']);lo=np.asarray(cert['dual_lower']);hi=np.asarray(cert['dual_upper'])
        c=np.r_[np.zeros(n),1.]; residual=c-matrix.T@y-lo-hi
        assert np.max(y)<1e-12 and np.min(lo)>-1e-12 and np.max(hi)<1e-12
        assert abs(residual[-1])<1e-12
        assert abs(lo[-1])+abs(hi[-1])<1e-12
        rigorous_bounded_coordinate_correction=np.minimum(residual[:-1],0).sum()
        bound=float(rhs@y+hi[:-1].sum()+rigorous_bounded_coordinate_correction)
        assert abs(bound-cert['dual_lower_bound'])<1e-12
        p=np.r_[cert['probabilities'],cert['minimum_worst_block_excess_per_row']]
        assert np.max(matrix@p-rhs)<1e-8
        assert abs(p[-1]-bound)<1e-8
        assert record['strict_lp']['success'] == (bound<=1e-8)
        checked+=1
    hobbies=next(r for r in result['records'] if r['group']=='HOBBIES')
    for key in KEYS[:2]:
        v=hobbies['development_bin_statistics'][key]
        occupied=np.asarray(v['count'])>0
        assert np.all(np.asarray(v['excess'])[occupied]>0)
        positive_demand=np.asarray(v['demand'])>0
        minimum=float(np.min(np.asarray(v['error'])[positive_demand]/np.asarray(v['demand'])[positive_demand]))
        assert minimum>R
        assert abs(minimum-hobbies['bin_budget_positivity'][key]['minimum_positive_demand_bin_wape'])<1e-12
    assert not result['all_groups_floor_feasible']
    assert hobbies['cap_only_lp']['minimum_calibration_coverage']==0
    assert hobbies['infeasibility_diagnostic']['dual_lower_bound']>.0142
    # A pooled negative budget cannot make a positive-budget category feasible.
    good={k:dict(count=np.array([1.]),excess=np.array([-R])) for k in KEYS[:2]}
    bad={k:dict(count=np.array([1.]),excess=np.array([1.-R])) for k in KEYS[:2]}
    assert (1.-2*R)/2<0
    assert rows_lp(good,FLOOR)['success']
    assert not rows_lp(bad,FLOOR)['success']
    assert rows_lp(bad,0.)['minimum_calibration_coverage']==0
    print(json.dumps(dict(status='PASS',coefficient_metric_checks=checked,
                          checked_group_dual_certificates=3,
                          positive_Hobbies_bin_budget_blocks=2,
                          no_cross_group_budget_borrowing_synthetic_check=True,
                          raw_cache_reads=0,external_reads=0),indent=2))


if __name__=='__main__':main()
