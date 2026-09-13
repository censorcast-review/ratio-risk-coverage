"""Independent recomputation of revision results; never changes frozen evidence."""
from pathlib import Path
from datetime import datetime
import argparse,json,tempfile,zipfile,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
from audit_observed_data import sha,rmsse_scale,metrics,severity,concentration,optimal_scale
from m5_capacity_reference import simulate_controlled_censoring
from run_m5_observable import candidate

def read(p):return json.loads(Path(p).read_text())
def near(a,b,tol=1e-10):assert np.isclose(a,b,rtol=0,atol=tol),(a,b)
def run(data):
    E=ROOT/'evidence/revision';checks=[];u=ROOT/'evidence/uci/run_001'
    a=read(E/'audit/UCI_AUDIT.json');pr=np.load(u/'FROZEN_PREDICTIONS.npz');y=np.load(u/'sealed_targets.npz')['truth'].astype(float)
    dev=np.load(u/'development.npz');context=np.load(u/'prediction_context.npz');days=pr['days']
    q=rmsse_scale(dev['truth'][:,:int(dev['train_days'][-1])+1])
    for k in ['baseline','proposal']:
        actual=metrics(y,pr[k],q)
        for key,value in actual.items():near(value,a['frozen_test_metrics'][k][key])
    assert severity(y,context['observed'][:,days],context['capacity'][:,days])==a['severity']['test']
    con=concentration(y,pr['baseline'],pr['proposal'])
    for x,z in zip(con,a['concentration']):
        for k,v in x.items():
            if isinstance(v,str):assert v==z[k]
            else:near(v,z[k])
    val=np.load(E/'audit/UCI_VALIDATION_PREDICTIONS.npz');yv=val['truth'];b=val['unscaled_l1']
    near(optimal_scale(yv,b),a['continuous_optimum_scale'])
    for record in a['validation_scale_diagnostic']:near(metrics(yv,record['scale']*b)['wape'],record['wape'])
    assert a['validation_scale_diagnostic'][1]['wape']<a['frozen_correction_validation_wape']
    assert a['test_new_predictions_computed']==0
    checks.append('UCI: new metrics, hit/strict/hidden mass, concentration and validation-only scale audit recomputed')
    z=np.load(data/'data/design_outcomes_v0_5.npz');s,c,h=simulate_controlled_censoring(z['truth'])
    assert np.array_equal(s,z['observed']) and np.array_equal(c,z['capacity']) and np.array_equal(h,z['censored'])
    ma=read(E/'audit/M5_AUDIT.json');q=rmsse_scale(z['truth'][:,:1313])
    for block in ['calibration_a','calibration_b','shadow']:
        cache=np.load(data/'cache'/f'{block}_aligned.npz');dd=cache['target_days'];yy=z['truth'][:,dd-1]
        assert np.array_equal(cache['truth'],yy)
        assert severity(yy,s[:,dd-1],c[:,dd-1])==ma[block]['severity']
        for name in ['baseline','proposal']:
            for key,v in metrics(yy,cache[name],q).items():near(v,ma[block]['metrics'][name][key])
    checks.append('M5: complete capacity/observation/strict-flag recurrence replay, severity and original additional metrics')
    base=E/'m5_observable';protocol=read(base/'PROTOCOL.json');results=read(base/'RESULTS.json')
    assert protocol['code_sha256']==sha(Path(__file__).with_name('run_m5_observable.py'))
    assert protocol['utility_sha256']==sha(Path(__file__).with_name('audit_observed_data.py'))
    assert results['protocol_sha256']==sha(base/'PROTOCOL.json')
    items=np.unique(z['item_id']);_,inv=np.unique(z['item_id'],return_inverse=True)
    selected=0
    for seed in protocol['seeds']:
        sd=base/f'seed_{seed}';fit=read(sd/'FIT_START.json');freeze=read(sd/'POLICY_FREEZE.json')
        assert datetime.fromisoformat(protocol['created_utc'])<datetime.fromisoformat(fit['created_utc'])<datetime.fromisoformat(freeze['created_utc'])
        assert fit['strict_flag_labels_used']==fit['hidden_target_labels_used']==0
        assert freeze['em_underflow_fallback_counts']==[0,0]
        for name,digest in freeze['model_sha256'].items():assert digest==sha(sd/f'{name}.txt')
        vh=np.load(sd/'validation_heads.npz');yy=z['truth'][:,vh['days']-1]
        for key,choice in freeze['choices'].items():
            pp=candidate(vh,choice['config'],z['cat_id']);near(metrics(yy,pp)['wape'],choice['wape'])
            if 'alpha' in choice['config']:assert choice['config']['alpha']==0;selected+=1
        for block in ['calibration_a','calibration_b','shadow']:
            row=next(v for v in results['rows'] if v['seed']==seed and v['block']==block)
            stats=np.load(sd/f'{block}_item_stats.npz');assert np.array_equal(items,stats['items'])
            for name in freeze['choices']:near(stats[name].sum()/stats['mass'].sum(),row['metrics'][name]['wape'])
            if seed==protocol['seeds'][0]:
                head=np.load(sd/f'{block}_heads.npz');yy=z['truth'][:,head['days']-1]
                for key,choice in freeze['choices'].items():
                    pp=candidate(head,choice['config'],z['cat_id'])
                    for k,v in metrics(yy,pp,q).items():near(v,row['metrics'][key][k])
                    np.testing.assert_allclose(stats[key],np.bincount(inv,weights=np.abs(yy-pp).sum(axis=1)),rtol=0,atol=1e-8)
        jp=E/'joint_calibration';jf=read(jp/f'seed_{seed}/POLICY_FREEZE.json')
        assert jf['source_heads_sha256']==sha(sd/'validation_heads.npz')
        for choice in jf['all_validation']:
            assert choice['gap']<=1.1e-7 and choice['lower_bound']<=choice['wape']+1e-9
            pp=choice['beta_base']*vh['l1'].astype(float)+choice['beta_uplift']*vh['hit'].astype(float)**choice['power']*np.maximum(vh[choice['mean']].astype(float)-vh['l1'].astype(float),0)
            near(metrics(z['truth'][:,vh['days']-1],pp)['wape'],choice['wape'])
    assert selected==9
    checks.append('M5 observable: three seed chronologies/model hashes, validation choices, all item-statistic scores, full first-seed prediction replay; 9 zero-uplift grid choices')
    checks.append('Continuous calibration: all 21 empirical objectives and numerical bound gaps checked; independent optimizer LP test is separate')
    out=dict(status='PASS',checks=checks,new_fits=0,new_uci_test_policies=0,frozen_evidence_modified=False)
    (ROOT/'verification/REVISION_VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data',type=Path);v=a.parse_args()
    if v.data:run(v.data)
    else:
        with tempfile.TemporaryDirectory(prefix='censorcast_verify_') as d:
            with zipfile.ZipFile(ROOT/'inputs/M5_DEVELOPMENT_INPUTS.zip') as z:z.extractall(d)
            run(Path(d))
