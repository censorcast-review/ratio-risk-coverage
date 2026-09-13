"""Frozen-model prediction after a separately authorized guardian opening.

Run under the recorded legacy sklearn environment. No estimator is fitted.
Context truth is unavailable and is never used: its observed-sales placeholder
exists only to satisfy the inherited panel loader's array-shape validation.
"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np,joblib,lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/design'))
from censorcast_v05.features import DesignPanel,FeatureBuilder
META=['id','item_id','dept_id','cat_id','store_id','state_id']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,**data):
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(p)

def main(a):
    freeze=json.loads(a.freeze.read_text());assert freeze['status']=='FROZEN_FOR_SEPARATE_COMPARATIVE_GUARDIAN_AUDIT'
    receipt=json.loads((a.output/'GUARDIAN_OPENED.json').read_text());assert receipt['freeze_sha256']==sha(a.freeze)
    assert sha(a.context)=='5ed3f06f69c570f2c6aff33a825b944f13b49144bedaf76e6f2f9666664ca62f'
    assert sha(a.outcomes)=='6bf62d608054063e7142cf47b15b75bec66765e1f04cc006138dc29037f0c827'
    with np.load(a.context,allow_pickle=False) as z:c={k:z[k] for k in z.files}
    with np.load(a.outcomes,allow_pickle=False) as z:h={k:z[k] for k in z.files}
    assert int(c['day_start'][0])==1 and int(c['day_end'][0])==1553
    assert int(h['day_start'][0])==1554 and int(h['day_end'][0])==1913
    for k in META:assert np.array_equal(c[k],h[k])
    assert len(c['id'])==6100 and len(np.unique(c['item_id']))==610
    with np.load(a.design/'design_outcomes_v0_5.npz',allow_pickle=False) as z:
        assert not (set(c['item_id'])&set(z['item_id']))
        # Codes are fitted by sorted labels. Require exact shared category vocabulary.
        for k in ['cat_id','dept_id','store_id','state_id']:assert np.array_equal(np.unique(c[k]),np.unique(z[k]))
    raw={k:c[k] for k in META}
    for k in ['observed','capacity','censored']:raw[k]=np.concatenate([c[k],h[k]],axis=1)
    raw['truth']=np.concatenate([c['observed'],h['truth']],axis=1)
    raw['day_start']=np.array([1]);raw['day_end']=np.array([1913])
    panelpath=a.output/'guardian_panel_for_prediction.npz';atomic(panelpath,**raw)
    panel=DesignPanel.load(panelpath,a.design/'calendar.csv',a.design/'sell_prices.csv');fb=FeatureBuilder(panel)
    # The exact inherited fit, and first prespecified L1 fit. No refitting on guardian items.
    bundle=joblib.load(ROOT/'inputs/forecast_bundle_v0_5.joblib')
    point=lgb.Booster(model_file=str(ROOT/'results/point_baselines/observed_l1_s20260906.txt'))
    for name,start,end in [('selection_base',1314,1433),('risk_train',1434,1553),('calibration_a',1554,1673),('calibration_b',1674,1793),('shadow',1794,1913)]:
        dest=a.output/(name+'_v0_5.npz')
        if dest.exists():continue
        days=np.arange(start,end+1);shape=(6100,len(days));v={k:np.empty(shape,np.float32) for k in ['baseline','proposal','raw_poisson_histgb','censored_poisson_em','censor_probability']}
        for st in range(0,len(days),7):
            ddays=days[st:st+7];ss=np.repeat(np.arange(6100),len(ddays));dd=np.tile(ddays,6100)
            x=fb.make_features(ss,dd,include_censor=False);cx=fb.make_features(ss,dd,include_censor=True)
            pp=np.maximum(point.predict(x,num_threads=4),0).reshape(6100,-1).astype(np.float32)
            legacy=bundle.predict(x,cx)
            for k,arr in legacy.items():v[k][:,st:st+len(ddays)]=arr.reshape(6100,-1)
            v['baseline'][:,st:st+len(ddays)]=pp
        v['proposal']=v['baseline']+2*v['censor_probability']**2*np.maximum(v['censored_poisson_em']-v['baseline'],0)
        v.update(observed=raw['observed'][:,days-1],censored=raw['censored'][:,days-1],truth=raw['truth'][:,days-1],target_days=days,
                 prior_observed_wape=np.zeros(shape,np.float32),instantaneous_absolute_error=np.zeros(shape,np.float32))
        atomic(dest,**v);print('Guardian fixed prediction block',name,flush=True)
    # Raw outcome panel name is explicit in the evaluation package; no design file is overwritten.
    destination=a.output/'design_outcomes_v0_5.npz'
    if not destination.exists():destination.symlink_to(panelpath.resolve())
    (a.output/'PREDICTION_RECEIPT.json').write_text(json.dumps({'status':'FIXED_GUARDIAN_PREDICTIONS_READY','training_calls':0,'context_truth_used_for_training_or_scoring':False,'prediction_days':[1314,1913],'evaluation_days':[1554,1913],'freeze_sha256':sha(a.freeze)},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['freeze','context','outcomes','design','output']:ap.add_argument('--'+k,type=Path,required=True)
    main(ap.parse_args())
