"""A separate, post-hoc development experiment with observable-label risk heads.

The fixed point forecaster is the existing observed-sales L1 model, WITHOUT
the truth-tuned adapter. Primary threshold selection uses model moments only.
Truth-label fits and truth-calibrated thresholds are separately labeled controls.
"""
from pathlib import Path
import argparse,json,time,hashlib,io,gc
import numpy as np
import lightgbm as lgb
from legacy_features.features import DesignPanel,FeatureBuilder
from r2_io import dump,write,sha
from r3_observable import fit_observable,predict_mean,nll,expected_abs,curve,select,stats

ROOT=Path(__file__).resolve().parents[1]
CAP=.85*.7530939208313988
SEEDS=[20260906,20260907,20260908]
BLOCKS={'calibration_a':(1554,1673),'calibration_b':(1674,1793),'shadow':(1794,1913)}
INPUT_HASHES={'design_outcomes_v0_5.npz':'5d63b6aff7854ed1811f6b5bffd2f397a09d671c6754e0f1345e2313b35e48fd',
 'calendar.csv':'d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5',
 'sell_prices.csv':'9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef'}
METHODS=[('observed_poisson',None),('complete_case_poisson',None),('censored_poisson',None),
 ('censored_nb_0p5',.5),('censored_nb_2',2.),('censored_nb_10',10.)]
def save_npz(p,**x):
    b=io.BytesIO();np.savez_compressed(b,**x);write(p,b.getvalue())
def origin(d):return d-(1+(d-1)%7)
def getdays(lo,hi,cut):
    d=np.arange(lo,hi+1);return d[origin(d)>=cut]
def feature_and_forecast(builder,point,s,d):
    x=builder.make_features(s,d,include_censor=True)
    f=np.maximum(point.predict(x[:,:28],num_threads=4),0).astype(np.float32)
    return np.column_stack([x,np.log1p(f)]).astype(np.float32),f
def prepare(data,out):
    cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
    if (cache/'READY.json').exists():return
    for name,h in INPUT_HASHES.items():assert sha(data/name)==h,name
    panel=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv')
    b=FeatureBuilder(panel);point_path=ROOT/'results/point_baselines/observed_l1_s20260906.txt';point=lgb.Booster(model_file=str(point_path))
    train_days=getdays(1434,1532,1433);valid_days=getdays(1533,1553,1532)
    protocol=dict(status='FROZEN_BEFORE_NEW_FITS',statistical_scope='POST_HOC_CONSUMED_DESIGN_ONLY',
      cap=CAP,min_row_coverage=.35,seeds=SEEDS,methods=METHODS,rows_per_seed=600000,trees=220,
      risk_fit_days=train_days.tolist(),dispersion_selection_days=valid_days.tolist(),
      point_model_sha256=sha(point_path),point_forecaster='fixed observed-sales L1; no adapter',
      primary_policy='model-moment threshold on calibration A/B; no hidden outcomes',
      negative_binomial_selection='minimum observed-data censored likelihood on validation days',
      diagnostic_controls=['truth Poisson fit','truth-calibrated thresholds'],
      hidden_demand_identification='requires specified count family and ignorable censoring conditional on X',
      source_hashes={n:sha(Path(__file__).parent/n) for n in ['run_observable_extension.py','r3_observable.py']},
      input_hashes=INPUT_HASHES,fresh_guardian_opened=False,external_opened=False,certificate_issued=False)
    dump(out/'PROTOCOL.json',protocol)
    save_npz(cache/'metadata.npz',**panel.metadata)
    for seed in SEEDS:
        idx=np.random.default_rng(seed).choice(panel.n_series*len(train_days),600000,replace=False)
        s=idx//len(train_days);d=train_days[idx%len(train_days)]
        x,f=feature_and_forecast(b,point,s,d)
        labels={k:getattr(panel,k)[s,d-1] for k in ['observed','censored','truth']}
        save_npz(cache/f'train_s{seed}.npz',x=x,f=f,series=s,days=d,**labels)
        print('Prepared training cache',seed,flush=True)
    # Validation uses all rows in its purged days; the three audit blocks use
    # the same origin purges as the existing paper.
    for key,days in [('validation',valid_days)]+[(k,getdays(lo,hi,lo-1)) for k,(lo,hi) in BLOCKS.items()]:
        xx=np.empty((panel.n_series,len(days),34),np.float32);ff=np.empty(xx.shape[:2],np.float32)
        for st in range(0,len(days),7):
            dd=days[st:st+7];s=np.repeat(np.arange(panel.n_series),len(dd));d=np.tile(dd,panel.n_series)
            x,f=feature_and_forecast(b,point,s,d);xx[:,st:st+len(dd)]=x.reshape(panel.n_series,len(dd),34);ff[:,st:st+len(dd)]=f.reshape(panel.n_series,len(dd))
        # Byte-buffered writes avoid partial workspace array files.
        stream=io.BytesIO();np.save(stream,xx);write(cache/(key+'_x.npy'),stream.getvalue());del xx,stream
        save_npz(cache/(key+'.npz'),f=ff,days=days,**{k:getattr(panel,k)[:,days-1] for k in ['observed','censored','truth']})
        if key in BLOCKS:
            old=ROOT/f'results/objectives/cache_design/{key}_aligned.npz'
            with np.load(old) as z:
                assert np.array_equal(days,z['target_days'])
                assert np.allclose(ff,z['baseline'],rtol=0,atol=1e-6)
        print('Prepared block',key,len(days),'baseline replay matched',flush=True)
    # Outcome values are irrelevant to feature generation; explicitly perturb
    # future observable values and confirm origin-limited features are unchanged.
    s=np.arange(16);d=np.full(16,1803);before=b.make_features(s,d,include_censor=True)
    oo=origin(d)[0];saved=panel.observed[:16,oo:].copy();panel.observed[:16,oo:]+=10000
    after=FeatureBuilder(panel).make_features(s,d,include_censor=True)
    assert np.array_equal(before,after);panel.observed[:16,oo:]=saved
    dump(cache/'READY.json',dict(status='OBSERVABLE_FEATURE_CACHE_READY',future_outcome_invariance=True,
        point_replay_matched=True,point_model_sha256=sha(point_path),features=34))

def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def eval_groups(v,s,t,meta):
    out={}
    for g in ['ALL']+sorted(set(meta['cat_id'])):
        rows=np.ones(len(meta['id']),bool) if g=='ALL' else meta['cat_id']==g
        a=np.zeros(s[rows].shape,bool) if t is None else s[rows]<=t
        out[g]=stats(v['truth'][rows],v['f'][rows],a)
    return out
def run(data,out):
    prepare(data,out);cache=out/'cache';models=out/'models';models.mkdir(exist_ok=True)
    meta=load_npz(cache/'metadata.npz');vv=load_npz(cache/'validation.npz');xv=np.load(cache/'validation_x.npy',mmap_mode='r').reshape(-1,34)
    dd={k:load_npz(cache/(k+'.npz')) for k in BLOCKS};records=[];selection=[]
    for seed in SEEDS:
        tr=load_npz(cache/f'train_s{seed}.npz');infos={};val=[]
        for method,shape in METHODS+[('truth_poisson_reference',None)]:
            p=models/f'{method}_s{seed}.txt';receipt=p.with_suffix('.json')
            if receipt.exists():
                info=json.loads(receipt.read_text());m=lgb.Booster(model_file=str(p))
            else:
                print('Fitting',method,seed,flush=True);start=time.time()
                if method=='truth_poisson_reference':
                    m,info=fit_observable(tr['x'],tr['truth'],np.zeros(len(tr['truth']),bool),'observed_poisson',seed)
                    info.update(method=method,hidden_labels_used_for_fit=len(tr['truth']))
                else:m,info=fit_observable(tr['x'],tr['observed'],tr['censored'],method,seed,shape)
                write(p,m.model_to_string().encode());info['fit_seconds']=time.time()-start
                info['model_sha256']=sha(p);dump(receipt,info)
            mv=predict_mean(m,xv,info)
            vloss=float(nll(mv,vv['observed'].ravel(),vv['censored'].ravel(),shape).mean())
            val.append(dict(method=method,shape=shape,observed_validation_nll=vloss));infos[method]=info
            scores={k:{} for k in ['excess','demand_normalized']};implied={}
            for k in BLOCKS:
                dest=out/f'{method}_s{seed}_{k}.npz'
                if dest.exists():v=load_npz(dest);mu=v['mu'];ee=v['expected_error']
                else:
                    xx=np.load(cache/(k+'_x.npy'),mmap_mode='r');mu=predict_mean(m,xx.reshape(-1,34),info).reshape(xx.shape[:2]);ee=expected_abs(mu,dd[k]['f'],shape)
                    save_npz(dest,mu=mu.astype(np.float64),expected_error=ee.astype(np.float64))
                scores['excess'][k]=ee-CAP*mu;scores['demand_normalized'][k]=ee/np.maximum(mu,.25)
                implied[k]=(ee,mu)
            for name,ss in scores.items():
                for calibration in ['model_implied_observable','truth_calibrated_diagnostic']:
                    cc=[]
                    for k in ['calibration_a','calibration_b']:
                        e,mu=implied[k] if calibration=='model_implied_observable' else (np.abs(dd[k]['truth']-dd[k]['f']),dd[k]['truth'])
                        cc.append(curve(ss[k],e,mu))
                    t=select(cc,CAP,.35)
                    record=dict(seed=seed,method=method,shape=shape,score=name,calibration=calibration,threshold=t,
                      hidden_labels_used_for_fit=info['hidden_labels_used_for_fit'],
                      hidden_labels_used_for_threshold=(0 if calibration=='model_implied_observable' else sum(dd[k]['truth'].size for k in ['calibration_a','calibration_b'])),
                      metrics={k:eval_groups(dd[k],ss[k],t,meta) for k in BLOCKS})
                    records.append(record)
            dump(out/f'records_{method}_s{seed}.json',records[-4:]);print('Completed',method,seed,'observed NLL',round(vloss,5),flush=True)
        chosen=min([v for v in val if v['method'].startswith('censored_nb_')],key=lambda v:v['observed_validation_nll'])
        selection.append(dict(seed=seed,selected_nb=chosen,all_validation=val))
        del tr;gc.collect()
    result=dict(status='OBSERVABLE_LABEL_DEVELOPMENT_COMPLETE',protocol_sha256=sha(out/'PROTOCOL.json'),records=records,
                validation_selection=selection,seeds=SEEDS,scope='POST_HOC_CONSUMED_DESIGN_ONLY',
                fresh_guardian_opened=False,external_opened=False,certificate_issued=False,
                model_identification_assumption='specified Poisson/NB family and conditionally ignorable censoring',
                point_forecaster='frozen observed-sales L1; no truth-tuned adapter')
    dump(out/'OBSERVABLE_RESULTS.json',result);print(result['status'],len(records),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(a.data,a.output)
