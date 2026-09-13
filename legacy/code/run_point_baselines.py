"""Metric-matched point forecast baselines on the already consumed M5 design shard.

No sealed file is loaded. The price schedule is assumed known at the forecast origin,
as in the inherited protocol. Outcome features are bounded by the weekly origin.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, time, zipfile
from pathlib import Path
import numpy as np
import lightgbm as lgb
from legacy_features.features import DesignPanel, FeatureBuilder

def dump(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(data,indent=2));q.replace(p)

def atomic_npz(p, arrays):
    temp=p.with_suffix('.npz.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    if not zipfile.is_zipfile(temp):raise RuntimeError('Incomplete prediction archive')
    temp.replace(p)

def metrics(y,p):
    e=np.asarray(p,dtype=np.float64)-y
    return dict(n=int(y.size),wape=float(np.abs(e).sum()/np.abs(y).sum()),
                wpe=float(e.sum()/np.abs(y).sum()),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.square(e).mean())))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    expected='5d63b6aff7854ed1811f6b5bffd2f397a09d671c6754e0f1345e2313b35e48fd'
    assert hashlib.sha256((args.input/'design_outcomes_v0_5.npz').read_bytes()).hexdigest()==expected
    print('Loading design panel only',flush=True)
    panel=DesignPanel.load(args.input/'design_outcomes_v0_5.npz',args.input/'calendar.csv',args.input/'sell_prices.csv')
    builder=FeatureBuilder(panel)
    blocks={'selection':np.arange(1314,1434),'shadow':np.arange(1794,1914)}
    # Any fitted quantity must exist by the forecast origin, including point-policy selection.
    cutoff={'selection':1313,'shadow':1433}
    blocks={k:d[d-builder.horizon_for_day(d)>=cutoff[k]] for k,d in blocks.items()}
    all_predictions={}
    for seed in (20260906,20260907,20260908):
        rng=np.random.default_rng(seed); nd=1313-365+1
        ix=rng.choice(panel.n_series*nd,1200000,replace=False)
        s=(ix//nd).astype(np.int32);d=(365+ix%nd).astype(np.int32)
        x=builder.make_features(s,d,include_censor=False)
        labels=builder.labels(s,d)
        for method,objective in [('observed_l1','regression_l1'),('observed_poisson','poisson'),('uncensored_l1','regression_l1')]:
            name=f'{method}_s{seed}';path=out/(name+'.npz')
            if path.exists() and zipfile.is_zipfile(path) and (out/(name+'.json')).exists():
                print('Reusing',name,flush=True);continue
            params=dict(objective=objective,n_estimators=300,num_leaves=63,learning_rate=.05,
                        min_child_samples=100,reg_lambda=2.0,n_jobs=4,random_state=seed,
                        verbosity=-1,deterministic=True,force_col_wise=True)
            keep=labels['censored']==0 if method=='uncensored_l1' else np.ones(len(s),bool)
            model=lgb.LGBMRegressor(**params)
            print('Fitting',name,int(keep.sum()),flush=True);start=time.time()
            saved_model=out/(name+'.txt')
            if saved_model.exists():
                booster=lgb.Booster(model_file=str(saved_model))
            else:
                model.fit(x[keep],labels['observed'][keep])
                model.booster_.save_model(str(saved_model));booster=model.booster_
            arrays={};met={}
            for block,days in blocks.items():
                pred=np.empty((panel.n_series,len(days)),np.float32)
                for startcol in range(0,len(days),7):
                    batch=days[startcol:startcol+7]; ss=np.repeat(np.arange(panel.n_series),len(batch));dd=np.tile(batch,panel.n_series)
                    xx=builder.make_features(ss,dd,include_censor=False)
                    pred[:,startcol:startcol+len(batch)]=np.maximum(booster.predict(xx,num_threads=4),0).reshape(panel.n_series,-1)
                arrays[block]=pred;arrays[block+'_days']=days
                met[block]=metrics(panel.truth[:,days-1],pred)
                print(name,block,met[block],flush=True)
            atomic_npz(path,arrays)
            dump(out/(name+'.json'),dict(method=method,seed=seed,training_rows=int(keep.sum()),parameters=params,metrics=met,seconds=time.time()-start))
    reports=[json.loads(p.read_text()) for p in sorted(out.glob('*_s*.json'))]
    best=min(reports,key=lambda r:r['metrics']['selection']['wape'])
    result=dict(status='DESIGN_POINT_BASELINES_COMPLETE',statistical_scope='POST_HOC_DEVELOPMENT',
                reports=reports,selected_on_selection=best,seconds=time.time()-t0,
                environment={'lightgbm':lgb.__version__,'numpy':np.__version__,'python':platform.python_version()},
                fresh_guardian_opened=False,external_opened=False)
    dump(out/'POINT_RESULTS.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
