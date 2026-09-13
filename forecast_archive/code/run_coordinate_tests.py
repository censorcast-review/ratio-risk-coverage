"""Fixed retrospective coordinate and model-transfer controls; no new holdout.

Every configuration is recorded before new fits/scoring. Original artifacts
are read only. Model transfer uses period A calibration because historical
foundation predictions do not cover the primary calibration interval.
"""
from pathlib import Path
import argparse, gc, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from replay_m5 import DesignPanel, FeatureBuilder, HierarchyFeatures, sha256, write_json, load_json, scores, apply_saved_policy
from reproduce_m5_missing_controls import weighted_scale, Progress, now
from analyze_m5_floor_split import fit_fixed


def wquant(y,b,tau=.5):
    y,b=np.asarray(y,dtype=float).ravel(),np.asarray(b,dtype=float).ravel(); pos=b>0
    if not pos.any(): return 1.
    v=y[pos]/b[pos];w=b[pos];ii=np.argsort(v,kind='stable')
    return float(v[ii][min(np.searchsorted(np.cumsum(w[ii]),tau*w.sum()),len(ii)-1)])


def policy(y,b,q,cats,tau=.5,mode='log',nb=8):
    edges=np.unique(np.quantile(q,np.arange(1,nb)/nb)); j=np.searchsorted(edges,q,side='right')
    cs={};ss={};cells=[]
    for cat in np.unique(cats):
        cm=cats==cat;parent=wquant(y[cm],b[cm],tau);cs[str(cat)]=parent
        for k in range(len(edges)+1):
            m=cm[:,None]&(j==k);bb=b[m].astype(float);med=wquant(y[m],bb,tau) if np.any(bb>0) else parent
            ess=float(bb.sum()**2/(np.square(bb).sum()+1e-30)); lam=.75 if mode!='ess_arithmetic' else .75*ess/(ess+100)
            s=float(np.exp((1-lam)*np.log(max(parent,1e-8))+lam*np.log(max(med,1e-8)))) if mode=='log' else (1-lam)*parent+lam*med
            ss[f'{cat}|{k}']=s;cells.append(dict(category=str(cat),bin=k+1,n=int(m.sum()),positive=int((bb>0).sum()),effective_n=ess,median=med,scale=s,shrink=lam))
    return dict(edges=edges.tolist(),scales=ss,category_scales=cs,cells=cells,tau=tau,mode=mode)


def category(b,cats,pol): return b*np.array([pol['category_scales'][str(c)] for c in cats])[:,None]


def isotonic_fit(y,b,cats,nb=32):
    result={}
    for cat in np.unique(cats):
        cm=cats==cat;v=b[cm].ravel();yy=y[cm].ravel(); edges=np.unique(np.quantile(v,np.arange(1,nb)/nb)); jj=np.searchsorted(edges,v,side='right'); stack=[]
        for j in range(len(edges)+1):
            vals=yy[jj==j]
            if not vals.size: vals=np.array([0.])
            stack.append([j,j,vals,float(np.median(vals))])
            while len(stack)>1 and stack[-2][3]>stack[-1][3]:
                a,z=stack[-2:]; vv=np.concatenate([a[2],z[2]]);stack[-2:]=[[a[0],z[1],vv,float(np.median(vv))]]
        values=np.zeros(len(edges)+1)
        for a,z,vv,med in stack:values[a:z+1]=med
        result[str(cat)]=dict(edges=edges.tolist(),values=values.tolist())
    return result


def isotonic_apply(b,cats,p):
    f=np.zeros_like(b,dtype=float)
    for c in np.unique(cats):
        m=cats==c;t=p[str(c)];f[m]=np.asarray(t['values'])[np.searchsorted(t['edges'],b[m],side='right')]
    return f


def paired(y,a,b,items,draws=4000):
    names,inv=np.unique(items,return_inverse=True)
    mass=np.bincount(inv,weights=y.sum(axis=1));diff=np.bincount(inv,weights=(np.abs(y-a)-np.abs(y-b)).sum(axis=1))
    rng=np.random.default_rng(20260909);ww=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=draws)
    vals=(ww@diff)/(ww@mass)
    return dict(estimate=float(diff.sum()/mass.sum()),ci95=np.quantile(vals,[.025,.975]).tolist(),clusters=len(names),draws=draws,seed=20260909)


def evaluate(y,b,q,cats,policies,items):
    preds={'base':category(b,cats,policies['risk'])}
    for name,pol in policies.items():
        if name=='isotonic32': preds[name]=isotonic_apply(b,cats,pol)
        else: preds[name]=apply_saved_policy(b,b if name=='forecast' else q,cats,pol)
    result={'metrics':{n:scores(y,f) for n,f in preds.items()},'categories':{str(c):{n:scores(y[cats==c],f[cats==c]) for n,f in preds.items()} for c in np.unique(cats)}}
    result['risk_minus_forecast']=paired(y,preds['risk'],preds['forecast'],items)
    result['risk_minus_base']=paired(y,preds['risk'],preds['base'],items)
    if 'isotonic32' in preds:result['risk_minus_isotonic']=paired(y,preds['risk'],preds['isotonic32'],items)
    return result,preds


def fit_family(y,b,q,cats,extra=False):
    pp={'risk':policy(y,b,q,cats),'forecast':policy(y,b,b,cats)}
    if extra:
        pp.update(arithmetic=policy(y,b,q,cats,mode='arithmetic'),ess_arithmetic=policy(y,b,q,cats,mode='ess_arithmetic'),isotonic32=isotonic_fit(y,b,cats))
    return pp


def features_predict(builder,hier,base,hit,days,drop,need_base=True):
    n=builder.panel.n_series;bf=np.zeros((n,len(days)),np.float32);q=bf.copy()
    keep=np.array([i for i in range(33) if i not in drop])
    for s0 in range(0,len(days),7):
        ds=days[s0:s0+7];ss=np.repeat(np.arange(n),len(ds));dd=np.tile(ds,n);common=builder.make_features(ss,dd,include_censor=True)[:,keep]
        q[:,s0:s0+len(ds)]=hit.predict(common,num_threads=4).reshape(n,-1)
        if need_base:bf[:,s0:s0+len(ds)]=np.maximum(base.predict(xgb.DMatrix(np.column_stack([common,hier.make(ss,dd)])))-1,0).reshape(n,-1)
    return bf,q


def main(a):
    root=a.package.resolve();out=root/'evidence/coordinate_tests';out.mkdir(exist_ok=True)
    run=out/'analysis';run.mkdir(exist_ok=False)
    drop=[30,31,32]
    write_json(run/'PROTOCOL.json',dict(created_utc=now(),script_sha256=sha256(__file__),scope='retrospective fixed controls; no new holdout',training_calls_planned=2,seed=20260906,training_rows=1200000,removed_features=['origin_capacity','origin_fill_ratio','capacity_vs_mean_28'],retained='historical hit labels/rates and observed-sales history; capacity remains mechanically determined, not an unidentified latent-capacity experiment',calibration='original point-validation full interval; separate fixed validation-half diagnostics',coordinate_bins=8,shrink=.75,standard_control='32 pooled-within-category forecast bins, L1 median isotonic PAVA; no tuning',arithmetic_controls=['fixed .75 arithmetic','arithmetic .75*n_eff/(n_eff+100)'],asymmetric_tau=[.8,.9],foundation='all 3 saved configurations; fit fixed calibrators on period A, evaluate Later; no model fine tuning',bootstrap=dict(seed=20260909,draws=4000,unit='item across stores')))
    panel=DesignPanel.load(a.inputs/'data/design_outcomes_v0_5.npz',a.inputs/'data/calendar.csv',a.inputs/'data/sell_prices.csv');panel.censored=(panel.observed>=panel.capacity).astype(np.uint8)
    cats=panel.metadata['cat_id'];items=panel.metadata['item_id'];builder=FeatureBuilder(panel);hier=HierarchyFeatures(panel)
    vv=np.load(root/'evidence/cell_audit/m5/point_validation_predictions.npz');tt=np.load(root/'evidence/cell_audit/m5/later_predictions.npz')
    vd=vv['days'];td=tt['days'];yv=panel.truth[:,vd-1].astype(float);yt=panel.truth[:,td-1].astype(float);bv=vv['raw'];bt=tt['raw'];qv=vv['risk'];qt=tt['risk']
    fit=vd<=1373;sel=~fit;results={};orig=fit_family(yv,bv,qv,cats,True)
    write_json(run/'ORIGINAL_POLICIES.json',orig)
    results['original'],pred=evaluate(yt,bt,qt,cats,orig,items)
    assert abs(results['original']['metrics']['risk']['wape']-.6685093103012937)<1e-10
    assert abs(results['original']['metrics']['forecast']['wape']-.67045)<1e-5
    results['validation_half'],_=evaluate(yv[:,sel],bv[:,sel],qv[:,sel],cats,fit_family(yv[:,fit],bv[:,fit],qv[:,fit],cats,True),items)
    write_json(run/'INTERIM_RESULTS.json',results);print('Original coordinate and standard/smoothing controls',results['original']['metrics'],flush=True)
    # Two fixed asymmetric loss levels; these are prediction-cost proxies, not inventory simulations.
    results['asymmetric']={}
    for tau in [.8,.9]:
        pp={k:policy(yv,bv,qv if k=='risk' else bv,cats,tau=tau) for k in ['risk','forecast']}
        ff={k:apply_saved_policy(bt,qt if k=='risk' else bt,cats,p) for k,p in pp.items()};ff['base']=category(bt,cats,pp['risk'])
        results['asymmetric'][str(tau)]={k:float(np.maximum(tau*(yt-f),(tau-1)*(yt-f)).sum()/yt.sum()) for k,f in ff.items()}
        write_json(run/f'QUANTILE_{tau}_POLICIES.json',pp)
    # Capacity-input removal: same observed targets and original sampled rows.
    train=np.load(root/'evidence/cell_audit/crossfit/TRAINING_SAMPLE.npz');ss=train['series'];dd=train['days'];keep=[i for i in range(33) if i not in drop]
    common=builder.make_features(ss,dd,include_censor=True)[:,keep];hy=hier.make(ss,dd);hh=(panel.observed[ss,dd-1]>=panel.capacity[ss,dd-1]).astype(int)
    hp=load_json(root/'evidence/revision/m5_observable/PROTOCOL.json')['parameters'];bp=load_json(root/'evidence/risk_calibration/attribution_controls/PROTOCOL.json')['xgboost']
    write_json(run/'FIT_START.json',dict(created_utc=now(),protocol_sha256=sha256(run/'PROTOCOL.json'),risk_params=hp,base_params=bp,training_sample_sha256=sha256(root/'evidence/cell_audit/crossfit/TRAINING_SAMPLE.npz')))
    hit=lgb.LGBMClassifier(objective='binary',random_state=20260906,**hp).fit(common,hh).booster_;hit.save_model(str(run/'hit_no_capacity.txt'))
    print('No-capacity risk fitted; fitting L1',flush=True)
    dm=xgb.DMatrix(np.column_stack([common,hy]),label=panel.observed[ss,dd-1]+1.);del common,hy;gc.collect()
    base=xgb.train(bp,dm,num_boost_round=750,callbacks=[Progress('no_capacity_l1',750)]);base.save_model(run/'base_no_capacity.ubj');del dm;gc.collect()
    nv,nqv=features_predict(builder,hier,base,hit,vd,drop);nt,nqt=features_predict(builder,hier,base,hit,td,drop)
    np.savez_compressed(run/'NO_CAPACITY_PREDICTIONS.npz',validation_days=vd,later_days=td,validation_base=nv,validation_q=nqv,later_base=nt,later_q=nqt)
    for name,fbv,fbt in [('risk_input_only',bv,bt),('both_inputs',nv,nt)]:
        pp=fit_family(yv,fbv,nqv,cats);write_json(run/f'{name}_POLICIES.json',pp);results[name],_=evaluate(yt,fbt,nqt,cats,pp,items)
    ht=(panel.observed[:,td-1]>=panel.capacity[:,td-1]).ravel();results['risk_discrimination']={k:dict(auc=float(roc_auc_score(ht,q.ravel())),brier=float(np.mean((q.ravel()-ht)**2))) for k,q in [('full',qt),('no_direct_capacity',nqt)]}
    write_json(run/'INTERIM_RESULTS.json',results);print('Capacity controls complete',results['risk_discrimination'],flush=True)
    # Foundation predictions: hash against the original execution receipt.
    saved=out/'foundation_predictions';receipts=load_json(root/'evidence/m5/bundle/PREDICTION_RECEIPTS.json');rh={(r['model'],r['origin']):r['sha256'] for r in receipts}
    ad=np.load(a.inputs/'cache/calibration_a_aligned.npz')['target_days'];ya=panel.truth[:,ad-1].astype(float)
    full_hit=lgb.Booster(model_file=str(root/'evidence/revision/m5_observable/seed_20260906/hit.txt'))
    _,qa=features_predict(builder,hier,None,full_hit,ad,[],False)
    results['foundation']={};verified=0
    for model in ['chronos_bolt_small','chronos2_univariate','chronos2_covariates']:
        byday={}
        for p in (saved/model).glob('*.npz'):
            origin=int(p.stem.split('_')[-1]);assert sha256(p)==rh[(model,origin)];verified+=1;z=np.load(p)
            for j,d in enumerate(z['days']):byday[int(d)]=z['forecast'][:,j]
        fa=np.column_stack([byday[int(d)] for d in ad]);ft=np.column_stack([byday[int(d)] for d in td]);pp=fit_family(ya,fa,qa,cats)
        write_json(run/f'{model}_POLICIES.json',pp);res,_=evaluate(yt,ft,qt,cats,pp,items);res['raw']=scores(yt,ft);results['foundation'][model]=res
        print('Foundation',model,res['metrics'],flush=True)
    results['foundation_prediction_files_verified']=verified
    # All-recorded-sales uncertainty uses identical saved predictions and no conditioning on stockout.
    df=pd.read_csv(root/'evidence/freshretail/frozen_eval/EVAL_PREDICTIONS.csv.gz');df['store']=df.series_id.str.split('::').str[0];df['product']=df.series_id.str.split('::').str[1]
    df['mass']=df.sale_amount;df['diff']=abs(df.risk_conditioned-df.sale_amount)-abs(df.base-df.sale_amount);results['fresh_all_clusters']={}
    for k in ['series_id','store','product','dt']:
        g=df.groupby(k,sort=True)[['mass','diff']].sum();n=len(g);ww=np.random.default_rng(20260909).multinomial(n,np.full(n,1/n),size=4000);vals=(ww@g['diff'])/(ww@g.mass)
        results['fresh_all_clusters'][k]=dict(clusters=n,estimate=float(g['diff'].sum()/g.mass.sum()),ci95=np.quantile(vals,[.025,.975]).tolist(),seed=20260909,draws=4000)
    results['finished_utc']=now();results['new_holdout_accesses']=0;results['new_model_fits']=2
    write_json(run/'RESULTS.json',results);write_json(run/'OUTPUT_HASHES.json',{str(p.relative_to(run)):sha256(p) for p in run.iterdir() if p.is_file()})
    print('COMPLETE',json.dumps({k:results[k] for k in ['asymmetric','fresh_all_clusters']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--inputs',type=Path,required=True);main(p.parse_args())
