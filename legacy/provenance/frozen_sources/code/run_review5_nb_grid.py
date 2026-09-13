"""Frozen post-hoc NB-shape diagnostic on consumed DESIGN data only.

Uses a fixed observed-sales forecast without the truth-tuned adapter. Four new
NB fits augment the retained kappa=2 model. A new supervised direct-excess head
has the identical forecast, features, train rows, tree budget, seed and splits.
Per-block count matching is a ranking diagnostic, not a deployable policy:
the target count comes from that block's NB acceptance, never its outcomes.
"""
from pathlib import Path
import argparse, gc, io, json, time, platform
from importlib.metadata import version
import numpy as np
import lightgbm as lgb
from legacy_features.features import DesignPanel, FeatureBuilder
from run_observable_extension import CAP, BLOCKS, INPUT_HASHES, getdays, load_npz, save_npz, feature_and_forecast
from run_capacity_hit_extension import likelihood_inputs, replace_history
from r3_observable import fit_observable, predict_mean, nll, expected_abs, curve, select
from r2_io import dump, write, sha
ROOT=Path(__file__).resolve().parents[1]
SEED=20260906
SHAPES=[1.,1.5,2.,3.,5.]
ANCHORS=[.5,10.]
OLD=ROOT/'results/capacity_hit_extension/models'

def method_name(shape):return 'capacity_hit_nb_'+str(shape).rstrip('0').rstrip('.').replace('.','p')
def arrsave(p,a):
    b=io.BytesIO();np.save(b,a);write(p,b.getvalue())
def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    point=ROOT/'results/point_baselines/observed_l1_s20260906.txt'
    spec={
      'schema':'review5-nb-grid-1','status':'FROZEN_BEFORE_NEW_FITS',
      'scope':'POST_HOC_CONSUMED_DESIGN_ONLY','seed':SEED,'shapes':SHAPES,
      'descriptive_anchors':ANCHORS,'trees':220,'training_rows':600000,
      'cap':CAP,'minimum_row_coverage':.35,
      'forecast':'frozen observed-sales L1; no adapter; identical across all comparisons',
      'point_model_sha256':sha(point),'input_hashes':INPUT_HASHES,
      'features':'34 original origin-safe features; columns28/29 use past capacity hits S>=C, not strict latent exceedance',
      'likelihood':'pmf(S) when S<C; Pr(Y>=C)=sf(C-1) when S>=C',
      'nb_selection':'minimum observed validation NLL over exactly [1,1.5,2,3,5]; no outcome labels',
      'threshold':'largest score threshold meeting both A/B WAPE<=cap and rows>=.35; NB uses implied moments, supervised control uses development truth',
      'supervised_control':'new 220-tree squared-loss direct excess head targeting abs(Y-f)-rY on identical training rows/features/forecast',
      'coverage_match':'for every shape/block, truncate supervised ranking to NB accepted count. Uniform boundary-tie selection with fixed seed, exact count. Also report analytic fractional tie expectation. Diagnostic uses block NB count, not outcomes; not a transferred deployable threshold.',
      'train_days':getdays(1434,1532,1433).tolist(),
      'validation_days':getdays(1533,1553,1532).tolist(),
      'blocks':{k:getdays(lo,hi,lo-1).tolist() for k,(lo,hi) in BLOCKS.items()},
      'source_hashes':{p:sha(Path(__file__).parent/p) for p in ['run_review5_nb_grid.py','r3_observable.py','run_capacity_hit_extension.py','run_observable_extension.py','legacy_features/features.py']},
      'retained_models':{str(k):{'path':str((OLD/(method_name(k)+f'_s{SEED}.txt')).relative_to(ROOT)), 'sha256':sha(OLD/(method_name(k)+f'_s{SEED}.txt'))} for k in [2.]+ANCHORS},
      'runtime':{p:version(p) for p in ['numpy','scipy','pandas','lightgbm','scikit-learn']},
      'fresh_guardian_opened':False,'external_opened':False,'certificate_issued':False}
    p=out/'PROTOCOL.json'
    if p.exists():assert json.loads(p.read_text())==spec,'Frozen protocol changed'
    else:dump(p,spec)
    return spec

def prepare(data,out,protocol):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    if (cache/'READY.json').exists():return cache
    for name,digest in INPUT_HASHES.items():assert sha(data/name)==digest,name
    panel=DesignPanel.load(data/'design_outcomes_v0_5.npz',data/'calendar.csv',data/'sell_prices.csv')
    b=FeatureBuilder(panel)
    point=lgb.Booster(model_file=str(ROOT/'results/point_baselines/observed_l1_s20260906.txt'))
    hits=panel.observed>=panel.capacity
    cs=np.pad(np.cumsum(hits,axis=1,dtype=np.int32),((0,0),(1,0)))
    days=np.asarray(protocol['train_days']);idx=np.random.default_rng(SEED).choice(panel.n_series*len(days),600000,replace=False)
    s=idx//len(days);d=days[idx%len(days)]
    x,f=feature_and_forecast(b,point,s,d);x=replace_history(x,s,d,cs)
    save_npz(cache/'train.npz',x=x,f=f,series=s,days=d,observed=panel.observed[s,d-1],capacity=panel.capacity[s,d-1],truth=panel.truth[s,d-1])
    save_npz(cache/'metadata.npz',**panel.metadata)
    del x,f
    for key,days in [('validation',protocol['validation_days'])]+list(protocol['blocks'].items()):
        days=np.asarray(days);xp=cache/(key+'_x.npy');yp=cache/(key+'.npz')
        if xp.exists() and yp.exists():continue
        xx=np.empty((panel.n_series,len(days),34),np.float32);ff=np.empty(xx.shape[:2],np.float32)
        for st in range(0,len(days),7):
            dd=days[st:st+7];s=np.repeat(np.arange(panel.n_series),len(dd));d=np.tile(dd,panel.n_series)
            x,f=feature_and_forecast(b,point,s,d);x=replace_history(x,s,d,cs)
            xx[:,st:st+len(dd)]=x.reshape(panel.n_series,len(dd),34);ff[:,st:st+len(dd)]=f.reshape(panel.n_series,len(dd))
        arrsave(xp,xx)
        save_npz(yp,f=ff,days=days,observed=panel.observed[:,days-1],capacity=panel.capacity[:,days-1],truth=panel.truth[:,days-1])
        if key in BLOCKS:
            old=load_npz(ROOT/f'results/review2/cache_design/{key}_aligned.npz')
            assert np.array_equal(days,old['target_days'])
            assert np.allclose(ff,old['baseline'],rtol=0,atol=1e-6),'point replay mismatch'
        del xx,ff;gc.collect();print('Prepared',key,flush=True)
    # Perturb all future observations: origin-restricted features stay fixed.
    from run_observable_extension import origin
    s=np.arange(16);d=np.full(16,1803);xx=b.make_features(s,d,include_censor=True)
    oo=origin(d)[0];keep=panel.observed[:16,oo:].copy();panel.observed[:16,oo:]+=10000
    assert np.array_equal(xx,FeatureBuilder(panel).make_features(s,d,include_censor=True))
    panel.observed[:16,oo:]=keep
    dump(cache/'READY.json',{'status':'READY','point_replay':True,'future_feature_invariance':True,'protocol_sha256':sha(out/'PROTOCOL.json'), 'files':{p.name:sha(p) for p in cache.iterdir() if p.is_file()}})
    return cache

def weighted_stats(y,f,w,implied=None):
    y=np.asarray(y,float);f=np.asarray(f,float);w=np.asarray(w,float)
    err=np.abs(y-f);mass=float(np.sum(w*y));error=float(np.sum(w*err));n=y.size;accepted=float(w.sum())
    v={'n':int(n),'accepted_n':accepted,'row_coverage':accepted/n,'demand_mass':mass,'error_mass':error,'total_demand':float(y.sum()),
       'demand_coverage':mass/float(y.sum()) if y.sum()>0 else None,'wape':error/mass if mass>0 else None}
    if implied is not None:
        ee,mu=implied;E=float(np.sum(w*ee));M=float(np.sum(w*mu))
        v.update(implied_error_mass=E,implied_demand_mass=M,implied_wape=E/M if M>0 else None,
          error_mass_relative_bias=E/error-1 if error>0 else None,demand_mass_relative_bias=M/mass-1 if mass>0 else None)
    return v

def groups(v,w,meta,implied=None):
    out={}
    for name in ['ALL']+sorted(set(meta['cat_id'])):
        ix=np.ones(len(meta['cat_id']),bool) if name=='ALL' else meta['cat_id']==name
        mm=None if implied is None else (implied[0][ix],implied[1][ix])
        out[name]=weighted_stats(v['truth'][ix],v['f'][ix],w[ix],mm)
    return out

def match_count(score,target_count,tie_seed):
    s=np.asarray(score,float).ravel();n=s.size;k=int(target_count)
    assert 0<=k<=n and np.isfinite(s).all()
    if k==0:return np.zeros(score.shape,bool),np.zeros(score.shape,float),{'target_count':0,'threshold':None,'boundary_probability':0.,'boundary_count':0}
    if k==n:return np.ones(score.shape,bool),np.ones(score.shape,float),{'target_count':n,'threshold':None,'boundary_probability':1.,'boundary_count':0}
    t=float(np.partition(s,k-1)[k-1]);below=s<t;ties=np.flatnonzero(s==t);remaining=k-int(below.sum());p=remaining/len(ties)
    a=below.copy();chosen=np.random.default_rng(tie_seed).choice(ties,remaining,replace=False);a[chosen]=True
    w=below.astype(float);w[ties]=p
    assert a.sum()==k and abs(w.sum()-k)<1e-8
    return a.reshape(score.shape),w.reshape(score.shape),{'target_count':k,'threshold':t,'boundary_probability':p,'boundary_count':int(len(ties)),'tie_seed':tie_seed}

def fit_supervised(tr,out):
    p=out/'models/supervised_same_forecast_excess.txt';receipt=p.with_suffix('.json');p.parent.mkdir(exist_ok=True)
    if receipt.exists():
        info=json.loads(receipt.read_text());assert sha(p)==info['model_sha256'];return lgb.Booster(model_file=str(p)),info
    target=np.abs(tr['truth']-tr['f'])-CAP*tr['truth']
    params=dict(objective='regression',learning_rate=.05,num_leaves=31,min_data_in_leaf=100,lambda_l2=3,num_threads=4,
      seed=SEED,deterministic=True,force_col_wise=True,verbosity=-1,feature_pre_filter=False)
    start=time.time();m=lgb.train(params,lgb.Dataset(tr['x'],label=target),num_boost_round=220)
    write(p,m.model_to_string().encode())
    info={'method':'supervised_same_forecast_excess','seed':SEED,'trees':220,'training_rows':len(target),'hidden_labels_used_for_fit':len(target),
      'target':'abs(Y-f)-rY','model_sha256':sha(p),'fit_seconds':time.time()-start,'protocol_sha256':sha(out/'PROTOCOL.json')}
    dump(receipt,info);return m,info

def run(data,out):
    protocol=freeze(out);cache=prepare(data,out,protocol)
    tr=load_npz(cache/'train.npz');vv=load_npz(cache/'validation.npz');meta=load_npz(cache/'metadata.npz')
    xv=np.load(cache/'validation_x.npy',mmap_mode='r').reshape(-1,34)
    dd={k:load_npz(cache/(k+'.npz')) for k in BLOCKS}
    oo,cc=likelihood_inputs(tr['observed'],tr['capacity']);vo,vc=likelihood_inputs(vv['observed'],vv['capacity'])
    ctrl,cinfo=fit_supervised(tr,out);control_scores={}
    for k in BLOCKS:
        xx=np.load(cache/(k+'_x.npy'),mmap_mode='r');control_scores[k]=ctrl.predict(xx.reshape(-1,34),num_threads=4).reshape(xx.shape[:2]);del xx
    ct=select([curve(control_scores[k],np.abs(dd[k]['truth']-dd[k]['f']),dd[k]['truth']) for k in ['calibration_a','calibration_b']],CAP,.35)
    control={'info':cinfo,'threshold':ct,'calibration':'truth_A_B','metrics':{k:groups(dd[k],np.zeros(control_scores[k].shape,bool) if ct is None else control_scores[k]<=ct,meta) for k in BLOCKS}}
    records=[];predhashes={};models=out/'models';models.mkdir(exist_ok=True)
    for shape in SHAPES+ANCHORS:
        method=method_name(shape);p=models/f'{method}_s{SEED}.txt';receipt=p.with_suffix('.json')
        if shape in [2.]+ANCHORS:
            p=OLD/p.name;receipt=p.with_suffix('.json');info=json.loads(receipt.read_text());assert sha(p)==info['model_sha256'];m=lgb.Booster(model_file=str(p));reused=True
        elif receipt.exists():
            info=json.loads(receipt.read_text());assert sha(p)==info['model_sha256'];m=lgb.Booster(model_file=str(p));reused=False
        else:
            print('Fitting NB',shape,flush=True);start=time.time();m,info=fit_observable(tr['x'],oo,cc,method,SEED,shape)
            write(p,m.model_to_string().encode());info.update(model_sha256=sha(p),fit_seconds=time.time()-start,protocol_sha256=sha(out/'PROTOCOL.json'));dump(receipt,info);reused=False
        vmu=predict_mean(m,xv,info);loss=float(nll(vmu,vo.ravel(),vc.ravel(),shape).mean());implied={};scores={}
        for k in BLOCKS:
            xx=np.load(cache/(k+'_x.npy'),mmap_mode='r');mu=predict_mean(m,xx.reshape(-1,34),info).reshape(xx.shape[:2]);ee=expected_abs(mu,dd[k]['f'],shape)
            scores[k]=ee-CAP*mu;implied[k]=(ee,mu);del xx
        t=select([curve(scores[k],*implied[k]) for k in ['calibration_a','calibration_b']],CAP,.35)
        rec={'shape':shape,'method':method,'reused_model':reused,'model_sha256':sha(p),'observed_validation_nll':loss,'threshold':t,'calibration':'model_implied_A_B',
          'hidden_labels_used_for_fit':0,'hidden_labels_used_for_threshold':0,'metrics':{},'matched_supervised':{}}
        for j,k in enumerate(BLOCKS):
            a=np.zeros(scores[k].shape,bool) if t is None else scores[k]<=t
            rec['metrics'][k]=groups(dd[k],a,meta,implied[k])
            ca,cw,match=match_count(control_scores[k],a.sum(),SEED+100+j)
            rec['matched_supervised'][k]={'matching':match,'exact_count_metrics':groups(dd[k],ca,meta),'fractional_boundary_metrics':groups(dd[k],cw,meta)}
            pp=cache/f'predictions_k{shape}_{k}.npz';save_npz(pp,score=scores[k],implied_error=implied[k][0],implied_demand=implied[k][1],supervised_score=control_scores[k]);predhashes[pp.name]=sha(pp)
            # kappa2 exact retained aggregate replay establishes feature/preprocessing identity.
            if shape==2.:
                old=json.loads((ROOT/f'results/capacity_hit_extension/records_capacity_hit_nb_2_s{SEED}.json').read_text())
                old=next(z for z in old if z['score']=='excess' and z['calibration']=='model_implied_observable')
                assert t==old['threshold'],(t,old['threshold'])
                for metric in ['row_coverage','demand_coverage','wape']:
                    for g in rec['metrics'][k]:assert abs(rec['metrics'][k][g][metric]-old['metrics'][k][g][metric])<1e-10,(k,g,metric)
        records.append(rec);dump(out/f'record_k{shape}.json',rec);print('Finished NB',shape,'validation',loss,'threshold',t,flush=True)
        del scores,implied,m;gc.collect()
    selected=min([r for r in records if r['shape'] in SHAPES],key=lambda r:r['observed_validation_nll'])
    result={'status':'COMPLETE','scope':protocol['scope'],'protocol_sha256':sha(out/'PROTOCOL.json'),'selected_shape_by_observed_validation':selected['shape'],
      'selection_candidates':SHAPES,'anchor_shapes_not_selected':ANCHORS,'records':records,'supervised_control':control,
      'cache_hashes':json.loads((cache/'READY.json').read_text())['files'],'prediction_hashes':predhashes,
      'identical_forecast_verified':True,'kappa2_retained_result_replay_verified':True,
      'fresh_guardian_opened':False,'external_opened':False,'certificate_issued':False,
      'matched_comparison_scope':'outcome-free per-block ranking diagnostic at NB realized row count; no test-time label selection, but target counts are not fixed on calibration'}
    dump(out/'RESULTS.json',result);print('COMPLETE selected shape',selected['shape'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--out',type=Path,default=ROOT/'results/review5_nb_grid');p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.freeze_only:freeze(a.out);print('FROZEN',sha(a.out/'PROTOCOL.json'))
    else:
        if a.data is None:p.error('--data required')
        run(a.data,a.out)
