"""Prospectively specified temporal test on a second public sales dataset.

This is a new observable-capacity correction implementation, not a replay of
the frozen M5 engine. No claim about naturally unobserved demand is supported.
"""
import argparse, hashlib, io, json, platform, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson

SEED = 20260907
PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=.05,
              min_child_samples=100, reg_lambda=10., verbosity=-1,
              n_jobs=4, random_state=SEED, deterministic=True, force_col_wise=True)
SCALES = [.5, .75, 1., 1.25, 1.5, 2.]
ALPHAS = [0., .25, .5, 1., 2.]
POWERS = [1., 2.]

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(path, data):
    # Receipts are append-only: never silently replace a protocol or result.
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
def npz(path, **data):
    with Path(path).open('xb') as f: np.savez_compressed(f, **data)
def score(y, f):
    return float(np.abs(y-f).sum(dtype=np.float64)/y.sum(dtype=np.float64))

def censor(y):
    observed = np.zeros_like(y, dtype=np.float32)
    capacity = np.ones_like(y, dtype=np.float32)
    level = np.zeros(y.shape[0], dtype=np.float64)
    for t in range(y.shape[1]):
        if t: level = .9*level + .1*observed[:, t-1]
        capacity[:, t] = np.maximum(1, np.ceil(1.15*level+.25*np.sqrt(level+1)))
        if t < 56:
            observed[:,t] = y[:,t]
            # Warmup is marked separately; never learn a fabricated hit.
        else: observed[:,t] = np.minimum(y[:,t], capacity[:,t])
    return observed, capacity

def features(obs, cap, dates, days):
    n = len(obs); d = np.asarray(days)
    origin = (d//7)*7-1
    if np.any(origin < 55): raise ValueError('Insufficient context')
    # Every feature below is a function of observations at/before origin.
    cols = [np.broadcast_to(np.arange(n)[:,None], (n,len(d))),
            np.broadcast_to(d-origin, (n,len(d)))]
    for lag in [0,1,6,7,13,27,55]: cols.append(obs[:,origin-lag])
    cs = np.pad(np.cumsum(obs,axis=1,dtype=np.float64),((0,0),(1,0)))
    sq = np.pad(np.cumsum(obs.astype(float)**2,axis=1),((0,0),(1,0)))
    ps = np.pad(np.cumsum(obs>0,axis=1),((0,0),(1,0)))
    hits = obs>=cap
    hits[:,:56] = False
    hs = np.pad(np.cumsum(hits,axis=1),((0,0),(1,0)))
    for w in [7,28,56]:
        mean = (cs[:,origin+1]-cs[:,origin+1-w])/w
        var = (sq[:,origin+1]-sq[:,origin+1-w])/w-mean**2
        cols.extend([mean,np.sqrt(np.maximum(var,0)),
                     (ps[:,origin+1]-ps[:,origin+1-w])/w,
                     (hs[:,origin+1]-hs[:,origin+1-w])/w])
    cols.append(cap[:,origin])
    for v in [dates.dayofweek.to_numpy(),dates.month.to_numpy(),dates.day.to_numpy()]:
        cols.append(np.broadcast_to(v[d], (n,len(d))))
    x = np.stack(cols,axis=-1).astype(np.float32)
    return x.reshape(-1,x.shape[-1])

def prepare(source, out):
    save(out/'PREPROCESSING_RECEIPT.json', {'created_utc':now(),
         'scope':'blind sharding; future outcomes processed without evaluation',
         'source_sha256':sha(source)})
    with zipfile.ZipFile(source) as z:
        with z.open('online_retail_II.xlsx') as f:
            sheets = pd.read_excel(io.BytesIO(f.read()),sheet_name=None,engine='openpyxl')
    df = pd.concat(sheets.values(),ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={'InvoiceNo':'Invoice','UnitPrice':'Price'})
    df['StockCode'] = df.StockCode.astype(str).str.strip().str.upper()
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)
    df['day'] = df.InvoiceDate.dt.normalize()
    eligible = (df.StockCode.str.fullmatch(r'\d{5}[A-Z]*').fillna(False)
                & ~df.Invoice.astype(str).str.upper().str.startswith('C')
                & (df.Quantity>0) & (df.Price>0))
    df = df.loc[eligible,['StockCode','day','Quantity']]
    tr = df[df.day <= '2010-12-31']
    active = tr.groupby('StockCode').day.nunique()
    ids = [s for s in active.index if active[s]>=14]
    ids = sorted(ids,key=lambda s:hashlib.sha256(('CENSORCAST_UCI_20260907:'+s).encode()).hexdigest())[:2000]
    ids = sorted(ids)
    if len(ids)<100: raise ValueError('Too few training-eligible products')
    dates = pd.date_range('2009-12-01','2011-11-30')
    panel = df[df.StockCode.isin(ids)].groupby(['StockCode','day']).Quantity.sum().unstack(fill_value=0)
    y = panel.reindex(index=ids,columns=dates,fill_value=0).to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid quantities')
    obs, cap = censor(y)
    fit_end = int(dates.get_loc('2010-12-31')); val_end = int(dates.get_loc('2011-06-30'))
    ds = np.arange(len(dates)); origins = (ds//7)*7-1
    train = ds[(ds>=63)&(ds<=fit_end)]
    valid = ds[(ds>fit_end)&(ds<=val_end)&(origins>=fit_end)]
    test = ds[(ds>val_end)&(origins>=val_end)]
    npz(out/'development.npz',observed=obs[:,:val_end+1],capacity=cap[:,:val_end+1],
        truth=y[:,:val_end+1],ids=np.array(ids),train_days=train,valid_days=valid)
    npz(out/'sealed_targets.npz',truth=y[:,test],days=test)
    npz(out/'prediction_context.npz',observed=obs,capacity=cap)
    save(out/'COHORT.json',{'ids':ids,'selection':'training active days >=14; hash sample max2000',
         'cohort_size':len(ids),'future_outcome_summary_released':False,
         'test_target_sha256':sha(out/'sealed_targets.npz')})
    print('Blind preprocessing complete. Training-selected products:',len(ids),flush=True)

def predict_candidate(pr, c):
    if c['kind']=='base': return c['scale']*pr[c['model']]
    b=pr['l1']; uplift=np.maximum(pr['censored_em']-b,0)
    return c['scale']*(b+c['alpha']*pr['hit']**c['power']*uplift)

def fit_develop(out):
    z=np.load(out/'development.npz'); obs=z['observed']; cap=z['capacity']
    dates=pd.date_range('2009-12-01',periods=obs.shape[1]); td=z['train_days']; vd=z['valid_days']
    x=features(obs,cap,dates,td); y=obs[:,td].ravel(); c=cap[:,td].ravel()
    keep=np.random.default_rng(SEED).choice(len(y),min(600000,len(y)),replace=False)
    x=x[keep]; y=y[keep]; c=c[keep]
    xv=features(obs,cap,dates,vd); yt=z['truth'][:,vd].ravel()
    save(out/'TRAINING_START.json',{'created_utc':now(),'rows':len(y),'test_evaluations':0})
    models={}; pr={}
    for name,objective in [('l1','regression_l1'),('poisson','poisson')]:
        m=lgb.LGBMRegressor(objective=objective,**PARAMS).fit(x,y,categorical_feature=[0])
        models[name]=m; pr[name]=np.maximum(m.predict(xv),0)
        print('Fitted',name,flush=True)
    hit=lgb.LGBMClassifier(objective='binary',**PARAMS).fit(x,(y>=c).astype(int),categorical_feature=[0])
    models['hit']=hit; pr['hit']=hit.predict_proba(xv)[:,1]
    em=models['poisson']; fallback=[]
    for iteration in range(2):
        mu=np.maximum(em.predict(x),1e-6); tail=poisson.sf(c-1,mu)
        numerator=mu*poisson.sf(c-2,mu)
        expectation=np.divide(numerator,tail,out=c.astype(float).copy(),where=tail>1e-250)
        expectation=np.maximum(expectation,c)
        target=np.where(y>=c,expectation,y)
        if not np.isfinite(target).all(): raise ValueError('Non-finite EM target')
        fallback.append(int(((y>=c)&(tail<=1e-250)).sum()))
        em=lgb.LGBMRegressor(objective='poisson',**PARAMS).fit(x,target,categorical_feature=[0])
        print('Fitted censored EM iteration',iteration+1,flush=True)
    models['censored_em']=em; pr['censored_em']=np.maximum(em.predict(xv),0)
    origin=(vd//7)*7-1
    pr['seasonal_naive']=obs[:,vd-7].ravel()
    pr['rolling_mean']=np.stack([obs[:,o-27:o+1].mean(axis=1) for o in origin],axis=1).ravel()
    candidates=[]
    for model in ['l1','poisson','seasonal_naive','rolling_mean']:
        for scale in SCALES: candidates.append(dict(kind='base',model=model,scale=scale))
    for scale in SCALES:
        for alpha in ALPHAS:
            for power in POWERS: candidates.append(dict(kind='correction',scale=scale,alpha=alpha,power=power))
    records=[dict(config=c,wape=score(yt,predict_candidate(pr,c))) for c in candidates]
    # Stable input order provides a deterministic tie-break (lower scale first).
    best_base=min([r for r in records if r['config']['kind']=='base'],key=lambda r:r['wape'])
    best_proposal=min([r for r in records if r['config']['kind']=='correction'],key=lambda r:r['wape'])
    save(out/'DEVELOPMENT_RESULTS.json',{'records':records,'baseline':best_base,'proposal':best_proposal,
         'em_underflow_fallback_counts':fallback,'created_utc':now()})
    (out/'models').mkdir()
    for name,m in models.items():m.booster_.save_model(str(out/'models'/f'{name}.txt'))
    save(out/'FINAL_FREEZE.json',{'created_utc':now(),'baseline':best_base['config'],
         'proposal':best_proposal['config'],'model_hashes':{p.name:sha(p) for p in sorted((out/'models').glob('*.txt'))},
         'code_sha256':sha(__file__),'protocol_sha256':sha(out/'STUDY_PROTOCOL.json'),
         'cohort_sha256':sha(out/'COHORT.json'),'development_sha256':sha(out/'DEVELOPMENT_RESULTS.json'),
         'test_sha256':sha(out/'sealed_targets.npz'),'test_evaluations':0,
         'comparison':'new observable-capacity correction versus validation-selected tabular baseline'})
    print('Final policy freeze written; test not evaluated.',flush=True)

def evaluate(out):
    freeze=json.loads((out/'FINAL_FREEZE.json').read_text())
    assert freeze['code_sha256']==sha(__file__)
    assert freeze['test_sha256']==sha(out/'sealed_targets.npz')
    for name,h in freeze['model_hashes'].items():assert sha(out/'models'/name)==h
    zz=np.load(out/'prediction_context.npz');obs=zz['observed'];cap=zz['capacity']
    dates=pd.date_range('2009-12-01',periods=obs.shape[1]);val_end=dates.get_loc('2011-06-30')
    ds=np.arange(len(dates));days=ds[(ds>val_end)&((ds//7)*7-1>=val_end)]
    xt=features(obs,cap,dates,days);pr={}
    for name in ['l1','poisson','censored_em','hit']:
        m=lgb.Booster(model_file=str(out/'models'/f'{name}.txt'))
        pr[name]=np.maximum(m.predict(xt,num_threads=4),0)
    origin=(days//7)*7-1
    pr['seasonal_naive']=obs[:,days-7].ravel()
    pr['rolling_mean']=np.stack([obs[:,o-27:o+1].mean(axis=1) for o in origin],axis=1).ravel()
    b=predict_candidate(pr,freeze['baseline']).reshape(len(obs),-1)
    p=predict_candidate(pr,freeze['proposal']).reshape(len(obs),-1)
    npz(out/'FROZEN_PREDICTIONS.npz',baseline=b,proposal=p,days=days)
    save(out/'FIRST_TEST_ACCESS.json',{'created_utc':now(),'prior_test_evaluations':0,
         'freeze_sha256':sha(out/'FINAL_FREEZE.json'),'prediction_sha256':sha(out/'FROZEN_PREDICTIONS.npz')})
    target=np.load(out/'sealed_targets.npz');assert np.array_equal(target['days'],days)
    y=target['truth'].astype(float); eb=np.abs(y-b);ep=np.abs(y-p)
    mass=y.sum(axis=1);diff=(ep-eb).sum(axis=1)
    rng=np.random.default_rng(SEED); boot=[];timeboot=[]
    for _ in range(2000):
        idx=rng.integers(0,len(mass),len(mass));boot.append(float(diff[idx].sum()/mass[idx].sum()))
    # Sensitivity to common calendar shocks, resampling whole origin weeks.
    weeks=np.unique(origin);wm=np.array([y[:,origin==w].sum() for w in weeks])
    wd=np.array([(ep-eb)[:,origin==w].sum() for w in weeks])
    for _ in range(2000):
        idx=rng.integers(0,len(wm),len(wm));timeboot.append(float(wd[idx].sum()/wm[idx].sum()))
    rows=[]
    for name,f in [('selected_baseline',b),('selected_correction',p)]:
        rows.append({'method':name,'wape':score(y,f),'mae':float(np.abs(y-f).mean()),
                     'volume_bias':float((f-y).sum()/y.sum()),'row_coverage':1.})
    ci=np.quantile(boot,[.025,.975]).tolist();ct=np.quantile(timeboot,[.025,.975]).tolist()
    save(out/'CONFIRMATION_RESULTS.json',{'status':'COMPLETED','created_utc':now(),
         'scope':'new-dataset temporal confirmation of validation-selected tabular contrast',
         'dataset':'UCI Online Retail II','series_count':len(obs),'rows':int(y.size),
         'days':len(days),'rows_results':rows,'delta_wape_proposal_minus_baseline':float(diff.sum()/mass.sum()),
         'relative_error_reduction':float(1-ep.sum()/eb.sum()),'paired_item_bootstrap_ci95':ci,
         'calendar_week_sensitivity_ci95':ct,'primary_improvement_supported':bool(ci[1]<0),
         'test_evaluations':1,'foundation_models_compared':False,'natural_lost_sales_validated':False,
         'interval_caveat':'bootstrap sampling assumptions; no distribution-free guarantee',
         'freeze_sha256':sha(out/'FINAL_FREEZE.json')})
    npz(out/'RECOMPUTABLE_TEST_STATISTICS.npz',mass_by_item=mass,error_baseline_by_item=eb.sum(axis=1),
        error_proposal_by_item=ep.sum(axis=1),mass_by_week=wm,error_difference_by_week=wd)
    print((out/'CONFIRMATION_RESULTS.json').read_text(),flush=True)

def run(source,out):
    out.mkdir(parents=True,exist_ok=True)
    save(out/'STUDY_PROTOCOL.json',{'created_utc':now(),'source_sha256':sha(source),
         'code_sha256':sha(__file__),'seed':SEED,'fit_end':'2010-12-31','validation_end':'2011-06-30',
         'test_end':'2011-11-30','max_products':2000,'training_active_days_min':14,
         'horizons':list(range(1,8)),'weekly_origin_purge':True,'parameters':PARAMS,
         'scales':SCALES,'alphas':ALPHAS,'powers':POWERS,'em_iterations':2,'train_row_cap':600000,
         'cohort_selection':'training only, deterministic hash sample',
         'non_merchandise_filter':'stock code digits{5} followed by optional letters; positive price and quantity; no cancellation invoices',
         'exact_duplicate_rows':'retained; transaction identity not assumed from matching fields',
         'primary_endpoint':'full coverage WAPE difference against validation-selected tabular baseline',
         'primary_ci':'2000 paired product-cluster bootstrap, two-sided 95%',
         'secondary_sensitivity':'2000 paired calendar-origin-week bootstrap',
         'model_fit_labels':'mechanically censored observed sales and observable capacity-hit only',
         'selection_labels':'development recorded-sales truth',
         'new_algorithm':'ported observable-capacity gated Poisson-EM correction; not frozen M5 engine',
         'versions':{'python':platform.python_version(),'lightgbm':lgb.__version__,'numpy':np.__version__,'pandas':pd.__version__}})
    prepare(source,out);fit_develop(out);evaluate(out)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();run(args.source,args.output)
