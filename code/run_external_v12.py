"""Pre-opening, one-run replication on Dominick's Oatmeal sales.

The protocol and source hashes must be initialized before downloading data.
No fitting/tuning uses calibration B or evaluation outcomes. Forecasts are
rolling one-week-ahead, so previous observed weeks may enter later features.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, zipfile, warnings
from urllib.request import urlopen
import numpy as np
import pandas as pd
import lightgbm as lgb
from policy_audit import METHODS, scores, mask, metrics, largest_threshold, choose

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260912
DATA_URL = 'https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/movement_csv-files/woat.zip'
WINDOWS = {'point': [53,208], 'head': [209,260], 'A': [261,312], 'B': [313,364], 'E': [365,400]}

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj): Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def model_params(trees, objective):
    return dict(objective=objective,alpha=.5,n_estimators=trees,num_leaves=31,
                learning_rate=.05,min_child_samples=50,n_jobs=4,random_state=SEED,
                verbosity=-1,deterministic=True,force_col_wise=True)

def initialize(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol = dict(version='B-v12-1',utc=now(),dataset='Dominicks Oatmeal movement CSV',
        url=DATA_URL,license='Academic research only; acknowledge Kilts Center, Chicago Booth',
        selection_reason='Public academic access without unavailable Kaggle credentials; one packaged-staple category limits computation. No raw outcomes inspected before choice.',
        prior_use='No use found in the current project or retrieved prior context; not a universal claim about every coauthor.',
        status='Locally frozen before first data download; not a public registry preregistration',
        windows=WINDOWS,forecast='Rolling one-week-ahead; past observed sales only; model fits remain fixed',
        target='Recorded unit sales MOVE, no artificial censoring; not latent demand',
        rows='OK=1, finite nonnegative MOVE; no missing-row zero imputation; duplicate store/UPC/week aborts',
        cohort='Store-UPC pairs with >=52 valid point-window records and positive point-window sales; decided without A/B/E',
        features='Store/UPC categorical codes, week, sin/cos52, lags1/2/4/13/26/52, rolling4/13/26 mean/std/max/missing; all sales strictly earlier than target week',
        seed=SEED,point_params=model_params(300,'quantile'),head_params=model_params(220,'regression'),
        samples='All eligible rows up to600000, fixed seeded nonreplacement draw if needed',
        menu=list(METHODS),head_pairs=[[220,220]],floor=.40,selection='Eq6 on A; complete ties; stable declared method order; no fallback after B',
        cap='r=.95*full-coverage A risk; design cap=.95*r; scores use r. Never inspect B to set r.',
        bootstrap=dict(unit='Circular moving blocks of4 calendar weeks, all stores and UPCs together',
                       B_draws=20000,E_draws=20000,B_seed=2026091201,E_seed=2026091202,
                       B_upper_tail=.05/5,quantile_method='linear',E_interval=[.025,.975]),
        screen='Five A-frozen candidates; B upper quantile of pooled signed excess L-rW <=0 and positive accepted exposure; report all candidates',
        primary_success='Both A-selected policies pass B and E paired95% interval upper(dc)<0 and lower(dd)>0',
        secondary='Point-sign exchange dc<0,dd>0; per-candidate E risk/exposure/case; E risk upper bounds descriptive',
        limitations=['Historical data, new to this analysis workflow','One external fit seed/category',
                     'Block bootstrap is approximate, conditional on cohort/fits; no population-risk guarantee',
                     'No evidence of uncensored latent demand','A-only cap differs from the retrospective joint-A/B design'],
        no_retuning='No alternate category, seed, floor, cap, model, or success criterion after opening; all failures retained',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'code/policy_audit.py']},
        runtime={k:v for k,v in [('lightgbm',lgb.__version__),('numpy',np.__version__),('pandas',pd.__version__)]})
    write(out/'PROTOCOL.json',protocol)
    (out/'PROTOCOL.sha256').write_text(sha(out/'PROTOCOL.json')+'  PROTOCOL.json\n')
    print('FROZEN',sha(out/'PROTOCOL.json'),flush=True)

def verify_protocol(out):
    p=json.loads((out/'PROTOCOL.json').read_text())
    assert sha(out/'PROTOCOL.json')==(out/'PROTOCOL.sha256').read_text().split()[0]
    for rel,h in p['code_sha256'].items(): assert sha(ROOT/rel)==h,rel
    assert p['runtime']==dict(lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__)
    return p

def download(out, data):
    verify_protocol(out)
    if data.exists() or (out/'INPUT_RECEIPT.json').exists(): raise FileExistsError('Preserve existing input')
    data.parent.mkdir(parents=True,exist_ok=True)
    with urlopen(DATA_URL,timeout=60) as src, data.open('xb') as dst:
        while chunk:=src.read(1024*1024): dst.write(chunk)
    write(out/'INPUT_RECEIPT.json',dict(utc=now(),url=DATA_URL,bytes=data.stat().st_size,
          sha256=sha(data),protocol_sha256=sha(out/'PROTOCOL.json'),opened_by_analysis=False))
    print('DOWNLOADED; outcomes not displayed',data.stat().st_size,flush=True)

def panel_from_raw(data):
    with zipfile.ZipFile(data) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        assert len(names)==1,names
        with z.open(names[0]) as f: raw=pd.read_csv(f)
    raw.columns=raw.columns.str.upper().str.strip()
    assert {'STORE','UPC','WEEK','MOVE','OK'}<=set(raw.columns)
    raw=raw[['STORE','UPC','WEEK','MOVE','OK']].copy()
    for col in raw.columns: raw[col]=pd.to_numeric(raw[col],errors='raise')
    raw=raw[(raw.WEEK>=1)&(raw.WEEK<=400)]
    assert not raw.duplicated(['STORE','UPC','WEEK']).any()
    valid=(raw.OK==1)&np.isfinite(raw.MOVE)&(raw.MOVE>=0)
    clean=raw[valid].copy()
    train=clean[clean.WEEK.between(*WINDOWS['point'])]
    grp=train.groupby(['STORE','UPC']).MOVE.agg(['count','sum'])
    cohort=grp[(grp['count']>=52)&(grp['sum']>0)].index.sort_values()
    assert len(cohort)>=20 and len(cohort.get_level_values('UPC').unique())>=5
    idx=pd.MultiIndex.from_frame(clean[['STORE','UPC']]); ids=cohort.get_indexer(idx)
    clean=clean[ids>=0];ids=ids[ids>=0]
    Y=np.full((len(cohort),401),np.nan,dtype=float)
    Y[ids,clean.WEEK.to_numpy(int)]=clean.MOVE.to_numpy(float)
    store,store_values=pd.factorize(cohort.get_level_values('STORE'),sort=True)
    upc,upc_values=pd.factorize(cohort.get_level_values('UPC'),sort=True)
    info=dict(raw_rows=int(len(raw)),invalid_rows=int((~valid).sum()),cohort_series=len(cohort),
              cohort_products=len(upc_values),cohort_stores=len(store_values),
              cohort_selected_on='point window only',normalised_features_used=False)
    return Y,store,upc,np.asarray(cohort.to_list(),dtype=np.int64),info

def make_features(Y, store, upc, ss, ww):
    cols=[store[ss],upc[ss],ww,np.sin(2*np.pi*ww/52),np.cos(2*np.pi*ww/52)]
    for lag in [1,2,4,13,26,52]: cols.append(Y[ss,ww-lag])
    for n in [4,13,26]:
        past=Y[ss[:,None],ww[:,None]-np.arange(1,n+1)[None,:]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            cols.extend([np.nanmean(past,axis=1),np.nanstd(past,axis=1),np.nanmax(past,axis=1),np.isnan(past).mean(axis=1)])
    return np.column_stack(cols).astype(np.float32)

def window_rows(Y,name):
    lo,hi=WINDOWS[name];ss,off=np.where(np.isfinite(Y[:,lo:hi+1]));ww=lo+off
    assert len(ss)>=1000,(name,len(ss))
    return ss,ww

def fit(X,y,trees,obj):
    m=lgb.LGBMRegressor(**model_params(trees,obj));m.fit(X,y,categorical_feature=[0,1]);return m

def block_counts(n,draws,seed,length=4):
    rng=np.random.default_rng(seed);starts=rng.integers(0,n,(draws,(n+length-1)//length))
    ix=((starts[:,:,None]+np.arange(length))%n).reshape(draws,-1)[:,:n]
    c=np.zeros((draws,n),dtype=np.int16)
    np.add.at(c,(np.repeat(np.arange(draws),n),ix.ravel()),1)
    return c

def weekly_stats(L,W,a,ww,lo,n):
    return np.column_stack([np.bincount(ww-lo,weights=v,minlength=n) for v in [a,a*W,a*L]])

def analyse_window(arr, plans, cap, T, name, out):
    L=np.abs(arr['y']-arr['f']);W=arr['y'];ww=arr['week'];lo,hi=WINDOWS[name];n=hi-lo+1
    counts=block_counts(n,20000,2026091201 if name=='B' else 2026091202)
    total=np.column_stack([np.bincount(ww-lo,minlength=n),np.bincount(ww-lo,weights=W,minlength=n)])
    assert np.all(total[:,0]>0) and np.all((counts@total)[:,1]>0)
    saved={'total':total,'counts':counts};result={}
    for m in METHODS:
        a=mask(scores(arr['e'],arr['w'],cap,m,T),plans[m]['threshold'])
        stat=weekly_stats(L,W,a,ww,lo,n);saved[m]=stat
        draw=counts@stat;nonzero=draw[:,1]>0
        upper=float(np.quantile(draw[:,2]-cap*draw[:,1],.99,method='linear'))
        risk_upper=float(np.quantile(draw[nonzero,2]/draw[nonzero,1],.99)) if nonzero.all() else None
        result[m]=dict(**metrics(L,W,a),signed_excess_U5=upper,risk_U5=risk_upper,
            passed=bool(plans[m]['threshold'] is not None and nonzero.all() and upper<=0))
    np.savez_compressed(out/f'{name}_WEEKLY_STATISTICS.npz',**saved)
    return result,counts,total,saved

def run(out,data):
    proto=verify_protocol(out)
    if (out/'START.json').exists(): raise FileExistsError('One analysis opening; preserve all partial runs')
    receipt=json.loads((out/'INPUT_RECEIPT.json').read_text());assert sha(data)==receipt['sha256']
    write(out/'START.json',dict(utc=now(),protocol_sha256=sha(out/'PROTOCOL.json'),input_sha256=sha(data)))
    Y,store,upc,cohort,info=panel_from_raw(data)
    rng=np.random.default_rng(SEED);trained={};samples={}
    for name in ['point','head']:
        ss,ww=window_rows(Y,name)
        if len(ss)>600000:
            ix=np.sort(rng.choice(len(ss),600000,replace=False));ss=ss[ix];ww=ww[ix]
        samples[name+'_series']=ss;samples[name+'_week']=ww
        X=make_features(Y,store,upc,ss,ww);y=Y[ss,ww]
        if name=='point': trained['f']=fit(X,y,300,'quantile')
        else:
            f=np.maximum(trained['f'].predict(X),0);T=float(y.mean())
            trained['e']=fit(np.column_stack([X,f]),np.abs(y-f),220,'regression')
            trained['w']=fit(X,y,220,'regression')
    assert T>0
    np.savez_compressed(out/'TRAINING_ROWS.npz',**samples,cohort=cohort)
    for k,m in trained.items():m.booster_.save_model(str(out/f'MODEL_{k}.txt'))
    def predict(name):
        ss,ww=window_rows(Y,name);X=make_features(Y,store,upc,ss,ww)
        f=np.maximum(trained['f'].predict(X),0)
        a=dict(f=f,e=np.maximum(trained['e'].predict(np.column_stack([X,f])),0),
               w=np.maximum(trained['w'].predict(X),0),y=Y[ss,ww],series=ss,week=ww)
        np.savez_compressed(out/f'{name}_ARRAYS.npz',**a);return a
    A=predict('A');L=np.abs(A['y']-A['f']);W=A['y'];full=float(L.sum()/W.sum());r=.95*full
    assert r>0 and np.isfinite(r)
    plans={}
    for m in METHODS:
        t,cal=largest_threshold(scores(A['e'],A['w'],r,m,T),L,W,np.zeros(len(W),int),.95*r,.40)
        plans[m]=dict(threshold=t,calibration=cal)
    selected={o:choose(plans,o) for o in ['c','d']}
    frozen=dict(utc=now(),T=T,full_A_risk=full,report_cap=r,design_cap=.95*r,plans=plans,
                selected=selected,model_sha256={k:sha(out/f'MODEL_{k}.txt') for k in trained},
                protocol_sha256=sha(out/'PROTOCOL.json'))
    write(out/'FROZEN_A.json',frozen)
    write(out/'B_OPEN.json',dict(utc=now(),frozen_A_sha256=sha(out/'FROZEN_A.json')))
    B=predict('B');br,*_=analyse_window(B,plans,r,T,'B',out);write(out/'B_SCREEN.json',br)
    write(out/'E_OPEN.json',dict(utc=now(),frozen_A_sha256=sha(out/'FROZEN_A.json'),B_screen_sha256=sha(out/'B_SCREEN.json')))
    E=predict('E');er,counts,total,saved=analyse_window(E,plans,r,T,'E',out)
    c,d=selected['c'],selected['d'];contrast=None;success=False
    if c is not None and d is not None:
        delta=(counts@(saved[d]-saved[c]))[:,:2]/(counts@total)
        ci=np.quantile(delta,[.025,.975],axis=0).T
        contrast=dict(dc=er[d]['c']-er[c]['c'],dd=er[d]['d']-er[c]['d'],ci95=ci.tolist(),
            both_pass_B=bool(br[c]['passed'] and br[d]['passed']))
        success=bool(contrast['both_pass_B'] and ci[0,1]<0 and ci[1,0]>0)
    info['window_rows']={n:len(window_rows(Y,n)[0]) for n in WINDOWS}
    result=dict(dataset=proto['dataset'],cohort=info,cap=r,design_cap=.95*r,selected=selected,
                B=br,E=er,contrast=contrast,primary_success=success,fit_calls=3,
                outcome='positive_prespecified_replication' if success else 'prespecified_joint_criterion_not_met',
                interpretation='Frozen historical external analysis; approximate risk summaries, no population guarantee')
    write(out/'RESULTS.json',result)
    write(out/'COMPLETE.json',dict(utc=now(),fit_calls=3,analysis_openings=1,
        files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--phase',choices=['initialize','download','run'],required=True);a=p.parse_args()
    if a.phase=='initialize': initialize(a.output)
    elif a.phase=='download': download(a.output,a.data)
    else: run(a.output,a.data)
