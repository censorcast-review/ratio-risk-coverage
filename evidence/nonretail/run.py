from pathlib import Path
import sys,json,datetime,hashlib,joblib,io,bz2,zipfile
import numpy as np,pandas as pd,lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from scipy.stats import kendalltau
R=Path(__file__).resolve().parent;name=sys.argv[1];P=R/name;P.mkdir(exist_ok=False);seed=20260910;now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):(P/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
dump('START.json',{'utc':now(),'protocol_sha256':sha(R/'PROTOCOL.json'),'script_sha256':sha(Path(__file__))})
if name=='delicious':
 lines=bz2.decompress((R/'delicious.bz2').read_bytes()).splitlines();ix=np.random.default_rng(seed).permutation(len(lines));n=len(ix);fit,cal,test=ix[:int(.6*n)],ix[int(.6*n):int(.8*n)],ix[int(.8*n):]
 def parse(ids):
  X,labels=load_svmlight_file(io.BytesIO(b'\n'.join(lines[i] for i in ids)),n_features=500,multilabel=True,zero_based=False);Y=np.zeros((len(ids),983),np.uint8)
  for j,ls in enumerate(labels):
   for k in ls:
    assert 0<=k<983;Y[j,int(k)]=1
  return X,Y
 X,Y=parse(fit);Xc,Yc=parse(cal);mods=[]
 for j in range(983):
  vals=np.unique(Y[:,j]);mods.append(float(vals[0]) if len(vals)==1 else LogisticRegression(C=1,solver='liblinear',max_iter=2000,random_state=seed).fit(X,Y[:,j]))
  if j%100==0:print('label',j,flush=True)
 def predict(X):return np.column_stack([np.full(X.shape[0],m) if isinstance(m,float) else m.predict_proba(X)[:,1] for m in mods])
 pc=predict(Xc);variants=[.5,.2];models=mods
 def fields(prob,Y,v):
  h=prob>=v;W=h.sum(1).astype(float);L=(h*(1-Y)).sum(1).astype(float);e=(h*(1-prob)).sum(1);return L,W,e,W
 caldata={str(v):fields(pc,Yc,v) for v in variants}
else:
 with zipfile.ZipFile(R/'bike.zip') as z:
  file=next(x for x in z.namelist() if x.endswith('hour.csv'));lines=z.read(file).splitlines();header=lines[0];lines=lines[1:]
 # Partition from date field only; held-out counts are not parsed here.
 dates=np.array([a.split(b',')[1].decode() for a in lines]);days=np.unique(dates);ix=np.random.default_rng(seed).permutation(len(days));a,b,c=[int(q*len(days)) for q in [.4,.6,.8]];groups=[days[ix[:a]],days[ix[a:b]],days[ix[b:c]],days[ix[c:]]];fit,head,cal,test=[np.flatnonzero(np.isin(dates,g)) for g in groups]
 cols=json.loads((R/'PROTOCOL.json').read_text())['bike']['features']
 def parse(ids):
  df=pd.read_csv(io.BytesIO(header+b'\n'+b'\n'.join(lines[i] for i in ids)));return df[cols].to_numpy(),df.cnt.to_numpy(float)
 X,Y=parse(fit);Xh,Yh=parse(head);Xc,Yc=parse(cal);params=dict(n_estimators=200,num_leaves=31,min_child_samples=30,learning_rate=.05,random_state=seed,n_jobs=4,verbosity=-1)
 base=lgb.LGBMRegressor(objective='quantile',alpha=.5,**params).fit(X,Y);mu=lgb.LGBMRegressor(objective='poisson',**params).fit(X,Y);error=lgb.LGBMRegressor(objective='regression',**params).fit(np.column_stack([Xh,base.predict(Xh)]),np.abs(Yh-base.predict(Xh)));models=[base,mu,error]
 def predict(X):
  f=np.maximum(base.predict(X),0);return f,np.maximum(mu.predict(X),0),np.maximum(error.predict(np.column_stack([X,f])),0)
 fc,mc,ec=predict(Xc);caldata={'default':(np.abs(Yc-fc),Yc,ec,mc)}
joblib.dump(models,P/'MODELS.joblib');np.savez_compressed(P/'SPLITS.npz',fit=fit,cal=cal,test=test)
def score(e,w,r,method,T):
 s=e-r*w
 if method=='error':s=e.copy()
 if method=='weight_descending':s=-w
 if method in ['mixed','demand']:
  den=.25+.75*w/T if method=='mixed' else w
  s=np.divide(s,den,out=np.where(s>0,np.inf,np.where(s<0,-np.inf,0.)),where=den>0)
 if name=='delicious':s[w==0]=-np.inf
 return s

def metrics(L,W,a):
 mass=float(W[a].sum());return dict(rows=int(a.sum()),c=float(a.mean()),d=mass/float(W.sum()),risk=float(L[a].sum()/mass) if mass else None,loss=float(L[a].sum()),weight=mass)
pols=[];factors=[.75,.85,.95,1.05];methods=['error','row','mixed','demand','weight_descending']
for variant,(L,W,e,w) in caldata.items():
 np.savez_compressed(P/('CALIBRATION_'+variant+'.npz'),L=L,W=W,e=e,w=w)
 T=float(w.mean());full=L.sum()/W.sum()
 for factor in factors:
  r=float(factor*full)
  for method in methods:
   s=score(e,w,r,method,T);ts=np.unique(s[np.isfinite(s)]);o=np.argsort(s,kind='stable');num=np.r_[0.,np.cumsum(L[o])];den=np.r_[0.,np.cumsum(W[o])];counts=np.searchsorted(s[o],ts,side='right');good=(den[counts]>0)&(counts/len(W)>=.35)&(num[counts]<=.95*r*den[counts]+1e-12);ind=np.flatnonzero(good);t=float(ts[ind[-1]]) if len(ind) else None
   if full<=.95*r:t='all'
   mask=np.ones(len(W),bool) if t=='all' else s<=t if t is not None else np.zeros(len(W),bool)
   pols.append(dict(variant=variant,factor=factor,cap=r,method=method,threshold=t,T=T,calibration=metrics(L,W,mask)))
dump('FROZEN.json',{'utc':now(),'policies':pols,'model_sha256':sha(P/'MODELS.joblib')});dump('TEST_ACCESS.json',{'utc':now(),'freeze_sha256':sha(P/'FROZEN.json'),'parse_count':1})
Xt,Yt=parse(test)
if name=='delicious':pt=predict(Xt);testdata={str(v):fields(pt,Yt,v) for v in variants};units=np.arange(len(test))
else:
 f,m,e=predict(Xt);testdata={'default':(np.abs(Yt-f),Yt,e,m)};units=np.unique(dates[test],return_inverse=True)[1]
results=[];masks={};taus={}
for variant,(L,W,e,w) in testdata.items():
 save=dict(L=L,W=W,e=e,w=w,units=units)
 for p in pols:
  if p['variant']!=variant:continue
  s=score(e,w,p['cap'],p['method'],p['T']);t=p['threshold'];mask=np.ones(len(W),bool) if t=='all' else s<=t if t is not None else np.zeros(len(W),bool);key=f"{p['factor']}_{p['method']}";save[key]=mask;masks[(variant,p['factor'],p['method'])]=mask;results.append({**p,'test':metrics(L,W,mask)})
 np.savez_compressed(P/('TEST_'+variant+'.npz'),**save)
 r=next(p['cap'] for p in pols if p['variant']==variant and p['factor']==.85);positive=w>0;taus[variant]=dict(kendall=float(kendalltau((e-r*w)[positive],(e/w.clip(1e-12))[positive]).statistic),weight_min=float(w.min()),weight_max=float(w.max()),full=metrics(L,W,np.ones(len(W),bool)))
contrasts=[];rng=np.random.default_rng(seed);n_units=int(units.max())+1;sample=rng.integers(0,n_units,(4000,n_units))
for variant,(L,W,e,w) in testdata.items():
 N=np.bincount(units);D=np.bincount(units,weights=W)
 for factor in factors:
  a=masks[(variant,factor,'row')];b=masks[(variant,factor,'demand')];delta=b.astype(int)-a.astype(int);dn=np.bincount(units,weights=delta);dd=np.bincount(units,weights=delta*W);vals=np.column_stack([dn[sample].sum(1)/N[sample].sum(1),dd[sample].sum(1)/D[sample].sum(1)])
  contrasts.append(dict(variant=variant,factor=factor,dc=float(delta.mean()),dd=float((delta*W).sum()/W.sum()),ci95=np.quantile(vals,[.025,.975],axis=0).T.tolist(),ci_family98_75=np.quantile(vals,[.00625,.99375],axis=0).T.tolist()))
dump('RESULTS.json',dict(dataset=name,n_test=len(test),n_units=n_units,policies=results,contrasts=contrasts,diagnostics=taus,completed_utc=now()));print(json.dumps({'dataset':name,'primary':[p for p in results if p['factor']==.85 and p['variant'] in ['default','0.5']],'contrasts':contrasts},indent=2))
