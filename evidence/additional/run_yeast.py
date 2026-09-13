from pathlib import Path
import numpy as np,json,hashlib,datetime,joblib
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
P=Path('additional_experiments/yeast');seed=20260909
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(n,v):(P/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
assert not (P/'FROZEN.json').exists()
dump('RUN_STARTED.json',{'created_utc':now(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((P/'PROTOCOL.json').read_bytes()).hexdigest()})
def load(path):
 a,meta=arff.loadarff(path);names=a.dtype.names;arr=np.column_stack([a[k].astype(float) for k in names]);return arr[:,:-14],arr[:,-14:].astype(int)
X,Y=load(P/'yeast-train.arff');assert X.shape[1]==103 and Y.shape[1]==14
ids=np.random.default_rng(seed).permutation(len(X));n=int(.75*len(X));fit,cal=ids[:n],ids[n:];sc=StandardScaler().fit(X[fit]);x=sc.transform(X);models=[]
for j in range(14):
 labels=np.unique(Y[fit,j]);mod=float(labels[0]) if len(labels)==1 else LogisticRegression(C=1,max_iter=2000,random_state=seed).fit(x[fit],Y[fit,j]);models.append(mod)
def pred(X):
 xx=sc.transform(X);prob=np.column_stack([np.full(len(X),m) if isinstance(m,float) else m.predict_proba(xx)[:,1] for m in models]);h=(prob>=.5).astype(int);w=h.sum(1);e=(h*(1-prob)).sum(1);return prob,h,w,e
prob,h,w,e=pred(X[cal]);loss=(h*(1-Y[cal])).sum(1);full=float(loss.sum()/w.sum());policies=[]
def scores(e,w,r,method):
 if method=='conditional_error':s=e.copy()
 elif method=='row_excess':s=e-r*w
 else:s=np.divide(e,w,out=np.zeros(len(w)),where=w>0)
 s[w==0]=-np.inf
 return s

def metrics(L,W,a):
 den=float(W[a].sum());return {'rows':int(a.sum()),'c':float(a.mean()),'d':float(W[a].sum()/W.sum()),'risk':float(L[a].sum()/den) if den else None,'loss':int(L[a].sum()),'weight':int(W[a].sum())}
for factor in [.75,.85,.95,1.05]:
 r=factor*full
 for method in ['conditional_error','row_excess','exposure_ratio']:
  s=scores(e,w,r,method);ts=np.unique(s[np.isfinite(s)]);best=None
  for t in ts:
   a=s<=t
   if a.mean()>=.35 and w[a].sum()>0 and loss[a].sum()<=r*w[a].sum()+1e-12:best=float(t)
  a=s<=best if best is not None else np.zeros(len(w),bool)
  policies.append({'factor':factor,'cap':r,'method':method,'threshold':best,'calibration':metrics(loss,w,a)})
joblib.dump({'scaler':sc,'models':models},P/'CLASSIFIER.joblib');np.savez_compressed(P/'CALIBRATION.npz',ids=cal,fit_ids=fit,L=loss,W=w,e=e,prob=prob,h=h,Y=Y[cal]);dump('FROZEN.json',{'created_utc':now(),'n_fit':len(fit),'n_cal':len(cal),'full_calibration_risk':full,'policies':policies,'classifier_sha256':hashlib.sha256((P/'CLASSIFIER.joblib').read_bytes()).hexdigest()})
# First parse of evaluation labels occurs after the model and thresholds are frozen.
dump('TEST_ACCESS.json',{'created_utc':now(),'test_sha256':hashlib.sha256((P/'yeast-test.arff').read_bytes()).hexdigest(),'opening_count':1,'frozen_sha256':hashlib.sha256((P/'FROZEN.json').read_bytes()).hexdigest()})
Xt,Yt=load(P/'yeast-test.arff');pt,ht,wt,et=pred(Xt);lt=(ht*(1-Yt)).sum(1);masks={};result=[]
for pol in policies:
 t=pol['threshold'];a=scores(et,wt,pol['cap'],pol['method'])<=t if t is not None else np.zeros(len(wt),bool);key=str(pol['factor'])+'_'+pol['method'];masks[key]=a;result.append({**pol,'test':metrics(lt,wt,a)})
np.savez_compressed(P/'TEST_PREDICTIONS.npz',L=lt,W=wt,e=et,h=ht,prob=pt,Y=Yt,**masks)
rng=np.random.default_rng(seed);bs=rng.integers(0,len(wt),(2000,len(wt)));contrasts=[]
for factor in [.75,.85,.95,1.05]:
 a=masks[str(factor)+'_row_excess'];b=masks[str(factor)+'_exposure_ratio'];dc=b.astype(float)-a.astype(float);dw=dc*wt;arr=np.column_stack([dc[bs].mean(1),dw[bs].sum(1)/wt[bs].sum(1)])
 contrasts.append({'factor':factor,'row_delta':float(dc.mean()),'exposure_delta':float(dw.sum()/wt.sum()),'ci95':np.quantile(arr,[.025,.975],axis=0).T.tolist()})
dump('RESULTS.json',{'n_test':len(wt),'full_test':metrics(lt,wt,np.ones(len(wt),bool)),'exposure_distribution':{str(i):int(sum(wt==i)) for i in np.unique(wt)},'policies':result,'contrasts':contrasts,'completed_utc':now()});print(json.dumps({'n_test':len(wt),'primary':[x for x in result if x['factor']==.85],'contrasts':contrasts},indent=2))
