"""Observable-label count likelihoods; no hidden outcome is an argument here.

The tail model is an assumption, not nonparametric identification. Censored
rows use P(Y > observed | X). The supplied flag means strict exceedance.
"""
import numpy as np
from scipy.stats import poisson, nbinom

def dist(mu, shape=None):
    return poisson(mu) if shape is None else nbinom(shape, shape/(shape+mu))

def tail_terms(mu, c, shape=None):
    mu=np.asarray(mu,dtype=float); c=np.asarray(c,dtype=float)
    d=dist(mu,shape); logsf=d.logsf(c); logpmf=d.logpmf(c)
    multiplier=mu if shape is None else mu*(c+shape)/(mu+shape)
    a=multiplier*np.exp(np.minimum(logpmf-logsf,700))
    bad=~np.isfinite(logsf)
    if bad.any():
        mm=mu[bad];cc=c[bad];term=np.ones(len(mm));series=term.copy()
        for j in range(1,1000):
            ratio=mm/(cc+1+j) if shape is None else (cc+j+shape)/(cc+1+j)*mm/(mm+shape)
            term*=ratio;series+=term
            if np.max(term/series)<1e-14:break
        else:raise RuntimeError('Tail expansion did not converge')
        a[bad]=(cc+1)/series
        logsf[bad]=dist(mm,shape).logpmf(cc+1)+np.log(series)
    return a,logsf

def nll(mu, observed, censored, shape=None):
    mu=np.maximum(np.asarray(mu,float),1e-10);o=np.asarray(observed,float);c=np.asarray(censored,bool)
    loss=-dist(mu,shape).logpmf(o)
    if c.any():loss[c]=-tail_terms(mu[c],o[c],shape)[1]
    return loss

def derivatives(logmu, observed, censored, shape=None):
    mu=np.exp(np.clip(np.asarray(logmu,float),-20,15));o=np.asarray(observed,float);c=np.asarray(censored,bool)
    if shape is None:g=mu-o;h=mu.copy()
    else:g=shape*(mu-o)/(shape+mu);h=shape*mu*(shape+o)/(shape+mu)**2
    if c.any():
        a,_=tail_terms(mu[c],o[c],shape)
        g[c]=-a
        h[c]=a*(mu[c]+a-o[c]-1) if shape is None else a*(a-shape*(o[c]+1-mu[c])/(shape+mu[c]))
    return g,np.maximum(h,1e-8)

def expected_abs(mu, forecast, shape=None):
    mu=np.maximum(np.asarray(mu,float),1e-10);f=np.maximum(np.asarray(forecast,float),0);k=np.floor(f)
    if shape is None:F=poisson.cdf(k,mu);tr=poisson.cdf(k-1,mu)
    else:
        p=shape/(shape+mu);F=nbinom.cdf(k,shape,p);tr=nbinom.cdf(k-1,shape+1,p)
    return np.maximum(mu-f+2*(f*F-mu*tr),0)

def fit_observable(x, observed, censored, method, seed, shape=None, rounds=220):
    import lightgbm as lgb
    observed=np.asarray(observed,float);censored=np.asarray(censored,bool)
    params=dict(learning_rate=.05,num_leaves=31,min_data_in_leaf=100,lambda_l2=3,
                num_threads=4,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1,
                poisson_max_delta_step=1e-12,feature_pre_filter=False)
    keep=~censored if method=='complete_case_poisson' else np.ones(len(observed),bool)
    xx=x[keep];oo=observed[keep];cc=censored[keep]
    bias=0.
    if method in ['observed_poisson','complete_case_poisson']:
        params['objective']='poisson'
    else:
        bias=float(np.log(max(oo.mean(),.05)))
        def objective(pred, dataset):return derivatives(pred+bias,oo,cc,shape)
        params['objective']=objective
    model=lgb.train(params,lgb.Dataset(xx,label=oo),num_boost_round=rounds)
    return model,dict(method=method,shape=shape,bias=bias,training_rows=int(keep.sum()),
                     hidden_labels_used_for_fit=0,seed=seed,trees=rounds)

def predict_mean(model, x, info):
    p=model.predict(x,num_threads=4)
    if info['method'] not in ['observed_poisson','complete_case_poisson','truth_poisson_reference']:
        p=np.exp(np.clip(p+info['bias'],-20,15))
    return np.maximum(p,1e-10)

def curve(score,error,mass):
    s=np.asarray(score,float).ravel();e=np.asarray(error,float).ravel();m=np.asarray(mass,float).ravel()
    if not (np.isfinite(s).all() and np.isfinite(e).all() and np.isfinite(m).all()):raise ValueError('Nonfinite frontier input')
    order=np.argsort(s,kind='stable');s=s[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    E=np.cumsum(e[order])[ends];M=np.cumsum(m[order])[ends]
    return dict(t=s[ends],coverage=(ends+1)/len(s),risk=np.divide(E,M,out=np.full(len(E),np.inf),where=M>0))

def select(curves, cap, floor):
    aa,bb=curves;ts=np.union1d(aa['t'],bb['t']);ia=np.searchsorted(aa['t'],ts,side='right')-1;ib=np.searchsorted(bb['t'],ts,side='right')-1
    valid=(ia>=0)&(ib>=0);ia=np.maximum(ia,0);ib=np.maximum(ib,0)
    good=valid&(aa['coverage'][ia]>=floor)&(bb['coverage'][ib]>=floor)&(aa['risk'][ia]<=cap)&(bb['risk'][ib]<=cap)
    return float(ts[np.flatnonzero(good)[-1]]) if good.any() else None

def stats(y,f,mask):
    y=np.asarray(y,float);f=np.asarray(f,float);m=np.asarray(mask,bool);mass=y[m].sum();tot=y.sum()
    return dict(n=int(y.size),accepted_n=int(m.sum()),row_coverage=float(m.mean()),
                demand_coverage=float(mass/tot) if tot else None,
                wape=float(np.abs(y[m]-f[m]).sum()/mass) if mass else None)
